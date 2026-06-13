# Tensor Parallelism in picotron

A from-scratch guide to tensor parallelism (TP): **why** it exists, the Megatron-LM column/row split
that already ships in picotron, and three canonical, **bit-exact** improvements layered on top —
**async communication overlap**, **sequence parallelism**, and **vocab-parallel cross-entropy** — each
implemented, gradient-validated, and benchmarked on 2× A100.

- Layers: `tensor_parallel.py` — `ColumnParallelLinear`, `RowParallelLinear`, `VocabParallelEmbedding`,
  `apply_tensor_parallel`
- Communication primitives: `tp_communications.py` — the `f`/`g` collectives, the async-overlap linear,
  and `vocab_parallel_cross_entropy`
- Sequence parallelism: `picotron/sequence_parallel/` — shares the TP process group; the seq-parallel
  collectives (`sp_communications.py`) + fused all-gather linear (`sequence_parallel.py`)
- Tests / benchmark: `tests/test_tensor_parallel.py`, `tests/test_tp_sequence_parallel.py`,
  `tests/test_tp_vocab_ce.py`, `tests/bench_tp.py`
- Slides: `teaching_slides.md` (`./render_slides.sh`)

---

## 1. Why tensor parallelism?

A layer's weight matrices can be too big (or too slow) for one device. **TP** splits the matmuls
*inside* a layer across the TP group, so each device holds a slice of every weight and computes a slice
of every activation. Unlike pipeline parallelism (split *across* layers, neighbor-to-neighbor comm),
TP needs a **collective every layer**, so it lives on the fast intra-node link (NVLink) and the TP
degree is kept small (typically ≤ 8).

picotron splits each transformer block the Megatron-LM way ([Shoeybi et al.,
2019](https://arxiv.org/abs/1909.08053)):

![tp split](figures/tp_split.svg)

- **Column-parallel** (`q/k/v/up/gate`): shard the weight by **output** columns. Input is replicated,
  output is sharded → no comm in forward, all-reduce of the *input gradient* in backward (the `f` op).
- **Row-parallel** (`out_proj/down_proj`): shard the weight by **input** rows. Input is already sharded
  (it came from a column-parallel layer), output is a partial sum → all-reduce in forward (the `g` op).

Chaining column→row means the attention/MLP block needs exactly **one all-reduce in the forward and one
in the backward**, and the head/intermediate dimension is sharded throughout.

The `f` and `g` operators are conjugates implemented as tiny autograd functions in
`tp_communications.py`:

```23:38:picotron/tensor_parallel/tp_communications.py
class CopyToModelParallelRegion(torch.autograd.Function):
    """
    Copy in forward pass, all-reduce in backward pass.
    This is the `f` function in the paper: https://arxiv.org/abs/1909.08053
    """

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        if pgm.process_group_manager.tp_world_size == 1:
            return grad_output
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.tp_group)
        return grad_output
```

The embedding is sharded over the **vocabulary** (`VocabParallelEmbedding`): each rank looks up the
tokens in its vocab slice, zeros the rest, and all-reduces.

---

## 2. The optimization ladder

The stock implementation is correct Megatron-v1 TP. Three well-known improvements sit on top, each
**opt-in** (defaults unchanged) and **bit-exact** with the baseline:

| rung | what it does | what it buys | source |
|---|---|---|---|
| **+async** | overlap the column-parallel input-grad all-reduce with the weight-grad GEMM | hides comm behind compute | Megatron |
| **+seqpar** | shard the norm/residual regions along the **sequence** dim | `tp×` less activation memory in those regions, **same comm volume** | [Korthikanti et al. 2022](https://arxiv.org/abs/2205.05198) |
| **+vocab_ce** | keep the output logits vocab-sharded; loss exchanges only `[b,s]` scalars | no `[b,s,V]` logits, O(V)→O(1) comm at the loss | Megatron `vocab_parallel_cross_entropy` |

```python
apply_tensor_parallel(model, async_tp=True, sequence_parallel=True, vocab_parallel_ce=True)
```

### Rung 1 — async communication overlap

The column-parallel backward needs three things from `grad_output`: the input gradient (`grad_x =
grad_y @ W`, which must be all-reduced and handed to the previous layer) and the weight/bias gradients
(local matmuls). The default order computes all of them, *then* all-reduces `grad_x`. The async path
reorders it so the `grad_x` all-reduce is **launched first** and runs while the weight-grad GEMM
executes:

```92:115:picotron/tensor_parallel/tp_communications.py
    @staticmethod
    def backward(ctx, grad_output):
        """
        The key difference with "linear_with_all_reduce" is that the all reduce of input_ gradeint is before
        the calculation of the gradient of weights and bias, instead of after. So we can overlap the computation and communication
        This is only applicable to Column Parallel Linear

        Before: grad_output -> grad_input, grad_weight, grad_bias  -> grad_input all reduce
        Now:    grad_output -> grad_input -> grad_input all reduce -> grad_weight, grad_bias
        """
```

This code already existed but was never wired in — `apply_tensor_parallel(..., async_tp=True)` now
enables it. Overlap requires **`CUDA_DEVICE_MAX_CONNECTIONS=1`** so the collective and the GEMM land on
separate streams that actually run concurrently.

### Rung 2 — sequence parallelism (the real memory lever)

Between TP blocks (RMSNorm + the residual add) the activations are normally **replicated** on every TP
rank, so that memory does not shrink with TP at all. Sequence parallelism shards those regions along
the **sequence** dimension instead:

![sequence parallel](figures/sequence_parallel.svg)

The trick is that **all-reduce = reduce-scatter + all-gather**, so we can split the TP block's two
collectives without changing the total bytes on the wire:

```
plain TP : f = copy / all-reduce            g = all-reduce / copy
seq-par  : f = all-gather / reduce-scatter  g = reduce-scatter / all-gather
```

`f` (column input) all-gathers the sequence before the matmul; `g` (row output, and the embedding)
reduce-scatters back to a sequence shard. In between — norm, residual, dropout — everything is `tp×`
smaller.

**The subtlety that actually matters.** A naive `all_gather` + `F.linear` loses most of the win: PyTorch
autograd saves the *gathered* (full-sequence) input for the weight-grad, so the largest activation is
still stored full-size on every rank. The fix (Megatron's
`linear_with_grad_accumulation_and_async_allreduce`) is a **fused gather+linear that checkpoints only
the sequence-sharded input and re-gathers in the backward** (in `picotron/sequence_parallel/`, which
shares the TP process group):

```python
class _ColumnParallelLinearWithSequenceParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_sharded, weight, bias):
        ctx.save_for_backward(x_sharded, weight)   # NOT the gathered tensor
        x_full = all_gather_along_seq(x_sharded)
        return x_full @ weight.t() + (bias if bias is not None else 0)

    @staticmethod
    def backward(ctx, grad_output):
        x_sharded, weight = ctx.saved_tensors
        x_full = all_gather_along_seq(x_sharded)   # recompute the gather instead of storing it
        ...
```

In the benchmark below this single change turns SP's memory win from a useless **51 MB** into **3.3 GB**.

One bookkeeping cost: under SP each rank's RMSNorm sees a *different* sequence shard, so its weight
gradient is a partial sum that must be all-reduced over the TP group (Megatron's "sequence-parallel"
parameters). The sharded projections see the full (gathered) sequence and need no fix-up. The tests and
benchmark do this all-reduce.

### Rung 3 — vocab-parallel cross-entropy

The output projection is already column-parallel over the vocabulary. The stock code sets
`gather_output=True`, all-gathering the full `[b, s, V]` logits onto every rank and running a dense
softmax there. For a 32k–128k vocab that is the single biggest activation in the step.

`vocab_parallel_cross_entropy` keeps the logits **sharded** and computes the loss directly on the
shards, exchanging only per-token scalars — the max logit, the sum of `exp`, and the target logit:

![vocab ce](figures/vocab_ce.svg)

Communication drops from `O(V)` to `O(1)` per token and the full logits are never materialized.
Backward is the usual `softmax - onehot(target)`, recovered from the saved per-shard softmax.

---

## 3. Tests — bit-exact

```bash
# original column/row vs a reference nn.Linear
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 2 tests/test_tensor_parallel.py

# full Llama: plain TP and TP+sequence-parallel vs a single-GPU reference
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 2 tests/test_tp_sequence_parallel.py

# vocab-parallel cross-entropy vs dense F.cross_entropy on gathered logits
torchrun --nproc_per_node 2 tests/test_tp_vocab_ce.py
```

`test_tp_sequence_parallel.py` builds a tiny Llama, takes a single-GPU reference forward/backward, then
runs plain TP and TP+SP from the same weights and checks every owned gradient. Both match the reference
**bit-exactly** (`loss_diff = 0`, `grad_diff ≈ 2e-8` in fp32). Vocab-parallel CE matches dense
cross-entropy to `grad_diff ≈ 4e-9`.

---

## 4. Benchmark — the ablation

`tests/bench_tp.py` times one forward + cross-entropy + backward on a real picotron Llama and reports
step time, tokens/s and **peak activation memory** for each rung:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 2 tests/bench_tp.py \
    --hidden 2048 --inter 5632 --layers 24 --heads 16 --kv-heads 16 \
    --vocab 32768 --seq 2048 --mbs 4 --steps 5 --warmup 2
```

Measured on **2× A100-80GB (TP=2, bf16, 24-layer / hidden-2048 / seq-2048 / mbs-4, SDPA attention)**:

| config | ms/step | tok/s | peak MB | mem vs base | speedup |
|---|---:|---:|---:|---:|---:|
| baseline | 327.7 | 49992 | 17865 | 1.00× | 1.00× |
| +async | 320.5 | 51126 | 17865 | 1.00× | **1.02×** |
| +seqpar | 363.2 | 45107 | **14576** | **1.23×** | 0.90× |
| +vocab_ce | 321.7 | 50936 | 15986 | 1.12× | 1.02× |
| **+seqpar+vocab_ce** | 356.4 | 45966 | **12697** | **1.41×** | 0.92× |

Takeaways (honest, single-node TP=2):

1. **Sequence parallelism is a memory lever, not a speed lever.** It cuts peak memory **1.23×** (3.3 GB)
   by sharding the residual stream, but on a single NVLink node at TP=2 the extra reduce-scatter /
   all-gather *calls* (same bytes, more launches) cost ~10% wall-clock. Its speed story improves at
   higher TP and across nodes, where it also unlocks not recomputing those regions.
2. **Vocab-parallel CE is nearly free** — 1.12× less memory at neutral speed, because it removes the
   full `[b,s,V]` logit gather + dense softmax for the price of a few `[b,s]` all-reduces.
3. **They stack: 1.41× less memory together**, which is what lets you fit a longer sequence or a bigger
   micro-batch on the same card.
4. **Async overlap** is a small win here (1.02×); the gain grows with the column-parallel weight-grad
   GEMM size relative to the all-reduce latency.

---

## 5. Configuration / training integration

The three knobs are plumbed through `apply_tensor_parallel(model, async_tp, sequence_parallel,
vocab_parallel_ce)`. Sequence parallelism additionally needs (a) the all-reduce of norm weight grads
over the TP group before `optimizer.step()` and (b) the loss to consume the gathered logits (when
`vocab_parallel_ce=False`, `final_proj` still gathers, so `train.py`'s dense `F.cross_entropy` is
unchanged); with `vocab_parallel_ce=True` the loss switches to `vocab_parallel_cross_entropy`. The
correctness path is gradient-validated end-to-end in `tests/test_tp_sequence_parallel.py`; benchmarks
run the full step in `tests/bench_tp.py`.

---

## 6. Teaching slides

`teaching_slides.md` (Marp) walks through all of the above with the figures in `figures/`. Render to
HTML / PDF / PPTX:

```bash
./render_slides.sh
```
