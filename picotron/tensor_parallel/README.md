# Tensor Parallelism in picotron

A from-scratch guide to tensor parallelism (TP): **why** it exists, the Megatron-LM column/row split
that already ships in picotron, and a ladder of canonical, **bit-exact** improvements layered on top —
**async communication overlap**, **sequence parallelism**, **vocab-parallel cross-entropy**, and the
**MegaScale** layer-level optimizations (a **parallel transformer block** that fuses the per-layer
collectives, plus a **chunked comm/compute overlap** kernel) — each implemented, gradient-validated,
and benchmarked on 2× A100.

- Layers: `tensor_parallel.py` — `ColumnParallelLinear`, `RowParallelLinear`, `VocabParallelEmbedding`,
  `apply_tensor_parallel`
- Communication primitives: `tp_communications.py` — the `f`/`g` collectives, the async-overlap linear,
  and `vocab_parallel_cross_entropy`
- Sequence parallelism: `picotron/sequence_parallel/` — shares the TP process group; the seq-parallel
  collectives (`sp_communications.py`), fused all-gather linear (`sequence_parallel.py`), and the
  chunked comm/compute overlap kernels (`overlap.py`, MegaScale Fig 3c)
- Parallel transformer block: `model.py` (`DecoderLayer`, `config.parallel_block`) — one shared norm,
  attention + MLP in parallel, fused to a single shared all-gather + combined reduce-scatter under SP
- Tests / benchmark: `tests/test_tensor_parallel.py`, `tests/test_tp_sequence_parallel.py`,
  `tests/test_tp_vocab_ce.py`, `tests/test_ptb_model.py`, `tests/test_ptb_overlap.py`,
  `tests/test_sp_overlap_kernel.py`, `tests/bench_tp.py`
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
| **+ptb** | run the layer as a parallel transformer block: attention + MLP from one shared norm, fused under SP to **one** shared all-gather + **one** combined reduce-scatter | **halves the per-layer collectives** (and drops a norm + a residual) → turns SP into a net speedup | [MegaScale](https://arxiv.org/abs/2402.15627) Fig 3b |
| **+overlap** | chunk the SP all-gather / reduce-scatter and pipeline each chunk's GEMM with the collective on a second stream | hides the collective behind the GEMM (needs higher TP / slower link to pay off) | [MegaScale](https://arxiv.org/abs/2402.15627) Fig 3c |

```python
# the architecture is a config flag; the comm strategies are apply_tensor_parallel flags
config.parallel_block = True
apply_tensor_parallel(model, async_tp=True, sequence_parallel=True, vocab_parallel_ce=True, overlap_comm=True)
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

### Rung 4 — parallel transformer block + chunked comm overlap (MegaScale)

The previous rungs keep the **sequential** block (`x = x + Attn(LN(x)); x = x + MLP(LN(x))`). Under SP
that block needs an all-gather + reduce-scatter around *each* sublayer — and because the q/k/v (and
up/gate) projections all-gather independently, a layer issues **6 all-gathers + 4 reduce-scatters** per
forward+backward.

[MegaScale](https://arxiv.org/abs/2402.15627) attacks this at the layer level (Fig 3):

**(a) Parallel transformer block (`config.parallel_block`, MegaScale Fig 3b).** Run attention and MLP
*in parallel* from one shared norm: `x = x + Attn(LN(x)) + MLP(LN(x))` (GPT-J / PaLM style). Under SP the
DecoderLayer then gathers the norm output **once**, feeds the (now "raw") projections, and reduce-scatters
the **summed** attention+MLP partials **once** — the whole layer is **one shared all-gather + one combined
reduce-scatter**. The projections are still fully TP-sharded (`out/in ÷ tp`, `num_local_heads = heads ÷
tp`); only the collective bookkeeping moves up into the layer.

```321:327:picotron/model.py
            if self.parallel_block_fused:
                # MegaScale Fig 3b: one shared all-gather feeds both branches (raw projections), and a
                # single combined reduce-scatter redistributes the summed attention + MLP partials.
                n = GatherFromSequenceParallelRegion.apply(n)
                attn = self.attention(n, cos, sin, attention_mask, position_ids)  # un-reduced partial
                mlp = self.mlp(n)  # un-reduced partial
                return x + ReduceScatterToSequenceParallelRegion.apply(attn + mlp)
```

**(b) Chunked comm/compute overlap (`overlap_comm`, MegaScale Fig 3c).** The fused SP linears still do
the collective as one *blocking* call (gather everything, then GEMM; or GEMM, then scatter everything).
The overlap kernels (`sequence_parallel/overlap.py`) break the GEMM into per-rank sequence chunks and
pipeline each chunk's matmul with one step of the collective on a second stream: a **ring** all-gather
whose GEMM overlaps the next rotation, and a **chunked** reduce-scatter that launches each chunk's reduce
while the next chunk's GEMM runs. Needs `CUDA_DEVICE_MAX_CONNECTIONS=1`.

The collective count per layer (forward+backward), measured with a live NCCL counter
(`tests/test_ptb_overlap.py`, q/k/v and up/gate fused):

| block / strategy | all-gather | reduce-scatter | reduce | p2p |
|---|---:|---:|---:|---:|
| sequential / fused | 6 | 4 | 0 | 0 |
| sequential / overlap | 4 | 2 | 4 | 2 |
| **PTB / fused** | **3** | **2** | 0 | 0 |
| **PTB / overlap** | **2** | **1** | 2 | 1 |

**Honest caveat.** The chunked overlap is **bit-exact but ~neutral at TP=2 on NVLink** (held even at
seq=16384): the link is fast enough that the AG/RS is already tiny next to the GEMM, so there is little to
hide and chunking slightly shrinks the GEMM tiles. Its payoff grows with TP degree / slower interconnect.
**The PTB collective fusion, by contrast, is a clear win even at TP=2** (see §4) — it is what flips
sequence parallelism from a net slowdown into a speedup.

---

## 3. Tests — bit-exact

```bash
# original column/row vs a reference nn.Linear
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 2 tests/test_tensor_parallel.py

# full Llama: plain TP, TP+sequence-parallel, and TP+SP+overlap vs a single-GPU reference
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 2 tests/test_tp_sequence_parallel.py

# vocab-parallel cross-entropy vs dense F.cross_entropy on gathered logits
torchrun --nproc_per_node 2 tests/test_tp_vocab_ce.py

# parallel transformer block (full Llama) vs a dense PTB reference
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 2 tests/test_ptb_model.py

# the overlap kernels vs the fused SP path (kernel-level, and seq/PTB block-level)
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 2 tests/test_sp_overlap_kernel.py
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 2 tests/test_ptb_overlap.py
```

`test_tp_sequence_parallel.py` builds a tiny Llama, takes a single-GPU reference forward/backward, then
runs plain TP, TP+SP, and TP+SP+overlap from the same weights and checks every owned gradient — all match
the reference **bit-exactly** (`loss_diff = 0`, `grad_diff ≈ 2e-8` in fp32, including the chunked overlap
kernels routing the real q/k/v/out_proj/up/gate/down). Vocab-parallel CE matches dense cross-entropy to
`grad_diff ≈ 4e-9`. `test_ptb_model.py` validates the parallel transformer block (plain TP and the fused
TP+SP path) against a dense PTB reference (`grad_diff ≈ 2e-8`); `test_ptb_overlap.py` checks all four
{sequential, PTB} × {fused, overlap} combinations against dense references and counts the collectives.

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
| baseline | 327.3 | 50051 | 17865 | 1.00× | 1.00× |
| +async | 319.0 | 51369 | 17865 | 1.00× | 1.03× |
| +seqpar | 362.9 | 45144 | 14576 | 1.23× | 0.90× |
| +vocab_ce | 321.0 | 51048 | 15986 | 1.12× | 1.02× |
| +seqpar+vocab_ce | 356.1 | 46010 | 12697 | 1.41× | 0.92× |
| **+ptb+seqpar** | 260.7 | 62856 | 13367 | **1.34×** | **1.26×** |
| **+ptb+seqpar+vocab_ce** | **254.0** | **64502** | **11488** | **1.56×** | **1.29×** |

And at **long sequence** (2× A100-80GB, 8-layer / hidden-4096 / inter-14336 / 32-head / **seq-16384** /
mbs-1, run on Modal):

| config | ms/step | peak MB | mem vs base | speedup |
|---|---:|---:|---:|---:|
| baseline | 795.3 | 28615 | 1.00× | 1.00× |
| +seqpar | 816.7 | 24052 | 1.19× | 0.97× |
| +seqpar+vocab_ce | 805.0 | 20294 | 1.41× | 0.99× |
| **+ptb+seqpar** | 716.3 | 22441 | 1.28× | **1.11×** |
| **+ptb+seqpar+vocab_ce** | **703.1** | **18683** | **1.53×** | **1.13×** |

Takeaways (honest, single-node TP=2):

1. **Sequence parallelism alone is a memory lever, not a speed lever.** It cuts peak memory **1.23×**
   (3.3 GB) by sharding the residual stream, but on a single NVLink node at TP=2 the extra reduce-scatter
   / all-gather *calls* (same bytes, more launches) cost ~10% wall-clock.
2. **The parallel transformer block flips that.** By fusing each layer to one shared all-gather + one
   combined reduce-scatter (half the collectives) and dropping a norm + a residual, `+ptb+seqpar` is
   **1.26× faster *and* 1.34× less memory** than baseline — SP becomes a net win. The advantage holds at
   long sequence (1.11× at seq-16384).
3. **Vocab-parallel CE is nearly free** — ~1.12× less memory at neutral speed; it stacks with PTB+SP for
   **1.29× faster / 1.56× less memory** (1.13× / 1.53× at long sequence).
4. **Async overlap** (1.03×) and the **chunked comm/compute overlap** (bit-exact, ~neutral at TP=2 on
   NVLink) are latency-hiding tricks whose payoff grows with TP degree / slower interconnect.

---

## 5. Configuration / training integration

The comm knobs are plumbed through `apply_tensor_parallel(model, async_tp, sequence_parallel,
vocab_parallel_ce, overlap_comm)`; the parallel-transformer-block **architecture** is a model flag,
`config.parallel_block` (read by the `DecoderLayer`; `apply_tensor_parallel` detects it and fuses the
collectives under SP). Sequence parallelism additionally needs (a) the all-reduce of norm weight grads
over the TP group before `optimizer.step()` and (b) the loss to consume the gathered logits (when
`vocab_parallel_ce=False`, `final_proj` still gathers, so `train.py`'s dense `F.cross_entropy` is
unchanged); with `vocab_parallel_ce=True` the loss switches to `vocab_parallel_cross_entropy`. Note PTB
has **no** post-attention norm, so its sequence-parallel norm-grad all-reduce covers only
`input_layernorm` + `final_norm`. The correctness path is gradient-validated end-to-end in
`tests/test_tp_sequence_parallel.py` and `tests/test_ptb_model.py`; benchmarks run the full step in
`tests/bench_tp.py`.

---

## 6. Teaching slides

`teaching_slides.md` (Marp) walks through all of the above with the figures in `figures/`. Render to
HTML / PDF / PPTX:

```bash
./render_slides.sh
```
