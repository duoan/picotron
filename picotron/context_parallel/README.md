# Context Parallelism in picotron

A from-scratch guide to context parallelism (CP): **why** it exists, the ring-attention implementation
that already ships in picotron, and a second, **bit-exact** algorithm layered alongside it —
**DeepSpeed-Ulysses** — implemented, gradient-validated, and benchmarked on 2x A100.

- Sequence sharding: the input sequence is split contiguously across the CP group (`data.py`); every
  layer is per-token **except attention**, which must see all tokens.
- Ring attention: `context_parallel.py` — keep the sequence sharded, rotate K/V around the CP ring
  (P2P), merge blocks with an online softmax. Each block uses the **fused flash kernel** when
  `FLASH_ATTEN=1` (`RingFlashAttentionFunc`), and a readable pure-python reference otherwise
  (`RingAttentionFunc`).
- Ulysses attention: `ulysses.py` — all-to-all to a head-parallel layout, run one dense local attention
  over the full sequence, all-to-all back. Shares the **same CP process group**; it is a different
  algorithm on the same axis.
- Tests / benchmark: `tests/test_cp_ulysses.py`, `tests/test_cp_ring_flash.py`, `tests/bench_cp.py`
- Slides: `teaching_slides.md` (`./render_slides.sh`)

Both algorithms use the `cp_group` and the contiguous sequence shard, so they are interchangeable behind
one switch: `CP_ATTENTION=ring` (default) or `CP_ATTENTION=ulysses`.

## Why context parallelism

Activation memory and attention compute grow with sequence length, eventually past what one GPU holds.
TP shards the hidden dimension and SP shards the norm/residual regions, but the attention score matrix
is still `O(S^2)` per head. Context parallelism shards the **sequence** itself across a dedicated grid
axis (`cp`), so each rank stores and computes on `S/cp` tokens.

![context parallel](figures/cp_split.svg)

The catch is attention: token `i` attends to all tokens `<= i`, which now live on other ranks. The two
algorithms below are two different answers to "how do we do attention when no rank has the full
sequence."

## Ring vs Ulysses

![ring vs ulysses](figures/ring_vs_ulysses.svg)

### Ring attention (`CP_ATTENTION=ring`, default)

Keep Q/K/V sharded as `[b, h, S/cp, d]`. Walk the CP ring `cp` times: at each step compute attention of
the local Q against the K/V block currently held, then P2P-send K/V to the next rank and receive the
next block. An **online softmax** merges the per-block partial outputs without ever materializing the
full `S x S` score matrix. Communication is P2P and there is **no limit on the head count**, so ring
scales to very long sequences.

Each per-step block attention dispatches to the **fused flash kernel** (`RingFlashAttentionFunc`, using
flash's low-level forward/backward that return/accept the softmax LSE — exactly what the online-softmax
merge needs) when `FLASH_ATTEN=1`; with `FLASH_ATTEN=0` it falls back to a readable pure-python
matmul + softmax reference (`RingAttentionFunc`, fp32/CPU-friendly). Both share the same ring structure
(K/V rotation, causal-step logic, online-softmax merge).

### Ulysses attention (`CP_ATTENTION=ulysses`)

Instead of moving K/V, move the **layout**. Two all-to-alls bracket a normal attention:

```
in   [b, num_heads,    S/cp, d]      sequence sharded
a2a  [b, num_heads/cp, S,    d]      head sharded, full sequence   (scatter heads, gather seq)
attn standard dense/flash attention over the full sequence, this rank's heads
a2a  [b, num_heads,    S/cp, d]      back to sequence sharded      (scatter seq, gather heads)
```

Because the all-to-all just reshuffles the same elements (`num_heads/cp * S == num_heads * S/cp`), the
per-rank tensor size is unchanged. The win is that the attention itself is an **ordinary kernel** — it
composes with flash-attn for free and skips the online-softmax bookkeeping. Communication is two
all-to-alls per attention, independent of sequence length.

**The backward is the same collective.** An all-to-all's transpose is another all-to-all with the
scatter/gather dims swapped, so one small autograd function (`_SeqAllToAll`) covers both directions:

```python
class _SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, x, scatter_dim, gather_dim):
        ctx.scatter_dim, ctx.gather_dim, ctx.group = scatter_dim, gather_dim, group
        return _all_to_all(x, scatter_dim, gather_dim, group)

    @staticmethod
    def backward(ctx, grad):
        return (None, _all_to_all(grad, ctx.gather_dim, ctx.scatter_dim, ctx.group), None, None)
```

**The limitation:** Ulysses parallelizes over heads, so `num_heads` (per TP rank) must be divisible by
`cp_world_size`. It cannot scale CP past the head count; ring has no such limit. GQA is handled
upstream — picotron expands K/V to the query-head count before dispatch, so Ulysses all-to-alls full
head tensors and the K/V head count is never the constraint.

## Correctness

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 2 tests/test_cp_ulysses.py
```

`test_cp_ulysses.py` builds a tiny Llama, runs a full-sequence forward/backward as a single-GPU
reference, then shards the sequence contiguously (exactly like `data.py`) and reruns with
`CP_ATTENTION=ulysses`. Under pure CP every parameter is replicated and each rank sees only its token
shard, so each gradient is a partial sum; summing gradients over the CP group (with a sum-reduced loss)
recovers the reference exactly.

| check (fp32, SDPA, cp=2) | result |
| --- | --- |
| forward logits vs reference | `max_diff = 4.8e-7` |
| loss (CP-summed) vs reference | `diff = 1.5e-5` |
| gradients (CP-summed) vs reference | `max_diff = 7.2e-7` |

The flash ring path is validated the same way in `tests/test_cp_ring_flash.py` (bf16, since flash is
half-precision only, so it uses a relative tolerance):

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 2 tests/test_cp_ring_flash.py
```

| check (bf16, flash, cp=2) | relative diff |
| --- | --- |
| forward logits vs full-sequence flash reference | `7.5e-3` |
| loss (CP-summed) vs reference | `3.5e-5` |
| gradients (CP-summed) vs reference | `8.0e-3` |

## Benchmark (2x A100, bf16, cp=2, flash in both)

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 FLASH_ATTEN=1 torchrun --nproc_per_node 2 tests/bench_cp.py \
    --layers 4 --hidden 2048 --inter 8192 --heads 16 --kv-heads 16 --vocab 32768 --seq 8192
```

Both paths now use flash kernels (`RingFlashAttentionFunc` and flash Ulysses), so this is an
apples-to-apples, **algorithm-vs-algorithm** comparison:

| seq (per rank) | attention | ms/step | tok/s | peak MB | speedup |
| --- | --- | --- | --- | --- | --- |
| 4096 (2048/rank) | ring | 48.6 | 84,301 | 2,593 | 1.00x |
| 4096 (2048/rank) | **ulysses** | **35.0** | **117,010** | 2,627 | **1.39x** |
| 8192 (4096/rank) | ring | 77.3 | 105,922 | 4,364 | 1.00x |
| 8192 (4096/rank) | **ulysses** | **62.2** | **131,802** | 4,431 | **1.24x** |
| 16384 (8192/rank) | ring | 167.9 | 97,577 | 7,905 | 1.00x |
| 16384 (8192/rank) | **ulysses** | **135.6** | **120,828** | 8,039 | **1.24x** |

**How to read this.** Once ring uses flash too, the gap collapses from the earlier ~2.8x (python ring vs
flash Ulysses — an implementation artifact) to a real **~1.2-1.4x** at cp=2 on a single node. That
residual is genuinely algorithmic:

1. **Ulysses does less, denser work per step.** It runs *one* dense flash attention over the full
   sequence (this rank's heads), bracketed by two all-to-alls — versus ring's `cp` smaller flash calls
   plus per-step P2P that doesn't perfectly overlap.
2. **Ring's causal sharding here is the contiguous (non-zigzag) variant**, so rank `r` computes `r+1`
   blocks: the last rank does `cp` blocks while rank 0 does 1. Zigzag sharding would rebalance this and
   shrink the gap further (a future optimization).
3. **Memory is now ~equal** — with flash, ring no longer materializes large score buffers, so its only
   memory overhead is the rotating K/V block.

The durable takeaway is unchanged: **Ulysses is modestly faster when it applies** (single node,
`cp <= num_heads`), while **ring's value is reach** — P2P and no head-count cap let it scale CP beyond
`num_heads`, across nodes, and to sequences where even Ulysses's head-parallel full-sequence attention
would not fit.

### Scaling to very long sequences (cp=2)

Pushing the sequence length until both OOM (same model, 2x A100 80GB):

| seq | ring ms | ulysses ms | speedup | ring peak MB | ulysses peak MB | mem gap |
| --- | --- | --- | --- | --- | --- | --- |
| 16k | 166.6 | 135.7 | 1.23x | 7,905 | 8,039 | 134 |
| 32k | 441.7 | 343.8 | 1.28x | 14,987 | 15,256 | 269 |
| 64k | 1341.4 | 1003.2 | 1.34x | 29,152 | 29,689 | 537 |
| 128k | 4568.0 | 3310.3 | 1.38x | 57,482 | 58,556 | 1,074 |
| 256k | OOM | OOM | — | — | — | — |

Two slightly counterintuitive results:

1. **Ulysses's lead *grows* with sequence length** (1.23x -> 1.38x), it does not shrink. At fixed small
   `cp` the attention `O(S^2)` term dominates more as `S` grows, and ring's contiguous-causal load
   imbalance (rank 1 does 2 blocks, rank 0 does 1) makes its bottleneck rank pay relatively more, while
   Ulysses does one balanced dense flash per rank. **Long-seq at small `cp` favors Ulysses, not ring.**
2. **Ring's memory edge is real but small** (~2%): the gap grows linearly with `S` (134 -> 1,074 MB) —
   exactly the Ulysses all-to-all transient (it briefly holds both the sequence-sharded and gathered
   full-sequence layouts). It is far too small to move the OOM point; both die at ~256k.

So on a single node at `cp=2` there is **no sequence length where ring beats Ulysses** — longer only
widens Ulysses's lead. Ring earns its keep in the regime we *cannot* show on 2 GPUs: `cp > num_heads`,
or scaling across nodes where Ulysses's full-sequence all-to-all gather becomes the bottleneck.

### With `torch.compile` (`--compile`)

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 FLASH_ATTEN=1 torchrun --nproc_per_node 2 tests/bench_cp.py \
    --layers 4 --hidden 2048 --inter 8192 --heads 16 --kv-heads 16 --vocab 32768 --seq 8192 --compile
```

| seq | attention | eager ms | compile ms | Δ | compile speedup (uly vs ring) |
| --- | --- | --- | --- | --- | --- |
| 4k | ring | 48.6 | 44.8 | -8% | 1.00x |
| 4k | **ulysses** | **35.0** | **36.1** | +3% | **1.24x** |
| 8k | ring | 77.3 | 73.0 | -6% | 1.00x |
| 8k | **ulysses** | **62.2** | **60.7** | -2% | **1.20x** |
| 16k | ring | 167.9 | 161.1 | -4% | 1.00x |
| 16k | **ulysses** | **135.6** | **132.2** | -3% | **1.22x** |

`torch.compile` is a **modest win** (~3-8%, a little more for ring than Ulysses) and trims peak memory
~5% for both. It does **not** change the ranking: Ulysses keeps its ~1.2-1.4x lead, essentially the same
as eager. The hot path is collectives + opaque flash / custom-autograd ops, which dynamo can't fuse
(it graph-breaks at the ring P2P and the all-to-alls); compile only fuses the dense regions
(linear / RMSNorm / rotary / MLP), whose share grows with model depth/width — so expect more upside on a
full-size model than on this 4-layer attention microbench.

> **Benchmark gotcha (fixed in `bench_cp.py`).** The ring/ulysses switch is an `os.getenv("CP_ATTENTION")`
> read, which dynamo constant-folds **without a guard**. So timing both in one process made the 2nd model
> reuse the 1st's compiled graph (Ulysses silently reran ring's graph — visible as identical ms *and*
> identical peak memory). `bench_cp.py` now calls `torch._dynamo.reset()` before each `torch.compile` so
> every config re-traces correctly. (Sanity-check by confirming each attention reports its own peak MB.)

## How to choose

| Situation | Use |
| --- | --- |
| Single node, `cp <= num_heads`, want speed | **`CP_ATTENTION=ulysses`** |
| `cp > num_heads`, or extreme sequence length, or multi-node scaling | `CP_ATTENTION=ring` (default) |

Set `cp_size` in the training config and pick the algorithm with the `CP_ATTENTION` env var (consistent
with `CONTEXT_PARALLEL` / `FLASH_ATTEN`):

```bash
CP_ATTENTION=ulysses CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 8 train.py --config cfg.yaml
```

The Ulysses path is gradient-validated end-to-end in `tests/test_cp_ulysses.py`; benchmarks are
reproducible with `tests/bench_cp.py`.
