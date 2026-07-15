# Pipeline Parallelism in picotron

A from-scratch guide to pipeline parallelism (PP): **why** it exists, how the naive version wastes the
cluster, how micro-batching and **1F1B** fix it, and then the three modern schedules
(**Zero-Bubble**, **Interleaved 1F1B**, **ZB-V / V-shape**) that attack the leftover pipeline *bubble*.

Everything here is implemented and gradient-validated on the gloo/CPU test harness, so you can read the
code, run the tests, and reproduce the benchmarks without a GPU cluster.

- Built-in schedules: `pipeline_parallel.py` — `train_step_pipeline_afab`, `train_step_pipeline_1f1b`
- Advanced schedules: `pp_schedules.py` — `train_step_pipeline_zb` / `train_step_pipeline_interleaved` / `train_step_pipeline_zbv`
- Communication primitives: `pp_communications.py`
- Tests / benchmark: `tests/test_pipeline_parallel.py`, `tests/bench_pp_schedules.py`
- Slides: `teaching_slides.md` (`./render_slides.sh`)

---

## 1. Why pipeline parallelism?

A large model's parameters + activations + optimizer state don't fit on one device. picotron already
shards along **data (DP)**, **tensor (TP)**, **context (CP)** and **expert (EP)** axes. **Pipeline
parallelism** adds another: split the model **by depth** — give each device a contiguous block of
layers (a *stage*).

- **TP** splits *within* a layer → needs a fast all-reduce every layer (intra-node, NVLink).
- **PP** splits *across* layers → a device only sends its output activations to the **next** stage.
- That is a single point-to-point hop per stage boundary → cheap, and scales across nodes.

PP is how you train a model deeper than one node's memory with only neighbor-to-neighbor traffic. In
practice the PP degree is modest (industry uses **8–16**); you push the micro-batch count `m` up to
hide the bubble (below), and let DP/FSDP scale the rest.

### Split the model into stages

![pp split](figures/pp_split.svg)

Forward: stage `s` computes, sends activations to `s+1`. Backward: stage `s+1` sends input-gradients
back to `s`. Only neighbors talk — that is `pipeline_communicate(send_forward / recv_forward / …)`.
Each rank holds a `PipelineParallel` stage: `embedding` (first stage), some `decoder_layers`,
`final_norm` + `final_proj` (last stage).

---

## 2. The naive pipeline wastes `p-1` of every `p` devices

Run one batch straight through the stages: stage 0 works while 1, 2, 3 wait; then stage 1 while the
rest wait; forward down the stages, backward back up.

![naive pipeline](figures/naive_pipeline.svg)

At any instant **one** of `p` stages is busy → utilization `~ 1/p`. With `p = 4` we waste 75% of the
cluster. We need all stages busy at once.

---

## 3. Micro-batching = pipelining (GPipe / AFAB)

Split the global batch into `m` **micro-batches** and feed them back-to-back. While stage 0 starts
micro-batch 1, stage 1 works on micro-batch 0 — the stages **overlap**.

![afab](figures/afab.svg)

**AFAB** (all-forward-all-backward) = `train_step_pipeline_afab`: push all `m` forwards, then all `m`
backwards. Simple — but it keeps **every** micro-batch's activations alive until backward, so
activation memory grows linearly with `m`.

### The bubble, quantified

The pipeline still needs `p-1` steps to **fill** and `p-1` to **drain** — the triangles where some
devices idle. Relative to the `m` useful steps:

```
bubble fraction = idle / busy = (p - 1) / m
```

- More micro-batches (`m` ↑) → thinner triangles → smaller bubble… but AFAB's activation memory ↑.
- More stages (`p` ↑) → bigger bubble.

So we want large `m` for throughput — but AFAB makes that expensive in memory. Enter 1F1B.

---

## 4. 1F1B: same bubble, bounded memory

Reorder so each stage, after a short warmup, does **one forward then one backward** in lockstep. Each
backward frees the activations of the oldest in-flight micro-batch, so a stage holds only `~p`
micro-batches' activations (not `m`).

![1f1b](figures/bubble_1f1b.svg)

Same `(p-1)/m` bubble as AFAB, but activation memory is capped by pipeline **depth**, not `m`. This is
picotron's default (`train_step_pipeline_1f1b`) and the foundation everything below builds on.

> Measured at p=8 (below): AFAB and 1F1B have the **same** throughput, but at `m=64` AFAB peaks at
> `15912 MB` vs 1F1B's flat `2207 MB` — a 7× memory difference. This is why real training always uses
> 1F1B, never AFAB.

---

## 5. Shrinking the bubble — three modern schedules

`pp_schedules.py` adds three schedules from the last two years of PP research, each trading a different
resource to shrink the `(p-1)/m` bubble:

| schedule | source | bubble | trades | wrapper + entry point |
|---|---|---|---|---|
| **Zero-Bubble (ZB-H1)** | Qi et al., ICLR 2024 ([2401.10241](https://arxiv.org/abs/2401.10241)) | `≈ 0` | extra activation memory for the W queue (~1× backward FLOPs) | `PipelineParallel` + `train_step_pipeline_zb` |
| **Interleaved 1F1B** | Narayanan et al., Megatron-LM ([2104.04473](https://arxiv.org/abs/2104.04473)) | `(p-1)/(m·v)` | `v`× more comm, needs `m % p == 0` | `InterleavedPipelineParallel` + `train_step_pipeline_interleaved` |
| **ZB-V (V-shape)** | Qi et al., ICLR 2024 ([2401.10241](https://arxiv.org/abs/2401.10241)) | `≈ 0` | 2 chunks/rank (params) + scheduling complexity; ~1F1B memory, no `v`× comm | `VShapePipelineParallel` + `train_step_pipeline_zbv` |

`p` = pipeline stages, `m` = micro-batches per step, `v` = virtual stages per rank.

### Lever 1 — Zero-Bubble (ZB-H1)

![zero bubble](figures/zero_bubble.svg)

A backward fuses **B** (gradient w.r.t. the stage *input* — the previous stage needs it, so it's on the
critical path) and **W** (gradient w.r.t. the *weights* — only needed before `optimizer.step()`, so
reschedulable). ZB queues the comm-free **W** work and drains it into the bubbles 1F1B would spend
idle. The **B** communication pattern is identical to 1F1B, so the schedule stays deadlock-free.

We split at the **Linear** level (`_DeferredLinear` in `pipeline_parallel.py`): the B pass returns
`grad_x = grad_y @ W` and stashes `(x, grad_y)` per Linear; the W pass later computes
`grad_W = grad_yᵀ @ x` directly from those saved tensors. Each matmul runs once, so this is a true
**~1× backward FLOPs** split (not a 2× double-traversal) and is bit-exact (`grad_diff = 0`). The cost
is extra activation memory: the W queue holds `(x, grad_y)` for every in-flight micro-batch.

### Lever 2 — Interleaved 1F1B (virtual pipeline)

![interleaved](figures/interleaved.svg)

Give each rank `v` **non-contiguous** chunks. With `G = p·v` virtual stages laid out round-robin
(virtual stage `g` on rank `g % p`, chunk `g // p`), each stage is `v`× smaller, so the fill/drain
triangles are `v`× thinner → bubble `(p-1)/(m·v)`. Cost: `v`× more cross-rank activation hops, and it
requires `m % p == 0`.

### Lever 3 — ZB-V (V-shape Zero-Bubble)

![zb-v placement](figures/zbv.svg)

Interleaved shrinks the bubble but pays `v`× comm; ZB-H1 removes the bubble but needs extra memory.
**ZB-V combines both** with a V-shaped placement. Each rank holds exactly **2** of the `G = 2p` chunks:
device `d` owns down-leg chunk `d` and up-leg chunk `2p-1-d` (`dev(g) = g if g < p else 2p-1-g`).
Forward walks `0→1→…→2p-1`, whose devices trace `0..p-1` then `p-1..0` — a "V"; the bottom turn
(`chunk p-1 → p`, both on device `p-1`) is a **local** hand-off, and the first + last chunk both live on
device 0 (the loss returns to where the input entered). Every backward is split into **B + W** (reusing
`_DeferredLinear`), and the comm-free W work fills the bubbles → **≈ zero bubble at ~1F1B memory**,
without interleaved's `v`× communication.

The optimal ZB-V ordering is intricate, so instead of a hand-written per-rank step table we **generate
the schedule deterministically** from `(p, m)` (a small list scheduler with priority B > F > W, an
in-flight cap for ~1F1B memory, and W drained into idle rounds — identical on every rank). Every
cross-device transfer is pinned to the round its producer runs, so `vshape_pipeline_communicate` posts
the matching send/recv **in the same round on both endpoints** — deadlock-free by construction and
bit-exact vs the full-model reference (`grad_diff = 0` at `p = 2/4/8`). The local turn is graph-cut like
any stage boundary (the consumer sees a fresh leaf), so each chunk backprops in isolation.

### Beyond these three — DualPipe (intentionally omitted)

DeepSeek-V3's **DualPipe** ([2412.19437](https://arxiv.org/abs/2412.19437)) pushes further: it runs
**two micro-batch streams in opposite directions**, each rank holding the two stages symmetric about
the middle so a forward of one stream fills the other's backward bubble (`≈ (p-1)/(2m)`). It is
powerful but **deliberately not implemented here**: it costs ~2× parameter memory (replicated stages),
needs per-direction communicators with careful CUDA-stream / connection ordering to stay deadlock-free,
and only pays off in a narrow regime (very large `p`, or large MoE models where it hides cross-node
expert-parallel comm). That machinery dwarfs the rest of this module, so we keep picotron's PP small
and readable and leave DualPipe as further reading.

---

## 6. Deadlock-free ring communication

Every schedule fuses its point-to-point ops into a single `dist.batch_isend_irecv` group
(`pipeline_communicate`, `interleaved_pipeline_communicate` in `pp_communications.py`). Synchronous
ring P2P otherwise deadlocks when every rank
blocks on a send whose matching receive sits behind the partner's own send; batching all ops into one
NCCL group is matched and deadlock-free. Every `send` in a schedule is count-matched by exactly one
`recv`, so there is never a phantom op to hang on.

For NCCL at larger `p`, pass `device_id` to `init_process_group` (see `tests/bench_pp_schedules.py`):
without it NCCL guesses the device from the global rank and lazily spins up per-pair communicators,
which can hang the pipeline at `p ≥ 8`.

---

## 7. Tests — bit-exact gradients

`tests/test_pipeline_parallel.py` builds a tiny Llama, takes a **non-pipelined** full-model gradient
reference on the same weights + micro-batches, shards the *same* modules across ranks, runs each
schedule, and compares the gradients each rank owns. Runs on gloo/CPU (no GPU needed):

```bash
torchrun --nproc_per_node 2 tests/test_pipeline_parallel.py   # p=2
torchrun --nproc_per_node 4 tests/test_pipeline_parallel.py   # p=4
torchrun --nproc_per_node 8 tests/test_pipeline_parallel.py   # p=8
```

Zero-Bubble, Interleaved and ZB-V all match the reference **exactly** (`grad_diff = 0`) at `p = 2/4/8`.

---

## 8. Benchmark — bubble vs wall clock

`tests/bench_pp_schedules.py` times one full `train_step` per schedule on the real picotron Llama stage
and prints measured step time, tokens/s, peak memory, and the empirical vs analytical bubble:

```bash
# CPU smoke test (bubble invisible — compute dominates)
torchrun --nproc_per_node 4 tests/bench_pp_schedules.py --layers 16 --grad-acc 16 --hidden 256
# GPU: pick backend + dtype + a model big enough that compute dominates launch overhead
torchrun --nproc_per_node 8 tests/bench_pp_schedules.py --backend nccl --dtype bf16 \
    --layers 32 --hidden 1024 --inter 4096 --seq 1024 --mbs 1 --grad-acc 32
```

Measured on **8× L4 (p=8, bf16, 32-layer / hidden-1024 / seq-1024)**, sweeping `m`:

| m | 1f1b | zb | interleaved | afab peak MB | 1f1b peak MB |
|---|---|---|---|---|---|
| 16 | 1.00× (bubble 0.32) | 1.00× (bubble 0.32) | **1.03×** (bubble 0.31) | 4103 | **2190** |
| 32 | 1.00× | 0.99× | 0.98× | 8039 | **2190** |
| 64 | 1.00× | 0.99× | 0.95× | 15912 | **2207** |

Takeaways: **(1)** 1F1B matches AFAB's throughput at a fraction of the memory (AFAB grows linearly with
`m`, 1F1B is flat). **(2)** interleaved wins most at small `m` / large `p` (biggest bubble). **(3)**
ZB-H1 ties 1F1B here: its true ~1× split removed the old 2× tax, but the framework-level per-Linear
W-deferral has enough Python overhead to roughly cancel the bubble-fill on L4, while costing more
activation memory — an honest ZB-H1 result on commodity GPUs.

---

## 9. Configuration / training integration

`create_config.py --pp_engine {afab,1f1b,zb,interleaved,zbv}` (with `--num_virtual_stages` for the
interleaved engine) writes the choice into the config. In `train.py`:

- `afab`, `1f1b`, `zb` are fully wired — Zero-Bubble is a drop-in over 1F1B because it reuses the
  `PipelineParallel` stage, so it shares the existing HF weight-materialization / checkpoint path.
- `interleaved` and `zbv` use multi-chunk stage wrappers (`InterleavedPipelineParallel` /
  `VShapePipelineParallel`), which the HF checkpoint-materialization path does not yet understand. Both
  are implemented and gradient-validated on the gloo/CPU and NCCL/GPU harnesses; wiring their
  multi-chunk layout into the HF training loop is left as follow-up.

---

## 10. Teaching slides

`teaching_slides.md` (Marp) walks through all of the above with the figures in `figures/`. Render to
HTML / PDF / PPTX:

```bash
./render_slides.sh
```
