# Expert Parallelism (EP) for picotron

Expert Parallelism distributes the experts of a Mixture-of-Experts (MoE) FFN across the EP group:
rank `r` owns experts `[r * num_local_experts, (r + 1) * num_local_experts)`. A replicated router
picks the top-`k` experts per token; tokens are exchanged with an **all-to-all** (dispatch), run
through the local experts, and sent back with a second all-to-all (combine), then weighted by the
router probabilities.

EP is the 5th axis of parallelism (the grid is `DP × PP × CP × EP × TP`). It composes with the
others: non-expert weights are replicated and their grads sync over the `cp_dp` group, while expert
weights are sharded across EP and not all-reduced.

This module keeps the picotron spirit — readable `torch.distributed` collectives — while adding two
families of MoE-systems optimization:
- **overlap** (hide the all-to-all behind compute): [DeepEP](https://github.com/deepseek-ai/DeepEP),
  [MegaScale-MoE](https://arxiv.org/abs/2505.11432), [UniEP](https://arxiv.org/abs/2604.19241)
- **comm-volume reduction** (send fewer bytes): FP8 dispatch
  ([DeepEP](https://github.com/deepseek-ai/DeepEP)) and LatentMoE
  ([NVIDIA Nemotron](https://research.nvidia.com/labs/nemotron/LatentMoE/))

## Files

| File | What it does |
| --- | --- |
| `expert_parallel.py` | `MoELayer`: router, sharded experts, dispatch/combine, path selection, FP8 + LatentMoE |
| `ep_communications.py` | differentiable + async + **FP8** all-to-all primitives (`all_to_all`, `all_to_all_async`, `all_to_all_fp8`, ...) |
| `tiled_moe.py` | `_TiledMoEFunction`: the MegaScale-style token-tiled pipeline (fwd **and** bwd overlap) |
| `megakernel.py` | Triton fused expert FFN (forward) + explicit `dgrad`/`wgrad` grouped-GEMM (backward) |
| `deepep_backend.py` | optional [DeepEP](https://github.com/deepseek-ai/DeepEP) dispatch/combine backend (Hopper SM90+, auto-fallback) |

## Execution paths

`MoELayer.forward` picks one of three numerically-identical **routed** paths (see
`tests/test_expert_parallel.py`):

- **naive** — blocking `dispatch -> experts -> combine`.
- **tiled-N** — MegaScale-style token-tiled pipeline: split routed tokens into `N` tiles and
  pipeline dispatch / expert GEMM / combine across them. Enabled by `ep_num_tiles > 1`.
- **deepep** — DeepEP's CUDA dispatch/combine kernels (Hopper SM90+). Enabled by
  `ep_backend="deepep"`; auto-falls back to naive when unavailable.

**Shared-expert overlap is orthogonal**, not a separate path. The shared expert (DeepSeek-style) is
comm-free local compute, so `ep_overlap=True` runs it on a side CUDA stream that overlaps the
dispatch/combine of *whichever* routed path is active — naive, tiled, **or** deepep. It is a no-op
(serial, bit-exact) on CPU/gloo and when there is no shared expert. This is most effective on the
deepep path, whose comm runs at near-zero SM occupancy and so leaves the GPU free for the shared FFN.

Config knobs (`create_config.py` → config → `train.py` → `model_config`):

```jsonc
"model": {
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "num_shared_experts": 1,
    "ep_overlap": true,       // shared-expert overlap (hide dispatch)
    "ep_num_tiles": 4,        // 1 = off; >1 enables the tiled pipeline
    "ep_fp8_dispatch": false, // FP8 (E4M3) dispatch  -> ~2x fewer dispatch bytes
    "moe_latent_dim": 0,      // LatentMoE latent dim l (0/>=hidden = off) -> route/compute in l
    "ep_backend": "torch"     // "torch" (portable) | "deepep" (Hopper SM90+, auto-fallback)
}
```

### DeepEP backend (`ep_backend="deepep"`)

[DeepEP](https://github.com/deepseek-ai/DeepEP) replaces the `torch.distributed` all-to-all with
hand-written CUDA kernels that saturate NVLink/RDMA bandwidth at near-zero SM occupancy (its main win
is *freeing SMs for the GEMM* and the cross-node RDMA path, not raw 2-GPU NVLink bandwidth).
`deepep_backend.py` wraps DeepEP's classic intranode `Buffer` (NVLink) dispatch/combine as autograd
functions. `dispatch` sends each token once to every rank owning one of its top-k experts; we run the
grouped expert FFN, apply the gate weights, then `combine` sums each token's per-rank contributions
back. `dispatch` and `combine` are linear transposes, so the autograd backward of one is the other.

We deliberately use the **classic `Buffer`, not the V2 `ElasticBuffer`**: ElasticBuffer routes through
the NCCL GIN (in-network) transport, which isn't exposed on every cluster (e.g. Modal's single-node
H100 boxes abort at init with *"NCCL GIN is unavailable"*). The classic intranode path only needs
NVLink P2P. The gate (router) is not differentiated through DeepEP's non-differentiable weight
transport; expert/projection grads flow normally.

**DeepEP requires Hopper (SM90) or newer** — on older GPUs (e.g. A100 = sm_80) `deepep_available()`
is False and the layer transparently falls back to the torch backend (so `ep_backend="deepep"` is
always safe to set). Set `EP_DISABLE_DEEPEP=1` to force the fallback.

Validated on **2× H100 (NVLink) via Modal** — see [Results (2× H100, NVLink)](#results-2-h100-sxm5-80gb-nvlink--deepep-backend)
below. The build (DeepEP from source: `libibverbs-dev`, NCCL ≥ 2.30.4, CUDA 13 + `g++`) is the default
job of `modal_run.py`, the general cloud-GPU runner.

## How the overlap works (and what we learned from the papers)

MegaScale-MoE makes a point we re-discovered the hard way: **do not implement the MoE backward with
`torch.autograd`.** Their backward is hand-written GPU operators, and the trick is to overlap the
*rematerialization recompute* and *weight-grad* with the gradient communication.

Accordingly the tiled backward (`tiled_moe.py` + `megakernel.py`) is built from **explicit
kernels**, one recompute pass, no double graph traversal:

- forward GEMM reused for `dgrad` by **swapping the W strides** (`A @ Wᵀ` → `A @ W`);
- a dedicated `wgrad` grouped-GEMM kernel (`Cᵀ @ A` per expert);
- per tile: `dgrad` first → launch the reverse-dispatch all-to-all → compute `wgrad` **while the
  a2a is in flight**; the combine-reverse a2a of tile `i+1` is prefetched to cover tile `i`'s
  recompute.

A first attempt that used `torch.autograd.grad` twice per tile regressed the full step to **0.87×**;
switching to the explicit-kernel scheme turned the same workload into a net win (table below).

On our hardware (2× L4, **PCIe**, no NVLink/NVSHMEM) we deliberately do *not* attempt UniEP's
kernel-fused device-side communication — that needs NVLink and is the right tool only there. We stay
at framework-level overlap (separate CUDA streams + explicit backward kernels).

## Results (2× L4, PCIe)

### Single MoE layer — `tests/bench_ep_overlap.py`
`hidden=4096 inter=4096 experts=8 tokens=8192 topk=2 shared=1` (dispatch ≈ 134 MB):

| path | forward | fwd+bwd |
| --- | --- | --- |
| plain | 1.00× | 1.00× |
| overlap(shared) | 1.10× | 1.02× |
| tiled-3 | 1.39× | 1.25× |
| **tiled-4** | **1.62×** | 1.27× |
| **tiled-6** | 1.54× | **1.32×** |

### End-to-end training step — `tests/bench_ep_train_step.py`
Full step (forward + cross-entropy + backward + AdamW) on the real `Llama` model, `experts=8 topk=2
shared=1`, best tiled vs plain:

| config | tokens/layer | best tiled | note |
| --- | --- | --- | --- |
| hidden2048 seq4096 mbs1 | 2048 | 1.05× | SDPA attention (O(seq²)) + 32k-vocab projection dominate |
| hidden4096 seq2048 mbs1 | 2048 | 0.78–0.94× | too few tokens — tiles become tiny, overhead > savings |
| **hidden2048 seq2048 mbs4** | **8192** | **tiled-4 1.28×** | tok/s 8636 → 11078 |

### Takeaway

The tiled overlap is a **large-batch / many-tokens-per-layer** optimization, exactly MegaScale-MoE's
target regime:

- with enough tokens per layer (≳ 8k) the per-tile overhead is amortized and the all-to-all is hidden
  → **~1.3× end-to-end** (the isolated MoE layer is 1.3–1.6×; attention / lm-head / optimizer dilute it);
- with few tokens per layer, splitting into tiles makes each grouped-GEMM tile inefficient and the
  per-tile metadata collectives dominate → **keep `ep_num_tiles=1`**.

The paper's headline 1.88× is on H800 **with NVLink** and kernel-fused tile communication; on PCIe
L4 the framework-level overlap above is what is achievable.

## Results (2× A100-SXM4-80GB, NVLink)

Same code, run on 2× A100 connected by **NV12 (12 links ≈ 300 GB/s)**. Here the story inverts:
framework-level (stream) overlap gives **nothing**, and tiling slightly regresses.

### Single MoE layer — `tests/bench_ep_overlap.py`

| path | fwd (134 MB) | fwd+bwd (134 MB) | fwd (537 MB) |
| --- | --- | --- | --- |
| plain | 1.00× (21.4 ms) | 1.00× (77.4 ms) | 1.00× (79.5 ms) |
| overlap(shared) | 1.00× | 1.01× | 1.01× |
| tiled-3 | 0.97× | 0.98× | 0.98× |
| tiled-4 | 0.98× | 0.99× | 0.96× |
| tiled-6/8 | 0.86–0.94× | 0.96× | 0.91–0.92× |

### End-to-end training step — `tests/bench_ep_train_step.py`
`hidden4096 inter4096 layers4 experts8 topk2 shared1`, best tiled vs plain:

| config | tokens/layer | plain tok/s | best overlap | speedup |
| --- | --- | --- | --- | --- |
| seq2048 mbs4 | 8192 | 16287 | tiled-4 16505 | 1.01× |
| seq2048 mbs8 | 16384 | 17761 | overlap 17620 | 0.98–0.99× |

### Why NVLink kills the framework-level win

The all-to-all that was the bottleneck on PCIe is now nearly free: even the 537 MB payload moves
dispatch+combine (~1 GB) in ~3.6 ms at 300 GB/s, i.e. **~4.5 % of the step**. With almost nothing to
hide, the per-tile metadata collectives and smaller grouped-GEMM tiles cost more than they save.

### Controlled proof: overlap is *bandwidth-bound*, not hardware-bound

Same A100 box, same `hidden4096 inter4096 tokens8192` config (dispatch ≈ 134 MB), forward-only — the
**only** variable is the interconnect bandwidth (`NCCL_P2P_DISABLE=1` forces the slow host path,
emulating a bandwidth-constrained cross-node / large-EP-degree link):

| interconnect | plain | best tiled | speedup |
| --- | --- | --- | --- |
| NVLink on (~300 GB/s) | 21.4 ms | 22.0 ms | **none** — comm ≈ 4.5 % of step, nothing to hide |
| NVLink off (slow path) | 39.2 ms | **30.5 ms** | **1.28×** — comm is the bottleneck again |
| 2× L4 PCIe (separate box) | — | — | **1.3–1.6×** |

This isolates the mechanism: the overlap is correct and effective **whenever
`comm_time / compute_time` is non-trivial**. On a 2-GPU single-node NVLink island that ratio is tiny,
so the right default there is **`ep_num_tiles=1`**. The win lands exactly where production MoE lives:
experts sharded across nodes, dispatch over IB/RoCE (effective BW an order of magnitude below NVLink),
large EP degree — i.e. the `NCCL_P2P_DISABLE=1` row above.

> Reducing `inter` (compute-light) does **not** reproduce the PCIe regime on NVLink — it leaves comm
> fast and merely exposes fixed per-tile launch overhead, so tiling *regresses* (tiled-6 → 0.43×). Only
> throttling bandwidth restores the comm bottleneck.

Going faster on NVLink itself requires **kernel-fused intra-operator overlap** (UniEP's megakernel /
NVSHMEM device-side comm), not stream pipelining — and even then the intra-node headroom is small,
which is exactly why MegaScale/UniEP target the cross-node / very-large-scale regime.

## Results (2× H100-SXM5-80GB, NVLink) — DeepEP backend

Run via `modal run modal_run.py` (its default job builds DeepEP from source on the Modal H100 image).
All correctness checks pass on real Hopper: `test_megakernel`, `test_fp8_dispatch`, and
`test_deepep_backend` (DeepEP vs torch, forward **and** backward).

### DeepEP backend vs torch all-to-all — `tests/test_deepep_backend.py`

`hidden=2048 inter=2048 experts=8 topk=2 shared=1 tokens=1024`; forward latency covers dispatch +
expert FFN + combine (grads also checked):

| model | fwd rel err | expert grad rel err | torch | deepep | speedup |
| --- | --- | --- | --- | --- | --- |
| dense | 0.0000 | 0.0000 | 2.00 ms | 1.64 ms | **1.22×** |
| LatentMoE (l=512) | 0.0000 | 0.0000 | 2.06 ms | 1.66 ms | **1.24×** |

DeepEP matches the torch path to bf16 and is ~1.2× faster even on a single NVLink node (its kernels
free SMs and skip our explicit permute); the larger wins are at scale / cross-node.

### Overlap & comm-volume on H100 (same NVLink story as A100)

Stream overlap is ~neutral on NVLink, while **LatentMoE wins because it cuts compute**, and FP8
dispatch slightly regresses on a fast fabric — identical conclusions to the A100 section:

`bench_ep_overlap` (`hidden4096 tokens8192`, fwd): plain 11.37 ms, overlap(shared) 1.01×,
tiled-2 1.03×, tiled-4 0.97×.

`bench_ep_compress` (`hidden4096 latent1024 inter4096 tokens8192`, fwd plain path):

| config | wire B/route | vs dense | H100 fwd | speedup |
| --- | --- | --- | --- | --- |
| dense bf16 | 8192 | 1.0× | 11.40 ms | 1.00× |
| dense + fp8 | 4100 | 2.0× | 12.71 ms | 0.90× |
| latent | 2048 | 4.0× | 4.72 ms | **2.41×** |
| latent + fp8 | 1028 | **8.0×** | 5.04 ms | 2.26× |

## Reducing communication *volume* (orthogonal to overlap)

Overlap *hides* the all-to-all; these two reduce the *bytes* sent, so they help even when there is no
compute left to hide behind. They stack with each other and with overlap.

### FP8 dispatch (DeepSeek / DeepEP) — `ep_fp8_dispatch=True`

The dispatch ships token features as **FP8 (E4M3) with a per-token scale** (1 byte/elem instead of
2), and the receiver dequantizes back to BF16 before the expert GEMM; **combine stays BF16** and the
backward is a straight-through BF16 reverse all-to-all. This is the DeepSeek-V3 "FP8 dispatch, BF16
combine" recipe. (NCCL has no portable FP8 collective, so the payload is bitcast to `uint8`. No FP8
tensor cores needed — the GEMM runs in BF16 — so it works on A100.) Measured: ~**2× fewer dispatch
bytes**, **relative L2 error 0.023** (`tests/test_fp8_dispatch.py`).

### LatentMoE (NVIDIA Nemotron) — `moe_latent_dim=l`

A shared **down-projection** `d → l` is applied before routing/dispatch; routing payloads, the
all-to-all, and the **expert weights** all live in the latent dim `l`; a shared **up-projection**
`l → d` runs after combine. The router and the shared expert stay in full `d`. This cuts dispatch
bytes **and** expert FLOPs/weights by `d/l`. (It is an *architecture* change — experts are
parameterized in `l` — so it must be trained this way; Nemotron reinvests the savings into more
experts / higher top-k for accuracy-per-FLOP.)

### Results (2× A100, `hidden=4096 latent=1024 inter=4096 tokens=8192`, plain path, fwd)

`tests/bench_ep_compress.py`, NVLink on (compute-bound) vs off (bandwidth-bound):

| config | wire B/route | vs dense | speedup (NVLink on) | speedup (NVLink off) |
| --- | --- | --- | --- | --- |
| dense bf16 | 8192 | 1.0× | 1.00× | 1.00× |
| dense + fp8 | 4100 | 2.0× | 0.91× | 1.07× |
| latent | 2048 | 4.0× | **2.14×** | **2.72×** |
| latent + fp8 | 1028 | **8.0×** | 2.05× | **2.86×** |

Takeaways:

- **LatentMoE wins even on NVLink (2.14×)** — unlike overlap/FP8, it also cuts *compute* (experts run
  in `l`), so it does not depend on a comm bottleneck. This is the one optimization here that pays off
  on a fast intra-node fabric. (Not iso-accuracy: it is a cheaper model; the apples-to-apples claim is
  the byte/FLOP reduction, which is then reinvested.)
- **FP8 dispatch is comm-bound, like overlap**: +7% off-NVLink, a slight *regression* on NVLink
  (quant/dequant cost > the tiny comm saved). Enable it for slow/cross-node fabrics.
- They **stack**: latent + fp8 = 8× fewer dispatch bytes; in the bandwidth-bound regime that compounds
  to 2.86×, and overlap can hide whatever comm remains on top.

## Correctness

- `tests/test_megakernel.py` — Triton forward + **explicit backward** vs autograd reference
  (`dx ≈ 1e-6`, weight grads `≈ 6–9e-6`; covers empty experts and `count > BLOCK_M`).
- `tests/test_expert_parallel.py` — multi-rank dispatch/combine, all three paths match the plain
  baseline (`grad_diff ≤ 5e-7`), **LatentMoE** paths agree, and full `Llama` integration. Runs on
  gloo/CPU by default; set `EP_TEST_BACKEND=nccl` for the GPU path.
- `tests/test_fp8_dispatch.py` — FP8 dispatch vs BF16 within quant tolerance + byte reduction (CUDA/NCCL).

```sh
# correctness (2 ranks)
torchrun --nproc_per_node 2 tests/test_expert_parallel.py
python tests/test_megakernel.py                       # GPU
torchrun --nproc_per_node 2 tests/test_fp8_dispatch.py  # GPU/NCCL

# benchmarks (2 GPUs)
torchrun --nproc_per_node 2 tests/bench_ep_overlap.py --tokens 8192 --tiles 3 4 6 --backward
torchrun --nproc_per_node 2 tests/bench_ep_train_step.py --mbs 4 --seq 2048 --tiles 3 4 6
torchrun --nproc_per_node 2 tests/bench_ep_compress.py --hidden 4096 --latent 1024 --tokens 8192

# DeepEP / Hopper path on cloud GPUs (builds DeepEP from source; see modal_run.py)
modal run modal_run.py                                   # default: full EP suite on 2x H100
modal run modal_run.py --command "torchrun --nproc_per_node 2 tests/test_deepep_backend.py"
```
