"""Two transformer blocks x two TP/SP comm strategies (MegaScale Figure 3).

This is the concrete MegaScale experiment on picotron's TP+SP:

  architectures
    seq : standard sequential block   x = x + Attn(LN(x)); x = x + MLP(LN(x))
    ptb : parallel transformer block  x = x + Attn(LN(x)) + MLP(LN(x))   (GPT-J/PaLM style)

  comm strategy (under sequence parallelism the TP collectives are all-gather `f` / reduce-scatter `g`)
    fused   : Fig 3b   one blocking all-gather, then the GEMM (or GEMM, then one blocking reduce-scatter)
    overlap : Fig 3c   chunk the GEMM and pipeline it with the collective on a second stream

Two things the experiment shows:
  1. PTB collapses the layer to ONE all-gather + ONE reduce-scatter (vs 2 + 2 for the sequential block),
     because attention and MLP share the same gathered input and their partials are summed before the
     single reduce-scatter. We count the NCCL collectives to prove it.
  2. The overlap kernels hide those collectives behind the GEMMs.

Every (arch, strategy) is checked bit-exact against a single-GPU dense reference of the same arch, then
benchmarked.

    CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 2 tests/test_ptb_overlap.py
    CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 2 tests/test_ptb_overlap.py \
        --bench --hidden 2048 --inter 5632 --heads 16 --seq 2048 --mbs 4 --steps 20
"""

import argparse
import datetime
import os
import time
import types

import torch
import torch.distributed as dist
import torch.nn.functional as F

local_rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
os.environ.setdefault("DEVICE", "cuda")

dist.init_process_group(
    rank=global_rank, world_size=world_size, backend="nccl", init_method="env://",
    device_id=device, timeout=datetime.timedelta(minutes=5),
)

from picotron.process_group_manager import setup_process_group_manager  # noqa: E402

setup_process_group_manager(tp_size=world_size, cp_size=1, pp_size=1, dp_size=1)

import picotron.process_group_manager as pgm  # noqa: E402
from picotron.sequence_parallel.overlap import (  # noqa: E402
    overlapped_column_parallel_linear_sp,
    overlapped_row_parallel_linear_sp,
)
from picotron.sequence_parallel.sequence_parallel import (  # noqa: E402
    ReduceScatterToSequenceParallelRegion,
    column_parallel_linear_sp,
)
from picotron.sequence_parallel.sp_communications import all_gather_along_seq  # noqa: E402

N = world_size
R = global_rank  # tp_rank (tp == world here)


# --------------------------------------------------------------------------------------------------
# Collective counter: monkeypatch the NCCL ops so we can prove the PTB collective-count win.
# --------------------------------------------------------------------------------------------------
class CollectiveCounter:
    def __init__(self):
        self.counts = {"all_gather": 0, "reduce_scatter": 0, "reduce": 0, "p2p": 0}

    def __enter__(self):
        self._orig = (dist.all_gather, dist.reduce_scatter, dist.reduce, dist.batch_isend_irecv)

        def wrap(name, fn):
            def inner(*a, **k):
                self.counts[name] += 1
                return fn(*a, **k)
            return inner

        dist.all_gather = wrap("all_gather", self._orig[0])
        dist.reduce_scatter = wrap("reduce_scatter", self._orig[1])
        dist.reduce = wrap("reduce", self._orig[2])
        dist.batch_isend_irecv = wrap("p2p", self._orig[3])
        return self

    def __exit__(self, *a):
        dist.all_gather, dist.reduce_scatter, dist.reduce, dist.batch_isend_irecv = self._orig


# --------------------------------------------------------------------------------------------------
# Weights (full, replicated) and their per-rank TP shards.
# --------------------------------------------------------------------------------------------------
def make_weights(cfg, dtype):
    h, inter, hd, heads = cfg.hidden, cfg.inter, cfg.head_dim, cfg.heads
    g = torch.Generator(device=device).manual_seed(0)

    def rand(*shape, fan_in):
        return (torch.rand(*shape, generator=g, device=device, dtype=dtype) * 2 - 1) / fan_in**0.5

    return types.SimpleNamespace(
        Wq=rand(heads * hd, h, fan_in=h), Wk=rand(heads * hd, h, fan_in=h), Wv=rand(heads * hd, h, fan_in=h),
        Wo=rand(h, heads * hd, fan_in=heads * hd),
        Wup=rand(inter, h, fan_in=h), Wgate=rand(inter, h, fan_in=h), Wdown=rand(h, inter, fan_in=inter),
        g1=torch.ones(h, device=device, dtype=dtype), g2=torch.ones(h, device=device, dtype=dtype),
    )


def col_shard(w):  # [out, in] -> this rank's output columns
    return w.chunk(N, dim=0)[R].clone()


def row_shard(w):  # [out, in] -> this rank's input rows
    return w.chunk(N, dim=1)[R].clone()


def as_params(W, sharded):
    """Build a dict of leaf Parameters; TP weights are sliced to this rank when `sharded`."""
    pick = {
        "Wq": col_shard, "Wk": col_shard, "Wv": col_shard, "Wo": row_shard,
        "Wup": col_shard, "Wgate": col_shard, "Wdown": row_shard,
        "g1": lambda w: w.clone(), "g2": lambda w: w.clone(),
    }
    out = {}
    for name, w in vars(W).items():
        t = pick[name](w) if sharded else w.clone()
        out[name] = torch.nn.Parameter(t.detach().requires_grad_())
    return out


def rms(x, weight, eps=1e-5):
    var = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + eps)).to(x.dtype) * weight


def attention(q, k, v, cfg, n_local_heads):
    b, s, _ = q.shape
    hd = cfg.head_dim
    q = q.view(b, s, n_local_heads, hd).transpose(1, 2)
    k = k.view(b, s, n_local_heads, hd).transpose(1, 2)
    v = v.view(b, s, n_local_heads, hd).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return out.transpose(1, 2).reshape(b, s, n_local_heads * hd)


# --------------------------------------------------------------------------------------------------
# Dense single-GPU reference (tp == 1 math on full weights).
# --------------------------------------------------------------------------------------------------
def dense_block(x, P, cfg, arch):
    n = rms(x, P["g1"])
    q, k, v = n @ P["Wq"].t(), n @ P["Wk"].t(), n @ P["Wv"].t()
    a = attention(q, k, v, cfg, cfg.heads) @ P["Wo"].t()
    if arch == "seq":
        x = x + a
        n2 = rms(x, P["g2"])
        m = (F.silu(n2 @ P["Wgate"].t()) * (n2 @ P["Wup"].t())) @ P["Wdown"].t()
        return x + m
    m = (F.silu(n @ P["Wgate"].t()) * (n @ P["Wup"].t())) @ P["Wdown"].t()
    return x + a + m


# --------------------------------------------------------------------------------------------------
# TP + sequence-parallel block. Input/output are sequence-sharded [b, s/tp, h].
# `strategy` in {"fused", "overlap"}; `arch` in {"seq", "ptb"}.
# --------------------------------------------------------------------------------------------------
def sp_block(x, P, cfg, arch, strategy):
    nlh = cfg.heads // N  # local heads
    qsz = nlh * cfg.head_dim

    def col(n_shard, weight):  # all-gather(seq) + matmul -> full-seq output
        if strategy == "overlap":
            return overlapped_column_parallel_linear_sp(n_shard, weight, None)
        return column_parallel_linear_sp(n_shard, weight, None)

    if arch == "seq":
        # --- attention sublayer: 1 all-gather (fused qkv) + 1 reduce-scatter ---
        n = rms(x, P["g1"])
        Wqkv = torch.cat([P["Wq"], P["Wk"], P["Wv"]], dim=0)
        qkv = col(n, Wqkv)
        q, k, v = qkv.split([qsz, qsz, qsz], dim=-1)
        a_local = attention(q, k, v, cfg, nlh)
        if strategy == "overlap":
            a = overlapped_row_parallel_linear_sp(a_local, P["Wo"])
        else:
            a = ReduceScatterToSequenceParallelRegion.apply(F.linear(a_local, P["Wo"]))
        x = x + a
        # --- MLP sublayer: 1 all-gather (fused up/gate) + 1 reduce-scatter ---
        n2 = rms(x, P["g2"])
        Wug = torch.cat([P["Wup"], P["Wgate"]], dim=0)
        ug = col(n2, Wug)
        up, gate = ug.split([cfg.inter // N, cfg.inter // N], dim=-1)
        hh = F.silu(gate) * up
        if strategy == "overlap":
            m = overlapped_row_parallel_linear_sp(hh, P["Wdown"])
        else:
            m = ReduceScatterToSequenceParallelRegion.apply(F.linear(hh, P["Wdown"]))
        return x + m

    # --- PTB: ONE shared all-gather (qkv + up/gate together), ONE combined reduce-scatter ---
    n = rms(x, P["g1"])
    Wcol = torch.cat([P["Wq"], P["Wk"], P["Wv"], P["Wup"], P["Wgate"]], dim=0)
    cols = col(n, Wcol)
    q, k, v, up, gate = cols.split([qsz, qsz, qsz, cfg.inter // N, cfg.inter // N], dim=-1)
    a_local = attention(q, k, v, cfg, nlh)
    attn_partial = F.linear(a_local, P["Wo"])  # [b, s, h] partial sum (not yet reduced)
    hh = F.silu(gate) * up
    if strategy == "overlap":
        # hide the down-proj GEMM behind the single reduce-scatter; fold in the attention partial.
        out = overlapped_row_parallel_linear_sp(hh, P["Wdown"], attn_partial)
    else:
        mlp_partial = F.linear(hh, P["Wdown"])
        out = ReduceScatterToSequenceParallelRegion.apply(attn_partial + mlp_partial)
    return x + out


# --------------------------------------------------------------------------------------------------
# Correctness: each (arch, strategy) bit-exact vs the dense reference of the same arch.
# --------------------------------------------------------------------------------------------------
def shard_seq(x):
    return x.chunk(N, dim=1)[R].contiguous()


def gather_seq(x):
    return all_gather_along_seq(x.contiguous())


def gather_full(grad_sharded, name):
    """Reassemble a full weight grad from per-rank shards for comparison with the dense grad."""
    if name in ("Wq", "Wk", "Wv", "Wup", "Wgate"):  # column: concat output dim
        chunks = [torch.empty_like(grad_sharded) for _ in range(N)]
        dist.all_gather(chunks, grad_sharded.contiguous())
        return torch.cat(chunks, dim=0)
    if name in ("Wo", "Wdown"):  # row: concat input dim
        chunks = [torch.empty_like(grad_sharded) for _ in range(N)]
        dist.all_gather(chunks, grad_sharded.contiguous())
        return torch.cat(chunks, dim=1)
    g = grad_sharded.clone()  # norm: sequence-sharded partial -> all-reduce
    dist.all_reduce(g)
    return g


def check(cfg, dtype, atol):
    torch.manual_seed(1234)
    x_full = torch.randn(cfg.mbs, cfg.seq, cfg.hidden, device=device, dtype=dtype)
    G_full = torch.randn(cfg.mbs, cfg.seq, cfg.hidden, device=device, dtype=dtype)
    W = make_weights(cfg, dtype)

    for arch in ("seq", "ptb"):
        Pd = as_params(W, sharded=False)
        out_ref = dense_block(x_full.clone().requires_grad_(), Pd, cfg, arch)
        (out_ref * G_full).sum().backward()

        for strategy in ("fused", "overlap"):
            Ps = as_params(W, sharded=True)
            xs = shard_seq(x_full).requires_grad_()
            out = sp_block(xs, Ps, cfg, arch, strategy)
            (out * shard_seq(G_full)).sum().backward()

            out_diff = (gather_seq(out) - out_ref).abs().max().item()
            gmax = 0.0
            for name, p in Ps.items():
                if p.grad is None:  # e.g. g2 is unused in the PTB arch
                    assert Pd[name].grad is None, name
                    continue
                gf = gather_full(p.grad, name)
                gmax = max(gmax, (gf - Pd[name].grad).abs().max().item())
            ok = out_diff < atol and gmax < atol
            if global_rank == 0:
                tag = f"{arch:>3} / {strategy:<7} {str(dtype).split('.')[-1]:>8}"
                print(f"  {tag}: out_diff={out_diff:.2e} grad_diff={gmax:.2e} {'OK' if ok else 'FAIL'}")
            assert ok, f"{arch}/{strategy}/{dtype} mismatch"


def count_collectives(cfg):
    """One fwd+bwd per (arch, strategy) under the counter -> prove PTB halves the collectives."""
    dtype = torch.float32
    x_full = torch.randn(cfg.mbs, cfg.seq, cfg.hidden, device=device, dtype=dtype)
    W = make_weights(cfg, dtype)
    if global_rank == 0:
        print("\ncollectives per block (fwd+bwd):")
        print(f"  {'config':<18}{'all_gather':>11}{'reduce_scatter':>16}{'reduce':>8}{'p2p':>6}")
    for arch in ("seq", "ptb"):
        for strategy in ("fused", "overlap"):
            Ps = as_params(W, sharded=True)
            xs = shard_seq(x_full).requires_grad_()
            with CollectiveCounter() as c:
                out = sp_block(xs, Ps, cfg, arch, strategy)
                out.sum().backward()
            if global_rank == 0:
                cc = c.counts
                print(
                    f"  {arch + '/' + strategy:<18}{cc['all_gather']:>11}{cc['reduce_scatter']:>16}"
                    f"{cc['reduce']:>8}{cc['p2p']:>6}"
                )


def bench(cfg, dtype, steps, warmup):
    x_full = torch.randn(cfg.mbs, cfg.seq, cfg.hidden, device=device, dtype=dtype)
    G = torch.randn(cfg.mbs, cfg.seq, cfg.hidden, device=device, dtype=dtype)
    W = make_weights(cfg, dtype)
    if global_rank == 0:
        print(
            f"\nbench  TP={N}  {str(dtype).split('.')[-1]}  hidden={cfg.hidden} inter={cfg.inter} "
            f"heads={cfg.heads} seq={cfg.seq} mbs={cfg.mbs}\n"
        )
        print(f"  {'config':<18}{'ms/step':>10}{'peak MB':>10}")
    rows = []
    for arch in ("seq", "ptb"):
        for strategy in ("fused", "overlap"):
            Ps = as_params(W, sharded=True)
            xs = shard_seq(x_full).requires_grad_()
            gs = shard_seq(G)

            def step():
                for p in Ps.values():
                    p.grad = None
                out = sp_block(xs, Ps, cfg, arch, strategy)
                (out * gs).sum().backward()

            for _ in range(warmup):
                step()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            dist.barrier()
            t0 = time.perf_counter()
            for _ in range(steps):
                step()
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / steps * 1e3
            peak = torch.tensor(torch.cuda.max_memory_allocated() / 1e6, device=device)
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            rows.append((f"{arch}/{strategy}", ms, peak.item()))
            if global_rank == 0:
                print(f"  {arch + '/' + strategy:<18}{ms:>10.2f}{peak.item():>10.0f}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", action="store_true")
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--inter", type=int, default=5632)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--mbs", type=int, default=4)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()

    assert args.heads % N == 0 and args.hidden % args.heads == 0
    assert args.inter % N == 0 and args.seq % N == 0

    # Small config for the correctness pass (kept tiny + fp32 for a tight tolerance).
    small = types.SimpleNamespace(hidden=32, inter=64, heads=4, head_dim=8, seq=8 * N, mbs=2)
    if global_rank == 0:
        print(f"[tp={N}] correctness (sp block vs dense reference)")
    check(small, torch.float32, atol=2e-4)
    check(small, torch.bfloat16, atol=5e-2)
    count_collectives(small)
    if global_rank == 0:
        print(f"\n[tp={N}] all blocks match their dense reference ✅")

    if args.bench:
        big = types.SimpleNamespace(
            hidden=args.hidden, inter=args.inter, heads=args.heads,
            head_dim=args.hidden // args.heads, seq=args.seq, mbs=args.mbs,
        )
        bench(big, torch.bfloat16, args.steps, args.warmup)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
