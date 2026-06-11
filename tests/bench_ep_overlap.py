"""Benchmark EP communication/computation overlap on real multi-GPU NCCL.

Compares the forward latency of one MoE layer under three execution paths (all numerically
identical, see tests/test_expert_parallel.py):

    plain     : blocking dispatch -> experts -> combine
    overlap   : dispatch hidden behind the shared-expert FFN (DeepSeek-style)
    tiled-N   : MegaScale-style token-tiled pipeline overlapping dispatch/GEMM/combine

Overlap only pays off when the all-to-all and the compute actually run concurrently, which needs
real GPUs on a real interconnect. On a single GPU (ep_size=1) there is no communication to hide.

Run (2 GPUs):
    torchrun --nproc_per_node 2 tests/bench_ep_overlap.py \
        --hidden 4096 --inter 4096 --experts 8 --tokens 8192 --topk 2 --shared 1
"""

import argparse
import os
import types

import torch
import torch.distributed as dist


def make_config(args):
    return types.SimpleNamespace(
        hidden_size=args.hidden,
        intermediate_size=args.inter,
        num_experts=args.experts,
        num_experts_per_tok=args.topk,
        num_shared_experts=args.shared,
        norm_topk_prob=True,
    )


def _step(moe, x, backward):
    if backward:
        moe.zero_grad(set_to_none=True)
        moe(x).float().pow(2).mean().backward()
    else:
        moe(x)


def time_step(moe, x, iters, warmup, backward):
    """Median per-iter latency (ms), averaged across ranks. fwd-only or fwd+bwd."""
    for _ in range(warmup):
        _step(moe, x, backward)
    torch.cuda.synchronize()
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        _step(moe, x, backward)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    t = torch.tensor(sorted(times)[len(times) // 2], device=x.device)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t.item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--inter", type=int, default=4096)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--tokens", type=int, default=8192)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--shared", type=int, default=1)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--tiles", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--backward", action="store_true", help="time fwd+bwd instead of fwd only")
    args = p.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    from picotron.process_group_manager import setup_process_group_manager

    setup_process_group_manager(tp_size=1, cp_size=1, ep_size=world_size, pp_size=1, dp_size=1)

    from picotron.expert_parallel.expert_parallel import MoELayer

    assert args.experts % world_size == 0, "num_experts must be divisible by ep_size"
    torch.manual_seed(0)
    cfg = make_config(args)
    moe = MoELayer(cfg).cuda().to(torch.bfloat16)

    seq = args.tokens
    x = torch.randn(1, seq, args.hidden, device="cuda", dtype=torch.bfloat16)

    # Payload moved per dispatch (and again per combine), for context.
    bytes_per_a2a = args.tokens * args.topk * args.hidden * 2  # bf16
    mode = "fwd+bwd" if args.backward else "fwd"
    if rank == 0:
        print(
            f"\n=== EP overlap benchmark [{mode}]: {world_size} GPUs | hidden={args.hidden} "
            f"inter={args.inter} experts={args.experts} tokens={args.tokens} topk={args.topk} "
            f"shared={args.shared} ==="
        )
        print(f"dispatch payload ~= {bytes_per_a2a / 1e6:.1f} MB (combine sends back the same)\n")

    bwd = args.backward
    results = {}

    moe.ep_num_tiles = 1
    moe.ep_overlap = False
    results["plain"] = time_step(moe, x, args.iters, args.warmup, bwd)

    if args.shared > 0:
        moe.ep_overlap = True
        results["overlap(shared)"] = time_step(moe, x, args.iters, args.warmup, bwd)
        moe.ep_overlap = False

    for n in args.tiles:
        moe.ep_num_tiles = n
        results[f"tiled-{n}"] = time_step(moe, x, args.iters, args.warmup, bwd)
    moe.ep_num_tiles = 1

    if rank == 0:
        base = results["plain"]
        print(f"{'path':<18}{mode + ' ms':>10}{'speedup':>10}")
        print("-" * 38)
        for name, ms in results.items():
            print(f"{name:<18}{ms:>10.3f}{base / ms:>9.2f}x")
        print()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
