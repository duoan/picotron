"""Benchmark the two communication-*volume* reducers for EP MoE (orthogonal to overlap):

    FP8 dispatch (DeepSeek/DeepEP) : ship token features at 1 byte instead of 2  -> ~2x less wire
    LatentMoE   (NVIDIA Nemotron)  : route/dispatch/compute in latent dim l < d  -> ~d/l less wire
                                                                                     AND d/l less expert FLOPs

They stack multiplicatively on dispatch bytes. This times the *plain* dispatch path (no overlap) for
the four combinations and prints dispatch payload + speedup, so the byte savings are isolated from the
overlap axis measured in bench_ep_overlap.py.

Run (2 GPUs); use NCCL_P2P_DISABLE=1 to emulate a bandwidth-bound (cross-node) link:
    torchrun --nproc_per_node 2 tests/bench_ep_compress.py --hidden 4096 --latent 1024 --tokens 8192
"""

import argparse
import os
import types

import torch
import torch.distributed as dist


def make_config(hidden, inter, experts, topk, shared, latent):
    return types.SimpleNamespace(
        hidden_size=hidden,
        intermediate_size=inter,
        num_experts=experts,
        num_experts_per_tok=topk,
        num_shared_experts=shared,
        norm_topk_prob=True,
        moe_latent_dim=latent,
    )


def time_fwd(moe, x, iters, warmup):
    for _ in range(warmup):
        moe(x)
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        moe(x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    t = torch.tensor(sorted(times)[len(times) // 2], device=x.device)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t.item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--latent", type=int, default=1024)
    p.add_argument("--inter", type=int, default=4096)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--tokens", type=int, default=8192)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--shared", type=int, default=1)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=8)
    args = p.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    from picotron.expert_parallel.expert_parallel import MoELayer
    from picotron.process_group_manager import setup_process_group_manager

    setup_process_group_manager(tp_size=1, cp_size=1, ep_size=world_size, pp_size=1, dp_size=1)
    assert args.experts % world_size == 0

    x = torch.randn(1, args.tokens, args.hidden, device="cuda", dtype=torch.bfloat16)

    def build(latent):
        torch.manual_seed(0)
        moe = MoELayer(make_config(args.hidden, args.inter, args.experts, args.topk, args.shared, latent))
        moe = moe.cuda().to(torch.bfloat16)
        moe.ep_overlap = False  # isolate byte-reduction from overlap
        moe.ep_num_tiles = 1
        return moe

    dense = build(0)
    latent = build(args.latent)

    # dispatch wire bytes per token-route (one routed row): payload + (fp8 scale).
    def wire_bytes(dim, fp8):
        return dim * (1 if fp8 else 2) + (4 if fp8 else 0)

    configs = [
        ("dense  bf16", dense, False, args.hidden),
        ("dense  fp8 ", dense, True, args.hidden),
        ("latent bf16", latent, False, args.latent),
        ("latent fp8 ", latent, True, args.latent),
    ]
    results = []
    for name, moe, fp8, dim in configs:
        moe.ep_fp8_dispatch = fp8
        ms = time_fwd(moe, x, args.iters, args.warmup)
        results.append((name, ms, wire_bytes(dim, fp8)))

    if rank == 0:
        base_ms = results[0][1]
        base_bytes = results[0][2]
        print(
            f"\n=== EP comm-volume reducers [fwd, plain path]: {world_size} GPUs | hidden={args.hidden} "
            f"latent={args.latent} inter={args.inter} experts={args.experts} tokens={args.tokens} ==="
        )
        print(f"{'config':<14}{'fwd ms':>9}{'speedup':>9}{'wire B/route':>14}{'vs dense bf16':>15}")
        print("-" * 61)
        for name, ms, wb in results:
            print(f"{name:<14}{ms:>9.3f}{base_ms / ms:>8.2f}x{wb:>14}{base_bytes / wb:>13.2f}x")
        print()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
