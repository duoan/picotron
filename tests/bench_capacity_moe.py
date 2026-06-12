"""Benchmark ONE routed-expert path under ONE routing skew: capacity vs uncapped, on real GPUs.

Single-path-per-process by design: the uncapped path can pile a huge, imbalanced token buffer on the
rank that owns the hot experts (the very memory-wall pathology capacity dispatch prevents). If that
rank stalls or OOMs inside an all-to-all it deadlocks the others, so we isolate each measurement in its
own process (see `bench_capacity_moe.sh`, which wraps every run in `timeout`); a stall there is itself
the headline result for the uncapped path.

What it reports for the chosen (path, skew): fwd+bwd latency (ms) and peak CUDA memory (MB). The point
is the *memory*: uncapped sizes expert buffers to the (imbalanced) per-expert counts, so peak swings
with the router; capacity runs fixed `C`-row buffers, so peak is constant.

    torchrun --nproc_per_node 4 tests/bench_capacity_moe.py --path naive --skew 1.0 \
        --hidden 4096 --inter 4096 --experts 16 --tokens 8192 --topk 2 --factor 1.0
"""

import argparse
import math
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
        num_shared_experts=0,
        norm_topk_prob=True,
    )


def make_routing(num_tokens, top_k, num_experts, skew, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    rank_w = 1.0 / torch.arange(1, num_experts + 1, dtype=torch.float32) ** skew
    probs = (rank_w / rank_w.sum()).expand(num_tokens, num_experts)
    topk_idx = torch.multinomial(probs, top_k, replacement=False, generator=g)
    topk_weights = torch.softmax(torch.rand(num_tokens, top_k, generator=g), dim=-1)
    return topk_idx.to(device), topk_weights.to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", choices=["naive", "dropless", "drop"], required=True)
    p.add_argument("--skew", type=float, required=True)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--inter", type=int, default=4096)
    p.add_argument("--experts", type=int, default=16)
    p.add_argument("--tokens", type=int, default=8192)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--factor", type=float, default=1.0)
    p.add_argument("--max_rounds", type=int, default=16)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    from picotron.expert_parallel.capacity_moe import compute_capacity
    from picotron.expert_parallel.expert_parallel import MoELayer
    from picotron.process_group_manager import setup_process_group_manager

    setup_process_group_manager(tp_size=1, cp_size=1, ep_size=world_size, pp_size=1, dp_size=1)
    assert args.experts % world_size == 0

    torch.manual_seed(0)
    moe = MoELayer(make_config(args)).cuda().to(torch.bfloat16)
    if args.path != "naive":
        moe.ep_capacity_factor = args.factor
        moe.ep_capacity_dropless = args.path == "dropless"
        moe.ep_max_rounds = args.max_rounds

    x = torch.randn(args.tokens, args.hidden, device="cuda", dtype=torch.bfloat16)
    idx, w = make_routing(args.tokens, args.topk, args.experts, args.skew, seed=100 + rank, device="cuda")
    C = compute_capacity(args.tokens, args.topk, args.experts, args.factor)
    counts = torch.bincount(idx.reshape(-1), minlength=args.experts)
    maxc, mean = int(counts.max()), counts.float().mean().item()

    def step():
        moe.zero_grad(set_to_none=True)
        out = moe._moe_naive(x, idx, w) if args.path == "naive" else moe._moe_capped(x, idx, w)
        out.float().pow(2).mean().backward()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    dist.barrier()

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(args.iters):
        start.record()
        step()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    torch.cuda.reset_peak_memory_stats()
    step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()

    t = torch.tensor(sorted(times)[len(times) // 2], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    m = torch.tensor(float(peak), device="cuda")
    dist.all_reduce(m, op=dist.ReduceOp.MAX)

    if rank == 0:
        rounds = math.ceil(maxc / C)
        drop = int(torch.clamp(counts - C, min=0).sum()) / (args.tokens * args.topk)
        tag = {"naive": "naive (uncapped)", "dropless": f"capped-dropless f={args.factor}", "drop": f"capped-drop f={args.factor}"}[args.path]
        print(f"RESULT skew={args.skew} path={tag:<22} maxc/mean={maxc / mean:.2f} "
              f"C={C} rounds={rounds} drop={100 * drop:.1f}% | fwd+bwd={t.item():.2f}ms peakMB={m.item() / 1e6:.0f}",
              flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
