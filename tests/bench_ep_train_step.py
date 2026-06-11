"""End-to-end training-step benchmark for the MoE / expert-parallel overlap.

Builds the *real* picotron Llama model with MoE layers and times a full optimizer step
(forward -> cross-entropy -> backward -> optimizer.step) under different overlap settings:

    plain     : ep_num_tiles=1, no shared-expert overlap
    tiled-N   : MegaScale-style token-tiled pipeline (forward + explicit-kernel backward overlap)

Unlike tests/bench_ep_overlap.py (single MoE layer), this measures the speedup as seen by a real
training loop, i.e. amortized over attention, embeddings, the optimizer and the autograd engine.

Run (2 GPUs):
    torchrun --nproc_per_node 2 tests/bench_ep_train_step.py \
        --hidden 2048 --inter 2048 --layers 4 --experts 8 --topk 2 --shared 1 --seq 2048 --mbs 1
"""

import argparse
import os
import types

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW


def make_config(args):
    return types.SimpleNamespace(
        vocab_size=args.vocab,
        hidden_size=args.hidden,
        intermediate_size=args.inter,
        num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads,
        num_hidden_layers=args.layers,
        max_position_embeddings=args.seq,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        num_experts=args.experts,
        num_experts_per_tok=args.topk,
        num_shared_experts=args.shared,
        norm_topk_prob=True,
    )


def set_overlap(model, num_tiles, shared_overlap):
    for layer in model.decoder_layers:
        mlp = layer.mlp
        if hasattr(mlp, "ep_num_tiles"):
            mlp.ep_num_tiles = num_tiles
            mlp.ep_overlap = shared_overlap


def train_step(model, optimizer, input_ids, target_ids):
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), target_ids.view(-1))
    loss.backward()
    optimizer.step()
    return loss


def time_step(model, optimizer, input_ids, target_ids, iters, warmup):
    for _ in range(warmup):
        train_step(model, optimizer, input_ids, target_ids)
    torch.cuda.synchronize()
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        train_step(model, optimizer, input_ids, target_ids)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    t = torch.tensor(sorted(times)[len(times) // 2], device=input_ids.device)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t.item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab", type=int, default=32000)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--inter", type=int, default=2048)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--kv_heads", type=int, default=4)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--shared", type=int, default=1)
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--mbs", type=int, default=1)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--tiles", type=int, nargs="+", default=[3, 4, 6])
    args = p.parse_args()

    os.environ.setdefault("FLASH_ATTEN", "0")  # use SDPA attention (no flash-attn dependency)
    os.environ.setdefault("DEVICE", "cuda")
    os.environ.setdefault("DTYPE", "bfloat16")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    os.environ.setdefault("LOCAL_RANK", str(rank))
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    from picotron.model import Llama
    from picotron.process_group_manager import setup_process_group_manager

    setup_process_group_manager(tp_size=1, cp_size=1, ep_size=world_size, pp_size=1, dp_size=1)

    assert args.experts % world_size == 0, "num_experts must be divisible by ep_size"
    torch.manual_seed(0)
    cfg = make_config(args)
    model = Llama(cfg)
    model.reset_parameters()
    model = model.to(torch.bfloat16).cuda()
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-4)

    torch.manual_seed(42)  # identical input on every EP rank (dp=1, replicated non-expert weights)
    input_ids = torch.randint(0, cfg.vocab_size, (args.mbs, args.seq), device="cuda")
    target_ids = torch.randint(0, cfg.vocab_size, (args.mbs, args.seq), device="cuda")

    tokens_per_step = args.mbs * args.seq
    if rank == 0:
        print(
            f"\n=== EP end-to-end train-step benchmark: {world_size} GPUs | hidden={args.hidden} "
            f"inter={args.inter} layers={args.layers} experts={args.experts} topk={args.topk} "
            f"shared={args.shared} seq={args.seq} mbs={args.mbs} ==="
        )
        print(
            f"tokens/step = {tokens_per_step}, dispatch payload/layer ~= "
            f"{args.seq * args.mbs * args.topk * args.hidden * 2 / 1e6:.1f} MB\n"
        )

    results = {}

    set_overlap(model, 1, False)
    results["plain"] = time_step(model, optimizer, input_ids, target_ids, args.iters, args.warmup)

    if args.shared > 0:
        set_overlap(model, 1, True)
        results["overlap(shared)"] = time_step(model, optimizer, input_ids, target_ids, args.iters, args.warmup)

    for n in args.tiles:
        set_overlap(model, n, False)
        results[f"tiled-{n}"] = time_step(model, optimizer, input_ids, target_ids, args.iters, args.warmup)

    if rank == 0:
        base = results["plain"]
        print(f"{'path':<18}{'step ms':>10}{'tok/s':>12}{'speedup':>10}")
        print("-" * 50)
        for name, ms in results.items():
            toks = tokens_per_step / (ms / 1000.0)
            print(f"{name:<18}{ms:>10.3f}{toks:>12.0f}{base / ms:>9.2f}x")
        print()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
