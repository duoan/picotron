"""Ablation benchmark for the tensor-parallel optimization ladder.

Times one forward + cross-entropy + backward on a real picotron Llama under tensor parallelism and
reports step time, tokens/s and peak activation memory for each rung of the ladder:

    baseline   plain Megatron TP (all-reduce f/g, full-logit gather, dense cross-entropy)
    +async     overlap the column-parallel input-grad all-reduce with the weight-grad GEMM
    +seqpar    sequence parallelism: shard norm/residual regions along the sequence
    +vocab_ce  vocab-parallel cross-entropy (logits stay vocab-sharded; only [b,s] scalars exchanged)
    +seqpar+vocab_ce   both memory levers together

    CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 2 tests/bench_tp.py \
        --layers 12 --hidden 2048 --inter 8192 --heads 16 --kv-heads 16 --vocab 32768 \
        --seq 4096 --mbs 1 --steps 10
"""

import argparse
import datetime
import os
import time
import types

import torch
import torch.distributed as dist
import torch.nn.functional as F

os.environ.setdefault("FLASH_ATTEN", "0")  # SDPA attention (no flash-attn dependency)


def make_cfg(a):
    return types.SimpleNamespace(
        vocab_size=a.vocab,
        hidden_size=a.hidden,
        intermediate_size=a.inter,
        num_attention_heads=a.heads,
        num_key_value_heads=a.kv_heads,
        num_hidden_layers=a.layers,
        max_position_embeddings=a.seq,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        num_experts=1,
        num_experts_per_tok=1,
        num_shared_experts=0,
        norm_topk_prob=True,
    )


NORM_SUFFIXES = ("input_layernorm.weight", "post_attention_layernorm.weight", "final_norm.weight")


def build(cfg, device, dtype, async_tp, sequence_parallel, vocab_parallel_ce, parallel_block=False):
    import copy

    from picotron.model import Llama
    from picotron.tensor_parallel.tensor_parallel import apply_tensor_parallel

    torch.manual_seed(0)
    cfg = copy.copy(cfg)
    cfg.parallel_block = parallel_block  # architecture is selected via config (read by the DecoderLayer)
    # Build + init directly on the GPU (CPU weight init dominates the wall clock otherwise).
    with torch.device(device):
        model = Llama(cfg)
        model = apply_tensor_parallel(
            model, async_tp=async_tp, sequence_parallel=sequence_parallel, vocab_parallel_ce=vocab_parallel_ce
        )
    return model.to(dtype).to(device)


def run_one(name, cfg, args, device, dtype, **flags):
    import picotron.process_group_manager as pgm
    from picotron.tensor_parallel.tp_communications import vocab_parallel_cross_entropy

    sequence_parallel = flags.get("sequence_parallel", False)
    vocab_ce = flags.get("vocab_parallel_ce", False)
    tp = pgm.process_group_manager.tp_world_size

    model = build(
        cfg, device, dtype, flags.get("async_tp", False), sequence_parallel, vocab_ce,
        parallel_block=flags.get("parallel_block", False),
    )
    g = torch.Generator().manual_seed(1234)
    input_ids = torch.randint(0, cfg.vocab_size, (args.mbs, args.seq), generator=g).to(device)
    target = torch.randint(0, cfg.vocab_size, (args.mbs, args.seq), generator=g).to(device).reshape(-1)

    def step():
        for p in model.parameters():
            p.grad = None
        logits = model(input_ids)
        logits = logits.reshape(args.mbs * args.seq, -1)
        if vocab_ce:
            loss = vocab_parallel_cross_entropy(logits.float(), target).mean()
        else:
            loss = F.cross_entropy(logits.float(), target)
        loss.backward()
        if sequence_parallel and tp > 1:  # SP norm grads are sequence-sharded -> all-reduce over TP
            for n, p in model.named_parameters():
                if p.grad is not None and n.endswith(NORM_SUFFIXES):
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.tp_group)
        return loss

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        loss = step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / args.steps

    peak = torch.cuda.max_memory_allocated() / 1e6  # MB
    peak_t = torch.tensor(peak, device=device)
    dist.all_reduce(peak_t, op=dist.ReduceOp.MAX)
    tokens = args.mbs * args.seq * tp  # global tokens / step (data is replicated across TP here)
    del model
    torch.cuda.empty_cache()
    return {"name": name, "ms": dt * 1e3, "tok_s": tokens / dt, "peak_mb": peak_t.item(), "loss": loss.item()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--inter", type=int, default=8192)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--kv-heads", type=int, default=16)
    p.add_argument("--vocab", type=int, default=32768)
    p.add_argument("--seq", type=int, default=4096)
    p.add_argument("--mbs", type=int, default=1)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    args = p.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    os.environ.setdefault("DEVICE", "cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    dist.init_process_group(
        rank=global_rank, world_size=world_size, backend="nccl", init_method="env://",
        device_id=device, timeout=datetime.timedelta(minutes=10),
    )
    from picotron.process_group_manager import setup_process_group_manager

    setup_process_group_manager(tp_size=world_size, cp_size=1, pp_size=1, dp_size=1)

    cfg = make_cfg(args)
    configs = [
        ("baseline", {}),
        ("+async", {"async_tp": True}),
        ("+seqpar", {"sequence_parallel": True}),
        ("+vocab_ce", {"vocab_parallel_ce": True}),
        ("+seqpar+vocab_ce", {"sequence_parallel": True, "vocab_parallel_ce": True}),
        ("+ptb+seqpar", {"sequence_parallel": True, "parallel_block": True}),
        ("+ptb+seqpar+vocab_ce", {"sequence_parallel": True, "parallel_block": True, "vocab_parallel_ce": True}),
    ]
    rows = [run_one(name, cfg, args, device, dtype, **flags) for name, flags in configs]

    if global_rank == 0:
        print(
            f"\nTP={world_size}  {args.dtype}  layers={args.layers} hidden={args.hidden} inter={args.inter} "
            f"vocab={args.vocab} seq={args.seq} mbs={args.mbs}\n"
        )
        base = rows[0]
        print(f"{'config':<18}{'ms/step':>10}{'tok/s':>12}{'peak MB':>11}{'mem vs base':>13}{'speedup':>10}")
        for r in rows:
            print(
                f"{r['name']:<18}{r['ms']:>10.1f}{r['tok_s']:>12.0f}{r['peak_mb']:>11.0f}"
                f"{base['peak_mb'] / r['peak_mb']:>12.2f}x{base['ms'] / r['ms']:>9.2f}x"
            )
        print()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
