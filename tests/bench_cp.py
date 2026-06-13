"""Ring vs DeepSpeed-Ulysses context parallelism.

Times one forward + cross-entropy + backward of a real picotron Llama under context parallelism, with
the sequence sharded across the CP group (exactly like ``data.py``), and reports step time, tokens/s and
peak memory for the two CP attention algorithms:

    ring      keep the sequence sharded; rotate K/V around the CP ring (P2P), online-softmax merge
    ulysses   all-to-all to head-parallel, one dense local attention, all-to-all back

    CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 2 tests/bench_cp.py \
        --layers 12 --hidden 2048 --inter 8192 --heads 16 --kv-heads 16 --vocab 32768 \
        --seq 8192 --mbs 1 --steps 10
"""

import argparse
import datetime
import os
import time
import types

import torch
import torch.distributed as dist
import torch.nn.functional as F

os.environ.setdefault("FLASH_ATTEN", "0")  # SDPA local attention (ring is python/SDPA-style too)


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


def build(cfg, device, dtype, compile_model=False):
    from picotron.model import Llama

    torch.manual_seed(0)
    with torch.device(device):  # CPU init dominates the wall clock otherwise
        model = Llama(cfg)
    model = model.to(dtype).to(device)
    if compile_model:
        # fullgraph=False: dynamo graph-breaks at the ring P2P / all-to-all collectives and the custom
        # attention autograd functions, and compiles the dense regions (linears, norm, rotary, MLP) around them.
        # IMPORTANT: reset the dynamo cache first. The ring/ulysses branch is an os.getenv() read that dynamo
        # constant-folds without a guard, so without a reset the 2nd model in a process reuses the 1st model's
        # compiled graph (e.g. ulysses would silently rerun ring's graph). Reset forces a correct re-trace.
        torch._dynamo.reset()
        model = torch.compile(model)
    return model


def run_one(attention, cfg, args, device, dtype):
    import picotron.process_group_manager as pgm

    cp = pgm.process_group_manager.cp_world_size
    cp_rank = pgm.process_group_manager.cp_rank
    s_local = args.seq // cp

    model = build(cfg, device, dtype, compile_model=args.compile)
    g = torch.Generator().manual_seed(1234)
    input_ids = torch.randint(0, cfg.vocab_size, (args.mbs, args.seq), generator=g)
    target = torch.randint(0, cfg.vocab_size, (args.mbs, args.seq), generator=g)
    # this rank's contiguous sequence shard
    local_input = input_ids[:, cp_rank * s_local : (cp_rank + 1) * s_local].contiguous().to(device)
    local_target = target[:, cp_rank * s_local : (cp_rank + 1) * s_local].contiguous().to(device).reshape(-1)

    os.environ["CONTEXT_PARALLEL"] = "1"
    os.environ["CP_ATTENTION"] = attention

    def step():
        for p in model.parameters():
            p.grad = None
        logits = model(local_input).reshape(args.mbs * s_local, -1)
        loss = F.cross_entropy(logits.float(), local_target)
        loss.backward()
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

    peak = torch.tensor(torch.cuda.max_memory_allocated() / 1e6, device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    tokens = args.mbs * args.seq  # the CP group collectively processes one global sequence per step
    del model
    torch.cuda.empty_cache()
    return {"name": attention, "ms": dt * 1e3, "tok_s": tokens / dt, "peak_mb": peak.item(), "loss": loss.item()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--inter", type=int, default=8192)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--kv-heads", type=int, default=16)
    p.add_argument("--vocab", type=int, default=32768)
    p.add_argument("--seq", type=int, default=8192)
    p.add_argument("--mbs", type=int, default=1)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--compile", action="store_true", help="wrap the model in torch.compile")
    p.add_argument("--attn", type=str, default="both", choices=["both", "ring", "ulysses"])
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

    setup_process_group_manager(tp_size=1, cp_size=world_size, pp_size=1, dp_size=1)
    assert args.heads % world_size == 0, "Ulysses needs num_heads divisible by cp_world_size"
    assert args.seq % world_size == 0, "seq must be divisible by cp_world_size"

    cfg = make_cfg(args)
    names = ("ring", "ulysses") if args.attn == "both" else (args.attn,)
    rows = [run_one(name, cfg, args, device, dtype) for name in names]

    if global_rank == 0:
        print(
            f"\nCP={world_size}  {args.dtype}  compile={args.compile}  layers={args.layers} "
            f"hidden={args.hidden} heads={args.heads} seq={args.seq} (={args.seq // world_size}/rank) "
            f"mbs={args.mbs}\n"
        )
        ring = rows[0]
        print(f"{'attention':<12}{'ms/step':>10}{'tok/s':>12}{'peak MB':>11}{'speedup':>10}")
        for r in rows:
            print(
                f"{r['name']:<12}{r['ms']:>10.1f}{r['tok_s']:>12.0f}{r['peak_mb']:>11.0f}"
                f"{ring['ms'] / r['ms']:>9.2f}x"
            )
        print()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
