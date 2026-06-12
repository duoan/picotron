"""Wall-clock + bubble benchmark for the advanced pipeline-parallel schedules.

Each schedule trades a different resource to shrink the ``(p-1)/m`` pipeline bubble of 1F1B. This
benchmark times one full ``train_step`` (the whole micro-batch sweep + backward) per schedule on the
*real* picotron Llama stage, and prints the measured step time next to the analytical bubble fraction
so the theory and the wall clock line up.

Directly comparable (each sweeps ``m = grad_acc_steps`` micro-batches once through the full model):
    afab, 1f1b, zb (Zero-Bubble), interleaved (virtual pipeline).

Metrics reported per schedule:
    step ms       : measured wall-clock of one full train_step (slowest stage, MAX-reduced).
    tok/s         : throughput = (mbs * seq * m) / step_time  — the bottom-line metric.
    peak MB       : peak CUDA memory (max_memory_allocated) — shows AFAB's m-activation cost vs 1F1B.
    bubble(emp)   : empirical pipeline bubble = 1 - compute_only / step, where compute_only is the same
                    stage running m fwd+bwd with NO pipeline comm/stagger (the zero-bubble lower bound).
    bubble(ana)   : analytical (p-1)/m, (p-1)/(m*v), ... for reference.

Backend auto-selects: NCCL+CUDA if available, else gloo/CPU. On CPU the bubble is invisible (compute
dominates, no real parallelism); use real GPUs to see it.

    torchrun --nproc_per_node 2 tests/bench_pp_schedules.py --backend nccl --dtype bf16 \
        --layers 24 --hidden 1024 --seq 1024 --grad-acc 8 --mbs 1
    torchrun --nproc_per_node 4 tests/bench_pp_schedules.py --layers 16 --grad-acc 16 --hidden 256
"""

import argparse
import os
import time
import types

import torch
import torch.distributed as dist


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


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
        num_experts=1,
        num_experts_per_tok=1,
        num_shared_experts=0,
        norm_topk_prob=True,
    )


class FakeDataLoader:
    """Deterministic in-memory micro-batch source (identical on every rank, replayable)."""

    def __init__(self, cfg, grad_acc_steps, mbs, seq, device):
        self.grad_acc_steps = grad_acc_steps
        self.seq_length_per_gpu = seq
        self.micro_batch_size = mbs
        g = torch.Generator().manual_seed(1234)
        self.batches = [
            {
                "input_ids": torch.randint(0, cfg.vocab_size, (mbs, seq), generator=g).to(device),
                "target_ids": torch.randint(0, cfg.vocab_size, (mbs, seq), generator=g).to(device),
                "position_ids": torch.arange(seq).unsqueeze(0).repeat(mbs, 1).to(device),
                "hidden_states": None,
            }
            for _ in range(grad_acc_steps)
        ]
        self.idx = 0

    def reset(self):
        self.idx = 0

    def __next__(self):
        batch = self.batches[self.idx % self.grad_acc_steps]
        self.idx += 1
        return batch


def bubble_fraction(engine, p, m, v):
    """Analytical steady-state bubble fraction (idle / busy) for each schedule."""
    if engine in ("afab", "1f1b"):
        return (p - 1) / m
    if engine == "interleaved":
        return (p - 1) / (m * v)  # virtual pipeline shrinks the bubble by v
    if engine == "zb":
        return 0.0  # ideal: W work fills the bubble (real ~ small, bounded by warmup)
    return float("nan")


def build_stage(engine, cfg, device, dtype, num_virtual_stages):
    from picotron.pipeline_parallel.pipeline_parallel import PipelineParallel
    from picotron.pipeline_parallel.pp_schedules import InterleavedPipelineParallel

    torch.manual_seed(0)
    from picotron.model import Llama

    model = Llama(cfg)
    model.reset_parameters()
    if engine == "interleaved":
        stage = InterleavedPipelineParallel(model, cfg, num_virtual_stages=num_virtual_stages)
    else:
        stage = PipelineParallel(model, cfg)
    return stage.to(dtype).to(device)


def schedule_fn(engine):
    from picotron.pipeline_parallel.pipeline_parallel import (
        train_step_pipeline_1f1b,
        train_step_pipeline_afab,
    )
    from picotron.pipeline_parallel.pp_schedules import (
        train_step_pipeline_interleaved,
        train_step_pipeline_zb,
    )

    return {
        "afab": train_step_pipeline_afab,
        "1f1b": train_step_pipeline_1f1b,
        "zb": train_step_pipeline_zb,
        "interleaved": train_step_pipeline_interleaved,
    }[engine]


def time_engine(engine, cfg, dl, device, dtype, num_virtual_stages, iters, warmup):
    stage = build_stage(engine, cfg, device, dtype, num_virtual_stages)
    fn = schedule_fn(engine)
    shapes = (dl.micro_batch_size, dl.seq_length_per_gpu, cfg.hidden_size)

    def one_step():
        for p_ in stage.parameters():
            p_.grad = None
        dl.reset()
        fn(stage, dl, shapes, device, dtype)

    for _ in range(warmup):
        one_step()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    _sync(device)
    dist.barrier()
    times = []
    for _ in range(iters):
        dist.barrier()
        _sync(device)
        t0 = time.perf_counter()
        one_step()
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    t = torch.tensor(sorted(times)[len(times) // 2], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)  # pipeline step time = slowest stage
    peak_mb = 0.0
    if device.type == "cuda":
        peak = torch.tensor(torch.cuda.max_memory_allocated() / 1e6, device=device)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        peak_mb = peak.item()
    return t.item(), peak_mb


def compute_only_ms(cfg, dl, device, dtype, iters, warmup):
    """Zero-bubble lower bound: this rank's stage running m forward+backward with NO pipeline comm.

    Feeds random activations for non-first stages, so it measures pure useful compute (no fill/drain,
    no cross-rank wait). ``1 - compute_only / step`` is then the empirical pipeline bubble + comm cost.
    """
    import torch.nn.functional as F

    import picotron.process_group_manager as pgm

    stage = build_stage("1f1b", cfg, device, dtype, 1)
    shapes = (dl.micro_batch_size, dl.seq_length_per_gpu, cfg.hidden_size)
    is_first = pgm.process_group_manager.pp_is_first_stage
    is_last = pgm.process_group_manager.pp_is_last_stage

    def one_step():
        for p_ in stage.parameters():
            p_.grad = None
        dl.reset()
        for _ in range(dl.grad_acc_steps):
            batch = next(dl)
            hidden = None if is_first else torch.randn(shapes, device=device, dtype=dtype, requires_grad=True)
            out = stage.forward(
                input_ids=batch["input_ids"].to(device) if is_first else None,
                position_ids=batch["position_ids"].to(device),
                hidden_states=hidden,
            )
            if is_last:
                out = F.cross_entropy(out.flatten(0, 1), batch["target_ids"].to(device).flatten(), reduction="mean")
            torch.autograd.backward(out, grad_tensors=torch.ones_like(out))

    for _ in range(warmup):
        one_step()
    _sync(device)
    dist.barrier()
    times = []
    for _ in range(iters):
        _sync(device)
        t0 = time.perf_counter()
        one_step()
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    t = torch.tensor(sorted(times)[len(times) // 2], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)  # ideal busy = slowest stage's compute
    return t.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--inter", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv_heads", type=int, default=2)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--mbs", type=int, default=2)
    ap.add_argument("--grad-acc", type=int, default=8, help="micro-batches per step (must be a multiple of pp_size)")
    ap.add_argument("--num-virtual-stages", type=int, default=2)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--backend", choices=["auto", "nccl", "gloo"], default="auto")
    ap.add_argument("--dtype", choices=["float32", "bf16", "fp16"], default="float32")
    args = ap.parse_args()

    os.environ.setdefault("FLASH_ATTEN", "0")

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "25600")

    backend = args.backend
    if backend == "auto":
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl":
        assert torch.cuda.is_available(), "nccl backend requested but no CUDA device is visible"
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
        device = torch.device("cuda", local_rank % torch.cuda.device_count())
    else:
        device = torch.device("cpu")
    dtype = {"float32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    # Bind the device eagerly for NCCL. Without device_id, NCCL guesses the device from the global
    # rank and lazily spins up a 2-rank communicator on the first P2P op, which can deadlock the
    # pipeline at larger pp_size (observed hanging at p=8 on L4). Eager binding avoids the guess.
    init_kwargs = {"backend": backend, "rank": rank, "world_size": world_size}
    if backend == "nccl":
        init_kwargs["device_id"] = device
    dist.init_process_group(**init_kwargs)

    from picotron.process_group_manager import setup_process_group_manager

    setup_process_group_manager(tp_size=1, cp_size=1, ep_size=1, pp_size=world_size, dp_size=1)

    cfg = make_config(args)
    p, m, v = world_size, args.grad_acc, args.num_virtual_stages
    assert m % p == 0, f"--grad-acc ({m}) must be a multiple of pp_size ({p}) for the interleaved engine"
    dl = FakeDataLoader(cfg, m, args.mbs, args.seq, device)
    tokens_per_step = args.mbs * args.seq * m

    ideal_ms = compute_only_ms(cfg, dl, device, dtype, args.iters, args.warmup)
    comparable = ["afab", "1f1b", "zb", "interleaved"]
    results = {e: time_engine(e, cfg, dl, device, dtype, v, args.iters, args.warmup) for e in comparable}

    if rank == 0:

        def emp_bubble(step):
            return max(0.0, 1.0 - ideal_ms / step)

        def row(name, ms, mem, ana):
            toks = tokens_per_step / (ms / 1000.0)
            print(f"{name:<13}{ms:>9.2f}{toks:>11.0f}{mem:>9.0f}{emp_bubble(ms):>12.3f}{ana:>11.3f}{base / ms:>9.2f}x")

        print(
            f"\n=== PP schedule benchmark: backend={backend} dtype={args.dtype} | p={p} m={m} v={v} | "
            f"layers={args.layers} hidden={args.hidden} seq={args.seq} mbs={args.mbs} ==="
        )
        print(f"zero-bubble lower bound (compute only, no pipeline comm): {ideal_ms:.2f} ms")
        base = results["1f1b"][0]
        print(
            f"\n{'engine':<13}{'step ms':>9}{'tok/s':>11}{'peak MB':>9}"
            f"{'bubble(emp)':>12}{'bubble(ana)':>11}{'vs 1f1b':>10}"
        )
        print("-" * 75)
        for e in comparable:
            ms, mem = results[e]
            row(e, ms, mem, bubble_fraction(e, p, m, v))
        print("-" * 75)
        print(
            "\nbubble(emp) = 1 - compute_only/step (measured); bubble(ana) = analytical (p-1)/m, "
            "(p-1)/(m*v), ...\n"
            "zb does a true ~1x B/W split (deferred per-Linear weight grads). peak MB is 0 on CPU "
            "(CUDA-only).\n"
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
