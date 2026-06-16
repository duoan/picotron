"""Bit-exact correctness for ZeRO-1/2/3 vs plain data parallelism.

Plain DP (all-reduce mean gradients, then a normal optimizer step replicated on every rank) is the
reference. ZeRO shards the optimizer states (1), gradients (2), and parameters (3) across the replica
group but must produce the *identical* parameter trajectory: each rank steps only its owned parameters,
on gradients that are all-reduced exactly as in plain DP, so the per-parameter update is unchanged.

Every rank sees a *different* data shard (that is the point of DP), and gradients are averaged across
ranks; summing/averaging must recover the same update as the single replicated optimizer would apply.

    CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 2 tests/test_data_parallel_zero.py
    CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 4 tests/test_data_parallel_zero.py
"""

import datetime
import os

import torch
import torch.distributed as dist
import torch.nn as nn

local_rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

backend = os.environ.get("DP_TEST_BACKEND", "nccl")
if backend == "nccl":
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    os.environ.setdefault("DEVICE", "cuda")
else:
    device = torch.device("cpu")
    os.environ["DEVICE"] = "cpu"

dist.init_process_group(
    rank=global_rank, world_size=world_size, backend=backend, init_method="env://",
    timeout=datetime.timedelta(minutes=3),
)

from picotron.process_group_manager import setup_process_group_manager  # noqa: E402

# Pure data parallelism: the whole world is one replica (cp_dp) group.
setup_process_group_manager(tp_size=1, cp_size=1, pp_size=1, dp_size=world_size)

import picotron.process_group_manager as pgm  # noqa: E402
from picotron.data_parallel.zero import DataParallelZero, ShardedOptimizer  # noqa: E402

STEPS = 5
LR = 1e-2
DIN, DH, DOUT = 32, 64, 32
MBS = 4


def build_model():
    torch.manual_seed(0)  # identical init on every rank
    model = nn.Sequential(
        nn.Linear(DIN, DH), nn.ReLU(), nn.Linear(DH, DH), nn.ReLU(), nn.Linear(DH, DOUT)
    )
    return model.to(torch.float32).to(device)


def batch(step):
    # Per-rank distinct data (DP), deterministic and identical between reference and ZeRO runs.
    g = torch.Generator().manual_seed(1000 * step + global_rank)
    x = torch.randn(MBS, DIN, generator=g).to(device)
    y = torch.randn(MBS, DOUT, generator=g).to(device)
    return x, y


def run_reference():
    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        x, y = batch(step)
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        for p in model.parameters():
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.cp_dp_group)
            p.grad /= world_size
        opt.step()
    return [p.detach().clone() for p in model.parameters()]


def run_zero(stage):
    model = build_model()
    dp = DataParallelZero(model, stage=stage)
    opt = ShardedOptimizer(dp, torch.optim.AdamW, lr=LR)
    for step in range(STEPS):
        opt.zero_grad()
        x, y = batch(step)
        loss = (dp(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
    return dp, dp.full_parameters()


def check_sharding(dp, stage):
    """Verify the model state is actually partitioned across ranks (not just numerically correct)."""
    total = sum(p.numel() for p in dp.module.parameters())
    owned = sum(p.numel() for p in dp.owned_params)
    # Owned shards across all ranks must cover the model exactly once (a clean partition).
    s = torch.tensor([float(owned)], device=device)
    dist.all_reduce(s, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.cp_dp_group)
    assert s.item() == total, f"owned shards sum {s.item()} != total {total}"
    assert owned < total, f"rank owns all {owned} params; expected a strict shard with dp>1"
    if stage == 3:
        # ZeRO-3: at rest, non-owned parameters hold no storage on this rank.
        at_rest = sum(p.untyped_storage().size() for p in dp.module.parameters())
        owned_bytes = sum(p.numel() * p.element_size() for p in dp.owned_params)
        assert at_rest == owned_bytes, f"stage-3 at-rest bytes {at_rest} != owned bytes {owned_bytes}"


def main():
    ref = run_reference()

    for stage in (1, 2, 3):
        dp, got = run_zero(stage)
        check_sharding(dp, stage)
        max_diff = max((a - b).abs().max().item() for a, b in zip(got, ref))
        d = torch.tensor([max_diff], device=device)
        dist.all_reduce(d, op=dist.ReduceOp.MAX, group=pgm.process_group_manager.cp_dp_group)
        max_diff = d.item()
        assert max_diff == 0.0, f"ZeRO-{stage} diverged from plain DP: max_diff={max_diff:.3e}"
        if global_rank == 0:
            print(f"  ZeRO-{stage} vs plain DP: PASSED  (sharded + bit-exact, max param diff = {max_diff:.1e})")

    if global_rank == 0:
        print(f"[dp={world_size}] All ZeRO-1/2/3 bit-exactness tests passed")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
