"""Bit-exact check for the overlap (Fig 3c) sequence-parallel kernels.

Compares the pipelined ring all-gather+GEMM and GEMM+reduce-scatter against the known-correct *fused*
sequence-parallel path (Fig 3b) already in picotron, on both forward and backward.

    CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 2 tests/test_sp_overlap_kernel.py
"""

import datetime
import os

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
    device_id=device, timeout=datetime.timedelta(minutes=3),
)

from picotron.process_group_manager import setup_process_group_manager  # noqa: E402

setup_process_group_manager(tp_size=world_size, cp_size=1, pp_size=1, dp_size=1)

from picotron.sequence_parallel.overlap import (  # noqa: E402
    overlapped_column_parallel_linear_sp,
    overlapped_row_parallel_linear_sp,
)
from picotron.sequence_parallel.sequence_parallel import (  # noqa: E402
    ReduceScatterToSequenceParallelRegion,
    column_parallel_linear_sp,
)

B, S, IN, OUT = 2, 8 * world_size, 16, 24


def report(tag, *diffs):
    if global_rank == 0:
        joined = ", ".join(f"{d:.2e}" for d in diffs)
        print(f"  {tag}: max_diff = {joined}")


def test_column(dtype, atol):
    """Column `f`: input is a seq shard [b, s/tp, in], weight is the local column shard [out, in]."""
    torch.manual_seed(100 + global_rank)  # each rank owns a different seq shard
    x = torch.randn(B, S // world_size, IN, device=device, dtype=dtype)
    w = torch.randn(OUT, IN, device=device, dtype=dtype) / IN**0.5
    bias = torch.randn(OUT, device=device, dtype=dtype)
    g = torch.randn(B, S, OUT, device=device, dtype=dtype)  # grad wrt full-seq output

    xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
    wa, wb = w.clone().requires_grad_(), w.clone().requires_grad_()
    ba, bb = bias.clone().requires_grad_(), bias.clone().requires_grad_()

    ref = column_parallel_linear_sp(xa, wa, ba)
    out = overlapped_column_parallel_linear_sp(xb, wb, bb)
    (ref * g).sum().backward()
    (out * g).sum().backward()

    report(
        f"column {dtype}",
        (out - ref).abs().max().item(),
        (xb.grad - xa.grad).abs().max().item(),
        (wb.grad - wa.grad).abs().max().item(),
        (bb.grad - ba.grad).abs().max().item(),
    )
    assert (out - ref).abs().max() < atol
    assert (xb.grad - xa.grad).abs().max() < atol
    assert (wb.grad - wa.grad).abs().max() < atol
    assert (bb.grad - ba.grad).abs().max() < atol


def test_row(dtype, atol):
    """Row `g`: input is full-seq, hidden-sharded [b, s, in/tp]; output is a reduced seq shard."""
    torch.manual_seed(200 + global_rank)
    x = torch.randn(B, S, IN, device=device, dtype=dtype)  # in == in/tp slice held by this rank
    w = torch.randn(OUT, IN, device=device, dtype=dtype) / IN**0.5
    g = torch.randn(B, S // world_size, OUT, device=device, dtype=dtype)  # grad wrt seq-shard output

    xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
    wa, wb = w.clone().requires_grad_(), w.clone().requires_grad_()

    ref = ReduceScatterToSequenceParallelRegion.apply(F.linear(xa, wa))
    out = overlapped_row_parallel_linear_sp(xb, wb)
    (ref * g).sum().backward()
    (out * g).sum().backward()

    report(
        f"row    {dtype}",
        (out - ref).abs().max().item(),
        (xb.grad - xa.grad).abs().max().item(),
        (wb.grad - wa.grad).abs().max().item(),
    )
    assert (out - ref).abs().max() < atol
    assert (xb.grad - xa.grad).abs().max() < atol
    assert (wb.grad - wa.grad).abs().max() < atol


def main():
    if global_rank == 0:
        print(f"[tp={world_size}] overlap kernel correctness")
    for dtype, atol in [(torch.float32, 1e-4), (torch.bfloat16, 5e-2)]:
        test_column(dtype, atol)
        test_row(dtype, atol)
    if global_rank == 0:
        print(f"[tp={world_size}] overlap kernels match fused SP ✅")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
