"""Communication/computation overlap for the sequence-parallel TP linears (MegaScale Figure 3c).

The fused sequence-parallel linears in ``sequence_parallel.py`` already collapse the standalone
all-gather / reduce-scatter into the Linears (MegaScale Figure 3b), but the collective still runs as a
*single blocking call* on the critical path: gather the whole sequence, then one big GEMM (or one big
GEMM, then scatter the whole sequence).

This module implements Figure 3c: break the GEMM into per-rank sequence chunks and **pipeline** each
chunk's matmul with one step of the collective so the communication hides behind compute.

    all-gather side (column `f`):  start with our own shard, ring-rotate the other shards in while we
                                   matmul the shard already in hand. After ``tp`` steps every output
                                   chunk is computed.
    reduce-scatter side (row `g`): compute output chunk ``i`` and launch its reduce to rank ``i``
                                   (async) while we matmul chunk ``i+1``.

Real overlap needs ``CUDA_DEVICE_MAX_CONNECTIONS=1`` so the NCCL P2P/collective and the GEMM land on
separate, concurrently-running streams (same constraint as the ``+async`` rung). Correctness does not
depend on it. Numerics are bit-exact with the fused path up to the summation reordering that any
reduce-scatter does (the tests use a small tolerance, like the existing SP test).
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F

import picotron.process_group_manager as pgm
from picotron.sequence_parallel.sp_communications import (
    SEQ_DIM,
    all_gather_along_seq,
    reduce_scatter_along_seq,
)


def _global_rank(group_local_rank: int) -> int:
    """Map a TP-group-local rank to its global rank (P2P peers are global ranks in torch.distributed)."""
    return pgm.process_group_manager.tp_group_ids[group_local_rank]


def ring_all_gather_matmul(x_shard: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Pipelined all-gather(seq) + ``x_full @ weight.t()`` (MegaScale Fig 3c, top).

    ``x_shard`` is this rank's sequence shard ``[b, s/tp, in]``; returns the full-sequence output
    ``[b, s, out]``. Because the matmul is independent per sequence row, the full result is just the
    concatenation of each shard's matmul, so we can compute a shard the moment it arrives.
    """
    pg = pgm.process_group_manager
    world, rank = pg.tp_world_size, pg.tp_rank
    if world == 1:
        return F.linear(x_shard, weight)

    group = pg.tp_group
    next_rank, prev_rank = _global_rank((rank + 1) % world), _global_rank((rank - 1) % world)
    send_buf = x_shard.contiguous()
    recv_buf = torch.empty_like(send_buf)
    out_chunks: list[torch.Tensor | None] = [None] * world

    for step in range(world):
        # We currently hold the shard that originated on rank `src` (we started with our own and
        # receive from the previous rank each step, so the origin walks backwards).
        src = (rank - step) % world
        if step + 1 < world:
            reqs = dist.batch_isend_irecv(
                [
                    dist.P2POp(dist.isend, send_buf, next_rank, group),
                    dist.P2POp(dist.irecv, recv_buf, prev_rank, group),
                ]
            )
        # GEMM on the shard in hand overlaps the in-flight rotation.
        out_chunks[src] = F.linear(send_buf, weight)
        if step + 1 < world:
            for req in reqs:
                req.wait()
            send_buf, recv_buf = recv_buf, torch.empty_like(recv_buf)

    return torch.cat(out_chunks, dim=SEQ_DIM)


def pipelined_matmul_reduce_scatter(
    x_full: torch.Tensor, weight: torch.Tensor, addend_full: torch.Tensor | None = None
) -> torch.Tensor:
    """Pipelined ``x_full @ weight.t()`` (+ ``addend_full``) + reduce-scatter(seq) (MegaScale Fig 3c).

    ``x_full`` is the full-sequence, hidden-sharded activation ``[b, s, in/tp]``; returns this rank's
    reduced sequence shard ``[b, s/tp, out]``. We split the output sequence into ``tp`` chunks (chunk
    ``i`` is destined for rank ``i``), and as each chunk's matmul finishes we launch its async reduce to
    the owner while the next chunk's matmul runs.

    ``addend_full`` is an optional full-sequence partial (same shape as the output) added in before the
    reduce; the parallel transformer block uses it to fold the attention branch into the FFN's single
    combined reduce-scatter.
    """
    pg = pgm.process_group_manager
    world, rank = pg.tp_world_size, pg.tp_rank
    if world == 1:
        return F.linear(x_full, weight) + (addend_full if addend_full is not None else 0)

    group = pg.tp_group
    assert x_full.size(SEQ_DIM) % world == 0, f"seq {x_full.size(SEQ_DIM)} not divisible by tp {world}"
    x_chunks = [c.contiguous() for c in torch.chunk(x_full, world, dim=SEQ_DIM)]
    add_chunks = torch.chunk(addend_full, world, dim=SEQ_DIM) if addend_full is not None else None

    partials: list[torch.Tensor] = []
    handles = []
    for i in range(world):
        # Local partial sum for the chunk that rank `i` will own.
        p = F.linear(x_chunks[i], weight)
        if add_chunks is not None:
            p = p + add_chunks[i]
        p = p.contiguous()
        # reduce(dst=i) sums this chunk across the TP group onto rank i; the matmul of the next chunk
        # overlaps this collective. Only rank i keeps a valid result for chunk i.
        handles.append(dist.reduce(p, dst=_global_rank(i), op=dist.ReduceOp.SUM, group=group, async_op=True))
        partials.append(p)

    for h in handles:
        h.wait()
    return partials[rank]


class OverlappedColumnParallelLinearSP(torch.autograd.Function):
    """Column-parallel `f` under sequence parallelism, with the forward all-gather pipelined.

    Forward mirrors ``_ColumnParallelLinearWithSequenceParallel`` but uses the ring all-gather+matmul.
    Backward recomputes the (cheap) gather and reduce-scatters the input gradient; it is left as plain
    collectives to keep the conjugate simple (the forward is the showcased Fig 3c path).
    """

    @staticmethod
    def forward(ctx, x_shard, weight, bias):
        ctx.save_for_backward(x_shard, weight)
        ctx.use_bias = bias is not None
        out = ring_all_gather_matmul(x_shard, weight)
        if bias is not None:
            out = out + bias
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x_shard, weight = ctx.saved_tensors
        x_full = all_gather_along_seq(x_shard)
        grad_input_full = grad_output @ weight
        grad_input = reduce_scatter_along_seq(grad_input_full)
        go = grad_output.reshape(-1, grad_output.size(-1))
        xf = x_full.reshape(-1, x_full.size(-1))
        grad_weight = go.t() @ xf
        grad_bias = go.sum(0) if ctx.use_bias else None
        return grad_input, grad_weight, grad_bias


class OverlappedRowParallelLinearSP(torch.autograd.Function):
    """Row-parallel `g` under sequence parallelism, with the forward reduce-scatter pipelined.

    Forward computes the local matmul and pipelines the reduce-scatter chunk-by-chunk. Backward
    all-gathers the (sequence-sharded) output gradient and does plain matmuls. An optional ``addend``
    (a full-sequence partial added before the reduce) lets the parallel transformer block fold its
    attention branch into this single combined reduce-scatter; its gradient is the gathered output grad.
    """

    @staticmethod
    def forward(ctx, x_full, weight, addend):
        ctx.save_for_backward(x_full, weight)
        ctx.has_addend = addend is not None
        return pipelined_matmul_reduce_scatter(x_full, weight, addend)

    @staticmethod
    def backward(ctx, grad_output):
        x_full, weight = ctx.saved_tensors
        grad_output_full = all_gather_along_seq(grad_output)
        grad_input = grad_output_full @ weight
        go = grad_output_full.reshape(-1, grad_output_full.size(-1))
        xf = x_full.reshape(-1, x_full.size(-1))
        grad_weight = go.t() @ xf
        grad_addend = grad_output_full if ctx.has_addend else None
        return grad_input, grad_weight, grad_addend


def overlapped_column_parallel_linear_sp(x, weight, bias):
    return OverlappedColumnParallelLinearSP.apply(x, weight, bias)


def overlapped_row_parallel_linear_sp(x, weight, addend=None):
    return OverlappedRowParallelLinearSP.apply(x, weight, addend)
