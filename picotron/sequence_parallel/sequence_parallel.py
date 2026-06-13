"""Sequence parallelism for picotron.

Sequence parallelism complements tensor parallelism: it shares the **tensor-parallel process group**
(there is no separate "SP group") and only operates inside a TP block, on the activations produced by
``ColumnParallelLinear`` / ``RowParallelLinear``. It cannot be enabled without tensor parallelism.

This module provides the two operators a TP layer needs when ``sequence_parallel=True``:
- ``column_parallel_linear_sp`` — the fused all-gather + matmul for the column-parallel ``f``.
- ``ReduceScatterToSequenceParallelRegion`` — the reduce-scatter ``g`` used at the output of a
  ``RowParallelLinear`` and of the vocab embedding.

See ``picotron/tensor_parallel/README.md`` for the full optimization ladder and benchmarks.
"""

import torch

from picotron.sequence_parallel.sp_communications import all_gather_along_seq, reduce_scatter_along_seq


class _ColumnParallelLinearWithSequenceParallel(torch.autograd.Function):
    """Fused all-gather + linear for the column-parallel `f` under sequence parallelism.

    A naive all-gather + ``F.linear`` works, but autograd then saves the *gathered* (full-sequence)
    input for the weight-grad — so the largest activation is still stored at full size on every rank and
    most of the sequence-parallel memory win is lost. Here we instead save only the sequence-**sharded**
    input (``tp_world_size`` smaller) and re-do the cheap all-gather in the backward pass. This is the
    trick that makes sequence parallelism actually reduce memory (Megatron's
    ``linear_with_grad_accumulation_and_async_allreduce``).
    """

    @staticmethod
    def forward(ctx, x_sharded, weight, bias):
        ctx.save_for_backward(x_sharded, weight)  # NOT the gathered tensor
        ctx.use_bias = bias is not None
        x_full = all_gather_along_seq(x_sharded)
        output = x_full @ weight.t()
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x_sharded, weight = ctx.saved_tensors
        x_full = all_gather_along_seq(x_sharded)  # recompute the gather instead of storing it
        grad_input_full = grad_output @ weight  # [b, s, in]
        grad_input = reduce_scatter_along_seq(grad_input_full)
        go = grad_output.reshape(-1, grad_output.size(-1))
        xf = x_full.reshape(-1, x_full.size(-1))
        grad_weight = go.t() @ xf
        grad_bias = go.sum(0) if ctx.use_bias else None
        return grad_input, grad_weight, grad_bias


def column_parallel_linear_sp(x, weight, bias):
    return _ColumnParallelLinearWithSequenceParallel.apply(x, weight, bias)


class ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """The seq-parallel `g`: reduce-scatter the sequence in forward, all-gather the grad in backward.

    Used at the *output* of a RowParallelLinear (and of the vocab embedding) when sequence parallelism
    is on: the local matmul produces a full-sequence partial sum, which we reduce across TP and scatter
    back to a sequence shard in one collective.
    """

    @staticmethod
    def forward(ctx, x):
        return reduce_scatter_along_seq(x)

    @staticmethod
    def backward(ctx, grad_output):
        return all_gather_along_seq(grad_output)
