import torch
import torch.distributed as dist
import torch.nn.functional as F

import picotron.process_group_manager as pgm


def merge_first_two_dims(grad_output: torch.Tensor, input_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge the first two dimensions of tensors."""
    return grad_output.contiguous().view(-1, *grad_output.shape[2:]), input_.contiguous().view(-1, *input_.shape[2:])


def split_tensor_along_last_dim(tensor, num_partitions):
    """Split a tensor along its last dimension into num_partitions chunks."""
    last_dim = tensor.dim() - 1
    assert tensor.size()[last_dim] % num_partitions == 0, (
        f"{tensor.size()[last_dim]} is not divisible by {num_partitions}"
    )
    last_dim_size = tensor.size()[last_dim] // num_partitions
    return torch.split(tensor, last_dim_size, dim=last_dim)


class CopyToModelParallelRegion(torch.autograd.Function):
    """
    Copy in forward pass, all-reduce in backward pass.
    This is the `f` function in the paper: https://arxiv.org/abs/1909.08053
    """

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        if pgm.process_group_manager.tp_world_size == 1:
            return grad_output
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.tp_group)
        return grad_output


class ReduceFromModelParallelRegion(torch.autograd.Function):
    """
    All-reduce in forward pass, identity in backward pass.
    This is the `g` function in the paper: https://arxiv.org/abs/1909.08053
    """

    @staticmethod
    def forward(ctx, x):
        if pgm.process_group_manager.tp_world_size == 1:
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.tp_group)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather in forward pass, split in backward pass."""

    @staticmethod
    def forward(ctx, x):
        if pgm.process_group_manager.tp_world_size == 1:
            return x
        last_dim = x.dim() - 1
        # Need contiguous tensors for collectives -> https://github.com/pytorch/pytorch/blob/main/torch/distributed/nn/functional.py#L321
        x = x.contiguous()
        tensor_list = [torch.empty_like(x) for _ in range(pgm.process_group_manager.tp_world_size)]
        tensor_list[pgm.process_group_manager.tp_rank] = x
        dist.all_gather(tensor_list, x, group=pgm.process_group_manager.tp_group)
        output = torch.cat(tensor_list, dim=last_dim).contiguous()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if pgm.process_group_manager.tp_world_size == 1:
            return grad_output
        # Split gradient according to TP size
        chunks = split_tensor_along_last_dim(grad_output, pgm.process_group_manager.tp_world_size)
        return chunks[pgm.process_group_manager.tp_rank].contiguous()


class LinearWithAsyncAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias):
        ctx.save_for_backward(input_, weight)
        ctx.use_bias = bias is not None
        output = input_ @ weight.t() + bias if bias is not None else input_ @ weight.t()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        The key difference with "linear_with_all_reduce" is that the all reduce of input_ gradeint is before
        the calculation of the gradient of weights and bias, instead of after. So we can overlap the computation and communication
        This is only applicable to Column Parallel Linear

        Before: grad_output -> grad_input, grad_weight, grad_bias  -> grad_input all reduce
        Now:    grad_output -> grad_input -> grad_input all reduce -> grad_weight, grad_bias
        """
        input_, weight = ctx.saved_tensors
        grad_input = grad_output @ weight  # (b, s, out_size) @ (out_size, input_size) = (b, s, input_size)
        # all-reduce input gradient.
        input_gradient_all_reduce_handle = dist.all_reduce(
            grad_input, group=pgm.process_group_manager.tp_group, async_op=True
        )
        # merge first two dims to allow matrix multiplication
        grad_output, input_ = merge_first_two_dims(
            grad_output, input_
        )  # grad_output, input_: (b, s, out_size), (b, s, input_size) -> (b*s, out_size), (b*s, input_size)
        grad_weight = grad_output.t() @ input_  # (out_size, b*s) @ (b*s, input_size) -> (out_size, input_size)
        grad_bias = grad_output.sum(0) if ctx.use_bias else None
        input_gradient_all_reduce_handle.wait()
        return grad_input, grad_weight, grad_bias


def linear_with_all_reduce(x, weight, bias):
    input_parallel = CopyToModelParallelRegion.apply(x)
    output = F.linear(input_parallel, weight, bias)  # XW_i^T + b, output is Y_i
    return output


def linear_with_async_all_reduce(x, weight, bias):
    return LinearWithAsyncAllReduce.apply(x, weight, bias)


# ---------------------------------------------------------------------------
# Vocab-parallel cross-entropy (Megatron `vocab_parallel_cross_entropy`).
#
# The output projection is column-parallel over the vocabulary, so each rank holds a [N, V/tp] slice
# of the logits (N = tokens). Instead of all-gathering the full [N, V] logits (huge for large V) and
# running a dense softmax on every rank, we compute the loss directly on the shards and exchange only
# per-token scalars: the max logit, the sum of exp, and the target logit. Communication drops from
# O(V) to O(1) per token, and the full logits are never materialized.
# ---------------------------------------------------------------------------
class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor, target: torch.Tensor):
        group = pgm.process_group_manager.tp_group
        tp_rank = pgm.process_group_manager.tp_rank
        tp_world_size = pgm.process_group_manager.tp_world_size
        partition_vocab_size = vocab_parallel_logits.size(-1)
        vocab_start = tp_rank * partition_vocab_size
        vocab_end = vocab_start + partition_vocab_size

        # 1) Global max per token (for numerical stability), then shift the logits.
        logits_max = vocab_parallel_logits.max(dim=-1).values
        if tp_world_size > 1:
            dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=group)
        logits = vocab_parallel_logits - logits_max.unsqueeze(-1)

        # 2) Gather this rank's contribution to the target logit (0 where the target is off-shard).
        target_mask = (target < vocab_start) | (target >= vocab_end)
        local_target = target.clone() - vocab_start
        local_target[target_mask] = 0
        predicted_logits = logits.gather(-1, local_target.unsqueeze(-1)).squeeze(-1)
        predicted_logits[target_mask] = 0.0
        if tp_world_size > 1:
            dist.all_reduce(predicted_logits, op=dist.ReduceOp.SUM, group=group)

        # 3) Global sum of exp over the full vocabulary.
        exp_logits = torch.exp(logits)
        sum_exp = exp_logits.sum(dim=-1)
        if tp_world_size > 1:
            dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=group)

        # loss = log(sum exp(z - max)) - (z_target - max)
        loss = torch.log(sum_exp) - predicted_logits

        # softmax (for backward): exp_logits / sum_exp.
        exp_logits.div_(sum_exp.unsqueeze(-1))
        ctx.save_for_backward(exp_logits, target_mask, local_target)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        softmax, target_mask, local_target = ctx.saved_tensors
        grad_input = softmax
        # grad = softmax - onehot(target); subtract 1 at the target position (only on the owning rank).
        keep = (~target_mask).float()
        grad_input.scatter_add_(
            -1, local_target.unsqueeze(-1), -keep.unsqueeze(-1).to(grad_input.dtype)
        )
        grad_input.mul_(grad_output.unsqueeze(-1))
        return grad_input, None


def vocab_parallel_cross_entropy(vocab_parallel_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-token cross-entropy on vocab-sharded logits. `vocab_parallel_logits`: [..., V/tp], `target`:
    [...] global vocab ids. Returns per-token loss [...] (reduce with `.mean()` like F.cross_entropy)."""
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target)
