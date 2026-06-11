"""Communication primitives for Expert Parallelism (EP).

The core of EP is an all-to-all exchange of tokens: every rank sends each routed token
to the rank that owns its target expert (dispatch), and later sends the expert outputs
back to the rank the token came from (combine). This mirrors the dispatch/combine design
of DeepEP and MegaScale-MoE, implemented here with plain ``torch.distributed`` collectives
for clarity rather than performance.
"""

import torch
import torch.distributed as dist


class _AllToAll(torch.autograd.Function):
    """Differentiable all-to-all with variable per-rank split sizes.

    The backward of an all-to-all is simply the reverse all-to-all (input/output
    split sizes swapped), which lets gradients flow back to the sending ranks.
    """

    @staticmethod
    def forward(ctx, group, x, output_split_sizes, input_split_sizes):
        ctx.group = group
        ctx.input_split_sizes = input_split_sizes
        ctx.output_split_sizes = output_split_sizes
        ctx.input_shape = x.shape

        x = x.contiguous()
        output = x.new_empty((sum(output_split_sizes), *x.shape[1:]))
        dist.all_to_all_single(output, x, output_split_sizes, input_split_sizes, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.new_empty(ctx.input_shape)
        dist.all_to_all_single(
            grad_input,
            grad_output.contiguous(),
            ctx.input_split_sizes,
            ctx.output_split_sizes,
            group=ctx.group,
        )
        return None, grad_input, None, None


def all_to_all(x, output_split_sizes, input_split_sizes, group):
    """Send ``input_split_sizes[r]`` rows to rank ``r`` and receive ``output_split_sizes[r]`` rows from it."""
    return _AllToAll.apply(group, x, output_split_sizes, input_split_sizes)


def all_to_all_no_grad(x, output_split_sizes, input_split_sizes, group):
    """Non-differentiable all-to-all, used for routing metadata (e.g. expert indices)."""
    x = x.contiguous()
    output = x.new_empty((sum(output_split_sizes), *x.shape[1:]))
    dist.all_to_all_single(output, x, output_split_sizes, input_split_sizes, group=group)
    return output


# E4M3 has a maximum representable magnitude of 448; we scale each token row so its absmax maps there.
_FP8_E4M3_MAX = 448.0


def _per_token_quant_fp8(x):
    """Cast rows of ``x`` to float8_e4m3 with a per-row (per-token) scale. Returns (fp8, scale)."""
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-4)  # [N, 1], avoid div-by-zero
    scale = amax / _FP8_E4M3_MAX  # dequant multiplier
    q = (x.float() / scale).clamp_(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return q, scale.squeeze(-1).float()  # scale: [N]


class _AllToAllFp8(torch.autograd.Function):
    """Dispatch all-to-all that ships token features in **FP8 (E4M3)** to halve communication volume.

    DeepSeek-V3 / DeepEP style: the forward quantizes each token row to FP8 with a per-token scale,
    exchanges the 1-byte payload (NCCL sees it as ``uint8`` for portability) plus the small fp32
    scales, then dequantizes back to the original dtype on the receiver. The backward keeps gradients
    in the original (BF16/FP32) precision via a straight-through reverse all-to-all -- i.e. only the
    forward *activation* dispatch is low-precision, exactly the "FP8 dispatch, BF16 combine" recipe.
    """

    @staticmethod
    def forward(ctx, group, x, output_split_sizes, input_split_sizes):
        ctx.group = group
        ctx.input_split_sizes = input_split_sizes
        ctx.output_split_sizes = output_split_sizes
        ctx.input_shape = x.shape
        ctx.dtype = x.dtype

        q, scale = _per_token_quant_fp8(x.contiguous())  # [N, H] fp8, [N] fp32
        n_out = sum(output_split_sizes)

        recv_q = q.new_empty((n_out, x.shape[1]))
        # NCCL has no native float8 collective on many builds; bitcast to uint8 (same 1-byte layout).
        dist.all_to_all_single(
            recv_q.view(torch.uint8), q.view(torch.uint8), output_split_sizes, input_split_sizes, group=group
        )
        recv_scale = scale.new_empty((n_out,))
        dist.all_to_all_single(recv_scale, scale, output_split_sizes, input_split_sizes, group=group)

        return (recv_q.to(ctx.dtype) * recv_scale.unsqueeze(-1).to(ctx.dtype)).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: gradients travel back in full precision (combine/backward stays BF16).
        grad_input = grad_output.new_empty(ctx.input_shape)
        dist.all_to_all_single(
            grad_input,
            grad_output.contiguous(),
            ctx.input_split_sizes,
            ctx.output_split_sizes,
            group=ctx.group,
        )
        return None, grad_input, None, None


def all_to_all_fp8(x, output_split_sizes, input_split_sizes, group):
    """FP8 (E4M3) dispatch all-to-all: ~2x less wire traffic than BF16, straight-through backward."""
    return _AllToAllFp8.apply(group, x, output_split_sizes, input_split_sizes)


class _AllToAllAsync(torch.autograd.Function):
    """Async differentiable all-to-all for communication/computation overlap.

    The forward launches the collective with ``async_op=True`` and returns immediately, handing the
    work handle back through ``work_box``. The caller overlaps independent compute, then calls
    ``work.wait()`` before reading the output. The backward is the (synchronous) reverse all-to-all.
    """

    @staticmethod
    def forward(ctx, group, x, output_split_sizes, input_split_sizes, work_box):
        ctx.group = group
        ctx.input_split_sizes = input_split_sizes
        ctx.output_split_sizes = output_split_sizes
        ctx.input_shape = x.shape

        x = x.contiguous()
        output = x.new_empty((sum(output_split_sizes), *x.shape[1:]))
        work = dist.all_to_all_single(output, x, output_split_sizes, input_split_sizes, group=group, async_op=True)
        work_box.append(work)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.new_empty(ctx.input_shape)
        dist.all_to_all_single(
            grad_input,
            grad_output.contiguous(),
            ctx.input_split_sizes,
            ctx.output_split_sizes,
            group=ctx.group,
        )
        return None, grad_input, None, None, None


def all_to_all_async(x, output_split_sizes, input_split_sizes, group):
    """Differentiable all-to-all launched asynchronously. Returns (output, work); wait before use."""
    work_box = []
    output = _AllToAllAsync.apply(group, x, output_split_sizes, input_split_sizes, work_box)
    return output, work_box[0]


def all_to_all_no_grad_async(x, output_split_sizes, input_split_sizes, group):
    """Non-differentiable async all-to-all (routing metadata). Returns (output, work)."""
    x = x.contiguous()
    output = x.new_empty((sum(output_split_sizes), *x.shape[1:]))
    work = dist.all_to_all_single(output, x, output_split_sizes, input_split_sizes, group=group, async_op=True)
    return output, work


def exchange_split_sizes(input_split_sizes, group):
    """All-to-all the per-rank send counts so every rank learns how many rows it will receive.

    ``input_split_sizes[r]`` is how many rows this rank sends to rank ``r``; the returned
    tensor's entry ``r`` is how many rows this rank receives from rank ``r``.
    """
    output_split_sizes = torch.empty_like(input_split_sizes)
    dist.all_to_all_single(output_split_sizes, input_split_sizes, group=group)
    return output_split_sizes
