"""Optional DeepEP backend for the MoE dispatch/combine all-to-all.

`DeepEP <https://github.com/deepseek-ai/DeepEP>`_ replaces our plain ``torch.distributed`` all-to-all
with hand-written CUDA kernels that saturate NVLink/RDMA bandwidth at near-zero SM occupancy. This
module wraps DeepEP's classic intranode ``Buffer`` (NVLink) so ``MoELayer`` can use it as a drop-in
backend, and **falls back to the torch path whenever DeepEP is unavailable**.

Why the classic ``Buffer`` and not the new V2 ``ElasticBuffer``: ElasticBuffer routes everything
through the NCCL GIN (in-network) transport, which isn't exposed on every cluster (e.g. Modal's
single-node H100 boxes), so it aborts at init. The classic ``Buffer`` intranode path only needs
NVLink P2P and works anywhere two Hopper GPUs share NVLink.

IMPORTANT — hardware: DeepEP requires **Hopper (SM90) or newer**. On older GPUs (e.g. A100 = sm_80)
``deepep_available()`` returns False and the layer transparently uses the torch backend.

Autograd: ``dispatch`` (gather tokens to their expert ranks) and ``combine`` (sum the copies back)
are linear transposes of each other, so we wrap them as ``autograd.Function``s whose backward is the
other op. The top-k gate weighting is applied in the (differentiable) expert step, so ``combine``
runs as a plain sum. Gradients therefore flow to the experts and the up/down projections; the router
gate is not differentiated through DeepEP's (non-differentiable) weight transport.
"""

import os

import torch

try:  # DeepEP is an optional dependency (Hopper SM90+); absence just disables this backend.
    import deep_ep  # type: ignore

    _IMPORT_OK = True
except Exception:
    deep_ep = None
    _IMPORT_OK = False

_NVL_BYTES = int(os.environ.get("EP_DEEPEP_NVL_BYTES", 1 << 30))  # 1 GiB NVLink staging buffer


def deepep_available():
    """True only if DeepEP is importable AND the current GPU is Hopper (SM90) or newer."""
    if not _IMPORT_OK or not torch.cuda.is_available():
        return False
    if os.environ.get("EP_DISABLE_DEEPEP", "0") == "1":
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


_buffer = None  # cached intranode Buffer (one per process; re-created if the group changes)


def _get_buffer(group):
    global _buffer
    if _buffer is None or _buffer.group is not group:
        _buffer = deep_ep.Buffer(group, _NVL_BYTES, 0)  # NVL-only: no RDMA bytes
    return _buffer


class _Dispatch(torch.autograd.Function):
    """Differentiable DeepEP dispatch. Backward (transpose of a gather) is a weightless combine."""

    @staticmethod
    def forward(ctx, buffer, x, topk_idx, topk_weights, num_experts, box):
        num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = buffer.get_dispatch_layout(
            topk_idx, num_experts
        )
        recv_x, recv_topk_idx, recv_topk_weights, counts, handle, _ = buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
        )
        ctx.buffer = buffer
        ctx.handle = handle
        # Non-tensor / non-differentiable outputs go back to the caller via a side channel.
        box["handle"] = handle
        box["recv_topk_idx"] = recv_topk_idx
        box["recv_topk_weights"] = recv_topk_weights
        box["counts"] = counts
        return recv_x

    @staticmethod
    def backward(ctx, grad_recv_x):
        grad_x, _, _ = ctx.buffer.combine(grad_recv_x.contiguous(), handle=ctx.handle)
        return None, grad_x, None, None, None, None


class _Combine(torch.autograd.Function):
    """Differentiable DeepEP combine. Backward (transpose of a sum) is a cached dispatch."""

    @staticmethod
    def forward(ctx, buffer, handle, y):
        combined, _, _ = buffer.combine(y, handle=handle)
        ctx.buffer = buffer
        ctx.handle = handle
        return combined

    @staticmethod
    def backward(ctx, grad_combined):
        grad_y, _, _, _, _, _ = ctx.buffer.dispatch(grad_combined.contiguous(), handle=ctx.handle)
        return None, None, grad_y


def dispatch(group, x, topk_idx, topk_weights, num_experts):
    """Route ``x`` [T, hidden] (bf16) to the experts via NVLink.

    Returns (recv_x, handle, recv_topk_idx, recv_topk_weights, counts) where ``recv_topk_idx`` holds
    local expert ids (``-1`` for experts not on this rank) and ``counts`` is the per-local-expert
    received-token count.
    """
    buffer = _get_buffer(group)
    box = {}
    recv_x = _Dispatch.apply(buffer, x, topk_idx.to(torch.int64), topk_weights.float(), num_experts, box)
    return recv_x, box["handle"], box["recv_topk_idx"], box["recv_topk_weights"], box["counts"]


def combine(group, y, handle):
    """Sum the expert outputs ``y`` [T_recv, hidden] (bf16) back to their source tokens."""
    return _Combine.apply(_get_buffer(group), handle, y)
