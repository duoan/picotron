"""Tiled MoE with a single custom autograd Function that pipelines BOTH forward and backward.

This is the "full" UniEP/MegaScale-style overlap: the dispatch / expert-GEMM / combine of one MoE
layer is split into ``num_tiles`` token tiles and run as a software pipeline so communication and
computation overlap in *both* directions.

Why one big autograd Function (instead of composing the differentiable a2a + expert ops)? Because
PyTorch's autograd engine schedules backward nodes serially and we cannot make it interleave the
reverse all-to-all of one stage with the compute of another. To overlap the backward we must own the
backward schedule. The key independence we exploit (Megatron's trick): the expert FFN backward
produces ``d_input`` and ``d_weight`` from the same upstream grad, and they are independent -- so we
launch the reverse-dispatch all-to-all on ``d_input`` and compute ``d_weight`` while it is in flight.

Forward pipeline   : dispatch(i+1) || experts(i) || combine(i-1)
Backward pipeline  : combine_rev(i+1) || [experts_bwd(i): launch dispatch_rev || compute d_weight]

Collective-count consistency: every rank issues exactly ``num_tiles`` dispatch/combine collectives
(and their reverses) regardless of how many tokens it actually routes -- empty tiles still call the
(zero-sized) collective so all ranks stay in lockstep.

Real overlap requires multi-GPU NCCL; on a single GPU / gloo the result is identical to the plain
path, just without the speedup.
"""

from dataclasses import dataclass

import torch

from picotron.expert_parallel import megakernel
from picotron.expert_parallel.ep_communications import all_to_all_no_grad_async


@dataclass
class TilePlan:
    """Per-tile dispatch plan: a contiguous row slice [lo, hi) of the routed tokens, sorted by
    destination rank, plus the all-to-all split sizes and each row's local expert id."""

    lo: int
    hi: int
    order: torch.Tensor  # sort routed[lo:hi] by destination rank
    inv_order: torch.Tensor  # undo `order`
    local_expert_sorted: torch.Tensor  # local expert id of each (sorted) row
    in_list: list  # rows sent to each rank (dispatch input splits)
    out_list: list  # rows received from each rank (dispatch output splits)


@dataclass
class TiledMeta:
    plans: list  # list[TilePlan]
    group: object  # EP process group
    num_local_experts: int


def _ffn_forward(x_sorted, counts, gate_w, up_w, down_w):
    """Expert SwiGLU FFN forward on expert-grouped tokens (Triton kernel on GPU, torch on CPU)."""
    if x_sorted.is_cuda and megakernel.cuda_available():
        return megakernel.fused_moe_forward(x_sorted, counts, gate_w, up_w, down_w)
    return megakernel.torch_fused_ffn(x_sorted, counts, gate_w, up_w, down_w, None)


class _TiledMoEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, routed, gate_w, up_w, down_w, meta):
        plans = meta.plans
        group = meta.group
        num_local_experts = meta.num_local_experts
        n = len(plans)

        def dispatch(p):
            routed_sorted = routed[p.lo : p.hi][p.order]
            recv, recv_work = all_to_all_no_grad_async(routed_sorted, p.out_list, p.in_list, group)
            recv_le, le_work = all_to_all_no_grad_async(p.local_expert_sorted, p.out_list, p.in_list, group)
            return recv, recv_work, recv_le, le_work

        recv, recv_work, recv_le, le_work = dispatch(plans[0])
        per_tile = [None] * n  # saved (recv_sorted, counts, rorder, rinv) for backward recompute
        combine_pending = []
        for i in range(n):
            nxt = dispatch(plans[i + 1]) if i + 1 < n else None
            recv_work.wait()
            le_work.wait()

            # Group received tokens by their local expert id for the grouped-GEMM kernel.
            rorder = torch.argsort(recv_le, stable=True)
            rinv = torch.empty_like(rorder)
            rinv[rorder] = torch.arange(rorder.numel(), device=rorder.device)
            recv_sorted = recv[rorder]
            counts = torch.bincount(recv_le, minlength=num_local_experts)

            out_sorted = _ffn_forward(recv_sorted, counts, gate_w, up_w, down_w)
            out_i = out_sorted[rinv]  # back to receive (source-grouped) order
            combined, combine_work = all_to_all_no_grad_async(out_i, plans[i].in_list, plans[i].out_list, group)
            combine_pending.append((combined, combine_work, plans[i].inv_order))
            per_tile[i] = (recv_sorted, counts, rorder, rinv)
            if nxt is not None:
                recv, recv_work, recv_le, le_work = nxt

        tiles_out = []
        for combined, combine_work, inv_order in combine_pending:
            combine_work.wait()
            tiles_out.append(combined[inv_order])
        y = torch.cat(tiles_out, dim=0)

        ctx.save_for_backward(gate_w, up_w, down_w)
        ctx.per_tile = per_tile
        ctx.meta = meta
        ctx.routed_shape = tuple(routed.shape)
        ctx.routed_dtype = routed.dtype
        ctx.routed_device = routed.device
        return y

    @staticmethod
    def backward(ctx, grad_y):
        plans = ctx.meta.plans
        group = ctx.meta.group
        gate_w, up_w, down_w = ctx.saved_tensors
        per_tile = ctx.per_tile
        n = len(plans)

        d_gate = torch.zeros_like(gate_w)
        d_up = torch.zeros_like(up_w)
        d_down = torch.zeros_like(down_w)
        d_routed = torch.zeros(ctx.routed_shape, dtype=ctx.routed_dtype, device=ctx.routed_device)

        def combine_rev(i):
            # y_tile = combined[inv_order]  =>  d_combined = d_y_tile[order]; reverse the combine a2a.
            d_y_tile = grad_y[plans[i].lo : plans[i].hi]
            d_combined = d_y_tile[plans[i].order]
            return all_to_all_no_grad_async(d_combined, plans[i].out_list, plans[i].in_list, group)

        use_kernel = grad_y.is_cuda and megakernel.cuda_available()
        H = gate_w.shape[2]

        def tile_backward(recv_sorted, counts, d_out_sorted):
            """Compute d_input now; return (d_recv_sorted, deferred_wgrad) where deferred_wgrad() ->
            (dg, du, dd). Splitting the two lets the caller overlap the reverse a2a (on d_input) with
            the weight-grad compute."""
            if recv_sorted.shape[0] == 0:
                return recv_sorted.new_zeros((0, H)), lambda: None
            if use_kernel:
                # Explicit dgrad first (single recompute, no autograd double traversal).
                dx_sorted, a_c, di_c = megakernel.fused_moe_dgrad(
                    recv_sorted, counts, gate_w, up_w, down_w, d_out_sorted
                )
                return dx_sorted, lambda: megakernel.fused_moe_wgrad(recv_sorted, counts, a_c, di_c, d_out_sorted)
            dx_sorted, dgw, duw, ddw = megakernel.torch_moe_backward(
                recv_sorted, counts, gate_w, up_w, down_w, d_out_sorted
            )
            return dx_sorted, lambda: (dgw, duw, ddw)

        d_out, a_work = combine_rev(0)
        dispatch_pending = []
        for i in range(n):
            nxt = combine_rev(i + 1) if i + 1 < n else None
            a_work.wait()  # overlaps combine_rev(i+1) with this tile's backward compute

            recv_sorted, counts, rorder, rinv = per_tile[i]
            dx_sorted, deferred_wgrad = tile_backward(recv_sorted, counts, d_out[rorder])
            d_recv = dx_sorted[rinv]  # back to receive (source-grouped) order (no-op when empty)

            # Launch reverse-dispatch a2a (sends d_input back to the token's origin)...
            d_routed_sorted, c_work = all_to_all_no_grad_async(d_recv, plans[i].in_list, plans[i].out_list, group)

            # ...and overlap it with the (independent) weight-gradient compute.
            grads = deferred_wgrad()
            if grads is not None:
                d_gate += grads[0]
                d_up += grads[1]
                d_down += grads[2]

            dispatch_pending.append((d_routed_sorted, c_work, plans[i].inv_order, plans[i].lo, plans[i].hi))
            if nxt is not None:
                d_out, a_work = nxt

        for d_routed_sorted, c_work, inv_order, lo, hi in dispatch_pending:
            c_work.wait()
            # routed_sorted = routed[lo:hi][order]  =>  d_routed[lo:hi] = d_routed_sorted[inv_order].
            d_routed[lo:hi] = d_routed_sorted[inv_order]

        return d_routed, d_gate, d_up, d_down, None


def tiled_moe(routed, gate_w, up_w, down_w, meta):
    """Differentiable tiled MoE (dispatch + experts + combine) with fwd/bwd comm-compute overlap."""
    return _TiledMoEFunction.apply(routed, gate_w, up_w, down_w, meta)
