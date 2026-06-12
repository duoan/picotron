"""Capacity-capped, static-memory MoE dispatch (GShard / Switch / MAI style).

The dropless paths (``_moe_naive`` / ``_moe_tiled``) size every buffer to the *actual* per-expert
token counts, which vary every step with the router's decisions. Under load imbalance this causes
memory swings, allocator fragmentation, and OOMs at scale. This module instead runs the experts on
**fixed-capacity** buffers: each (rank, expert) processes exactly ``capacity`` tokens per round, so
all activations -- forward and backward -- have a constant, predictable shape.

Two modes (both reuse the same fixed buffers):

* **capacity-capped** (``dropless=False``): one round. Tokens beyond ``capacity`` for an expert are
  *dropped* (they skip the MoE and contribute 0). This is the classic Switch/GShard behaviour.
* **static-memory dropless** (``dropless=True``): multiple capped rounds. Round ``r`` processes an
  expert's tokens ``[r*capacity, (r+1)*capacity)``; the loop repeats until every token is processed
  (or ``max_rounds`` is hit). No tokens are dropped, yet each round's memory is still fixed. The
  round count is reduced with a MAX over the EP group so all ranks issue the same collectives.

Everything here is plain autograd (a differentiable equal-split all-to-all + index_add scatter /
gather), so no custom backward is needed -- the backward also runs on fixed-capacity buffers.

Layout note: global expert ``e`` is owned by rank ``e // num_local_experts`` with local id
``e % num_local_experts``; experts are contiguous per rank, so a ``[num_experts, C, d]`` send buffer
reshapes to ``[ep, num_local_experts, C, d]`` and the EP all-to-all is a balanced exchange of equal
``num_local_experts * C`` row blocks.
"""

import torch
import torch.distributed as dist

from picotron.expert_parallel.ep_communications import all_to_all


def compute_capacity(num_tokens, top_k, num_experts, capacity_factor):
    """Per-(rank, expert) token capacity for one round.

    ``capacity_factor * (tokens routed on this rank / num_experts)``, rounded up, min 1. Deterministic
    from config (``num_tokens`` is the fixed per-rank token count), so identical on every rank -- the
    all-to-all stays balanced.
    """
    avg = (num_tokens * top_k) / num_experts
    return max(1, int(capacity_factor * avg + 0.999))


def _slot_assignment(expert_idx, num_experts):
    """For each routed row, its position within its expert's queue (0,1,2,... per expert).

    Returns ``within`` (int64 [N]): row r is the ``within[r]``-th token routed to expert ``expert_idx[r]``
    on this rank. Used to bucket tokens into fixed-capacity slots / rounds.
    """
    order = torch.argsort(expert_idx, stable=True)
    sorted_e = expert_idx[order]
    counts = torch.bincount(sorted_e, minlength=num_experts)
    offsets = torch.cumsum(counts, 0) - counts  # start index of each expert in the sorted array
    within_sorted = torch.arange(expert_idx.numel(), device=expert_idx.device) - offsets[sorted_e]
    within = torch.empty_like(within_sorted)
    within[order] = within_sorted
    return within, counts


def capacity_moe(
    routed,
    expert_idx,
    weights,
    token_idx,
    num_tokens,
    expert_dim,
    num_experts,
    num_local_experts,
    ep_world_size,
    group,
    expert_fn,
    capacity,
    dropless,
    max_rounds,
):
    """Run the routed experts on fixed-capacity buffers and scatter the result back to tokens.

    Args:
        routed: [N, expert_dim] features, one row per (token, selected expert).
        expert_idx: [N] global expert id for each routed row.
        weights: [N] gate weight for each routed row.
        token_idx: [N] source token id for each routed row.
        num_tokens: T, number of tokens (output rows).
        expert_fn: callable(x_grouped [num_local_experts*ep*C, d], counts [num_local_experts]) -> y,
            running the local experts on rows grouped by local expert (uniform ``ep*C`` counts here).
        capacity: C, tokens per (rank, expert) per round.
        dropless: if True, loop rounds until all tokens are processed (no drop); else one round (drop).
        max_rounds: hard cap on rounds (bounds worst-case cost; overflow beyond this is dropped).

    Returns:
        [num_tokens, expert_dim] combined output (sum of gate-weighted expert outputs per token).
    """
    device = routed.device
    within, counts = _slot_assignment(expert_idx, num_experts)

    if dropless:
        # Rounds needed locally, then MAX across EP ranks so everyone issues the same collectives.
        local_rounds = int((counts.max().item() + capacity - 1) // capacity) if counts.numel() else 1
        rounds = max(1, min(local_rounds, max_rounds))
        if ep_world_size > 1:
            t = torch.tensor([rounds], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
            rounds = int(t.item())
    else:
        rounds = 1

    output = torch.zeros(num_tokens, expert_dim, dtype=routed.dtype, device=device)
    for r in range(rounds):
        round_slot = within - r * capacity  # slot within this round's buffer
        keep = (round_slot >= 0) & (round_slot < capacity)
        if not bool(keep.any()):
            # Still must participate in the collectives (lockstep) with empty buffers.
            if ep_world_size > 1:
                _round_alltoall_empty(expert_dim, num_experts, num_local_experts, ep_world_size, capacity, group, routed.dtype, device, expert_fn)
            continue
        output = output + _one_round(
            routed, expert_idx, weights, token_idx, round_slot, keep,
            num_tokens, expert_dim, num_experts, num_local_experts, ep_world_size, group, expert_fn, capacity,
        )
    return output


def _scatter_to_buffer(routed, target_index, e_c_rows, expert_dim):
    """Place kept routed rows into a flat [num_experts*C, expert_dim] buffer at ``e*C + slot``."""
    buf = torch.zeros(e_c_rows, expert_dim, dtype=routed.dtype, device=routed.device)
    return buf.index_add(0, target_index, routed)


def _one_round(
    routed, expert_idx, weights, token_idx, round_slot, keep,
    num_tokens, expert_dim, num_experts, num_local_experts, ep_world_size, group, expert_fn, capacity,
):
    e_c = num_experts * capacity
    kept_e = expert_idx[keep]
    kept_slot = round_slot[keep]
    target = (kept_e * capacity + kept_slot).to(torch.int64)  # [num_experts*C] addressing

    # Scatter token features into the fixed [num_experts*C, d] send buffer.
    send = _scatter_to_buffer(routed[keep], target, e_c, expert_dim)  # [E*C, d]

    if ep_world_size > 1:
        splits = [num_local_experts * capacity] * ep_world_size
        recv = all_to_all(send, splits, splits, group)  # [E*C, d]; block s = rank s's tokens for our experts
        recv = recv.view(ep_world_size, num_local_experts, capacity, expert_dim)
    else:
        recv = send.view(1, num_local_experts, capacity, expert_dim)

    # Group rows by LOCAL expert for the grouped GEMM: [num_local_experts, ep*C, d].
    grouped = recv.permute(1, 0, 2, 3).reshape(num_local_experts * ep_world_size * capacity, expert_dim)
    cnt = torch.full((num_local_experts,), ep_world_size * capacity, device=routed.device, dtype=torch.int64)
    out_grouped = expert_fn(grouped, cnt)
    out = out_grouped.view(num_local_experts, ep_world_size, capacity, expert_dim).permute(1, 0, 2, 3).reshape(e_c, expert_dim)

    if ep_world_size > 1:
        out = all_to_all(out, splits, splits, group)  # back to the sender frame: [E*C, d]

    # Gather each kept row's expert output, weight it, and scatter-add back to its token.
    contrib = out[target] * weights[keep].unsqueeze(-1).to(out.dtype)
    y = torch.zeros(num_tokens, expert_dim, dtype=routed.dtype, device=routed.device)
    return y.index_add(0, token_idx[keep], contrib)


def _round_alltoall_empty(expert_dim, num_experts, num_local_experts, ep_world_size, capacity, group, dtype, device, expert_fn):
    """Issue this round's collectives with zero-filled fixed buffers (keeps ranks in lockstep)."""
    e_c = num_experts * capacity
    send = torch.zeros(e_c, expert_dim, dtype=dtype, device=device)
    splits = [num_local_experts * capacity] * ep_world_size
    recv = all_to_all(send, splits, splits, group).view(ep_world_size, num_local_experts, capacity, expert_dim)
    grouped = recv.permute(1, 0, 2, 3).reshape(num_local_experts * ep_world_size * capacity, expert_dim)
    cnt = torch.full((num_local_experts,), ep_world_size * capacity, device=device, dtype=torch.int64)
    out_grouped = expert_fn(grouped, cnt)
    out = out_grouped.view(num_local_experts, ep_world_size, capacity, expert_dim).permute(1, 0, 2, 3).reshape(e_c, expert_dim)
    all_to_all(out, splits, splits, group)
