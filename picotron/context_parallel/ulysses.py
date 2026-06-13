"""DeepSpeed-Ulysses context parallelism (https://arxiv.org/abs/2309.14509).

Ring attention (``context_parallel.py``) keeps the sequence sharded the whole time and rotates K/V
around the CP ring. Ulysses takes the opposite approach: it briefly **re-shards the activations from
sequence-parallel to head-parallel** with two all-to-all collectives, runs an ordinary (full-sequence)
local attention on a subset of heads, then re-shards back.

    in  : [batch, num_heads, seq/cp, head_dim]        (sequence sharded across the CP group)
    a2a : [batch, num_heads/cp, seq, head_dim]         (head sharded, full sequence)
    attn: standard local attention over the full sequence, this rank's heads
    a2a : [batch, num_heads, seq/cp, head_dim]         (back to sequence sharded)

Trade-offs vs ring attention:
- Communication is O(2 * all-to-all) per attention, independent of sequence length, and uses the highly
  optimized all-to-all primitive instead of P2P sends — usually faster on a single node.
- The local attention is a plain dense/flash kernel (no online-softmax bookkeeping), so it composes with
  flash-attn for free.
- The catch: ``num_heads`` (per TP rank) must be divisible by ``cp_world_size`` — Ulysses parallelizes
  over heads, so it cannot scale past the head count, whereas ring attention has no such limit.

Ulysses shares the **CP process group** with ring attention (it is the same CP axis, a different
algorithm), which is why it lives in ``context_parallel/``.
"""

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

import picotron.process_group_manager as pgm

try:
    from flash_attn.flash_attn_interface import flash_attn_func
except ImportError:
    flash_attn_func = None


def _all_to_all(x: torch.Tensor, scatter_dim: int, gather_dim: int, group) -> torch.Tensor:
    """Split ``x`` along ``scatter_dim`` into one chunk per rank, exchange, and concatenate the received
    chunks along ``gather_dim``."""
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return x
    assert x.size(scatter_dim) % world_size == 0, (
        f"all-to-all scatter dim {scatter_dim} (size {x.size(scatter_dim)}) not divisible by cp {world_size}"
    )
    input_list = [t.contiguous() for t in torch.chunk(x, world_size, dim=scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


class _SeqAllToAll(torch.autograd.Function):
    """All-to-all that is its own inverse in the backward pass (swap scatter/gather dims)."""

    @staticmethod
    def forward(ctx, group, x, scatter_dim, gather_dim):
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        return _all_to_all(x, scatter_dim, gather_dim, group)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _all_to_all(grad_output, ctx.gather_dim, ctx.scatter_dim, ctx.group)
        return (None, grad_input, None, None)


# Tensor layout is [batch, heads, seq, head_dim]; we scatter heads (dim 1) and gather seq (dim 2).
HEAD_DIM_IDX = 1
SEQ_DIM_IDX = 2


def _local_attention(q, k, v, causal):
    """Standard attention on the full (gathered) sequence for this rank's heads. [b, h, s, d] -> same."""
    if os.getenv("FLASH_ATTEN", "1") == "1":
        if flash_attn_func is None:
            raise ImportError("FLASH_ATTEN=1 but flash-attn is not installed; set FLASH_ATTEN=0 for SDPA.")
        # flash-attn expects [b, s, h, d]
        out = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=causal)
        return out.transpose(1, 2)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def ulysses_attention(q, k, v, causal):
    """Ulysses attention. q/k/v are [batch, num_heads, seq/cp, head_dim] (GQA already expanded upstream).

    Returns [batch, num_heads, seq/cp, head_dim] to match ``ring_attention``'s contract.
    """
    group = pgm.process_group_manager.cp_group
    world_size = pgm.process_group_manager.cp_world_size
    if world_size == 1:
        return _local_attention(q, k, v, causal)

    num_heads = q.size(HEAD_DIM_IDX)
    assert num_heads % world_size == 0, (
        f"Ulysses needs num_heads ({num_heads}) divisible by cp_world_size ({world_size}); "
        f"use ring attention or reduce CP for head counts below the CP degree."
    )

    # sequence-sharded -> head-sharded (scatter heads, gather seq)
    q = _SeqAllToAll.apply(group, q, HEAD_DIM_IDX, SEQ_DIM_IDX)
    k = _SeqAllToAll.apply(group, k, HEAD_DIM_IDX, SEQ_DIM_IDX)
    v = _SeqAllToAll.apply(group, v, HEAD_DIM_IDX, SEQ_DIM_IDX)

    out = _local_attention(q, k, v, causal)  # [b, num_heads/cp, seq, head_dim]

    # head-sharded -> sequence-sharded (scatter seq, gather heads)
    out = _SeqAllToAll.apply(group, out, SEQ_DIM_IDX, HEAD_DIM_IDX)
    return out
