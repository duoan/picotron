"""Communication primitives for sequence parallelism.

Sequence parallelism (Megatron, "Reducing Activation Recomputation in Large Transformer Models",
https://arxiv.org/abs/2205.05198) is *not* a separate parallelism axis: it has no process group of its
own and reuses the **tensor-parallel group**. In the regions between TP blocks (RMSNorm + residual) the
activations are normally replicated on every TP rank; sequence parallelism shards them along the
sequence dimension instead, so that activation memory drops by ``tp_world_size``. The price is paid by
swapping the TP block's collectives:

    plain TP : f = copy / all-reduce            g = all-reduce / copy
    seq-par  : f = all-gather / reduce-scatter  g = reduce-scatter / all-gather

Since ``all-reduce = reduce-scatter + all-gather`` over the same TP ranks, the total communication
volume is unchanged — only the activation memory shrinks.

picotron activations are [batch, seq, hidden]; we shard along dim=1 (seq).
"""

import torch
import torch.distributed as dist

import picotron.process_group_manager as pgm

SEQ_DIM = 1


def all_gather_along_seq(x: torch.Tensor) -> torch.Tensor:
    """Concatenate the per-rank sequence shards into the full sequence (over the TP group)."""
    world_size = pgm.process_group_manager.tp_world_size
    if world_size == 1:
        return x
    x = x.contiguous()
    tensor_list = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(tensor_list, x, group=pgm.process_group_manager.tp_group)
    return torch.cat(tensor_list, dim=SEQ_DIM).contiguous()


def reduce_scatter_along_seq(x: torch.Tensor) -> torch.Tensor:
    """Sum the (partial) tensors across TP ranks and keep only this rank's seq shard."""
    world_size = pgm.process_group_manager.tp_world_size
    if world_size == 1:
        return x
    assert x.size(SEQ_DIM) % world_size == 0, f"seq dim {x.size(SEQ_DIM)} not divisible by tp {world_size}"
    chunks = [c.contiguous() for c in torch.chunk(x, world_size, dim=SEQ_DIM)]
    output = torch.empty_like(chunks[pgm.process_group_manager.tp_rank])
    dist.reduce_scatter(output, chunks, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.tp_group)
    return output
