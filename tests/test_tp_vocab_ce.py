"""Bit-exact check for vocab-parallel cross-entropy.

Each rank builds the *same* full logits, computes the dense reference loss/grad with
``F.cross_entropy``, then shards the logits along the vocabulary and compares the loss and gradient
produced by ``vocab_parallel_cross_entropy`` (which only communicates per-token scalars) against the
reference. This is the loss that pairs with a vocab-sharded ``final_proj`` (``gather_output=False``)
to avoid ever materializing the full [tokens, vocab] logits.

    torchrun --nproc_per_node 2 tests/test_tp_vocab_ce.py
    torchrun --nproc_per_node 4 tests/test_tp_vocab_ce.py
"""

import datetime
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

local_rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

backend = os.environ.get("TP_TEST_BACKEND", "nccl")
if backend == "nccl":
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
else:
    device = torch.device("cpu")

dist.init_process_group(
    rank=global_rank, world_size=world_size, backend=backend, init_method="env://",
    timeout=datetime.timedelta(minutes=3),
)

from picotron.process_group_manager import setup_process_group_manager  # noqa: E402

setup_process_group_manager(tp_size=world_size, cp_size=1, pp_size=1, dp_size=1)

import picotron.process_group_manager as pgm  # noqa: E402
from picotron.tensor_parallel.tp_communications import vocab_parallel_cross_entropy  # noqa: E402

N, VOCAB = 16, 64
assert VOCAB % world_size == 0


def main():
    # Identical full logits on every rank (seeded), then take this rank's vocab shard.
    torch.manual_seed(0)
    logits_full = torch.randn(N, VOCAB, device=device)
    target = torch.randint(0, VOCAB, (N,), device=device, generator=torch.Generator(device=device).manual_seed(7))

    # Reference: dense cross-entropy on the full logits.
    ref_logits = logits_full.clone().requires_grad_(True)
    ref_loss = F.cross_entropy(ref_logits, target, reduction="mean")
    ref_loss.backward()

    # Vocab-parallel: this rank holds a [N, VOCAB/tp] slice.
    shard = logits_full.chunk(world_size, dim=-1)[pgm.process_group_manager.tp_rank].clone().requires_grad_(True)
    loss = vocab_parallel_cross_entropy(shard, target).mean()
    loss.backward()

    loss_diff = abs(loss.item() - ref_loss.item())
    ref_grad_shard = ref_logits.grad.chunk(world_size, dim=-1)[pgm.process_group_manager.tp_rank]
    grad_diff = (shard.grad - ref_grad_shard).abs().max().item()

    assert loss_diff < 1e-5, f"loss diff {loss_diff:.3e}"
    assert grad_diff < 1e-6, f"grad diff {grad_diff:.3e}"
    if global_rank == 0:
        print(f"[tp={world_size}] vocab-parallel CE PASSED (loss_diff={loss_diff:.2e}, grad_diff={grad_diff:.2e}) ✅")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
