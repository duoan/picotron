"""Bit-exact correctness for DeepSpeed-Ulysses context parallelism.

Every rank builds the *same* full (replicated) Llama as a single-GPU reference and runs a full-sequence
forward + backward. It then builds an identical model, shards the sequence contiguously across the CP
group (exactly how ``data.py`` shards it), and runs the forward/backward with ``CP_ATTENTION=ulysses``.

Under pure context parallelism every parameter is replicated across CP ranks, and each rank only sees a
slice of the tokens, so each rank's gradient is a *partial sum* over its token shard. Summing the
gradients over the CP group (and using a sum-reduced loss) recovers the single-GPU reference exactly.

    CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 2 tests/test_cp_ulysses.py
    CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 4 tests/test_cp_ulysses.py
"""

import datetime
import os
import types

import torch
import torch.distributed as dist
import torch.nn.functional as F

os.environ.setdefault("FLASH_ATTEN", "0")  # SDPA + LlamaRMSNorm, no flash-attn / triton dependency

local_rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

backend = os.environ.get("CP_TEST_BACKEND", "nccl")
if backend == "nccl":
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    os.environ.setdefault("DEVICE", "cuda")
else:
    device = torch.device("cpu")
    os.environ["DEVICE"] = "cpu"

dist.init_process_group(
    rank=global_rank, world_size=world_size, backend=backend, init_method="env://",
    timeout=datetime.timedelta(minutes=3),
)

from picotron.process_group_manager import setup_process_group_manager  # noqa: E402

setup_process_group_manager(tp_size=1, cp_size=world_size, pp_size=1, dp_size=1)

import picotron.process_group_manager as pgm  # noqa: E402
from picotron.model import Llama  # noqa: E402
from picotron.utils import set_all_seed  # noqa: E402

set_all_seed(42)

BS, SEQ = 2, 16


def make_config():
    return types.SimpleNamespace(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        max_position_embeddings=SEQ,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        num_experts=1,
        num_experts_per_tok=1,
        num_shared_experts=0,
        norm_topk_prob=True,
    )


class force_cp1:
    """Build/run the single-GPU reference as if cp_world_size == 1 (full sequence, un-sliced RoPE)."""

    def __enter__(self):
        self.cp_ws = pgm.process_group_manager.cp_world_size
        self.cp_r = pgm.process_group_manager.cp_rank
        pgm.process_group_manager.cp_world_size = 1
        pgm.process_group_manager.cp_rank = 0

    def __exit__(self, *a):
        pgm.process_group_manager.cp_world_size = self.cp_ws
        pgm.process_group_manager.cp_rank = self.cp_r


def build_model(cfg):
    torch.manual_seed(0)
    model = Llama(cfg).to(torch.float32).to(device)
    model.reset_parameters()
    return model


def main():
    cfg = make_config()
    cp_rank, cp = pgm.process_group_manager.cp_rank, pgm.process_group_manager.cp_world_size
    s_local = SEQ // cp

    g = torch.Generator().manual_seed(1234)
    input_ids = torch.randint(0, cfg.vocab_size, (BS, SEQ), generator=g).to(device)
    target_ids = torch.randint(0, cfg.vocab_size, (BS, SEQ), generator=g).to(device)

    # --- single-GPU reference (full sequence) ---
    os.environ["CONTEXT_PARALLEL"] = "0"
    with force_cp1():
        ref_model = build_model(cfg)
        for p in ref_model.parameters():
            p.grad = None
        ref_logits = ref_model(input_ids)  # [B, S, V]
        ref_loss = F.cross_entropy(ref_logits.reshape(-1, ref_logits.size(-1)), target_ids.reshape(-1), reduction="sum")
        ref_loss.backward()
    ref_grads = {n: p.grad.detach().clone() for n, p in ref_model.named_parameters() if p.grad is not None}
    if global_rank == 0:
        print(f"[cp={cp}] reference loss (sum) = {ref_loss.item():.4f}")

    # --- Ulysses context parallel (sequence sharded contiguously, exactly like data.py) ---
    cp_model = build_model(cfg)  # RoPE is sliced to this rank's shard in DecoderLayer.__init__
    # Same seed builds identical weights; copy explicitly to be safe (RoPE is an attribute, not in state_dict).
    cp_model.load_state_dict({k: v.detach().clone() for k, v in ref_model.state_dict().items()})

    local_input = input_ids[:, cp_rank * s_local : (cp_rank + 1) * s_local].contiguous()
    local_target = target_ids[:, cp_rank * s_local : (cp_rank + 1) * s_local].contiguous()

    os.environ["CONTEXT_PARALLEL"] = "1"
    os.environ["CP_ATTENTION"] = "ulysses"
    for p in cp_model.parameters():
        p.grad = None
    local_logits = cp_model(local_input)  # [B, s_local, V]
    local_loss = F.cross_entropy(
        local_logits.reshape(-1, local_logits.size(-1)), local_target.reshape(-1), reduction="sum"
    )
    local_loss.backward()
    os.environ["CONTEXT_PARALLEL"] = "0"

    atol = 2e-4  # fp32; all-to-all + sharded summation reorder the float accumulation

    # forward: this rank's logits must match the reference slice
    ref_slice = ref_logits[:, cp_rank * s_local : (cp_rank + 1) * s_local]
    fwd_diff = (local_logits - ref_slice).abs().max().item()

    # loss: sum over CP ranks == reference sum loss
    total_loss = torch.tensor(local_loss.item(), device=device)
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.cp_group)
    loss_diff = abs(total_loss.item() - ref_loss.item())

    # grads: every param is replicated; sum the partial token-shard grads over the CP group
    grad_diff = 0.0
    for name, p in cp_model.named_parameters():
        if p.grad is None:
            continue
        g_full = p.grad.detach().clone()
        dist.all_reduce(g_full, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.cp_group)
        grad_diff = max(grad_diff, (g_full - ref_grads[name]).abs().max().item())

    # reduce diagnostics across ranks so a failure on any rank is visible
    diffs = torch.tensor([fwd_diff, grad_diff], device=device)
    dist.all_reduce(diffs, op=dist.ReduceOp.MAX, group=pgm.process_group_manager.cp_group)
    fwd_diff, grad_diff = diffs[0].item(), diffs[1].item()

    assert fwd_diff < atol, f"Ulysses forward diff {fwd_diff:.3e} >= {atol}"
    assert loss_diff < atol, f"Ulysses loss diff {loss_diff:.3e} >= {atol}"
    assert grad_diff < atol, f"Ulysses grad diff {grad_diff:.3e} >= {atol}"

    if global_rank == 0:
        print(
            f"  Ulysses context parallel: PASSED  "
            f"(fwd_diff={fwd_diff:.2e}, loss_diff={loss_diff:.2e}, grad_diff={grad_diff:.2e})"
        )
        print(f"[cp={cp}] All Ulysses context-parallel tests passed ✅")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
