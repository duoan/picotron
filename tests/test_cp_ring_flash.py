"""Correctness for flash-based ring attention (RingFlashAttentionFunc).

Same idea as ``test_cp_ulysses.py``: a full-sequence single-GPU run is the reference, and a
sequence-sharded context-parallel run with ``CP_ATTENTION=ring`` (which dispatches to the flash ring
when ``FLASH_ATTEN=1``) must reproduce it. Flash kernels are bf16-only, so this runs in bf16 and uses a
relative tolerance instead of the fp32 bit-exact bound used for the SDPA paths.

    CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 2 tests/test_cp_ring_flash.py
"""

import datetime
import os
import types

import torch
import torch.distributed as dist
import torch.nn.functional as F

os.environ["FLASH_ATTEN"] = "1"  # flash ring + TritonRMSNorm + flash rotary
os.environ.setdefault("DEVICE", "cuda")

local_rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

dist.init_process_group(
    rank=global_rank, world_size=world_size, backend="nccl", init_method="env://",
    timeout=datetime.timedelta(minutes=3),
)

from picotron.process_group_manager import setup_process_group_manager  # noqa: E402

setup_process_group_manager(tp_size=1, cp_size=world_size, pp_size=1, dp_size=1)

import picotron.process_group_manager as pgm  # noqa: E402
from picotron.model import Llama  # noqa: E402
from picotron.utils import set_all_seed  # noqa: E402

set_all_seed(42)

BS, SEQ = 2, 64
DTYPE = torch.bfloat16


def make_config():
    return types.SimpleNamespace(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
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
    with torch.device(device):
        model = Llama(cfg)
    model.reset_parameters()
    return model.to(DTYPE).to(device)


def reldiff(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-6)


def main():
    cfg = make_config()
    cp_rank, cp = pgm.process_group_manager.cp_rank, pgm.process_group_manager.cp_world_size
    s_local = SEQ // cp

    g = torch.Generator().manual_seed(1234)
    input_ids = torch.randint(0, cfg.vocab_size, (BS, SEQ), generator=g).to(device)
    target_ids = torch.randint(0, cfg.vocab_size, (BS, SEQ), generator=g).to(device)

    # --- single-GPU reference (full sequence, flash) ---
    os.environ["CONTEXT_PARALLEL"] = "0"
    with force_cp1():
        ref_model = build_model(cfg)
        for p in ref_model.parameters():
            p.grad = None
        ref_logits = ref_model(input_ids)
        ref_loss = F.cross_entropy(
            ref_logits.float().reshape(-1, ref_logits.size(-1)), target_ids.reshape(-1), reduction="sum"
        )
        ref_loss.backward()
    ref_grads = {n: p.grad.detach().clone() for n, p in ref_model.named_parameters() if p.grad is not None}
    if global_rank == 0:
        print(f"[cp={cp}] reference loss (sum) = {ref_loss.item():.4f}")

    # --- flash ring context parallel (sequence sharded) ---
    cp_model = build_model(cfg)
    cp_model.load_state_dict({k: v.detach().clone() for k, v in ref_model.state_dict().items()})

    local_input = input_ids[:, cp_rank * s_local : (cp_rank + 1) * s_local].contiguous()
    local_target = target_ids[:, cp_rank * s_local : (cp_rank + 1) * s_local].contiguous()

    os.environ["CONTEXT_PARALLEL"] = "1"
    os.environ["CP_ATTENTION"] = "ring"
    for p in cp_model.parameters():
        p.grad = None
    local_logits = cp_model(local_input)
    local_loss = F.cross_entropy(
        local_logits.float().reshape(-1, local_logits.size(-1)), local_target.reshape(-1), reduction="sum"
    )
    local_loss.backward()
    os.environ["CONTEXT_PARALLEL"] = "0"

    ref_slice = ref_logits[:, cp_rank * s_local : (cp_rank + 1) * s_local]
    fwd_rel = reldiff(local_logits.float(), ref_slice.float())

    total_loss = torch.tensor(local_loss.item(), device=device)
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.cp_group)
    loss_rel = abs(total_loss.item() - ref_loss.item()) / (abs(ref_loss.item()) + 1e-6)

    grad_rel = 0.0
    for name, p in cp_model.named_parameters():
        if p.grad is None:
            continue
        g_full = p.grad.detach().float().clone()
        dist.all_reduce(g_full, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.cp_group)
        grad_rel = max(grad_rel, reldiff(g_full, ref_grads[name].float()))

    diffs = torch.tensor([fwd_rel, loss_rel, grad_rel], device=device)
    dist.all_reduce(diffs, op=dist.ReduceOp.MAX, group=pgm.process_group_manager.cp_group)
    fwd_rel, loss_rel, grad_rel = (d.item() for d in diffs)

    atol = 2e-2  # bf16 flash vs blocked flash + sharded summation
    assert fwd_rel < atol, f"ring-flash forward rel diff {fwd_rel:.3e} >= {atol}"
    assert loss_rel < atol, f"ring-flash loss rel diff {loss_rel:.3e} >= {atol}"
    assert grad_rel < atol, f"ring-flash grad rel diff {grad_rel:.3e} >= {atol}"

    if global_rank == 0:
        print(
            f"  flash ring attention: PASSED  "
            f"(fwd_rel={fwd_rel:.2e}, loss_rel={loss_rel:.2e}, grad_rel={grad_rel:.2e})"
        )
        print(f"[cp={cp}] flash ring-attention test passed")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
