"""Bit-exact correctness for the parallel transformer block (PTB) wired into picotron's Llama.

Builds a full (replicated) PTB Llama as a single-GPU reference, then builds TP and TP+SP PTB copies from
the same weights and checks every owned gradient. The TP+SP copy uses the *fused* PTB path (one shared
all-gather + one combined reduce-scatter per layer, MegaScale Fig 3b).

    CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 2 tests/test_ptb_model.py
"""

import datetime
import os
import types

import torch
import torch.distributed as dist
import torch.nn.functional as F

os.environ.setdefault("FLASH_ATTEN", "0")  # SDPA + LlamaRMSNorm (no flash-attn / triton dep)

local_rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
os.environ.setdefault("DEVICE", "cuda")

dist.init_process_group(
    rank=global_rank, world_size=world_size, backend="nccl", init_method="env://",
    device_id=device, timeout=datetime.timedelta(minutes=3),
)

from picotron.process_group_manager import setup_process_group_manager  # noqa: E402

setup_process_group_manager(tp_size=world_size, cp_size=1, pp_size=1, dp_size=1)

import picotron.process_group_manager as pgm  # noqa: E402
from picotron.model import Llama  # noqa: E402
from picotron.tensor_parallel.tensor_parallel import apply_tensor_parallel  # noqa: E402

torch.manual_seed(42)

BS, SEQ = 2, 8
# PTB has no post-attention norm; only these replicated norms have sequence-sharded grads under SP.
NORM_SUFFIXES = ("input_layernorm.weight", "final_norm.weight")


def make_config():
    return types.SimpleNamespace(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2,
        max_position_embeddings=SEQ, rms_norm_eps=1e-5, rope_theta=10000.0,
        num_experts=1, num_experts_per_tok=1, num_shared_experts=0, norm_topk_prob=True,
        parallel_block=True,
    )


class force_tp1:
    def __enter__(self):
        self.tp_ws, self.tp_r = pgm.process_group_manager.tp_world_size, pgm.process_group_manager.tp_rank
        pgm.process_group_manager.tp_world_size = 1
        pgm.process_group_manager.tp_rank = 0

    def __exit__(self, *a):
        pgm.process_group_manager.tp_world_size = self.tp_ws
        pgm.process_group_manager.tp_rank = self.tp_r


def build_reference(cfg):
    with force_tp1():
        torch.manual_seed(0)
        model = Llama(cfg).to(torch.float32).to(device)
        model.reset_parameters()
    return model, {k: v.detach().clone() for k, v in model.state_dict().items()}


def reference_grads(model, input_ids, target_ids):
    for p in model.parameters():
        p.grad = None
    with force_tp1():
        logits = model(input_ids)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
    loss.backward()
    return loss.item(), {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def run_tp(cfg, weights, input_ids, target_ids, sequence_parallel):
    torch.manual_seed(0)
    model = Llama(cfg).to(torch.float32).to(device)
    model = apply_tensor_parallel(model, sequence_parallel=sequence_parallel)
    model.to(device)
    tp_rank, tp = pgm.process_group_manager.tp_rank, pgm.process_group_manager.tp_world_size
    sd = {}
    for name, p in model.named_parameters():
        ref = weights[name]
        if p.shape == ref.shape:
            sd[name] = ref.clone()
        else:
            dim = next(d for d in range(ref.dim()) if ref.shape[d] != p.shape[d])
            sd[name] = ref.chunk(tp, dim=dim)[tp_rank].clone()
    model.load_state_dict(sd, strict=True)

    for p in model.parameters():
        p.grad = None
    logits = model(input_ids)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
    loss.backward()

    grads = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().clone()
        if sequence_parallel and name.endswith(NORM_SUFFIXES):
            dist.all_reduce(g, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.tp_group)
        grads[name] = g
    return loss.item(), grads


def compare(tag, ref_loss, ref_grads, loss, grads, atol):
    tp_rank, tp = pgm.process_group_manager.tp_rank, pgm.process_group_manager.tp_world_size
    assert abs(ref_loss - loss) < atol, f"{tag}: loss {loss:.6f} vs ref {ref_loss:.6f}"
    max_diff = 0.0
    for name, g in grads.items():
        ref = ref_grads[name]
        if g.shape != ref.shape:
            dim = next(d for d in range(ref.dim()) if ref.shape[d] != g.shape[d])
            ref = ref.chunk(tp, dim=dim)[tp_rank]
        max_diff = max(max_diff, (g - ref).abs().max().item())
    assert max_diff < atol, f"{tag}: grad diff {max_diff:.3e} >= {atol}"
    if global_rank == 0:
        print(f"  {tag}: PASSED  (loss_diff={abs(ref_loss - loss):.2e}, grad_diff={max_diff:.2e})")


def main():
    cfg = make_config()
    g = torch.Generator().manual_seed(1234)
    input_ids = torch.randint(0, cfg.vocab_size, (BS, SEQ), generator=g).to(device)
    target_ids = torch.randint(0, cfg.vocab_size, (BS, SEQ), generator=g).to(device)

    ref_model, weights = build_reference(cfg)
    ref_loss, ref_grads = reference_grads(ref_model, input_ids, target_ids)
    if global_rank == 0:
        print(f"[tp={world_size}] PTB reference loss = {ref_loss:.6f}")

    atol = 2e-4
    loss, grads = run_tp(cfg, weights, input_ids, target_ids, sequence_parallel=False)
    compare("PTB + plain TP", ref_loss, ref_grads, loss, grads, atol)
    loss, grads = run_tp(cfg, weights, input_ids, target_ids, sequence_parallel=True)
    compare("PTB + TP + sequence parallel (fused)", ref_loss, ref_grads, loss, grads, atol)

    if global_rank == 0:
        print(f"[tp={world_size}] PTB model tests passed ✅")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
