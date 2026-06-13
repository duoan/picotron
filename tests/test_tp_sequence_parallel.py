"""Bit-exact correctness for tensor parallelism, including **sequence parallelism**.

Every rank builds the *same* full (replicated) Llama as a single-GPU reference, runs a forward +
cross-entropy + backward on the full batch, then builds two tensor-parallel copies from the reference
weights — plain TP (all-reduce) and TP + sequence parallelism — and checks that each rank's owned
gradients match the reference.

The only subtlety is the norm weights: under sequence parallelism each rank's RMSNorm sees a different
sequence shard, so its weight gradient is a *partial* sum that must be all-reduced over the TP group to
recover the replicated reference gradient (this is exactly what Megatron marks as a "sequence-parallel"
parameter). Sharded weights (attention/MLP projections, embedding, final_proj) see the full sequence
and need no fix-up.

    CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 2 tests/test_tp_sequence_parallel.py
    CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 4 tests/test_tp_sequence_parallel.py
"""

import datetime
import os
import types

import torch
import torch.distributed as dist
import torch.nn.functional as F

os.environ.setdefault("FLASH_ATTEN", "0")  # use SDPA + LlamaRMSNorm (no flash-attn / triton dep)

local_rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

backend = os.environ.get("TP_TEST_BACKEND", "nccl")
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

setup_process_group_manager(tp_size=world_size, cp_size=1, pp_size=1, dp_size=1)

import picotron.process_group_manager as pgm  # noqa: E402
from picotron.model import Llama  # noqa: E402
from picotron.tensor_parallel.tensor_parallel import apply_tensor_parallel  # noqa: E402
from picotron.utils import set_all_seed  # noqa: E402

set_all_seed(42)

BS, SEQ = 2, 8


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


# Norm weights are the only replicated params whose grad is sequence-sharded under SP.
NORM_SUFFIXES = ("input_layernorm.weight", "post_attention_layernorm.weight", "final_norm.weight")


def is_norm(name):
    return name.endswith(NORM_SUFFIXES)


class force_tp1:
    """Build/run the single-GPU reference as if tp_world_size == 1.

    `Attention.__init__` sizes `num_local_heads = num_attention_heads // tp_world_size`, so a plain
    (non-parallel) Llama can only be constructed when the manager reports tp=1. The reference has no TP
    collectives, so it is safe to run with this temporarily patched."""

    def __enter__(self):
        self.tp_ws = pgm.process_group_manager.tp_world_size
        self.tp_r = pgm.process_group_manager.tp_rank
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
    weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return model, weights


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
    # Load the reference weights, slicing the tensor-parallel params to this rank's shard.
    tp_rank, tp = pgm.process_group_manager.tp_rank, pgm.process_group_manager.tp_world_size
    sd = {}
    for name, p in model.named_parameters():
        ref = weights[name]
        if p.shape == ref.shape:
            sd[name] = ref.clone()
        else:  # find the sharded dim and take this rank's slice
            dim = next(d for d in range(ref.dim()) if ref.shape[d] != p.shape[d])
            sd[name] = ref.chunk(tp, dim=dim)[tp_rank].clone()
    model.load_state_dict(sd, strict=True)

    for p in model.parameters():
        p.grad = None
    logits = model(input_ids)  # full [b, s, V] (final_proj gathers seq + vocab)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
    loss.backward()

    grads = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().clone()
        if sequence_parallel and is_norm(name):
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
        print(f"[tp={world_size}] reference loss = {ref_loss:.6f}")

    atol = 2e-4  # fp32, but reduce-scatter/all-gather reorder the summation order
    loss, grads = run_tp(cfg, weights, input_ids, target_ids, sequence_parallel=False)
    compare("plain TP (all-reduce)", ref_loss, ref_grads, loss, grads, atol)
    loss, grads = run_tp(cfg, weights, input_ids, target_ids, sequence_parallel=True)
    compare("TP + sequence parallel", ref_loss, ref_grads, loss, grads, atol)

    if global_rank == 0:
        print(f"[tp={world_size}] All tensor/sequence-parallel tests passed ✅")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
