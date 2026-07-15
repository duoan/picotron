"""Correctness tests for the advanced pipeline-parallel schedules.

Every schedule (Zero-Bubble, Interleaved 1F1B) must produce **bit-exact gradients** vs a
non-pipelined full-model reference on the same weights + micro-batches. We build a tiny Llama, take a
full-model gradient reference, then shard the *same* modules across ``pp_size`` ranks (restoring the
reference weights), run each schedule, and compare the gradients of the layers this rank owns.

Runs on gloo/CPU by default (portable, no CUDA needed):
    torchrun --nproc_per_node 2 tests/test_pipeline_parallel.py    # pp_size=2
    torchrun --nproc_per_node 4 tests/test_pipeline_parallel.py    # pp_size=4
"""

import os
import types

import torch
import torch.distributed as dist
import torch.nn.functional as F

GRAD_ACC_STEPS = 8  # multiple of pp_size (interleaved 1F1B requires m % p == 0)
MBS = 2
SEQ = 8


def make_model_config():
    return types.SimpleNamespace(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=8,
        max_position_embeddings=SEQ,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        num_experts=1,
        num_experts_per_tok=1,
        num_shared_experts=0,
        norm_topk_prob=True,
    )


class FakeDataLoader:
    """Deterministic in-memory micro-batch source: identical batches on every rank, replayable."""

    def __init__(self, cfg, grad_acc_steps, mbs, seq, device):
        self.grad_acc_steps = grad_acc_steps
        self.seq_length_per_gpu = seq
        self.micro_batch_size = mbs
        g = torch.Generator().manual_seed(1234)
        self.batches = []
        for _ in range(grad_acc_steps):
            self.batches.append(
                {
                    "input_ids": torch.randint(0, cfg.vocab_size, (mbs, seq), generator=g).to(device),
                    "target_ids": torch.randint(0, cfg.vocab_size, (mbs, seq), generator=g).to(device),
                    "position_ids": torch.arange(seq).unsqueeze(0).repeat(mbs, 1).to(device),
                    "hidden_states": None,
                }
            )
        self.idx = 0

    def reset(self):
        self.idx = 0

    def __next__(self):
        batch = self.batches[self.idx % self.grad_acc_steps]
        self.idx += 1
        return batch


def full_reference_grads(model, dl, device):
    """Non-pipelined reference: run the full model on each micro-batch and accumulate grads."""
    dl.reset()
    for _ in range(dl.grad_acc_steps):
        batch = next(dl)
        logits = model(batch["input_ids"].to(device))
        loss = F.cross_entropy(logits.flatten(0, 1), batch["target_ids"].to(device).flatten(), reduction="mean")
        loss.backward()
    return {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}


def build_reference_and_stage(cfg, dl, device, wrapper_cls, **kwargs):
    """Build full Llama, snapshot weights + reference grads, then wrap the *same* modules into a
    pipeline stage with the reference weights restored (so reference and pipeline share weights)."""
    from picotron.model import Llama

    torch.manual_seed(0)
    model = Llama(cfg)
    model.reset_parameters()
    model.to(device)
    weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ref_grads = full_reference_grads(model, dl, device)
    for p in model.parameters():
        p.grad = None
    stage = wrapper_cls(model, cfg, **kwargs)  # may re-init the shared modules
    model.load_state_dict(weights)  # restore reference weights into the shared params
    return model, stage, ref_grads


def run_schedule(stage, schedule_fn, dl, cfg, device):
    for p in stage.parameters():
        p.grad = None
    dl.reset()
    tensor_shapes = (MBS, SEQ, cfg.hidden_size)
    return schedule_fn(stage, dl, tensor_shapes, device, torch.float32)


def compare_owned_grads(model, ref_grads, tag, atol=1e-5):
    import picotron.process_group_manager as pgm

    owned = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
    assert owned, f"{tag}: this rank computed no gradients"
    max_diff = 0.0
    for name in owned:
        assert name in ref_grads, f"{tag}: {name} missing from reference"
        max_diff = max(max_diff, (owned[name] - ref_grads[name]).abs().max().item())
    assert max_diff < atol, f"{tag}: grad diff {max_diff:.3e} >= {atol}"
    print(
        f"[rank {pgm.process_group_manager.pp_rank}] {tag} PASSED ({len(owned)} owned params, grad_diff={max_diff:.2e})"
    )


def test_zero_bubble(cfg, device):
    from picotron.pipeline_parallel.pipeline_parallel import PipelineParallel
    from picotron.pipeline_parallel.pp_schedules import train_step_pipeline_zb

    dl = FakeDataLoader(cfg, GRAD_ACC_STEPS, MBS, SEQ, device)
    model, stage, ref_grads = build_reference_and_stage(cfg, dl, device, PipelineParallel)
    run_schedule(stage, train_step_pipeline_zb, dl, cfg, device)
    compare_owned_grads(model, ref_grads, "zero_bubble")


def test_interleaved(cfg, device, num_virtual_stages=2):
    from picotron.pipeline_parallel.pp_schedules import (
        InterleavedPipelineParallel,
        train_step_pipeline_interleaved,
    )

    dl = FakeDataLoader(cfg, GRAD_ACC_STEPS, MBS, SEQ, device)
    model, stage, ref_grads = build_reference_and_stage(
        cfg, dl, device, InterleavedPipelineParallel, num_virtual_stages=num_virtual_stages
    )
    run_schedule(stage, train_step_pipeline_interleaved, dl, cfg, device)
    compare_owned_grads(model, ref_grads, f"interleaved(v={num_virtual_stages})")


def test_zbv(cfg, device):
    from picotron.pipeline_parallel.pp_schedules import (
        VShapePipelineParallel,
        train_step_pipeline_zbv,
    )

    dl = FakeDataLoader(cfg, GRAD_ACC_STEPS, MBS, SEQ, device)
    model, stage, ref_grads = build_reference_and_stage(cfg, dl, device, VShapePipelineParallel)
    run_schedule(stage, train_step_pipeline_zbv, dl, cfg, device)
    compare_owned_grads(model, ref_grads, "zbv")


def main():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "25513")
    os.environ.setdefault("LOCAL_RANK", str(rank))
    os.environ.setdefault("FLASH_ATTEN", "0")
    os.environ.setdefault("DEVICE", "cpu")
    os.environ.setdefault("DTYPE", "float32")

    import faulthandler

    faulthandler.dump_traceback_later(120, exit=True)  # dump stacks + abort if a schedule hangs

    backend = os.environ.get("PP_TEST_BACKEND", "gloo")
    init_kwargs = {"backend": backend, "rank": rank, "world_size": world_size}
    if backend == "nccl":
        local = rank % torch.cuda.device_count()
        torch.cuda.set_device(local)
        init_kwargs["device_id"] = torch.device("cuda", local)  # eager NCCL bind (avoids p>=8 lazy-comm hang)
    dist.init_process_group(**init_kwargs)

    from picotron.process_group_manager import setup_process_group_manager

    setup_process_group_manager(tp_size=1, cp_size=1, ep_size=1, pp_size=world_size, dp_size=1)

    device = torch.device("cpu") if backend == "gloo" else torch.device("cuda", rank % torch.cuda.device_count())
    cfg = make_model_config()

    assert world_size > 1, "pipeline tests need pp_size >= 2 (torchrun --nproc_per_node 2)"
    test_zero_bubble(cfg, device)
    test_interleaved(cfg, device, num_virtual_stages=2)
    test_zbv(cfg, device)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
