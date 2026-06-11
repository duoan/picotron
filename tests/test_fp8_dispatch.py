"""Correctness test for FP8 (E4M3) dispatch in the EP MoE layer (CUDA + NCCL only).

FP8 dispatch ships token features at 1 byte instead of 2 (BF16), halving the dispatch all-to-all
volume; combine stays BF16. This checks the FP8 path matches the BF16 path within FP8 quantization
tolerance (forward) and that gradients flow (backward), and prints the wire-byte reduction.

Run (2 GPUs):
    torchrun --nproc_per_node 2 tests/test_fp8_dispatch.py
"""

import os
import types

import torch
import torch.distributed as dist


def make_config(hidden=1024, inter=2048, num_experts=8, top_k=2, n_shared=1):
    return types.SimpleNamespace(
        hidden_size=hidden,
        intermediate_size=inter,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        num_shared_experts=n_shared,
        norm_topk_prob=True,
    )


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    os.environ.setdefault("LOCAL_RANK", str(rank))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "25513")
    os.environ.setdefault("FLASH_ATTEN", "0")
    os.environ.setdefault("DEVICE", "cuda")
    os.environ.setdefault("DTYPE", "bfloat16")

    assert torch.cuda.is_available(), "FP8 dispatch test requires CUDA"
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    from picotron.expert_parallel.expert_parallel import MoELayer
    from picotron.process_group_manager import setup_process_group_manager

    setup_process_group_manager(tp_size=1, cp_size=1, ep_size=world_size, pp_size=1, dp_size=1)

    torch.manual_seed(0)
    cfg = make_config()
    moe = MoELayer(cfg).to(torch.bfloat16).cuda()
    moe.ep_overlap = False  # use the plain dispatch/combine path (where fp8 is wired)

    torch.manual_seed(123)
    tokens = torch.randn(512, cfg.hidden_size, dtype=torch.bfloat16, device="cuda")
    x = tokens.view(8, 64, cfg.hidden_size)

    # BF16 dispatch (baseline).
    moe.ep_fp8_dispatch = False
    out_bf16 = moe(x).view(-1, cfg.hidden_size)
    out_bf16.float().sum().backward()
    moe.zero_grad(set_to_none=True)

    # FP8 dispatch.
    moe.ep_fp8_dispatch = True
    out_fp8 = moe(x.clone().requires_grad_(False)).view(-1, cfg.hidden_size)
    out_fp8.float().sum().backward()

    grad_ok = (
        moe.local_experts[0].up_proj.weight.grad is not None
        and torch.isfinite(moe.local_experts[0].up_proj.weight.grad).all()
    )

    diff = (out_bf16 - out_fp8).abs()
    rel = (diff.float().norm() / out_bf16.float().norm()).item()
    max_abs = diff.max().item()

    bf16_bytes = 2 * cfg.hidden_size
    fp8_bytes = 1 * cfg.hidden_size + 4  # 1-byte payload per elem + one fp32 scale per token row

    if rank == 0:
        print(f"FP8 dispatch: relative L2 error = {rel:.4f}, max abs = {max_abs:.4f}")
        print(
            f"dispatch wire bytes/token: BF16={bf16_bytes}  FP8={fp8_bytes}  reduction = {bf16_bytes / fp8_bytes:.2f}x"
        )
        assert rel < 0.05, f"FP8 dispatch relative error too high: {rel}"
        assert grad_ok, "FP8 dispatch backward produced no / non-finite grads"
        print("test_fp8_dispatch PASSED")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
