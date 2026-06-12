"""Correctness tests for the capacity-capped / static-memory MoE dispatch.

The headline guarantee: **static-memory dropless** (multiple capped rounds, no tokens dropped) must
compute exactly the same thing as the dropless variable-sized ``_moe_naive`` path -- it only changes
*how* the experts are batched (fixed-capacity buffers), not the math. We also check the classic
capacity-capped mode (single round, drops overflow) runs and stays finite, and that capacity works
with LatentMoE.

Run:
    python tests/test_capacity_moe.py                          # ep_size=1
    torchrun --nproc_per_node 2 tests/test_capacity_moe.py     # ep_size=2 (fixed-capacity all-to-all)
"""

import os
import types

import torch
import torch.distributed as dist


def make_config(hidden=16, inter=32, num_experts=4, top_k=2, n_shared=0, latent=0):
    return types.SimpleNamespace(
        hidden_size=hidden,
        intermediate_size=inter,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        num_shared_experts=n_shared,
        norm_topk_prob=True,
        moe_latent_dim=latent,
    )


def _run(moe, x):
    moe.zero_grad(set_to_none=True)
    out = moe(x).view(-1, moe.hidden_size)
    out.sum().backward()
    grads = {n: p.grad.detach().clone() for n, p in moe.named_parameters() if p.grad is not None}
    return out.detach().clone(), grads


def _check_dropless_matches_naive(latent):
    import picotron.process_group_manager as pgm
    from picotron.expert_parallel.expert_parallel import MoELayer

    ep = pgm.process_group_manager.ep_world_size
    cfg = make_config(num_experts=4, top_k=2, n_shared=1, latent=latent)
    torch.manual_seed(0)
    moe = MoELayer(cfg)

    torch.manual_seed(123)
    tokens = torch.randn(16, cfg.hidden_size)
    x = tokens.view(2, 8, cfg.hidden_size)

    # Baseline: dropless variable-sized path.
    moe.ep_capacity_factor = 0.0
    moe.ep_overlap = False
    out_base, g_base = _run(moe, x)

    # Static-memory dropless: fixed-capacity buffers, enough rounds to process every token.
    moe.ep_capacity_factor = 1.0
    moe.ep_capacity_dropless = True
    moe.ep_max_rounds = 64
    out_cap, g_cap = _run(moe, x)

    od = (out_base - out_cap).abs().max().item()
    gd = max((g_base[n] - g_cap[n]).abs().max().item() for n in g_base)
    assert od < 1e-5, f"(latent={latent}) fwd diff {od}"
    assert gd < 1e-5, f"(latent={latent}) grad diff {gd}"
    print(
        f"[rank {pgm.process_group_manager.ep_rank}] capacity dropless==naive PASSED "
        f"(ep_size={ep}, latent={latent}, out_diff={od:.2e}, grad_diff={gd:.2e})"
    )


def test_capacity_dropless_matches_naive():
    _check_dropless_matches_naive(latent=0)


def test_capacity_dropless_matches_naive_latent():
    _check_dropless_matches_naive(latent=8)


def test_capacity_capped_drops_and_is_finite():
    """Capped mode (single round, tiny capacity) drops tokens -> differs from naive but stays finite."""
    import picotron.process_group_manager as pgm
    from picotron.expert_parallel.expert_parallel import MoELayer

    ep = pgm.process_group_manager.ep_world_size
    cfg = make_config(num_experts=4, top_k=2, n_shared=1)
    torch.manual_seed(0)
    moe = MoELayer(cfg)

    torch.manual_seed(123)
    tokens = torch.randn(16, cfg.hidden_size)
    x = tokens.view(2, 8, cfg.hidden_size)

    moe.ep_capacity_factor = 0.0
    out_base, _ = _run(moe, x)

    moe.ep_capacity_factor = 0.5  # capacity below average -> guaranteed drops
    moe.ep_capacity_dropless = False
    out_capped, g_capped = _run(moe, x)

    assert torch.isfinite(out_capped).all(), "capped output has non-finite values"
    assert all(torch.isfinite(g).all() for g in g_capped.values()), "capped grads non-finite"
    dropped_diff = (out_base - out_capped).abs().max().item()
    assert dropped_diff > 0, "capped mode should drop some tokens (expected a difference)"
    print(
        f"[rank {pgm.process_group_manager.ep_rank}] capacity capped PASSED "
        f"(ep_size={ep}, diff_vs_naive={dropped_diff:.2e}, finite)"
    )


def main():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "25514")
    os.environ.setdefault("LOCAL_RANK", str(rank))
    os.environ.setdefault("FLASH_ATTEN", "0")
    os.environ.setdefault("DEVICE", "cpu")
    os.environ.setdefault("DTYPE", "float32")

    backend = os.environ.get("EP_TEST_BACKEND", "gloo")
    if backend == "nccl":
        torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    from picotron.process_group_manager import setup_process_group_manager

    setup_process_group_manager(tp_size=1, cp_size=1, ep_size=world_size, pp_size=1, dp_size=1)

    test_capacity_dropless_matches_naive()
    test_capacity_dropless_matches_naive_latent()
    test_capacity_capped_drops_and_is_finite()

    dist.barrier() if world_size > 1 else None
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
