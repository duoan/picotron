"""Correctness tests for the Mixture-of-Experts / Expert-Parallel layer.

Single-process (ep_size=1) tests run anywhere and check the routing + expert math and autograd.
The multi-rank dispatch/combine test is launched with torchrun and checks that the all-to-all
based EP path produces the same result as running every expert locally.

Run:
    python tests/test_expert_parallel.py                         # local ep_size=1 checks
    torchrun --nproc_per_node 2 tests/test_expert_parallel.py    # ep_size=2 dispatch/combine check
"""

import os
import types

import torch
import torch.distributed as dist
import torch.nn.functional as F


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


def reference_moe(moe, tokens):
    """Plain (non-distributed) reference: gather every expert and route locally."""
    logits = moe.gate(tokens)
    probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    topk_w, topk_i = torch.topk(probs, moe.top_k, dim=-1)
    if moe.top_k > 1:
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
    topk_w = topk_w.to(tokens.dtype)

    out = torch.zeros_like(tokens)
    for t in range(tokens.shape[0]):
        for k in range(moe.top_k):
            e = topk_i[t, k].item()
            out[t] += topk_w[t, k] * moe.local_experts[e](tokens[t : t + 1]).squeeze(0)
    if moe.shared_expert is not None:
        out = out + moe.shared_expert(tokens)
    return out


def test_local_ep1():
    import picotron.process_group_manager as pgm
    from picotron.expert_parallel.expert_parallel import MoELayer

    torch.manual_seed(0)
    cfg = make_config(n_shared=1)
    moe = MoELayer(cfg)

    tokens = torch.randn(8, cfg.hidden_size)
    x = tokens.view(2, 4, cfg.hidden_size)

    out = moe(x).view(-1, cfg.hidden_size)
    ref = reference_moe(moe, tokens)

    assert torch.allclose(out, ref, atol=1e-5), f"max diff {(out - ref).abs().max().item()}"

    # backward should populate grads for gate and all local experts
    out.sum().backward()
    assert moe.gate.weight.grad is not None
    assert all(e.up_proj.weight.grad is not None for e in moe.local_experts)
    print(f"[rank {pgm.process_group_manager.ep_rank}] test_local_ep1 PASSED (ep_size=1)")


def test_local_ep1_cuda():
    """ep_size=1 on GPU: exercises the custom CUDA grouped-GEMM path inside MoELayer."""
    import picotron.process_group_manager as pgm
    from picotron.expert_parallel import megakernel
    from picotron.expert_parallel.expert_parallel import MoELayer

    if not torch.cuda.is_available():
        print("[skip] test_local_ep1_cuda (no CUDA)")
        return
    assert megakernel.cuda_available(), "Triton MegaKernel unavailable"

    torch.manual_seed(0)
    cfg = make_config(n_shared=1)
    moe_cpu = MoELayer(cfg)
    tokens = torch.randn(8, cfg.hidden_size)
    x = tokens.view(2, 4, cfg.hidden_size)
    ref = reference_moe(moe_cpu, tokens)

    moe = moe_cpu.cuda()
    out = moe(x.cuda()).view(-1, cfg.hidden_size).cpu()
    diff = (out - ref).abs().max().item()
    assert torch.allclose(out, ref, atol=1e-4), f"cuda kernel diff {diff}"

    out.sum().backward()
    assert all(e.up_proj.weight.grad is not None for e in moe.local_experts)
    print(f"[rank {pgm.process_group_manager.ep_rank}] test_local_ep1_cuda PASSED (kernel, diff={diff:.2e})")


def test_distributed_dispatch_combine():
    import picotron.process_group_manager as pgm
    from picotron.expert_parallel.expert_parallel import MoELayer

    ep = pgm.process_group_manager.ep_world_size
    cfg = make_config(num_experts=4, top_k=2)
    # Same weights on every rank, then each rank keeps only its expert shard.
    torch.manual_seed(0)
    moe = MoELayer(cfg)

    # Build a global reference set of experts (all of them) on every rank to check correctness.
    torch.manual_seed(123)
    tokens = torch.randn(16, cfg.hidden_size)
    x = tokens.view(2, 8, cfg.hidden_size)

    out = moe(x).view(-1, cfg.hidden_size)

    # Reference: all-gather every rank's experts so we can route locally.
    nle = moe.num_local_experts
    all_experts = [None] * cfg.num_experts

    class _Holder:
        def __init__(self, experts):
            self.gate = moe.gate
            self.top_k = moe.top_k
            self.local_experts = experts
            self.shared_expert = moe.shared_expert

    # Gather expert state dicts across ranks.
    local_states = [e.state_dict() for e in moe.local_experts]
    gathered = [None] * ep
    dist.all_gather_object(gathered, local_states, group=pgm.process_group_manager.ep_group)
    from picotron.model import MLP

    for r in range(ep):
        for li in range(nle):
            e = MLP(cfg)
            e.load_state_dict(gathered[r][li])
            all_experts[r * nle + li] = e

    ref = reference_moe(_Holder(all_experts), tokens)

    max_diff = (out - ref).abs().max().item()
    assert torch.allclose(out, ref, atol=1e-3), f"max diff {max_diff}"

    # backward sanity
    out.sum().backward()
    assert moe.gate.weight.grad is not None
    print(
        f"[rank {pgm.process_group_manager.ep_rank}] test_distributed_dispatch_combine PASSED "
        f"(ep_size={ep}, max_diff={max_diff:.2e})"
    )


def test_overlap_matches_baseline():
    """The dispatch/shared-expert overlap path must match the plain path bit-for-bit (fwd + grads)."""
    import picotron.process_group_manager as pgm
    from picotron.expert_parallel.expert_parallel import MoELayer

    ep = pgm.process_group_manager.ep_world_size
    if ep == 1:
        print("[skip] test_overlap_matches_baseline (ep_size=1, no comm to overlap)")
        return

    cfg = make_config(num_experts=4, top_k=2, n_shared=1)  # shared expert needed for overlap
    torch.manual_seed(0)
    moe = MoELayer(cfg)

    torch.manual_seed(123)
    tokens = torch.randn(16, cfg.hidden_size)
    x = tokens.view(2, 8, cfg.hidden_size)

    # Baseline: synchronous dispatch, shared expert after.
    moe.ep_overlap = False
    out_base = moe(x).view(-1, cfg.hidden_size)
    out_base.sum().backward()
    grads_base = {n: p.grad.detach().clone() for n, p in moe.named_parameters()}
    moe.zero_grad(set_to_none=True)

    # Overlap: async dispatch hidden behind the shared-expert FFN.
    moe.ep_overlap = True
    out_over = moe(x).view(-1, cfg.hidden_size)
    out_over.sum().backward()
    grads_over = {n: p.grad.detach().clone() for n, p in moe.named_parameters()}

    out_diff = (out_base - out_over).abs().max().item()
    assert torch.allclose(out_base, out_over, atol=1e-6), f"forward diff {out_diff}"
    g_diff = max((grads_base[n] - grads_over[n]).abs().max().item() for n in grads_base)
    assert g_diff < 1e-6, f"grad diff {g_diff}"
    print(
        f"[rank {pgm.process_group_manager.ep_rank}] test_overlap_matches_baseline PASSED "
        f"(out_diff={out_diff:.2e}, grad_diff={g_diff:.2e})"
    )


def test_tiled_matches_baseline():
    """The MegaScale-style tiled pipeline must match the plain path bit-for-bit (fwd + grads)."""
    import picotron.process_group_manager as pgm
    from picotron.expert_parallel.expert_parallel import MoELayer

    ep = pgm.process_group_manager.ep_world_size
    if ep == 1:
        print("[skip] test_tiled_matches_baseline (ep_size=1, no comm to tile)")
        return

    for n_shared in (0, 1):  # tiling must work with and without a shared expert
        cfg = make_config(num_experts=4, top_k=2, n_shared=n_shared)
        torch.manual_seed(0)
        moe = MoELayer(cfg)

        torch.manual_seed(123)
        tokens = torch.randn(16, cfg.hidden_size)  # 32 routed rows; 3 tiles -> uneven 10/11/11
        x = tokens.view(2, 8, cfg.hidden_size)

        moe.ep_overlap = False
        moe.ep_num_tiles = 1
        out_base = moe(x).view(-1, cfg.hidden_size)
        out_base.sum().backward()
        grads_base = {n: p.grad.detach().clone() for n, p in moe.named_parameters()}
        moe.zero_grad(set_to_none=True)

        moe.ep_num_tiles = 3
        out_tiled = moe(x).view(-1, cfg.hidden_size)
        out_tiled.sum().backward()
        grads_tiled = {n: p.grad.detach().clone() for n, p in moe.named_parameters()}

        out_diff = (out_base - out_tiled).abs().max().item()
        assert torch.allclose(out_base, out_tiled, atol=1e-6), f"(n_shared={n_shared}) fwd diff {out_diff}"
        g_diff = max((grads_base[n] - grads_tiled[n]).abs().max().item() for n in grads_base)
        assert g_diff < 1e-6, f"(n_shared={n_shared}) grad diff {g_diff}"
        print(
            f"[rank {pgm.process_group_manager.ep_rank}] test_tiled_matches_baseline PASSED "
            f"(n_shared={n_shared}, out_diff={out_diff:.2e}, grad_diff={g_diff:.2e})"
        )


def test_tiled_empty_tiles():
    """More tiles than routed rows -> some tiles are empty; fwd/bwd must stay correct & in lockstep."""
    import picotron.process_group_manager as pgm
    from picotron.expert_parallel.expert_parallel import MoELayer

    ep = pgm.process_group_manager.ep_world_size
    if ep == 1:
        print("[skip] test_tiled_empty_tiles (ep_size=1)")
        return

    cfg = make_config(num_experts=4, top_k=2, n_shared=1)
    torch.manual_seed(0)
    moe = MoELayer(cfg)

    torch.manual_seed(7)
    tokens = torch.randn(2, cfg.hidden_size)  # 4 routed rows
    x = tokens.view(1, 2, cfg.hidden_size)

    moe.ep_num_tiles = 1
    out_base = moe(x).view(-1, cfg.hidden_size)
    out_base.sum().backward()
    grads_base = {n: p.grad.detach().clone() for n, p in moe.named_parameters()}
    moe.zero_grad(set_to_none=True)

    moe.ep_num_tiles = 8  # > 4 rows -> empty tiles
    out_tiled = moe(x).view(-1, cfg.hidden_size)
    out_tiled.sum().backward()
    grads_tiled = {n: p.grad.detach().clone() for n, p in moe.named_parameters()}

    out_diff = (out_base - out_tiled).abs().max().item()
    assert torch.allclose(out_base, out_tiled, atol=1e-6), f"fwd diff {out_diff}"
    g_diff = max((grads_base[n] - grads_tiled[n]).abs().max().item() for n in grads_base)
    assert g_diff < 1e-6, f"grad diff {g_diff}"
    print(
        f"[rank {pgm.process_group_manager.ep_rank}] test_tiled_empty_tiles PASSED "
        f"(out_diff={out_diff:.2e}, grad_diff={g_diff:.2e})"
    )


def test_latent_paths_match():
    """LatentMoE: all three EP paths (plain / overlap / tiled) must agree (fwd + grads).

    The down/up projection changes the dispatched dim from hidden to latent; this checks that the
    latent dim is threaded correctly through every path and that they stay numerically identical.
    """
    import picotron.process_group_manager as pgm
    from picotron.expert_parallel.expert_parallel import MoELayer

    ep = pgm.process_group_manager.ep_world_size
    if ep == 1:
        print("[skip] test_latent_paths_match (ep_size=1, no dispatch)")
        return

    cfg = make_config(hidden=16, inter=32, num_experts=4, top_k=2, n_shared=1, latent=8)
    torch.manual_seed(0)
    moe = MoELayer(cfg)
    assert moe.latent and moe.expert_dim == 8
    # Experts live in the latent dim (8), not the model hidden dim (16).
    assert moe.local_experts[0].gate_proj.weight.shape[1] == 8

    torch.manual_seed(123)
    tokens = torch.randn(16, cfg.hidden_size)
    x = tokens.view(2, 8, cfg.hidden_size)

    def run(num_tiles, overlap):
        moe.zero_grad(set_to_none=True)
        moe.ep_num_tiles = num_tiles
        moe.ep_overlap = overlap
        out = moe(x).view(-1, cfg.hidden_size)
        out.sum().backward()
        grads = {n: p.grad.detach().clone() for n, p in moe.named_parameters()}
        return out.detach().clone(), grads

    out_plain, g_plain = run(1, False)
    out_over, g_over = run(1, True)
    out_tiled, g_tiled = run(3, False)

    for name, out, g in (("overlap", out_over, g_over), ("tiled", out_tiled, g_tiled)):
        od = (out_plain - out).abs().max().item()
        gd = max((g_plain[n] - g[n]).abs().max().item() for n in g_plain)
        assert od < 1e-6, f"latent {name} fwd diff {od}"
        assert gd < 1e-6, f"latent {name} grad diff {gd}"

    print(
        f"[rank {pgm.process_group_manager.ep_rank}] test_latent_paths_match PASSED "
        f"(latent={moe.latent_dim}, hidden={cfg.hidden_size})"
    )


def make_model_config():
    return types.SimpleNamespace(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        max_position_embeddings=8,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        norm_topk_prob=True,
    )


def test_model_integration():
    """Build the real Llama model with MoE layers and run forward/backward across EP ranks."""
    import picotron.process_group_manager as pgm
    from picotron.model import Llama

    ep = pgm.process_group_manager.ep_world_size
    torch.manual_seed(0)
    cfg = make_model_config()
    model = Llama(cfg)
    model.reset_parameters()

    torch.manual_seed(42)  # same input on every EP rank
    input_ids = torch.randint(0, cfg.vocab_size, (2, cfg.max_position_embeddings))
    logits = model(input_ids)

    # EP ranks process identical inputs with replicated non-expert weights, so logits must match.
    if ep > 1:
        gathered = [torch.empty_like(logits) for _ in range(ep)]
        dist.all_gather(gathered, logits.contiguous(), group=pgm.process_group_manager.ep_group)
        for r in range(1, ep):
            d = (gathered[0] - gathered[r]).abs().max().item()
            assert d < 1e-4, f"logits differ across EP ranks: {d}"

    loss = logits.float().pow(2).mean()
    loss.backward()
    # router gate and local experts should have gradients
    assert model.decoder_layers[0].mlp.gate.weight.grad is not None
    assert model.decoder_layers[0].mlp.local_experts[0].up_proj.weight.grad is not None
    assert torch.isfinite(loss).item()
    print(
        f"[rank {pgm.process_group_manager.ep_rank}] test_model_integration PASSED "
        f"(ep_size={ep}, loss={loss.item():.4f})"
    )


def main():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "25512")
    os.environ.setdefault("LOCAL_RANK", str(rank))
    # CPU-friendly model settings (disable flash attention / CUDA paths) for portable testing.
    os.environ.setdefault("FLASH_ATTEN", "0")
    os.environ.setdefault("DEVICE", "cpu")
    os.environ.setdefault("DTYPE", "float32")

    # Default to gloo/CPU so the test is portable (the EP dispatch/combine logic is
    # backend-agnostic). Set EP_TEST_BACKEND=nccl to exercise the GPU path on real hardware.
    backend = os.environ.get("EP_TEST_BACKEND", "gloo")
    if backend == "nccl":
        torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    from picotron.process_group_manager import setup_process_group_manager

    setup_process_group_manager(tp_size=1, cp_size=1, ep_size=world_size, pp_size=1, dp_size=1)

    if world_size == 1:
        test_local_ep1()
        test_local_ep1_cuda()
        test_model_integration()
    else:
        test_distributed_dispatch_combine()
        test_overlap_matches_baseline()
        test_tiled_matches_baseline()
        test_tiled_empty_tiles()
        test_latent_paths_match()
        test_model_integration()

    dist.barrier() if world_size > 1 else None
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
