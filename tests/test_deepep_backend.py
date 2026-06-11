"""Integration test for the DeepEP dispatch/combine backend (Hopper SM90+ + NCCL).

Compares ``ep_backend="deepep"`` against the reference ``ep_backend="torch"`` path on the same
weights/inputs: forward outputs and parameter grads must match within BF16 + DeepEP-reduction
tolerance. Also prints a quick latency comparison of the two backends.

Requires real DeepEP on a Hopper (SM90+) GPU; meant to be launched on Modal (see
``modal_deepep_test.py``) or any 2x H100 box:

    torchrun --nproc_per_node 2 tests/test_deepep_backend.py
"""

import os
import types

import torch
import torch.distributed as dist


def make_config(hidden=2048, inter=2048, num_experts=8, top_k=2, n_shared=1, latent=0):
    return types.SimpleNamespace(
        hidden_size=hidden,
        intermediate_size=inter,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        num_shared_experts=n_shared,
        norm_topk_prob=True,
        moe_latent_dim=latent,
    )


def _bench(moe, x, iters=20, warmup=5):
    for _ in range(warmup):
        moe(x)
    torch.cuda.synchronize()
    dist.barrier()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        start.record()
        moe(x)
        end.record()
        torch.cuda.synchronize()
        ts.append(start.elapsed_time(end))
    t = torch.tensor(sorted(ts)[len(ts) // 2], device=x.device)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t.item()


def run(latent):
    import picotron.process_group_manager as pgm
    from picotron.expert_parallel import deepep_backend
    from picotron.expert_parallel.expert_parallel import MoELayer

    rank = pgm.process_group_manager.ep_rank
    assert deepep_backend.deepep_available(), "DeepEP not available (needs SM90+ and the deep_ep package)"

    torch.manual_seed(0)
    cfg = make_config(latent=latent)
    moe = MoELayer(cfg).to(torch.bfloat16).cuda()

    torch.manual_seed(123)
    tokens = torch.randn(1024, cfg.hidden_size, dtype=torch.bfloat16, device="cuda")
    x = tokens.view(8, 128, cfg.hidden_size)

    def forward_backward(backend):
        moe.zero_grad(set_to_none=True)
        moe.ep_backend = backend
        out = moe(x).view(-1, cfg.hidden_size)
        out.float().sum().backward()
        grads = {n: p.grad.detach().clone() for n, p in moe.named_parameters() if p.grad is not None}
        return out.detach().clone(), grads

    out_torch, g_torch = forward_backward("torch")
    out_deep, g_deep = forward_backward("deepep")

    rel = (out_torch - out_deep).float().norm() / out_torch.float().norm()

    # Expert + up/down-projection grads must match. DeepEP does not differentiate through its
    # (non-differentiable) gate-weight transport, so the router 'gate' grad is reported but not gated on.
    def grad_rel(name):
        return (g_torch[name] - g_deep[name]).float().norm() / (g_torch[name].float().norm() + 1e-9)

    common = [n for n in g_torch if n in g_deep]  # gate has no grad on the DeepEP path
    g_rel = max(grad_rel(n) for n in common if "gate" not in n)
    gate_rel = max((grad_rel(n) for n in common if "gate" in n), default=float("nan"))

    moe.ep_backend = "torch"
    t_torch = _bench(moe, x)
    moe.ep_backend = "deepep"
    t_deep = _bench(moe, x)

    if rank == 0:
        tag = f"latent={latent}" if latent else "dense"
        print(f"[{tag}] fwd rel={rel:.4f} | expert grad rel={g_rel:.4f} | gate grad rel={gate_rel:.4f}")
        print(f"[{tag}] fwd ms: torch={t_torch:.3f}  deepep={t_deep:.3f}  speedup={t_torch / t_deep:.2f}x")
        assert rel < 0.03, f"[{tag}] DeepEP forward mismatch: {rel}"
        assert g_rel < 0.05, f"[{tag}] DeepEP expert grad mismatch: {g_rel}"
        print(f"[{tag}] test_deepep_backend PASSED")


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    os.environ.setdefault("LOCAL_RANK", str(rank))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "25514")
    os.environ.setdefault("FLASH_ATTEN", "0")
    os.environ.setdefault("DEVICE", "cuda")
    os.environ.setdefault("DTYPE", "bfloat16")

    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    from picotron.process_group_manager import setup_process_group_manager

    setup_process_group_manager(tp_size=1, cp_size=1, ep_size=world_size, pp_size=1, dp_size=1)

    run(latent=0)
    run(latent=512)  # LatentMoE + DeepEP stacked

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
