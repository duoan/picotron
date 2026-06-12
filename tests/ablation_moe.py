"""Ablation study for the static-memory capacity dispatch and the router load balancing.

This is a *correctness / behaviour* ablation (runs on gloo/CPU, portable) — it isolates the knobs
introduced for training at scale and reports the metrics that matter for those features (token drops,
round count, buffer determinism, load imbalance), not wall-clock (the timing story lives in the GPU
`bench_ep_*` scripts and the H100/A100 results tables).

Run:
    python tests/ablation_moe.py                          # ep=1 (both ablations)
    torchrun --nproc_per_node 2 tests/ablation_moe.py     # ep=2 (capacity correctness across ranks)
"""

import math
import os
import types

import torch
import torch.distributed as dist


def cfg(hidden=32, inter=64, num_experts=8, top_k=2, aux=0.0, bias_rate=0.0):
    return types.SimpleNamespace(
        hidden_size=hidden,
        intermediate_size=inter,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        num_shared_experts=0,
        norm_topk_prob=True,
        router_aux_loss_coef=aux,
        router_bias_update_rate=bias_rate,
    )


def make_routing(num_tokens, top_k, num_experts, skew, seed):
    """Synthesize a top-k routing with controllable imbalance.

    Experts are drawn from a Zipf distribution p_e ~ 1/(e+1)^skew: skew=0 is uniform, larger skew
    concentrates tokens on a few experts (the load-imbalance / memory-spike regime). Returns
    (topk_idx [T,k], topk_weights [T,k]) on CPU.
    """
    g = torch.Generator().manual_seed(seed)
    rank_w = 1.0 / torch.arange(1, num_experts + 1, dtype=torch.float32) ** skew
    probs = (rank_w / rank_w.sum()).expand(num_tokens, num_experts)
    topk_idx = torch.multinomial(probs, top_k, replacement=False, generator=g)  # [T,k] distinct
    topk_weights = torch.softmax(torch.rand(num_tokens, top_k, generator=g), dim=-1)
    return topk_idx, topk_weights


# --------------------------------------------------------------------------------------------------
# Ablation 1: capacity dispatch — drops, rounds, buffer determinism, dropless correctness
# --------------------------------------------------------------------------------------------------
def ablation_capacity():
    import picotron.process_group_manager as pgm
    from picotron.expert_parallel.capacity_moe import compute_capacity
    from picotron.expert_parallel.expert_parallel import MoELayer

    ep = pgm.process_group_manager.ep_world_size
    E, k, T = 8, 2, 512
    torch.manual_seed(0)
    moe = MoELayer(cfg(num_experts=E, top_k=k))
    moe.eval()
    tokens = torch.randn(T, moe.hidden_size)

    def metrics(topk_idx, factor):
        counts = torch.bincount(topk_idx.reshape(-1), minlength=E)
        maxc = int(counts.max())
        mean = counts.float().mean().item()
        C = compute_capacity(T, k, E, factor)
        dropped = int(torch.clamp(counts - C, min=0).sum())
        drop_frac = dropped / (T * k)
        rounds = math.ceil(maxc / C)
        return counts, maxc, mean, C, drop_frac, rounds

    def correctness(topk_idx, topk_w, factor):
        """dropless capacity (enough rounds) must equal the naive dropless path bit-for-bit."""
        moe.ep_capacity_factor = 0.0
        out_naive = moe._moe_naive(tokens, topk_idx, topk_w)
        moe.ep_capacity_factor, moe.ep_capacity_dropless, moe.ep_max_rounds = factor, True, 64
        out_cap = moe._moe_capped(tokens, topk_idx, topk_w)
        moe.ep_capacity_factor = 0.0
        return (out_naive - out_cap).abs().max().item()

    if pgm.process_group_manager.ep_rank == 0:
        print(f"\n=== Ablation 1: capacity dispatch (E={E}, top_k={k}, tokens/rank={T}, ep={ep}) ===")
        print("\n[A] vary routing imbalance (capacity_factor = 1.0)")
        print(f"{'skew':>5} {'max/mean':>9} {'capacity C':>11} {'naive worst-buf':>16} "
              f"{'capped drop%':>13} {'dropless rounds':>16} {'dropless==naive':>16}")
    for skew in (0.0, 0.5, 1.0, 1.5, 2.0):
        topk_idx, topk_w = make_routing(T, k, E, skew, seed=100 + int(skew * 10))
        counts, maxc, mean, C, drop_frac, rounds = metrics(topk_idx, 1.0)
        diff = correctness(topk_idx, topk_w, 1.0)
        if pgm.process_group_manager.ep_rank == 0:
            print(f"{skew:>5.1f} {maxc / mean:>9.2f} {C:>11d} {maxc:>16d} "
                  f"{100 * drop_frac:>12.1f}% {rounds:>16d} {diff:>16.1e}")

    if pgm.process_group_manager.ep_rank == 0:
        print("\n[B] vary capacity_factor (skew = 1.0, imbalanced)")
        print(f"{'factor':>7} {'capacity C':>11} {'capped drop%':>13} {'dropless rounds':>16}")
    topk_idx, topk_w = make_routing(T, k, E, 1.0, seed=110)
    for factor in (0.5, 1.0, 1.25, 2.0):
        _, maxc, mean, C, drop_frac, rounds = metrics(topk_idx, factor)
        if pgm.process_group_manager.ep_rank == 0:
            print(f"{factor:>7.2f} {C:>11d} {100 * drop_frac:>12.1f}% {rounds:>16d}")

    if pgm.process_group_manager.ep_rank == 0:
        print("\n  takeaway: capacity C is FIXED (static memory); naive worst-buf grows with imbalance.")
        print("  capacity-capped trades a few dropped tokens for that fixed budget; dropless pays")
        print("  extra rounds instead of drops and stays BIT-EXACT with naive.")


# --------------------------------------------------------------------------------------------------
# Ablation 2: load balancing — none / aux / bias / both on a small clustered-input training task
# --------------------------------------------------------------------------------------------------
def ablation_load_balance():
    import picotron.process_group_manager as pgm
    from picotron.expert_parallel.expert_parallel import MoELayer, collect_aux_loss

    if pgm.process_group_manager.ep_world_size > 1:
        return  # single-rank training ablation

    E, k, hidden = 8, 2, 32
    G = 8  # input clusters: a balanced router would map clusters -> experts ~uniformly
    torch.manual_seed(0)
    centers = torch.randn(G, hidden) * 3.0
    teacher = [torch.randn(hidden, hidden) * (1.0 / hidden**0.5) for _ in range(G)]  # per-cluster target map

    def batch(n, seed):
        g = torch.Generator().manual_seed(seed)
        c = torch.randint(0, G, (n,), generator=g)
        x = centers[c] + 0.3 * torch.randn(n, hidden, generator=g)
        y = torch.stack([x[i] @ teacher[c[i]] for i in range(n)])
        return x.unsqueeze(0), y  # x as [1, n, hidden] for the MoE, y as [n, hidden]

    def run(aux, bias_rate, steps=400):
        torch.manual_seed(0)
        moe = MoELayer(cfg(hidden=hidden, num_experts=E, top_k=k, aux=aux, bias_rate=bias_rate))
        moe.train()
        opt = torch.optim.AdamW(moe.parameters(), lr=3e-3)
        for s in range(steps):
            x, y = batch(256, seed=1000 + s)
            out = moe(x).view(-1, hidden)
            loss = torch.nn.functional.mse_loss(out, y)
            a = collect_aux_loss(moe)
            total = loss + a if a is not None else loss
            opt.zero_grad()
            total.backward()
            opt.step()

        # Eval routing load over a fresh fixed batch.
        moe.eval()
        xe, ye = batch(2048, seed=7)
        with torch.no_grad():
            tok = xe.reshape(-1, hidden)
            probs = torch.softmax(moe.gate(tok), dim=-1)
            sel = torch.topk(probs + moe.expert_bias, k, dim=-1).indices
            load = torch.bincount(sel.reshape(-1), minlength=E).float()
            mse = torch.nn.functional.mse_loss(moe(xe).view(-1, hidden), ye).item()
        maxvio = (load.max() / load.mean()).item()  # 1.0 = perfectly balanced
        dead = int((load == 0).sum())
        return mse, maxvio, dead, load

    print("\n=== Ablation 2: router load balancing (E={}, top_k={}, {} input clusters) ===".format(E, k, G))
    print(f"{'config':>14} {'task MSE':>10} {'MaxVio (max/mean)':>18} {'dead experts':>13}")
    for name, aux, br in (("none", 0.0, 0.0), ("aux", 1e-2, 0.0), ("bias", 0.0, 1e-3), ("aux+bias", 1e-2, 1e-3)):
        mse, maxvio, dead, load = run(aux, br)
        print(f"{name:>14} {mse:>10.4f} {maxvio:>18.2f} {dead:>13d}")
    print("  MaxVio = busiest expert's load / mean load (1.0 = perfect); dead = experts with 0 tokens.")


def main():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "25516")
    os.environ.setdefault("LOCAL_RANK", str(rank))
    os.environ.setdefault("FLASH_ATTEN", "0")
    os.environ.setdefault("DEVICE", "cpu")
    os.environ.setdefault("DTYPE", "float32")

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    from picotron.process_group_manager import setup_process_group_manager

    setup_process_group_manager(tp_size=1, cp_size=1, ep_size=world_size, pp_size=1, dp_size=1)

    ablation_capacity()
    ablation_load_balance()

    dist.barrier() if world_size > 1 else None
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
