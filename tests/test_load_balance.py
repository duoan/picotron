"""Tests for MoE router load balancing: the aux loss and the loss-free bias controller.

* aux loss   -> a differentiable scalar that flows gradient to the gate; ``None`` when disabled.
* loss-free  -> a non-trainable per-expert bias updated by a sign rule; over-loaded experts get a
                lower selection bias and under-loaded experts a higher one, which flattens the load.

Run:
    python tests/test_load_balance.py
"""

import os
import types

import torch
import torch.distributed as dist


def make_config(hidden=16, inter=32, num_experts=8, top_k=2, aux=0.0, bias_rate=0.0):
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


def test_aux_loss_grad():
    import picotron.process_group_manager as pgm
    from picotron.expert_parallel.expert_parallel import MoELayer, collect_aux_loss

    torch.manual_seed(0)
    moe = MoELayer(make_config(aux=0.01))
    moe.train()

    torch.manual_seed(1)
    x = torch.randn(4, 8, moe.hidden_size)
    out = moe(x)

    aux = collect_aux_loss(moe)
    assert aux is not None and aux.item() > 0, "aux loss should be a positive scalar in training mode"
    # A balanced router gives sum_e f_e*P_e ~ 1/E per term * E ~ 1, scaled by coef.
    (out.float().pow(2).mean() + aux).backward()
    assert moe.gate.weight.grad is not None, "aux loss must backprop into the gate"

    # Disabled (coef = 0) -> no aux loss stashed.
    moe2 = MoELayer(make_config(aux=0.0))
    moe2.train()
    moe2(x)
    assert collect_aux_loss(moe2) is None, "aux loss should be None when coef=0"
    print(f"[rank {pgm.process_group_manager.ep_rank}] test_aux_loss_grad PASSED (aux={aux.item():.4f})")


def _selection_load(moe, x):
    """Per-expert count of how many tokens the (biased) router currently selects."""
    with torch.no_grad():
        probs = torch.softmax(moe.gate(x.reshape(-1, moe.hidden_size)), dim=-1)
        sel = torch.topk(probs + moe.expert_bias, moe.top_k, dim=-1).indices
        return torch.bincount(sel.reshape(-1), minlength=moe.num_experts).float()


def test_loss_free_bias_balances():
    import picotron.process_group_manager as pgm
    from picotron.expert_parallel.expert_parallel import MoELayer

    # One controller step moves the bias the right way (over-loaded down, under-loaded up).
    torch.manual_seed(0)
    moe = MoELayer(make_config(num_experts=8, top_k=2, bias_rate=0.01))
    moe.train()
    assert moe.use_loss_free
    torch.manual_seed(2)
    x = torch.randn(8, 16, moe.hidden_size)
    load0 = _selection_load(moe, x)
    moe(x)
    most, least = int(load0.argmax()), int(load0.argmin())
    assert moe.expert_bias[most] < moe.expert_bias[least], "bias controller moved the wrong way"

    # Over many steps with a fixed batch the bang-bang controller oscillates, but the TIME-AVERAGED
    # load is much more uniform than with no controller -- that is what load balancing buys.
    def cumulative_load(bias_rate):
        torch.manual_seed(0)
        m = MoELayer(make_config(num_experts=8, top_k=2, bias_rate=bias_rate))
        m.train()
        total = torch.zeros(m.num_experts)
        for _ in range(300):
            m(x)
            total += _selection_load(m, x)
        return total

    off = cumulative_load(0.0)
    on = cumulative_load(0.01)
    assert on.std() < 0.5 * off.std(), f"controller did not balance load: std {off.std():.0f} -> {on.std():.0f}"
    print(
        f"[rank {pgm.process_group_manager.ep_rank}] test_loss_free_bias_balances PASSED "
        f"(time-averaged load std {off.std():.0f} -> {on.std():.0f})"
    )


def main():
    rank = int(os.environ.get("RANK", 0))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "25515")
    os.environ.setdefault("LOCAL_RANK", str(rank))
    os.environ.setdefault("FLASH_ATTEN", "0")
    os.environ.setdefault("DEVICE", "cpu")
    os.environ.setdefault("DTYPE", "float32")

    dist.init_process_group(backend="gloo", rank=rank, world_size=int(os.environ.get("WORLD_SIZE", 1)))
    from picotron.process_group_manager import setup_process_group_manager

    setup_process_group_manager(tp_size=1, cp_size=1, ep_size=1, pp_size=1, dp_size=1)

    test_aux_loss_grad()
    test_loss_free_bias_balances()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
