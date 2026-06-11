"""Numerical tests for the fused MoE MegaKernel (Triton): forward + explicit dgrad/wgrad backward.

Run:
    python tests/test_megakernel.py
"""

import torch
import torch.nn.functional as F

from picotron.expert_parallel import megakernel


def reference_ffn(x_sorted, counts, gate_w, up_w, down_w, row_weight=None):
    outs, offset = [], 0
    for e, n in enumerate(counts.tolist()):
        x = x_sorted[offset : offset + n]
        h = F.silu(x @ gate_w[e].T) * (x @ up_w[e].T)
        outs.append(h @ down_w[e].T)
        offset += n
    y = torch.cat(outs, dim=0)
    if row_weight is not None:
        y = y * row_weight[:, None]
    return y


def _clone(t):
    return t.detach().clone().requires_grad_(True)


def test_megakernel_fwd_bwd():
    assert torch.cuda.is_available(), "needs CUDA"
    torch.manual_seed(0)
    device = "cuda"

    E, H, inter = 4, 48, 96
    counts = torch.tensor([5, 0, 130, 17], device=device)  # ragged incl. empty + > BLOCK_M
    N = int(counts.sum())

    x = torch.randn(N, H, device=device, requires_grad=True)
    gate_w = (torch.randn(E, inter, H, device=device) * 0.1).requires_grad_(True)
    up_w = (torch.randn(E, inter, H, device=device) * 0.1).requires_grad_(True)
    down_w = (torch.randn(E, H, inter, device=device) * 0.1).requires_grad_(True)

    xr, gwr, uwr, dwr = _clone(x), _clone(gate_w), _clone(up_w), _clone(down_w)

    y = megakernel.fused_moe(x, counts, gate_w, up_w, down_w)
    yr = reference_ffn(xr, counts, gwr, uwr, dwr)
    fwd = (y - yr).abs().max().item()
    assert torch.allclose(y, yr, atol=1e-4, rtol=1e-4), f"forward mismatch {fwd}"

    g = torch.randn_like(y)
    y.backward(g)
    yr.backward(g)
    diffs = {
        "dx": (x.grad - xr.grad).abs().max().item(),
        "dgate": (gate_w.grad - gwr.grad).abs().max().item(),
        "dup": (up_w.grad - uwr.grad).abs().max().item(),
        "ddown": (down_w.grad - dwr.grad).abs().max().item(),
    }
    for name, d in diffs.items():
        assert d < 1e-3, f"{name} grad mismatch {d}"
    print(f"test_megakernel_fwd_bwd PASSED fwd={fwd:.2e} " + " ".join(f"{k}={v:.2e}" for k, v in diffs.items()))


def test_megakernel_combine_epilogue():
    """Fused combine: per-row gating weight multiplied inside the kernel epilogue."""
    torch.manual_seed(1)
    device = "cuda"
    E, H, inter = 3, 32, 64
    counts = torch.tensor([20, 33, 7], device=device)
    N = int(counts.sum())
    x = torch.randn(N, H, device=device)
    gate_w = torch.randn(E, inter, H, device=device) * 0.1
    up_w = torch.randn(E, inter, H, device=device) * 0.1
    down_w = torch.randn(E, H, inter, device=device) * 0.1
    rw = torch.rand(N, device=device)

    y = megakernel.fused_moe(x, counts, gate_w, up_w, down_w, row_weight=rw)
    yr = reference_ffn(x, counts, gate_w, up_w, down_w, row_weight=rw)
    diff = (y - yr).abs().max().item()
    assert torch.allclose(y, yr, atol=1e-4), f"combine mismatch {diff}"
    print(f"test_megakernel_combine_epilogue PASSED diff={diff:.2e}")


def test_megakernel_bf16():
    torch.manual_seed(2)
    device = "cuda"
    E, H, inter = 4, 64, 128
    counts = torch.tensor([16, 16, 16, 16], device=device)
    N = int(counts.sum())
    x = torch.randn(N, H, device=device, dtype=torch.bfloat16)
    gate_w = (torch.randn(E, inter, H, device=device) * 0.1).bfloat16()
    up_w = (torch.randn(E, inter, H, device=device) * 0.1).bfloat16()
    down_w = (torch.randn(E, H, inter, device=device) * 0.1).bfloat16()
    y = megakernel.fused_moe(x, counts, gate_w, up_w, down_w)
    yr = reference_ffn(x, counts, gate_w, up_w, down_w)
    diff = (y.float() - yr.float()).abs().max().item()
    assert diff < 5e-2, f"bf16 mismatch {diff}"
    print(f"test_megakernel_bf16 PASSED diff={diff:.2e}")


def main():
    if not torch.cuda.is_available():
        print("[skip] CUDA megakernel tests (no GPU)")
        return
    test_megakernel_fwd_bwd()
    test_megakernel_combine_epilogue()
    test_megakernel_bf16()


if __name__ == "__main__":
    main()
