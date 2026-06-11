"""Industrial-style fused MoE expert FFN in Triton (vLLM / MegaBlocks pattern).

The expert computation is the SwiGLU FFN  down( silu(x @ gate^T) * (x @ up^T) )  run per expert on
tokens that have been grouped (sorted) by their destination expert. We implement it the way
production MoE kernels do:

  * An autotuned, tiled **grouped GEMM** (`_grouped_gemm_kernel`) that processes fixed-size token
    blocks, each block belonging to a single expert (block-aligned tiling, as in MegaBlocks). Tile
    sizes / num_warps / num_stages are picked by `triton.autotune` — the main perf lever.
  * **gate and up projections fused into one GEMM** by stacking their weights into `w13 = [E, 2I, H]`,
    then a Triton **silu_and_mul** activation, then the **down** GEMM — the standard two-GEMM vLLM
    structure.
  * An optional **routed-weight epilogue** (multiply each row by its gating weight) fused into the
    down GEMM's store.

Forward runs the Triton kernels. Backward is also explicit Triton (no autograd): the forward GEMM is
reused for the input-grad by swapping the weight strides (`A @ Wᵀ` -> `A @ W`, see ``_grouped_gemm_dx``)
and a dedicated grouped ``_grouped_wgrad_kernel`` computes the per-expert weight grads (`Cᵀ @ A`). A
torch fallback (``torch_fused_ffn`` / ``torch_moe_backward``) is used when CUDA/Triton is unavailable.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

BLOCK_M = 64  # token block; one block belongs to a single expert


def _gemm_configs():
    return [
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 16}, num_warps=2, num_stages=2),
    ]


@triton.autotune(configs=_gemm_configs(), key=["N", "K"])
@triton.jit
def _grouped_gemm_kernel(
    A,
    W,
    C,
    ROWW,
    blk_expert_ptr,
    blk_row_start_ptr,
    blk_row_count_ptr,
    N,
    K,
    stride_am,
    stride_ak,
    stride_we,
    stride_wn,
    stride_wk,
    stride_cm,
    stride_cn,
    HAS_ROWW: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # C[m, :] = A[m, :] @ W[expert(block), :, :]^T  for the token block this program owns.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    e = tl.load(blk_expert_ptr + pid_m)
    row0 = tl.load(blk_row_start_ptr + pid_m)
    rcount = tl.load(blk_row_count_ptr + pid_m)

    offs_m = row0 + tl.arange(0, BLOCK_M)
    m_mask = tl.arange(0, BLOCK_M) < rcount
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    w_base = W + e * stride_we
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        a = tl.load(
            A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        w = tl.load(
            w_base + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(a, w, allow_tf32=ALLOW_TF32)

    if HAS_ROWW:
        acc = acc * tl.load(ROWW + offs_m, mask=m_mask, other=0.0)[:, None]

    tl.store(
        C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(C.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _silu_mul_kernel(X, Y, n_cols, D, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < D
    g = tl.load(X + row * n_cols + col, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(X + row * n_cols + D + col, mask=mask, other=0.0).to(tl.float32)
    y = (g * tl.sigmoid(g)) * u
    tl.store(Y + row * D + col, y.to(Y.dtype.element_ty), mask=mask)


def _build_block_schedule(counts, device):
    """Per token-block (expert, row_start, row_count); one block never spans two experts."""
    be, brs, brc = [], [], []
    offset = 0
    for e, n in enumerate(counts):
        rt = 0
        while rt < n:
            be.append(e)
            brs.append(offset + rt)
            brc.append(min(BLOCK_M, n - rt))
            rt += BLOCK_M
        offset += n
    i32 = lambda xs: torch.tensor(xs, dtype=torch.int32, device=device)  # noqa: E731
    return i32(be), i32(brs), i32(brc)


def _gg(A, W, sched, M, N, K, stride_we, stride_wn, stride_wk, row_weight=None):
    """Grouped GEMM C[M,N] = sum_k A[m,k] * W[expert(block)][n,k], with explicit W strides.

    Passing forward strides (we, wn, wk) computes ``A @ W[e]^T``; swapping (wn, wk) computes
    ``A @ W[e]`` (the dgrad pattern), reusing the same kernel.
    """
    be, brs, brc = sched
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    num_m_blocks = be.numel()
    has_roww = row_weight is not None
    roww = row_weight if has_roww else A
    grid = lambda meta: (num_m_blocks, triton.cdiv(N, meta["BLOCK_N"]))  # noqa: E731
    _grouped_gemm_kernel[grid](
        A,
        W,
        C,
        roww,
        be,
        brs,
        brc,
        N,
        K,
        A.stride(0),
        A.stride(1),
        stride_we,
        stride_wn,
        stride_wk,
        C.stride(0),
        C.stride(1),
        HAS_ROWW=has_roww,
        ALLOW_TF32=(A.dtype != torch.float32),
        BLOCK_M=BLOCK_M,
    )
    return C


def _grouped_gemm(A, W, sched, M, N, K, row_weight):
    """Forward: C[M,N] = A @ W[e]^T, with W stored as [E, N, K]."""
    return _gg(A, W, sched, M, N, K, W.stride(0), W.stride(1), W.stride(2), row_weight)


def _grouped_gemm_dx(A, W, sched, M, N, K):
    """Input-grad: C[M,N] = A @ W[e], with W stored as [E, K, N] (contraction over the K=dim-1)."""
    return _gg(A, W, sched, M, N, K, W.stride(0), W.stride(2), W.stride(1), None)


def _silu_and_mul(x):
    M, two_d = x.shape
    D = two_d // 2
    y = torch.empty((M, D), device=x.device, dtype=x.dtype)
    grid = lambda meta: (M, triton.cdiv(D, meta["BLOCK"]))  # noqa: E731
    _silu_mul_kernel[grid](x, y, two_d, D, BLOCK=256)
    return y


def _silu_and_mul_bwd(inter13, da):
    """Backward of silu_and_mul. inter13=[g|u] [N,2I], da [N,I] -> d_inter13=[dg|du] [N,2I]."""
    D = da.shape[1]
    g = inter13[:, :D].float()
    u = inter13[:, D:].float()
    sig = torch.sigmoid(g)
    silu = g * sig
    daf = da.float()
    dg = daf * u * (sig * (1 + g * (1 - sig)))
    du = daf * silu
    return torch.cat([dg, du], dim=1).to(inter13.dtype)


def _wgrad_configs():
    return [
        triton.Config({"BLOCK_P": 64, "BLOCK_Q": 64, "BLOCK_M": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_P": 32, "BLOCK_Q": 64, "BLOCK_M": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_P": 64, "BLOCK_Q": 32, "BLOCK_M": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_P": 32, "BLOCK_Q": 32, "BLOCK_M": 32}, num_warps=2, num_stages=2),
    ]


@triton.autotune(configs=_wgrad_configs(), key=["P", "Q"])
@triton.jit
def _grouped_wgrad_kernel(
    C,
    A,
    D,
    row_start_ptr,
    row_count_ptr,
    P,
    Q,
    stride_cm,
    stride_cp,
    stride_am,
    stride_aq,
    stride_de,
    stride_dp,
    stride_dq,
    ALLOW_TF32: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # D[e, p, q] = sum_{m in expert e} C[m, p] * A[m, q]   (weight gradient: C^T @ A per expert).
    e = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_q = tl.program_id(2)

    row0 = tl.load(row_start_ptr + e)
    cnt = tl.load(row_count_ptr + e)

    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    p_mask = offs_p < P
    q_mask = offs_q < Q

    acc = tl.zeros([BLOCK_P, BLOCK_Q], dtype=tl.float32)
    for m0 in range(0, cnt, BLOCK_M):
        offs_m = row0 + m0 + tl.arange(0, BLOCK_M)
        m_mask = tl.arange(0, BLOCK_M) < (cnt - m0)
        c = tl.load(  # [BLOCK_P, BLOCK_M] (transposed load of C[m, p])
            C + offs_m[None, :] * stride_cm + offs_p[:, None] * stride_cp,
            mask=p_mask[:, None] & m_mask[None, :],
            other=0.0,
        )
        a = tl.load(  # [BLOCK_M, BLOCK_Q]
            A + offs_m[:, None] * stride_am + offs_q[None, :] * stride_aq,
            mask=m_mask[:, None] & q_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(c, a, allow_tf32=ALLOW_TF32)

    tl.store(
        D + e * stride_de + offs_p[:, None] * stride_dp + offs_q[None, :] * stride_dq,
        acc.to(D.dtype.element_ty),
        mask=p_mask[:, None] & q_mask[None, :],
    )


def _grouped_wgrad(C, A, counts, E, P, Q):
    """Per-expert weight grad D[E,P,Q] = C[:,P]^T @ A[:,Q] grouped by expert (counts [E])."""
    D = torch.empty((E, P, Q), device=A.device, dtype=A.dtype)
    row_count = counts.to(torch.int32)
    row_start = torch.zeros_like(row_count)
    row_start[1:] = torch.cumsum(row_count, 0)[:-1]
    grid = lambda meta: (E, triton.cdiv(P, meta["BLOCK_P"]), triton.cdiv(Q, meta["BLOCK_Q"]))  # noqa: E731
    _grouped_wgrad_kernel[grid](
        C,
        A,
        D,
        row_start,
        row_count,
        P,
        Q,
        C.stride(0),
        C.stride(1),
        A.stride(0),
        A.stride(1),
        D.stride(0),
        D.stride(1),
        D.stride(2),
        ALLOW_TF32=(A.dtype != torch.float32),
    )
    return D


def fused_moe_forward(x_sorted, counts, gate_w, up_w, down_w, row_weight=None):
    """x_sorted [N,H] (rows grouped by expert); gate_w/up_w [E,I,H]; down_w [E,H,I]; counts [E]."""
    N, H = x_sorted.shape
    inter_dim = gate_w.shape[1]
    if N == 0:
        return torch.empty((0, H), device=x_sorted.device, dtype=x_sorted.dtype)

    sched = _build_block_schedule(counts.tolist(), x_sorted.device)
    w13 = torch.cat([gate_w, up_w], dim=1).contiguous()  # [E, 2I, H]
    inter13 = _grouped_gemm(x_sorted, w13, sched, N, 2 * inter_dim, H, None)  # [N, 2I]
    inter = _silu_and_mul(inter13)  # [N, I]
    return _grouped_gemm(inter, down_w.contiguous(), sched, N, H, inter_dim, row_weight)  # [N, H]


def torch_fused_ffn(x_sorted, counts, gate_w, up_w, down_w, row_weight):
    outs, offset = [], 0
    for e, n in enumerate(counts.tolist()):
        x = x_sorted[offset : offset + n]
        h = F.silu(x @ gate_w[e].T) * (x @ up_w[e].T)
        outs.append(h @ down_w[e].T)
        offset += n
    y = torch.cat(outs, dim=0) if outs else x_sorted.new_empty((0, x_sorted.shape[1]))
    if row_weight is not None:
        y = y * row_weight[:, None]
    return y


def torch_moe_backward(x_sorted, counts, gate_w, up_w, down_w, dy):
    """Reference / CPU backward via autograd over the torch FFN (used when Triton is unavailable)."""
    with torch.enable_grad():
        xs = x_sorted.detach().requires_grad_(True)
        gw = gate_w.detach().requires_grad_(True)
        uw = up_w.detach().requires_grad_(True)
        dw = down_w.detach().requires_grad_(True)
        y = torch_fused_ffn(xs, counts, gw, uw, dw, None)
        return torch.autograd.grad(y, [xs, gw, uw, dw], dy)


def fused_moe_dgrad(x_sorted, counts, gate_w, up_w, down_w, dy):
    """Explicit input-grad (Triton). Returns (dx, a, d_inter13); a/d_inter13 feed ``fused_moe_wgrad``.

    Recomputes the forward intermediates (rematerialization) once, then runs explicit dgrad GEMMs --
    no autograd, no double graph traversal.
    """
    N, H = x_sorted.shape
    inter_dim = gate_w.shape[1]
    sched = _build_block_schedule(counts.tolist(), x_sorted.device)
    w13 = torch.cat([gate_w, up_w], dim=1).contiguous()  # [E, 2I, H]
    inter13 = _grouped_gemm(x_sorted, w13, sched, N, 2 * inter_dim, H, None)  # recompute
    a = _silu_and_mul(inter13)
    da = _grouped_gemm_dx(dy, down_w, sched, N, inter_dim, H)  # dy @ down_w[e]
    d_inter13 = _silu_and_mul_bwd(inter13, da)
    dx = _grouped_gemm_dx(d_inter13, w13, sched, N, H, 2 * inter_dim)  # d_inter13 @ w13[e]
    return dx, a, d_inter13


def fused_moe_wgrad(x_sorted, counts, a, d_inter13, dy):
    """Explicit weight-grads (Triton) from the cached forward intermediates. Returns (dg, du, dd)."""
    inter_dim = a.shape[1]
    E = counts.numel()
    H = x_sorted.shape[1]
    d_down = _grouped_wgrad(dy, a, counts, E, H, inter_dim)  # [E, H, I]
    d_w13 = _grouped_wgrad(d_inter13, x_sorted, counts, E, 2 * inter_dim, H)  # [E, 2I, H]
    return d_w13[:, :inter_dim, :].contiguous(), d_w13[:, inter_dim:, :].contiguous(), d_down


def fused_moe_backward(x_sorted, counts, gate_w, up_w, down_w, dy):
    """Single-pass MoE backward: explicit Triton dgrad+wgrad on GPU, torch autograd on CPU."""
    if x_sorted.shape[0] == 0:
        return (
            torch.zeros_like(x_sorted),
            torch.zeros_like(gate_w),
            torch.zeros_like(up_w),
            torch.zeros_like(down_w),
        )
    if x_sorted.is_cuda and cuda_available():
        dx, a, d_inter13 = fused_moe_dgrad(x_sorted, counts, gate_w, up_w, down_w, dy)
        d_gate, d_up, d_down = fused_moe_wgrad(x_sorted, counts, a, d_inter13, dy)
        return dx, d_gate, d_up, d_down
    return torch_moe_backward(x_sorted, counts, gate_w, up_w, down_w, dy)


class _FusedMoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_sorted, gate_w, up_w, down_w, counts, row_weight):
        y = fused_moe_forward(x_sorted, counts, gate_w, up_w, down_w, row_weight)
        ctx.save_for_backward(x_sorted, gate_w, up_w, down_w, row_weight)
        ctx.counts = counts
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x_sorted, gate_w, up_w, down_w, row_weight = ctx.saved_tensors
        if row_weight is None:
            # Fast path: explicit single-pass dgrad/wgrad kernels.
            dx, dgw, duw, ddw = fused_moe_backward(x_sorted, ctx.counts, gate_w, up_w, down_w, grad_y)
            return dx, dgw, duw, ddw, None, None
        # row-weight epilogue (rare): recompute with autograd.
        with torch.enable_grad():
            xs = x_sorted.detach().requires_grad_(True)
            gw = gate_w.detach().requires_grad_(True)
            uw = up_w.detach().requires_grad_(True)
            dw = down_w.detach().requires_grad_(True)
            rw = row_weight.detach().requires_grad_(True)
            y = torch_fused_ffn(xs, ctx.counts, gw, uw, dw, rw)
            grads = torch.autograd.grad(y, [xs, gw, uw, dw, rw], grad_y)
        return grads[0], grads[1], grads[2], grads[3], None, grads[4]


def fused_moe(x_sorted, counts, gate_w, up_w, down_w, row_weight=None):
    """Differentiable industrial fused MoE expert FFN. counts [E] = per-expert token counts."""
    return _FusedMoE.apply(x_sorted, gate_w, up_w, down_w, counts, row_weight)


def cuda_available():
    """Whether the Triton fused path can run (Triton is a hard import, so this just gates on CUDA)."""
    return torch.cuda.is_available()
