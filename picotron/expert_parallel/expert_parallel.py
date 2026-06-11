"""Expert Parallelism (EP) for Mixture-of-Experts layers.

Each MoE layer holds ``num_experts`` experts that are sharded across the EP group:
rank ``r`` owns experts ``[r * num_local_experts, (r + 1) * num_local_experts)``. A trainable
router (gate) picks the top-k experts for every token. Tokens are then exchanged with an
all-to-all (dispatch), processed by the local experts, and sent back with a second all-to-all
(combine) before being weighted by the router probabilities. This dispatch -> expert GEMM ->
combine pattern is the one used by DeepEP / MegaScale-MoE / UniEP, here written with plain
``torch.distributed`` collectives for readability rather than peak performance.
"""

import types

import torch
import torch.nn as nn
import torch.nn.functional as F

import picotron.process_group_manager as pgm
from picotron.expert_parallel import deepep_backend, megakernel
from picotron.expert_parallel.ep_communications import (
    all_to_all,
    all_to_all_fp8,
    all_to_all_no_grad,
    exchange_split_sizes,
)
from picotron.expert_parallel.tiled_moe import TiledMeta, TilePlan, tiled_moe


def _argsort_inv(key):
    """Stable argsort plus its inverse permutation (so ``x[order][inv_order] == x``)."""
    order = torch.argsort(key, stable=True)
    inv_order = torch.empty_like(order)
    inv_order[order] = torch.arange(order.numel(), device=order.device)
    return order, inv_order


class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Imported lazily to avoid a circular import (model.py imports this module).
        from picotron.model import MLP

        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = getattr(config, "norm_topk_prob", True)

        self.ep_world_size = pgm.process_group_manager.ep_world_size
        self.ep_rank = pgm.process_group_manager.ep_rank
        assert self.num_experts % self.ep_world_size == 0, (
            f"num_experts ({self.num_experts}) must be divisible by ep_size ({self.ep_world_size})"
        )
        self.num_local_experts = self.num_experts // self.ep_world_size

        # LatentMoE (NVIDIA Nemotron): project tokens to a smaller latent dim before routing/dispatch,
        # run the experts and all-to-all in latent space, then project back up after combine. Shrinks
        # BOTH the dispatch bytes AND the expert weights by hidden/latent. 0 (or >= hidden) disables it.
        self.latent_dim = getattr(config, "moe_latent_dim", 0)
        self.latent = 0 < self.latent_dim < self.hidden_size
        # Dimension the experts + dispatch/combine operate in (latent if enabled, else full hidden).
        self.expert_dim = self.latent_dim if self.latent else self.hidden_size

        # Router over all experts, in the FULL hidden dim (Nemotron keeps routing at full precision).
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        # Shared down/up projections around the routed path (only for LatentMoE), replicated per rank.
        if self.latent:
            self.down_proj_latent = nn.Linear(self.hidden_size, self.latent_dim, bias=False)
            self.up_proj_latent = nn.Linear(self.latent_dim, self.hidden_size, bias=False)
        # Only the experts owned by this rank are materialized here, in ``expert_dim``.
        expert_config = types.SimpleNamespace(
            hidden_size=self.expert_dim,
            intermediate_size=config.intermediate_size,
        )
        self.local_experts = nn.ModuleList([MLP(expert_config) for _ in range(self.num_local_experts)])
        # Optional always-on shared expert(s) (DeepSeek-style), in full hidden dim, replicated per rank.
        # N shared experts of width ``intermediate_size`` are merged into a single MLP of width
        # N * intermediate_size: a SwiGLU MLP's intermediate channels are independent and ``down_proj``
        # sums over them, so the wide MLP equals the sum of N separate shared experts (one GEMM).
        self.num_shared_experts = getattr(config, "num_shared_experts", 0)
        if self.num_shared_experts > 0:
            shared_config = types.SimpleNamespace(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size * self.num_shared_experts,
            )
            self.shared_expert = MLP(shared_config)
        else:
            self.shared_expert = None

        # Overlap the dispatch/combine all-to-all with the shared-expert FFN (DeepSeek-style). The
        # shared expert is comm-free local compute, so it is run on a side CUDA stream and composes
        # with EVERY routed path (naive / tiled / deepep). Only helps when there is a shared expert
        # and EP actually communicates; no-op (serial) on CPU/gloo so results stay bit-exact.
        self.ep_overlap = getattr(config, "ep_overlap", True)
        self._shared_stream = None  # lazily created CUDA stream for the overlapped shared expert
        # MegaScale-style token-tiled pipeline: split routed tokens into this many tiles and overlap
        # dispatch / expert GEMM / combine across them. 1 disables it (falls back to the naive path).
        self.ep_num_tiles = getattr(config, "ep_num_tiles", 1)
        # DeepSeek/DeepEP-style FP8 dispatch: ship token features in FP8 (E4M3) to ~halve the dispatch
        # all-to-all volume; combine stays BF16. Reduces *bytes* (orthogonal to the overlap above).
        self.ep_fp8_dispatch = getattr(config, "ep_fp8_dispatch", False)
        # Dispatch/combine backend: "torch" (our portable all-to-all) or "deepep" (DeepEP's CUDA kernels,
        # Hopper SM90+ only). "deepep" transparently falls back to torch when DeepEP is unavailable.
        self.ep_backend = getattr(config, "ep_backend", "torch")

        self.reset_parameters()

    def reset_parameters(self):
        # NOTE: experts share the same RNG stream across EP ranks, so corresponding local-index
        # experts start from identical weights. This keeps the non-expert RNG state in sync across
        # ranks (required for replicated params) and is fine for an educational setup; a production
        # implementation would seed each global expert independently for more init diversity.
        self.gate.reset_parameters()
        if self.latent:
            self.down_proj_latent.reset_parameters()
            self.up_proj_latent.reset_parameters()
        for expert in self.local_experts:
            expert.reset_parameters()
        if self.shared_expert is not None:
            self.shared_expert.reset_parameters()

    def forward(self, x):
        batch_size, seq_length, _ = x.shape
        tokens = x.reshape(-1, self.hidden_size)  # [T, hidden]

        router_logits = self.gate(tokens)  # [T, num_experts]
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_idx = torch.topk(routing_weights, self.top_k, dim=-1)  # [T, k]
        if self.norm_topk_prob and self.top_k > 1:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(tokens.dtype)

        # Pick the routed-expert execution path (all produce identical results). Each path also
        # finalizes with the shared expert, which `ep_overlap` runs on a side stream so it is hidden
        # behind the dispatch/combine comm regardless of which path is chosen:
        #   * deepep   -> DeepEP's CUDA dispatch/combine kernels (Hopper SM90+)
        #   * tiled    -> overlap dispatch/GEMM/combine across token tiles (MegaScale-style)
        #   * naive    -> blocking dispatch -> experts -> combine
        if self.ep_world_size > 1 and self.ep_backend == "deepep" and deepep_backend.deepep_available():
            expert_out = self._moe_deepep(tokens, topk_idx, topk_weights)
        elif self.ep_world_size > 1 and self.ep_num_tiles > 1:
            expert_out = self._moe_tiled(tokens, topk_idx, topk_weights)
        else:
            expert_out = self._moe_naive(tokens, topk_idx, topk_weights)  # [T, hidden]

        return expert_out.view(batch_size, seq_length, self.hidden_size)

    def _expand_routed(self, tokens, topk_idx, topk_weights):
        """Expand to one routed row per (token, selected expert), laid out token-major."""
        token_idx = torch.arange(tokens.shape[0], device=tokens.device).repeat_interleave(self.top_k)  # [T*k]
        return tokens[token_idx], topk_idx.reshape(-1), topk_weights.reshape(-1)  # routed, expert_idx, weights

    def _combine_topk(self, y, weights, num_tokens):
        """Weight each routed row by its gating prob and sum the top-k contributions per token."""
        y = y * weights.unsqueeze(-1)
        # Rows are token-major (token 0's k experts, token 1's k experts, ...), so a reshape-sum
        # recovers per-token output (in ``expert_dim``: latent if LatentMoE, else full hidden).
        return y.view(num_tokens, self.top_k, self.expert_dim).sum(dim=1)

    def _maybe_down(self, tokens):
        """LatentMoE down-projection: full hidden -> latent before routing/dispatch (no-op otherwise)."""
        return self.down_proj_latent(tokens) if self.latent else tokens

    def _shared_start(self, tokens):
        """Start the shared-expert FFN, overlapped with the routed path's all-to-all when possible.

        The shared expert needs no communication, so on CUDA (with ``ep_overlap`` and EP comm) it is
        launched on a side stream and runs concurrently with the dispatch/combine of whichever routed
        path is active. Returns ``(output, stream)``; ``stream`` is ``None`` when run serially (no
        shared expert, CPU/gloo, or single rank), in which case ``_finalize`` just adds the result.
        """
        if self.shared_expert is None:
            return None, None
        overlap = self.ep_overlap and self.ep_world_size > 1 and tokens.is_cuda
        if not overlap:
            return self.shared_expert(tokens), None
        if self._shared_stream is None:
            self._shared_stream = torch.cuda.Stream()
        self._shared_stream.wait_stream(torch.cuda.current_stream())  # shared reads tokens once ready
        with torch.cuda.stream(self._shared_stream):
            out = self.shared_expert(tokens)
        return out, self._shared_stream

    def _finalize(self, y_expert, shared):
        """Up-project the combined expert output back to hidden (LatentMoE) and add the shared expert."""
        y = self.up_proj_latent(y_expert) if self.latent else y_expert
        shared_out, stream = shared
        if shared_out is None:
            return y
        if stream is not None:
            torch.cuda.current_stream().wait_stream(stream)  # join the overlapped shared expert
        return y + shared_out

    def _dest_plan(self, expert_idx):
        """Sort routed rows by destination EP rank; return the sort order, its inverse, the dispatch
        split sizes (as lists), and each sorted row's local expert id."""
        group = pgm.process_group_manager.ep_group
        dest_rank = expert_idx // self.num_local_experts
        order, inv_order = _argsort_inv(dest_rank)
        input_splits = torch.bincount(dest_rank, minlength=self.ep_world_size)
        output_splits = exchange_split_sizes(input_splits, group)
        local_expert_sorted = (expert_idx[order] % self.num_local_experts).to(torch.int64)
        return order, inv_order, input_splits.tolist(), output_splits.tolist(), local_expert_sorted

    def _moe_naive(self, tokens, topk_idx, topk_weights):
        num_tokens = tokens.shape[0]
        shared = self._shared_start(tokens)  # overlaps the dispatch/combine below
        latent = self._maybe_down(tokens)  # full hidden, or latent for LatentMoE
        routed, expert_idx, weights = self._expand_routed(latent, topk_idx, topk_weights)

        if self.ep_world_size == 1:
            # All experts are local: no token exchange needed.
            y = self._run_local_experts(routed, expert_idx)
        else:
            y = self._dispatch_combine(routed, expert_idx)

        y = self._combine_topk(y, weights, num_tokens)
        return self._finalize(y, shared)

    def _grouped_experts(self, x_sorted, counts):
        """Run the local experts on rows already grouped/sorted by local expert; ``counts[e]`` rows go
        to expert ``e``. Returns the expert outputs in the same (sorted) order."""
        if x_sorted.shape[0] == 0:
            return x_sorted
        if x_sorted.is_cuda and megakernel.cuda_available():
            # Fast path: UniEP-style fused MegaKernel (one launch for the whole expert FFN).
            gate_w = torch.stack([e.gate_proj.weight for e in self.local_experts])  # [E, I, H]
            up_w = torch.stack([e.up_proj.weight for e in self.local_experts])  # [E, I, H]
            down_w = torch.stack([e.down_proj.weight for e in self.local_experts])  # [E, H, I]
            return megakernel.fused_moe(x_sorted, counts, gate_w, up_w, down_w)
        # Fallback: run each expert as a separate matmul.
        counts_list = counts.tolist()
        outputs, offset = [], 0
        for expert_id in range(self.num_local_experts):
            count = counts_list[expert_id]
            outputs.append(self.local_experts[expert_id](x_sorted[offset : offset + count]))
            offset += count
        return torch.cat(outputs, dim=0)

    def _run_local_experts(self, x, local_expert_idx):
        """Run the local experts on ``x`` given each row's local expert id, preserving input order."""
        if x.shape[0] == 0:
            return x
        order = torch.argsort(local_expert_idx, stable=True)  # group rows by expert
        inv_order = torch.empty_like(order)
        inv_order[order] = torch.arange(order.numel(), device=order.device)

        counts = torch.bincount(local_expert_idx, minlength=self.num_local_experts)
        y_sorted = self._grouped_experts(x[order], counts)
        return y_sorted[inv_order]

    def _moe_deepep(self, tokens, topk_idx, topk_weights):
        """MoE using DeepEP's classic intranode (NVLink) dispatch/combine kernels (Hopper SM90+).

        DeepEP's ``dispatch`` sends each token once to every rank owning one of its top-k experts and
        returns, per received token, the local expert ids it selected (``recv_topk_idx``, ``-1`` for
        non-local) and their gate weights. We run the experts, apply the gate weights, then ``combine``
        sums each token's per-rank contributions back into the full top-k output. See ``deepep_backend``.
        """
        group = pgm.process_group_manager.ep_group
        shared = self._shared_start(tokens)  # overlaps DeepEP's dispatch/combine (near-zero SM comm)
        latent = self._maybe_down(tokens).contiguous()  # full hidden, or latent for LatentMoE (bf16)
        recv_x, handle, recv_idx, recv_w, _ = deepep_backend.dispatch(
            group, latent, topk_idx, topk_weights, self.num_experts
        )
        expert_out = self._deepep_experts(recv_x, recv_idx, recv_w)  # [T_recv, expert_dim], gate-weighted
        y = deepep_backend.combine(group, expert_out, handle)  # [T, expert_dim], summed over ranks
        return self._finalize(y, shared)

    def _deepep_experts(self, recv_x, recv_topk_idx, recv_topk_weights):
        """Run local experts on DeepEP-dispatched tokens, weighted by their gate probs.

        Each received row may select several local experts (``recv_topk_idx``, ``-1`` masked). We
        expand to one (token, expert) pair per valid selection, group by expert for the fused FFN,
        scale by the gate weight, then scatter-add back to one row per received token.
        """
        n, top_k = recv_x.shape[0], recv_topk_idx.shape[1]
        token_row = torch.arange(n, device=recv_x.device).repeat_interleave(top_k)  # [n*k]
        valid = recv_topk_idx.reshape(-1) >= 0
        sel_expert = recv_topk_idx.reshape(-1)[valid].to(torch.int64)
        sel_token = token_row[valid]
        sel_w = recv_topk_weights.reshape(-1)[valid].to(recv_x.dtype)

        order = torch.argsort(sel_expert, stable=True)  # group rows by local expert
        sel_token, sel_w, sel_expert = sel_token[order], sel_w[order], sel_expert[order]
        counts = torch.bincount(sel_expert, minlength=self.num_local_experts)

        y_sorted = self._grouped_experts(recv_x[sel_token], counts) * sel_w.unsqueeze(-1)
        y = torch.zeros_like(recv_x)
        return y.index_add(0, sel_token, y_sorted)

    def _dispatch_combine(self, routed, expert_idx):
        group = pgm.process_group_manager.ep_group
        order, inv_order, in_list, out_list, local_expert_sorted = self._dest_plan(expert_idx)
        routed_sorted = routed[order]

        # Dispatch: send token features and their (local) target expert id to the owning rank.
        dispatch = all_to_all_fp8 if self.ep_fp8_dispatch else all_to_all
        recv = dispatch(routed_sorted, out_list, in_list, group)
        recv_local_expert = all_to_all_no_grad(local_expert_sorted, out_list, in_list, group)

        # Local expert computation on received tokens.
        recv_out = self._run_local_experts(recv, recv_local_expert)

        # Combine: send the expert outputs back to where the tokens came from (split sizes swapped).
        combined = all_to_all(recv_out, in_list, out_list, group)

        # Undo the destination-rank grouping to restore the original routed order.
        return combined[inv_order]

    def _moe_tiled(self, tokens, topk_idx, topk_weights):
        """MoE with a MegaScale/UniEP-style token-tiled pipeline overlapping dispatch / GEMM / combine.

        Routed rows are split into ``ep_num_tiles`` contiguous tiles run as a software pipeline that
        overlaps communication and computation in BOTH forward and backward (see ``tiled_moe.py``).
        Smaller tiles overlap more but send smaller (less bandwidth-efficient) messages -- tune
        ``ep_num_tiles`` per fabric. Real overlap needs multi-GPU NCCL; the result is identical to the
        non-tiled path.

        ``num_tiles`` is fixed (not clamped to this rank's token count) so every rank issues the same
        number of all-to-all collectives and stays in lockstep; tiles with no tokens are simply empty.
        """
        num_tokens = tokens.shape[0]
        group = pgm.process_group_manager.ep_group
        shared = self._shared_start(tokens)  # overlaps the tiled dispatch/combine pipeline
        latent = self._maybe_down(tokens)  # full hidden, or latent for LatentMoE
        routed, expert_idx, weights = self._expand_routed(latent, topk_idx, topk_weights)

        num_rows = routed.shape[0]
        num_tiles = max(1, self.ep_num_tiles)
        bounds = [(i * num_rows) // num_tiles for i in range(num_tiles + 1)]

        # Per-tile destination sort + split-size exchange (tiny metadata collectives).
        plans = []
        for i in range(num_tiles):
            lo, hi = bounds[i], bounds[i + 1]
            order, inv_order, in_list, out_list, local_expert_sorted = self._dest_plan(expert_idx[lo:hi])
            plans.append(TilePlan(lo, hi, order, inv_order, local_expert_sorted, in_list, out_list))

        # Stacked expert weights; grads route back to each expert via torch.stack's backward.
        gate_w = torch.stack([ex.gate_proj.weight for ex in self.local_experts])  # [E, I, H]
        up_w = torch.stack([ex.up_proj.weight for ex in self.local_experts])  # [E, I, H]
        down_w = torch.stack([ex.down_proj.weight for ex in self.local_experts])  # [E, H, I]
        meta = TiledMeta(plans, group, self.num_local_experts)

        y = tiled_moe(routed, gate_w, up_w, down_w, meta)  # [num_rows, expert_dim], routed order
        y = self._combine_topk(y, weights, num_tokens)
        return self._finalize(y, shared)
