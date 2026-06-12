"""Advanced pipeline-parallel schedules built on top of the picotron 1F1B engine.

Stock picotron ships two schedules in ``pipeline_parallel.py``:

- **AFAB** (all-forward-all-backward) — simplest, largest activation memory and bubble.
- **1F1B** (one-forward-one-backward) — steady-state interleave, bubble ``(p-1)/m``.

This module adds three schedules from the last two years of pipeline-parallelism research, each
trading a different resource to shrink the ``(p-1)/m`` pipeline bubble of 1F1B:

- **Zero Bubble (ZB-H1)** — Qi et al., ICLR 2024 (https://arxiv.org/abs/2401.10241). Splits the
  backward into B (input-activation grad, on the critical path) and W (weight grad, reschedulable),
  then fills bubbles with the comm-free W work. ``train_step_pipeline_zb``.
- **Interleaved 1F1B / virtual pipeline** — Narayanan et al. (Megatron-LM,
  https://arxiv.org/abs/2104.04473). Each rank owns ``v`` non-contiguous chunks, shrinking the
  bubble by ``v`` at the cost of ``v`` x more pipeline flushes of comm.
  ``InterleavedPipelineParallel`` + ``train_step_pipeline_interleaved``.
- **DualPipe** — DeepSeek-V3 (https://arxiv.org/abs/2412.19437). Two micro-batch streams run in
  opposite directions through the pipeline; each rank holds the two stages symmetric about the
  middle and their (replicated) gradients are summed across the symmetric rank pair. The two streams
  are *interleaved in time* so a forward of one overlaps a backward of the other and the bubble
  roughly halves. We run the two streams as cooperating generators driven concurrently — on NCCL each
  on its own CUDA stream (single host thread), on gloo each on its own host thread — over independent
  per-stream communicators, which keeps the interleave deadlock-free on both backends and bit-exact.
  ``DualPipeParallel`` + ``train_step_pipeline_dualpipe``.

All three are validated to produce **bit-exact gradients** vs the AFAB/1F1B baseline in
``tests/test_pipeline_parallel.py``.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import picotron.process_group_manager as pgm
from picotron.pipeline_parallel import pipeline_parallel
from picotron.pipeline_parallel.pp_communications import (
    _device_synchronize,
    bidirectional_pipeline_communicate,
    interleaved_pipeline_communicate,
    pipeline_communicate,
    ring_exchange,
)


def _forward_step(model, data_loader, input_tensor, device, chunk_id=None):
    """Run one micro-batch forward through ``model`` (or one of its chunks for interleaving).

    Returns ``(output_tensor, loss_item)`` where ``output_tensor`` is the cross-entropy loss on the
    last stage (and the hidden activation otherwise), and ``loss_item`` is the per-microbatch loss
    contribution to log (0.0 on non-last stages).
    """
    batch = next(data_loader)
    batch["hidden_states"] = input_tensor.to(device) if input_tensor is not None else input_tensor
    forward = model.forward if chunk_id is None else (lambda **kw: model.forward(chunk_id=chunk_id, **kw))
    output_tensor = forward(
        input_ids=batch["input_ids"].to(device),
        position_ids=batch["position_ids"].to(device),
        hidden_states=batch["hidden_states"],
    )

    is_last = (
        model.is_last_stage(chunk_id) if hasattr(model, "is_last_stage") else pgm.process_group_manager.pp_is_last_stage
    )
    if is_last:
        output_tensor = F.cross_entropy(
            output_tensor.flatten(0, 1), batch["target_ids"].to(device).flatten(), reduction="mean"
        )
        return output_tensor, output_tensor.item() / data_loader.grad_acc_steps
    return output_tensor, 0.0


def train_step_pipeline_zb(model, data_loader, tensor_shapes, device, dtype):
    """Zero-Bubble (ZB-H1) pipeline schedule.

    Same forward/B communication pattern as 1F1B — so the comm schedule and its dependencies are
    unchanged and deadlock-free — but each backward is split into B (``backward_input``) and W
    (``backward_weight``). The W passes are comm-free, so they are queued and drained into the
    bubbles that 1F1B would spend idle (one W per steady-state and cooldown step, the rest flushed at
    the tail). Activation memory stays close to 1F1B because the W queue is bounded by ``num_warmup``.

    The B/W split is a true ZB-H1 split (deferred per-Linear weight grads, ~1x backward FLOPs); the
    ``_ZB_DEFER`` flag must be on while building the forward graphs so the Linears become deferrable.
    """
    pipeline_parallel._ZB_DEFER = True
    try:
        return _train_step_pipeline_zb_impl(model, data_loader, tensor_shapes, device, dtype)
    finally:
        pipeline_parallel._ZB_DEFER = False


def _train_step_pipeline_zb_impl(model, data_loader, tensor_shapes, device, dtype):
    pp = pgm.process_group_manager
    num_warmup = min(pp.pp_world_size - pp.pp_rank - 1, data_loader.grad_acc_steps)
    num_remaining = data_loader.grad_acc_steps - num_warmup
    logging_loss = 0.0
    input_tensors, output_tensors = [], []  # pending B (FIFO, same as 1F1B)
    w_queue = []  # pending W: (output_tensor, output_tensor_grad)
    requires_grad_sync = pp.cp_dp_world_size > 1
    if requires_grad_sync:
        model.require_backward_grad_sync = False

    def run_w(sync):
        if requires_grad_sync:
            model.require_backward_grad_sync = sync
        w_out, w_grad = w_queue.pop(0)
        model.backward_weight(w_out, w_grad)

    # Warmup: fill the pipeline with forwards.
    for _ in range(num_warmup):
        input_tensor = pipeline_communicate("recv_forward", device, dtype, shapes=tensor_shapes)
        output_tensor, loss = _forward_step(model, data_loader, input_tensor, device)
        logging_loss += loss
        pipeline_communicate("send_forward", device, dtype, tensor=output_tensor)
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    if num_remaining > 0:
        input_tensor = pipeline_communicate("recv_forward", device, dtype, shapes=tensor_shapes)

    # Steady state: one forward, one B, drain one W to keep memory bounded.
    for i in range(num_remaining):
        is_last_iteration = i == num_remaining - 1
        output_tensor, loss = _forward_step(model, data_loader, input_tensor, device)
        logging_loss += loss
        output_tensor_grad = bidirectional_pipeline_communicate(
            "send_fwd_recv_bwd", output_tensor, tensor_shapes, device, dtype
        )
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

        b_input, b_output = input_tensors.pop(0), output_tensors.pop(0)
        input_tensor_grad = model.backward_input(b_input, b_output, output_tensor_grad)
        w_queue.append((b_output, output_tensor_grad))

        if is_last_iteration:
            input_tensor = None
            pipeline_communicate("send_backward", device, dtype, tensor=input_tensor_grad)
        else:
            input_tensor = bidirectional_pipeline_communicate(
                "send_bwd_recv_fwd", input_tensor_grad, tensor_shapes, device, dtype
            )
        # Drain a W into the slot the steady state would otherwise spend idle, bounding the W queue.
        if len(w_queue) > num_warmup:
            run_w(sync=False)

    # Cooldown: remaining B passes, each followed by a W to fill the drain bubble.
    for _ in range(num_warmup):
        b_input, b_output = input_tensors.pop(0), output_tensors.pop(0)
        output_tensor_grad = pipeline_communicate("recv_backward", device, dtype, shapes=tensor_shapes)
        input_tensor_grad = model.backward_input(b_input, b_output, output_tensor_grad)
        w_queue.append((b_output, output_tensor_grad))
        pipeline_communicate("send_backward", device, dtype, tensor=input_tensor_grad)
        run_w(sync=False)

    # Flush remaining W passes (these cover the tail bubble); sync grads on the very last one.
    while w_queue:
        run_w(sync=(len(w_queue) == 1))

    return logging_loss


# =============================================================================================
# Interleaved 1F1B (virtual pipeline) — Narayanan et al., Megatron-LM (arXiv:2104.04473)
# =============================================================================================


def _distribute_layers(num_layers, num_stages):
    """Split ``num_layers`` into ``num_stages`` contiguous groups, as evenly as possible."""
    per_stage = [num_layers // num_stages + (1 if i < num_layers % num_stages else 0) for i in range(num_stages)]
    groups, start = [], 0
    for n in per_stage:
        groups.append(list(range(start, start + n)))
        start += n
    return groups


class InterleavedPipelineParallel(nn.Module):
    """Virtual-pipeline model wrapper: each rank owns ``v`` non-contiguous layer chunks.

    With ``p`` ranks and ``v`` virtual stages per rank there are ``G = p*v`` virtual stages laid out
    round-robin: virtual stage ``g`` lives on rank ``g % p``, chunk ``g // p``. So rank ``r`` owns
    virtual stages ``r, r+p, r+2p, …`` (its chunks ``0..v-1``). Activations flow ``0→1→…→G-1`` which,
    projected onto ranks, is the ring ``0→1→…→p-1→0→…``. Shrinking each stage by ``v`` shrinks the
    pipeline bubble by ``v`` (at the cost of ``v`` x more cross-rank hops).
    """

    def __init__(self, model, config, num_virtual_stages=2):
        super().__init__()
        pg = pgm.process_group_manager
        p, rank = pg.pp_world_size, pg.pp_rank
        self.num_virtual_stages = num_virtual_stages
        self.global_num_stages = g_total = p * num_virtual_stages
        groups = _distribute_layers(config.num_hidden_layers, g_total)
        # Global virtual-stage id of each local chunk.
        self.chunk_global_id = [c * p + rank for c in range(num_virtual_stages)]

        self.embeddings = nn.ModuleList()
        self.decoder_chunks = nn.ModuleList()
        self.final_norms = nn.ModuleList()
        self.final_projs = nn.ModuleList()
        for gid in self.chunk_global_id:
            self.embeddings.append(model.embedding if gid == 0 else nn.Identity())
            self.decoder_chunks.append(nn.ModuleDict({str(i): model.decoder_layers[i] for i in groups[gid]}))
            self.final_norms.append(model.final_norm if gid == g_total - 1 else nn.Identity())
            self.final_projs.append(model.final_proj if gid == g_total - 1 else nn.Identity())

    def is_first_stage(self, chunk_id):
        return self.chunk_global_id[chunk_id] == 0

    def is_last_stage(self, chunk_id):
        return self.chunk_global_id[chunk_id] == self.global_num_stages - 1

    def forward(self, input_ids, position_ids, hidden_states, chunk_id):
        x = hidden_states if hidden_states is not None else input_ids
        x = self.embeddings[chunk_id](x)
        for layer in self.decoder_chunks[chunk_id].values():
            x = layer(x, position_ids=position_ids)
        x = self.final_norms[chunk_id](x)
        return self.final_projs[chunk_id](x)

    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        if input_tensor is not None:
            input_tensor.retain_grad()
        if output_tensor_grad is None:
            output_tensor_grad = torch.ones_like(output_tensor, memory_format=torch.preserve_format)
        torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad)
        return input_tensor.grad if input_tensor is not None else None


def _interleaved_forward(model, data_loader, input_tensor, device, chunk_id):
    """Forward one chunk. Only the global-first chunk reads ``input_ids`` and the global-last chunk
    reads ``target_ids`` / computes the loss; every other chunk is a pure activation forward."""
    if model.is_first_stage(chunk_id):
        batch = next(data_loader)
        output_tensor = model.forward(
            input_ids=batch["input_ids"].to(device),
            position_ids=batch["position_ids"].to(device),
            hidden_states=None,
            chunk_id=chunk_id,
        )
        return output_tensor, 0.0
    if model.is_last_stage(chunk_id):
        batch = next(data_loader)
        output_tensor = model.forward(
            input_ids=None,
            position_ids=batch["position_ids"].to(device),
            hidden_states=input_tensor.to(device),
            chunk_id=chunk_id,
        )
        loss = F.cross_entropy(output_tensor.flatten(0, 1), batch["target_ids"].to(device).flatten(), reduction="mean")
        return loss, loss.item() / data_loader.grad_acc_steps
    output_tensor = model.forward(
        input_ids=None, position_ids=None, hidden_states=input_tensor.to(device), chunk_id=chunk_id
    )
    return output_tensor, 0.0


def train_step_pipeline_interleaved(model, data_loader, tensor_shapes, device, dtype):
    """Interleaved 1F1B schedule over ``v`` virtual stages per rank (Megatron-LM)."""
    pg = pgm.process_group_manager
    p, rank = pg.pp_world_size, pg.pp_rank
    v = model.num_virtual_stages
    assert data_loader.grad_acc_steps % p == 0, (
        f"interleaved 1F1B requires grad_acc_steps ({data_loader.grad_acc_steps}) divisible by pp_size ({p})"
    )
    total = data_loader.grad_acc_steps * v
    logging_loss = 0.0
    requires_grad_sync = pg.cp_dp_world_size > 1
    if requires_grad_sync:
        model.require_backward_grad_sync = False

    # Per-chunk FIFOs of stage inputs / outputs / received output-grads.
    input_tensors = [[] for _ in range(v)]
    output_tensors = [[] for _ in range(v)]
    output_grads = [[] for _ in range(v)]

    def fwd_chunk(step):
        return (step % (p * v)) // p

    def bwd_chunk(step):
        return v - 1 - fwd_chunk(step)

    num_warmup = min((p - rank - 1) * 2 + (v - 1) * p, total)
    num_remaining = total - num_warmup

    def forward_step_helper(step):
        nonlocal logging_loss
        cid = fwd_chunk(step)
        if model.is_first_stage(cid):
            input_tensors[cid].append(None)  # global-first chunk reads the data loader, no recv
        input_tensor = input_tensors[cid][-1]
        output_tensor, loss = _interleaved_forward(model, data_loader, input_tensor, device, cid)
        logging_loss += loss
        output_tensors[cid].append(output_tensor)
        return None if model.is_last_stage(cid) else output_tensor  # loss stays local

    def backward_step_helper(step):
        cid = bwd_chunk(step)
        if model.is_last_stage(cid):
            output_grads[cid].append(None)  # global-last chunk: loss grad is implicit (ones)
        input_tensor = input_tensors[cid].pop(0)
        output_tensor = output_tensors[cid].pop(0)
        output_grad = output_grads[cid].pop(0)
        input_tensor_grad = model.backward(input_tensor, output_tensor, output_grad)
        return None if model.is_first_stage(cid) else input_tensor_grad  # no upstream to send to

    # --- Warmup: forwards only. Prefetch chunk-0's first input, then prefetch each next input. ---
    if not model.is_first_stage(fwd_chunk(0)):
        recv_fwd, _ = interleaved_pipeline_communicate(device, dtype, tensor_shapes, recv_forward=True)
        input_tensors[fwd_chunk(0)].append(recv_fwd)

    for step in range(num_warmup):
        output_tensor = forward_step_helper(step)
        next_step = step + 1
        recv_prev = next_step < total and not model.is_first_stage(fwd_chunk(next_step))
        last_warmup = step == num_warmup - 1
        if last_warmup:
            # End of warmup: also kick off the first backward-grad receive (chunk v-1). This must
            # happen even when num_remaining == 0 (warmup consumed every forward, e.g. small m /
            # large p), otherwise the first cooldown backward pops an empty output-grad FIFO.
            recv_next = not model.is_last_stage(v - 1)
            recv_fwd, recv_bwd = interleaved_pipeline_communicate(
                device,
                dtype,
                tensor_shapes,
                send_forward_tensor=output_tensor,
                recv_forward=recv_prev,
                recv_backward=recv_next,
            )
            if recv_next:
                output_grads[v - 1].append(recv_bwd)
        else:
            recv_fwd, _ = interleaved_pipeline_communicate(
                device, dtype, tensor_shapes, send_forward_tensor=output_tensor, recv_forward=recv_prev
            )
        if recv_prev:
            input_tensors[fwd_chunk(next_step)].append(recv_fwd)

    # --- Steady state: 1 forward + 1 backward per step (combined comm in both directions). ---
    for i in range(num_remaining):
        fwd_step = num_warmup + i
        bwd_step = i
        output_tensor = forward_step_helper(fwd_step)
        input_tensor_grad = backward_step_helper(bwd_step)

        next_fwd_step = fwd_step + 1
        recv_prev = next_fwd_step < total and not model.is_first_stage(fwd_chunk(next_fwd_step))
        next_bwd_step = bwd_step + 1
        recv_next = next_bwd_step < total and not model.is_last_stage(bwd_chunk(next_bwd_step))

        recv_fwd, recv_bwd = interleaved_pipeline_communicate(
            device,
            dtype,
            tensor_shapes,
            send_forward_tensor=output_tensor,
            send_backward_tensor=input_tensor_grad,
            recv_forward=recv_prev,
            recv_backward=recv_next,
        )
        if recv_prev:
            input_tensors[fwd_chunk(next_fwd_step)].append(recv_fwd)
        if recv_next:
            output_grads[bwd_chunk(next_bwd_step)].append(recv_bwd)

    # --- Cooldown: remaining backwards. ---
    for i in range(num_remaining, total):
        if requires_grad_sync and i == total - 1:
            model.require_backward_grad_sync = True
        input_tensor_grad = backward_step_helper(i)
        next_bwd_step = i + 1
        recv_next = next_bwd_step < total and not model.is_last_stage(bwd_chunk(next_bwd_step))
        _, recv_bwd = interleaved_pipeline_communicate(
            device, dtype, tensor_shapes, send_backward_tensor=input_tensor_grad, recv_backward=recv_next
        )
        if recv_next:
            output_grads[bwd_chunk(next_bwd_step)].append(recv_bwd)

    return logging_loss


# =============================================================================================
# DualPipe — bidirectional pipeline (DeepSeek-V3, arXiv:2412.19437)
# =============================================================================================


class DualPipeParallel(nn.Module):
    """DualPipe model wrapper: each rank holds the two stages symmetric about the pipeline middle.

    The model is split into ``p`` contiguous stages. Rank ``r`` holds **stage ``r``** (chunk 0, fed by
    the "down" micro-batch stream that enters at rank 0) and **stage ``p-1-r``** (chunk 1, fed by the
    "up" stream that enters at rank ``p-1``). Stage ``s`` therefore lives on both rank ``s`` and rank
    ``p-1-s`` — parameters are replicated across that symmetric pair (DualPipe's ~2x parameter-memory
    cost), and their gradients are summed across the pair (see ``dualpipe_reduce_grads``). Running both
    opposing streams and summing the replicated stages' gradients reproduces the full-batch gradient
    bit-exactly. Interleaving the two streams in time (so one's forwards fill the other's backward
    bubbles, roughly halving the bubble) is DualPipe's performance win; we keep the streams sequential
    here so the engine is deadlock-free on gloo/CPU and leave the overlap as the documented extension.
    """

    def __init__(self, model, config):
        super().__init__()
        pg = pgm.process_group_manager
        p, rank = pg.pp_world_size, pg.pp_rank
        self.num_stages = p
        groups = _distribute_layers(config.num_hidden_layers, p)
        self.chunk_stage = [rank, p - 1 - rank]  # chunk 0 = down-stream stage, chunk 1 = up-stream stage

        self.embeddings = nn.ModuleList()
        self.decoder_chunks = nn.ModuleList()
        self.final_norms = nn.ModuleList()
        self.final_projs = nn.ModuleList()
        for stage in self.chunk_stage:
            self.embeddings.append(model.embedding if stage == 0 else nn.Identity())
            self.decoder_chunks.append(nn.ModuleDict({str(i): model.decoder_layers[i] for i in groups[stage]}))
            self.final_norms.append(model.final_norm if stage == p - 1 else nn.Identity())
            self.final_projs.append(model.final_proj if stage == p - 1 else nn.Identity())

    def is_first_stage(self, chunk_id):
        return self.chunk_stage[chunk_id] == 0

    def is_last_stage(self, chunk_id):
        return self.chunk_stage[chunk_id] == self.num_stages - 1

    def forward(self, input_ids, position_ids, hidden_states, chunk_id):
        x = hidden_states if hidden_states is not None else input_ids
        x = self.embeddings[chunk_id](x)
        for layer in self.decoder_chunks[chunk_id].values():
            x = layer(x, position_ids=position_ids)
        x = self.final_norms[chunk_id](x)
        return self.final_projs[chunk_id](x)

    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        if input_tensor is not None:
            input_tensor.retain_grad()
        if output_tensor_grad is None:
            output_tensor_grad = torch.ones_like(output_tensor, memory_format=torch.preserve_format)
        torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad)
        return input_tensor.grad if input_tensor is not None else None


def _dualpipe_chunk_params(model, chunk_id):
    """All parameters of a DualPipe chunk in a stage-deterministic order (so the two ranks holding a
    replicated stage iterate it identically): embedding, decoder layers, final norm, final proj."""
    params = list(model.embeddings[chunk_id].parameters())
    params += list(model.decoder_chunks[chunk_id].parameters())
    params += list(model.final_norms[chunk_id].parameters())
    params += list(model.final_projs[chunk_id].parameters())
    return params


def dualpipe_reduce_grads(model, device=None, dtype=None):
    """Sum the gradients of each replicated stage across its symmetric rank pair (rank ``r`` ↔ rank
    ``p-1-r``). Stage ``s`` lives on rank ``s`` (one chunk) and rank ``p-1-s`` (the other chunk); after
    the schedule each holds a partial sum (from its own micro-batch stream), so the pair exchanges and
    adds them. The lower rank iterates ``[stage_min, stage_max]`` and the higher rank
    ``[stage_max, stage_min]`` so both walk the same stage at the same step.
    """
    pg = pgm.process_group_manager
    p, rank = pg.pp_world_size, pg.pp_rank
    partner = pg.pp_group_ids[p - 1 - rank]
    if partner == pg.pp_group_ids[rank]:
        return  # odd-length pipeline: the middle rank owns one stage twice, grads already complete
    lower = pg.pp_group_ids[rank] < partner
    ordered_chunks = (0, 1) if lower else (1, 0)  # both ranks visit stage_min first, then stage_max
    for chunk_id in ordered_chunks:
        for prm in _dualpipe_chunk_params(model, chunk_id):
            grad = prm.grad if prm.grad is not None else torch.zeros_like(prm)
            (recv,) = ring_exchange(
                [(grad.contiguous(), partner)], [(partner, grad.shape)], device or grad.device, dtype or grad.dtype
            )
            prm.grad = grad + recv


def _dualpipe_stream_groups():
    """Two **independent** pp communicators (one per opposing micro-batch stream), created once and
    cached on the process-group manager.

    DualPipe runs both streams concurrently, and on a given pipeline edge *both* streams move data in
    the *same* direction at overlapping times (the down-stream's forward activations and the
    up-stream's forward activations both flow ``r → r+1``). NCCL matches point-to-point ops by their
    submission order per peer — **not** by tag — so if the two streams shared one communicator their
    ops could be paired up wrongly (mismatched tensors, or a hang) whenever the two ranks of an edge
    disagree on the interleaving order (which they do, because the per-rank warmup lengths differ).
    Giving each stream its own communicator makes its ordering a single, self-consistent sequence per
    edge — exactly like a standalone 1F1B — so the interleaved schedule stays deadlock-free.

    Built with the same enumeration picotron uses for ``pp_group`` (all world ranks participate, every
    pipeline replica gets its own pair), so this is safe to call collectively from the train step.
    """
    pg = pgm.process_group_manager
    cached = getattr(pg, "_dualpipe_stream_groups", None)
    if cached is not None:
        return cached
    dp_size, _pp_size, cp_size, ep_size, tp_size = pg.grid.shape
    enumeration = [
        pg.grid[d, :, c, e, t].tolist()
        for d in range(dp_size)
        for c in range(cp_size)
        for e in range(ep_size)
        for t in range(tp_size)
    ]
    groups = (
        dist.new_subgroups_by_enumeration(enumeration)[0],
        dist.new_subgroups_by_enumeration(enumeration)[0],
    )
    pg._dualpipe_stream_groups = groups
    _dualpipe_warmup_p2p(groups)
    return groups


def _dualpipe_warmup_p2p(groups):
    """Pre-establish every point-to-point connection each stream will use, **symmetrically**.

    NCCL builds its send/recv channels lazily on the *first* P2P op between a pair, and that handshake
    blocks until both peers post their matching op. DualPipe's real schedule is asymmetric (per-rank
    warmup lengths differ), so the first sends/recvs do not pair up in time and the lazy connection
    setup itself deadlocks at ``p >= 3`` (``batch_isend_irecv`` hangs in ``group_end``). Here every rank
    posts a 1-element ``isend`` *and* ``irecv`` to **both** neighbors in a single batched group per
    stream communicator — fully matched, so the channels are built deadlock-free up front; afterwards
    the real (asymmetric) ``batch_isend_irecv`` calls just enqueue and return. (A ring ``all_reduce``
    warmup is not enough — it builds the collective channels, not the P2P send/recv ones.)
    """
    if dist.get_backend(groups[0]) != dist.Backend.NCCL:
        return
    pg = pgm.process_group_manager
    p, rank, ids = pg.pp_world_size, pg.pp_rank, pg.pp_group_ids
    neighbors = [ids[rank + d] for d in (-1, 1) if 0 <= rank + d < p]
    dev = torch.cuda.current_device()
    for g in groups:
        ops = []
        for nb in neighbors:
            ops.append(dist.P2POp(dist.irecv, torch.empty(1, device=dev), nb, group=g))
            ops.append(dist.P2POp(dist.isend, torch.zeros(1, device=dev), nb, group=g))
        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()
    torch.cuda.synchronize()


def _dualpipe_stream(
    model, batches, chunk_id, warmup, fwd_send, fwd_recv, bwd_send, bwd_recv, shapes, device, dtype, group, loss_box
):
    """One DualPipe stream as a **generator**: a standard 1F1B that ``yield``s its batched P2P ops at
    every communication point instead of waiting on them inline.

    Yielding (rather than blocking) is what lets :func:`_drive_dualpipe` overlap the two streams from a
    single thread: it launches this stream's ``isend``/``irecv`` asynchronously and runs the *other*
    stream's compute while these bytes are on the wire. All P2P uses this stream's private ``group``
    (see :func:`_dualpipe_stream_groups`), so each stream is a self-consistent FIFO per edge — matched
    independently of the other stream, exactly like a standalone 1F1B. Activations flow
    ``fwd_recv → here → fwd_send`` and gradients ``bwd_recv → here → bwd_send``; ``None`` peers mark the
    stream's pipeline ends. Each yield is ``(ops, recv_buf)``; the driver waits ``ops`` then hands the
    (now-filled) ``recv_buf`` back via ``.send``. Accumulates the stream loss into ``loss_box[0]``.
    """
    m = len(batches)
    is_first = fwd_recv is None
    is_last = fwd_send is None
    warmup = min(warmup, m)
    remaining = m - warmup
    input_tensors, output_tensors = [], []
    f_idx = 0

    def fwd(input_tensor):
        nonlocal f_idx
        batch = batches[f_idx]
        f_idx += 1
        output = model.forward(
            input_ids=batch["input_ids"].to(device) if is_first else None,
            position_ids=batch["position_ids"].to(device),
            hidden_states=input_tensor.to(device) if input_tensor is not None else None,
            chunk_id=chunk_id,
        )
        if is_last:
            loss = F.cross_entropy(output.flatten(0, 1), batch["target_ids"].to(device).flatten(), reduction="mean")
            loss_box[0] += loss.item() / m
            return loss
        return output

    def make_ops(send_tensor, send_peer, recv_peer):
        ops, recv_buf = [], None
        if recv_peer is not None:
            recv_buf = torch.empty(shapes, requires_grad=True, device=device, dtype=dtype)
            ops.append(dist.P2POp(dist.irecv, recv_buf, recv_peer, group=group))
        if send_peer is not None and send_tensor is not None:
            ops.append(dist.P2POp(dist.isend, send_tensor.contiguous(), send_peer, group=group))
        return ops, recv_buf

    # Warmup: forwards only.
    for _ in range(warmup):
        if not is_first:
            input_tensor = yield make_ops(None, None, fwd_recv)
        else:
            input_tensor = None
        output_tensor = fwd(input_tensor)
        if not is_last:
            yield make_ops(output_tensor, fwd_send, None)
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    if remaining > 0:
        input_tensor = (yield make_ops(None, None, fwd_recv)) if not is_first else None

    # Steady state: one forward + one backward, combining comm in opposite directions per step.
    for i in range(remaining):
        last_iter = i == remaining - 1
        output_tensor = fwd(input_tensor)
        output_tensor_grad = (yield make_ops(output_tensor, fwd_send, bwd_recv)) if not is_last else None
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)
        b_input, b_output = input_tensors.pop(0), output_tensors.pop(0)
        input_tensor_grad = model.backward(b_input, b_output, output_tensor_grad)
        if last_iter:
            input_tensor = None
            if not is_first:
                yield make_ops(input_tensor_grad, bwd_send, None)
        elif is_first:
            input_tensor = yield make_ops(None, None, fwd_recv)
        else:
            input_tensor = yield make_ops(input_tensor_grad, bwd_send, fwd_recv)

    # Cooldown: remaining backwards.
    for _ in range(warmup):
        b_input, b_output = input_tensors.pop(0), output_tensors.pop(0)
        output_tensor_grad = (yield make_ops(None, None, bwd_recv)) if not is_last else None
        input_tensor_grad = model.backward(b_input, b_output, output_tensor_grad)
        if not is_first:
            yield make_ops(input_tensor_grad, bwd_send, None)


def _drive_dualpipe_async(streams):
    """Pump the two stream generators from **one thread**, each on its **own CUDA stream**, so compute
    and communication of the two opposing directions overlap on the GPU (the NCCL path).

    The critical detail is the per-stream ``torch.cuda.Stream``. If both directions' P2P were enqueued
    on the *same* (default) CUDA stream, the GPU would run them in enqueue order: on rank ``r`` the
    down-stream ``irecv`` from ``r-1`` would sit ahead of the up-stream ops and block the whole stream
    until ``r-1`` sends — but ``r-1`` is symmetrically blocked behind *its* down-recv, whose release
    needs ``r``'s up-send that is stuck behind the very op we wait on. That circular wait through
    CUDA-stream ordering deadlocks at ``p >= 3`` (it happens to be acyclic, hence fine, at ``p = 2``).
    Separate CUDA streams let the two directions' NCCL kernels make progress independently, breaking the
    cycle — which is exactly DeepSeek's "two interleaved streams" at the CUDA-stream level.

    A stream is resumed only once *its own* outstanding ops complete — polled with ``Work.is_completed``
    (queries the CUDA events without blocking) so a stream that finishes its (shorter) schedule never
    forces the other to wait. Single-threaded ⇒ no concurrent NCCL P2P submission from multiple host
    threads on one device (which corrupts/hangs); the concurrency lives on the CUDA streams instead.
    (gloo has no async P2P progress, so gloo uses the threaded driver below.)
    """
    n = len(streams)
    alive = [True] * n
    reqs = [None] * n  # outstanding async P2P per stream (None = nothing pending)
    bufs = [None] * n  # recv buffer to feed back into each stream on resume
    cuda_streams = [torch.cuda.Stream() for _ in range(n)]

    def advance(i, send_val, prime=False):
        # Run this stream's compute + launch its P2P on its dedicated CUDA stream, so the recv buffer,
        # the forward/backward kernels reading it, and the next send are all ordered on one stream while
        # staying independent of the other direction's stream.
        with torch.cuda.stream(cuda_streams[i]):
            ops, recv_buf = next(streams[i]) if prime else streams[i].send(send_val)
            bufs[i] = recv_buf
            reqs[i] = dist.batch_isend_irecv(ops) if ops else None

    for i in range(n):
        try:
            advance(i, None, prime=True)
        except StopIteration:
            alive[i] = False
    while any(alive):
        for i in range(n):
            if not alive[i]:
                continue
            if reqs[i] is not None:
                if not all(req.is_completed() for req in reqs[i]):
                    continue  # this stream's comm still in flight — let the other stream run
                for req in reqs[i]:
                    req.wait()  # finalize (and surface any error)
                reqs[i] = None
            try:
                advance(i, bufs[i])  # resume: compute overlaps the other's in-flight comm
            except StopIteration:
                alive[i] = False
    _device_synchronize()


def _run_stream_blocking(stream):
    """Drive one stream generator with blocking waits (one comm point at a time)."""
    try:
        buf = None
        while True:
            ops, recv_buf = stream.send(buf)
            if ops:
                for req in dist.batch_isend_irecv(ops):
                    req.wait()
                _device_synchronize()
            buf = recv_buf
    except StopIteration:
        pass


def _drive_dualpipe_threads(streams):
    """Pump the two streams on **two threads** (the gloo path).

    gloo makes no async P2P progress outside ``wait()``, so the single-thread async driver would hang;
    instead each stream blocks on its own communicator in its own thread. gloo releases the GIL inside
    the blocking wait, so the two threads' waits progress concurrently and one stream's compute fills
    the other's 1F1B bubble. (NCCL serializes/corrupts concurrent multi-thread P2P on one device, so
    NCCL uses the single-thread async driver instead.)
    """
    import threading

    errors = []

    def worker(stream):
        try:
            _run_stream_blocking(stream)
        except BaseException as exc:  # surface a stream's failure instead of hanging the join
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(s,), daemon=True) for s in streams]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]


def _drive_dualpipe(streams):
    """Overlap the two opposing streams, picking the driver that is deadlock-free on this backend:
    single-thread async for NCCL (CUDA-stream progress), two threads for gloo (blocking-wait progress).
    """
    backend = dist.get_backend(pgm.process_group_manager.pp_group)
    if backend == dist.Backend.NCCL:
        _drive_dualpipe_async(streams)
    else:
        _drive_dualpipe_threads(streams)


def train_step_pipeline_dualpipe(model, data_loader, tensor_shapes, device, dtype):
    """DualPipe schedule: two opposing micro-batch streams interleaved in time (DeepSeek-V3).

    The global batch is split in two: micro-batches ``[0, m)`` feed the "down" stream (chunk 0,
    activations head→tail) and ``[m, 2m)`` feed the "up" stream (chunk 1, activations tail→head). The
    two streams run as cooperating generators pumped by :func:`_drive_dualpipe` from a single thread,
    which overlaps one stream's compute with the other's in-flight communication so the combined bubble
    drops from 1F1B's ``(p-1)/m`` toward ``(p-1)/(2m)``. Each stream owns a private pp communicator
    (:func:`_dualpipe_stream_groups`) so the two never contend for P2P ordering — deadlock-free on both
    gloo and NCCL.

    Each rank holds the two stages symmetric about the middle (stage ``r`` and stage ``p-1-r``), so a
    stage is replicated across the pair ``(r, p-1-r)``; after the step the caller must call
    :func:`dualpipe_reduce_grads` to sum each replicated stage's partial gradients across that pair.
    Summing the two streams' gradients reproduces the full ``2m``-micro-batch gradient bit-exactly.
    """
    pg = pgm.process_group_manager
    p, rank = pg.pp_world_size, pg.pp_rank
    total_microbatches = data_loader.grad_acc_steps
    assert total_microbatches % 2 == 0, "DualPipe needs an even grad_acc_steps (two micro-batch streams)"
    m = total_microbatches // 2
    pp_ids = pg.pp_group_ids
    up_rank = pp_ids[rank + 1] if rank + 1 < p else None  # toward the pipeline tail (higher stage)
    down_rank = pp_ids[rank - 1] if rank - 1 >= 0 else None  # toward the pipeline head (lower stage)

    batches = [next(data_loader) for _ in range(total_microbatches)]
    group_down, group_up = _dualpipe_stream_groups()
    loss_down, loss_up = [0.0], [0.0]

    # Down stream (chunk 0 = stage r): forward head→tail, so warmup = #downstream stages = p-1-r.
    stream_down = _dualpipe_stream(
        model, batches[:m], chunk_id=0, warmup=p - 1 - rank,
        fwd_send=up_rank, fwd_recv=down_rank, bwd_send=down_rank, bwd_recv=up_rank,
        shapes=tensor_shapes, device=device, dtype=dtype, group=group_down, loss_box=loss_down,
    )
    # Up stream (chunk 1 = stage p-1-r): forward tail→head (mirror), so warmup = #downstream stages = r.
    stream_up = _dualpipe_stream(
        model, batches[m:], chunk_id=1, warmup=rank,
        fwd_send=down_rank, fwd_recv=up_rank, bwd_send=up_rank, bwd_recv=down_rank,
        shapes=tensor_shapes, device=device, dtype=dtype, group=group_up, loss_box=loss_up,
    )
    _drive_dualpipe([stream_down, stream_up])
    return loss_down[0] + loss_up[0]
