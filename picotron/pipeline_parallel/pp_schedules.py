"""Advanced pipeline-parallel schedules built on top of the picotron 1F1B engine.

Stock picotron ships two schedules in ``pipeline_parallel.py``:

- **AFAB** (all-forward-all-backward) — simplest, largest activation memory and bubble.
- **1F1B** (one-forward-one-backward) — steady-state interleave, bubble ``(p-1)/m``.

This module adds two schedules from the last two years of pipeline-parallelism research, each
trading a different resource to shrink the ``(p-1)/m`` pipeline bubble of 1F1B:

- **Zero Bubble (ZB-H1)** — Qi et al., ICLR 2024 (https://arxiv.org/abs/2401.10241). Splits the
  backward into B (input-activation grad, on the critical path) and W (weight grad, reschedulable),
  then fills bubbles with the comm-free W work. ``train_step_pipeline_zb``.
- **Interleaved 1F1B / virtual pipeline** — Narayanan et al. (Megatron-LM,
  https://arxiv.org/abs/2104.04473). Each rank owns ``v`` non-contiguous chunks, shrinking the
  bubble by ``v`` at the cost of ``v`` x more pipeline flushes of comm.
  ``InterleavedPipelineParallel`` + ``train_step_pipeline_interleaved``.

Both are validated to produce **bit-exact gradients** vs the AFAB/1F1B baseline in
``tests/test_pipeline_parallel.py``.
"""

import torch.nn as nn
import torch.nn.functional as F

import picotron.process_group_manager as pgm
from picotron.pipeline_parallel import pipeline_parallel
from picotron.pipeline_parallel.pipeline_parallel import distribute_layers, pp_autograd_backward
from picotron.pipeline_parallel.pp_communications import (
    bidirectional_pipeline_communicate,
    interleaved_pipeline_communicate,
    pipeline_communicate,
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


class InterleavedPipelineParallel(nn.Module):
    """Virtual-pipeline model wrapper: each rank owns ``v`` non-contiguous layer chunks.

    With ``p`` ranks and ``v`` virtual stages per rank there are ``G = p*v`` virtual stages laid out
    round-robin: virtual stage ``g`` lives on rank ``g % p``, chunk ``g // p``. So rank ``r`` owns
    virtual stages ``r, r+p, r+2p, …`` (its chunks ``0..v-1``). Activations flow ``0→1→…→G-1`` which,
    projected onto ranks, is the ring ``0→1→…→p-1→0→…``. Shrinking each stage by ``v`` shrinks the
    pipeline bubble by ``v`` (at the cost of ``v`` x more cross-rank hops).

    Each local chunk materializes only the modules its global stage owns: the embedding on global
    stage 0, the final norm/proj on the last global stage, decoder layers everywhere.
    """

    def __init__(self, model, config, num_virtual_stages=2):
        super().__init__()
        pg = pgm.process_group_manager
        p, rank = pg.pp_world_size, pg.pp_rank
        self.num_virtual_stages = num_virtual_stages
        self.num_global_stages = p * num_virtual_stages
        self.chunk_global_id = [c * p + rank for c in range(num_virtual_stages)]  # global stage of each chunk
        groups = distribute_layers(config.num_hidden_layers, self.num_global_stages)

        self.embeddings = nn.ModuleList()
        self.decoder_chunks = nn.ModuleList()
        self.final_norms = nn.ModuleList()
        self.final_projs = nn.ModuleList()
        for gid in self.chunk_global_id:
            self.embeddings.append(model.embedding if gid == 0 else nn.Identity())
            self.decoder_chunks.append(nn.ModuleDict({str(i): model.decoder_layers[i] for i in groups[gid]}))
            self.final_norms.append(model.final_norm if gid == self.num_global_stages - 1 else nn.Identity())
            self.final_projs.append(model.final_proj if gid == self.num_global_stages - 1 else nn.Identity())

    def is_first_stage(self, chunk_id):
        return self.chunk_global_id[chunk_id] == 0

    def is_last_stage(self, chunk_id):
        return self.chunk_global_id[chunk_id] == self.num_global_stages - 1

    def forward(self, input_ids, position_ids, hidden_states, chunk_id):
        x = hidden_states if hidden_states is not None else input_ids
        x = self.embeddings[chunk_id](x)
        for layer in self.decoder_chunks[chunk_id].values():
            x = layer(x, position_ids=position_ids)
        x = self.final_norms[chunk_id](x)
        return self.final_projs[chunk_id](x)

    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        return pp_autograd_backward(input_tensor, output_tensor, output_tensor_grad)


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
