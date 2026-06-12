import os

import torch
import torch.distributed as dist

import picotron.process_group_manager as pgm

STEP, VERBOSE = 0, os.environ.get("VERBOSE", "0") == "1"


def _device_synchronize():
    # No-op on CPU/gloo so the pipeline schedules stay portable (tests run on CPU).
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def pipeline_communicate(operation, device, dtype, tensor=None, shapes=None):
    global STEP
    global VERBOSE
    if operation == "recv_forward":
        if pgm.process_group_manager.pp_is_first_stage:
            return None
        tensor = torch.empty(shapes, requires_grad=True, device=device, dtype=dtype)
        src = pgm.process_group_manager.pp_prev_rank
    elif operation == "send_forward":
        if pgm.process_group_manager.pp_is_last_stage:
            return
        dest = pgm.process_group_manager.pp_next_rank
    elif operation == "recv_backward":
        if pgm.process_group_manager.pp_is_last_stage:
            return None
        tensor = torch.empty(shapes, requires_grad=True, device=device, dtype=dtype)
        src = pgm.process_group_manager.pp_next_rank
    elif operation == "send_backward":
        if pgm.process_group_manager.pp_is_first_stage:
            return
        dest = pgm.process_group_manager.pp_prev_rank
    is_send = operation.startswith("send")
    peer_rank = dest if is_send else src
    op = dist.P2POp(dist.isend if is_send else dist.irecv, tensor, peer_rank)
    if VERBOSE:
        print(
            f"{operation} | {'sending' if is_send else 'receiving'} {operation.split('_')[1]} {pgm.process_group_manager.pp_rank} {'→' if is_send else '←'} {peer_rank} | STEP:{STEP} | RANK:{pgm.process_group_manager.pp_rank}",
            flush=True,
        )
    [req.wait() for req in dist.batch_isend_irecv([op])]
    _device_synchronize()
    if VERBOSE:
        STEP += 1
    return tensor if not is_send else None


def interleaved_pipeline_communicate(
    device,
    dtype,
    shapes,
    send_forward_tensor=None,
    send_backward_tensor=None,
    recv_forward=False,
    recv_backward=False,
):
    """Combined ring send/recv for interleaved & DualPipe schedules (deadlock-free).

    Batches up to four P2P ops in a single :func:`dist.batch_isend_irecv` so every rank posts its
    sends and matching receives together — the only safe pattern for synchronous point-to-point on a
    ring. Forward activations move toward ``ring_next``; backward grads move toward ``ring_prev``.
    Returns ``(recv_forward_tensor, recv_backward_tensor)`` (``None`` for directions not requested).
    """
    pg = pgm.process_group_manager
    ring_next = pg.pp_group_ids[(pg.pp_rank + 1) % pg.pp_world_size]
    ring_prev = pg.pp_group_ids[(pg.pp_rank - 1) % pg.pp_world_size]

    # Coalesce every direction into a SINGLE batch_isend_irecv (one ncclGroupStart/End). This is
    # deadlock-free on a ring and, crucially, avoids NCCL's eager-mode serialization: standalone
    # dist.isend/irecv are each treated as an independent size-P collective and serialized, which
    # stalls the pipeline at large pp_size. Batching fuses them into one group op (same primitive
    # afab/1f1b use). Forward activations move toward ring_next; backward grads toward ring_prev.
    ops = []
    recv_forward_tensor = recv_backward_tensor = None
    if send_forward_tensor is not None:
        ops.append(dist.P2POp(dist.isend, send_forward_tensor, ring_next))
    if send_backward_tensor is not None:
        ops.append(dist.P2POp(dist.isend, send_backward_tensor, ring_prev))
    if recv_forward:
        recv_forward_tensor = torch.empty(shapes, requires_grad=True, device=device, dtype=dtype)
        ops.append(dist.P2POp(dist.irecv, recv_forward_tensor, ring_prev))
    if recv_backward:
        recv_backward_tensor = torch.empty(shapes, requires_grad=True, device=device, dtype=dtype)
        ops.append(dist.P2POp(dist.irecv, recv_backward_tensor, ring_next))
    if ops:
        for req in dist.batch_isend_irecv(ops):
            req.wait()
        _device_synchronize()
    return recv_forward_tensor, recv_backward_tensor


def ring_exchange(send_ops, recv_specs, device, dtype):
    """Generic deadlock-free batched P2P used by DualPipe's two opposing micro-batch streams.

    ``send_ops`` is a list of ``(tensor, peer_rank)`` and ``recv_specs`` a list of ``(peer_rank,
    shape)``; returns the received tensors in ``recv_specs`` order. As in
    :func:`interleaved_pipeline_communicate`, every op is fused into one batch_isend_irecv group so
    merging both streams' communication into one call cannot deadlock and is not serialized by
    NCCL's eager mode at large pp_size.
    """
    ops = [dist.P2POp(dist.isend, tensor, peer) for tensor, peer in send_ops]
    recv_tensors = []
    for peer, shape in recv_specs:
        t = torch.empty(shape, requires_grad=True, device=device, dtype=dtype)
        recv_tensors.append(t)
        ops.append(dist.P2POp(dist.irecv, t, peer))
    if ops:
        for req in dist.batch_isend_irecv(ops):
            req.wait()
        _device_synchronize()
    return recv_tensors


def bidirectional_pipeline_communicate(operation, send_tensor, recv_shapes, device, dtype):
    global STEP
    global VERBOSE
    is_fwd = operation == "send_fwd_recv_bwd"
    if (is_fwd and pgm.process_group_manager.pp_is_last_stage) or (
        not is_fwd and pgm.process_group_manager.pp_is_first_stage
    ):
        return None
    peer_rank = pgm.process_group_manager.pp_next_rank if is_fwd else pgm.process_group_manager.pp_prev_rank
    recv_tensor = torch.empty(recv_shapes, requires_grad=True, device=device, dtype=dtype)
    reqs = dist.batch_isend_irecv(
        [dist.P2POp(dist.isend, send_tensor, peer_rank), dist.P2POp(dist.irecv, recv_tensor, peer_rank)]
    )
    if VERBOSE:
        print(
            f"{operation} | sending {'next' if is_fwd else 'prev'} {pgm.process_group_manager.pp_rank} -> {peer_rank} | "
            f"receiving {'next' if is_fwd else 'prev'} {peer_rank} -> {pgm.process_group_manager.pp_rank} | "
            f"STEP {STEP=} | RANK:{pgm.process_group_manager.pp_rank}",
            flush=True,
        )
    [req.wait() for req in reqs]
    _device_synchronize()
    if VERBOSE:
        STEP += 1
    return recv_tensor
