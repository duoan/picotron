"""ZeRO / FSDP-style data parallelism (memory sharding of the model state).

Plain data parallelism (`DataParallelNaive` / `DataParallelBucket`) replicates *everything* on every
data-parallel rank: parameters, gradients, and optimizer states (Adam m/v, fp32 master). ZeRO removes
that redundancy by partitioning the model state across the `N` replica ranks, in three cumulative stages
(Rajbhandari et al., 2019, https://arxiv.org/abs/1910.02054):

    stage 1  shard optimizer states         (params + grads replicated)
    stage 2  + shard gradients              (params replicated)
    stage 3  + shard parameters  (== FSDP)  (nothing fully replicated at rest)

Partitioning here is at **whole-parameter granularity**: each parameter is assigned to one owner rank
(greedily balanced by element count). This is the simplest correct partitioning; production ZeRO/FSDP
flatten and split parameters into equal byte-shards for perfect balance. See the trade-off note below.

Gradient reduction uses ``all_reduce`` (then drops the non-owner copies) rather than the
``reduce_scatter`` a production implementation would use. ``all_reduce`` is numerically identical to the
plain-DP reference, which lets the test assert *bit-exact* equality; ``reduce_scatter`` additionally cuts
communication volume and the gradient-reduction peak, and is the only thing separating this from a
comm-optimal implementation. The headline ZeRO property — model-state memory shrinks ~linearly with the
DP degree — is fully realized either way.

    sharding group:   process_group_manager.cp_dp_group  (DP * CP replicas, same as plain-DP grad sync)
"""

import contextlib

import torch
import torch.distributed as dist
from torch import nn

import picotron.process_group_manager as pgm


def _group():
    return pgm.process_group_manager.cp_dp_group


def _world_size():
    return pgm.process_group_manager.cp_dp_world_size


def _global_rank(group_rank: int) -> int:
    """Map a rank index inside the replica group to its global (world) rank for src/dst args."""
    return pgm.process_group_manager.cp_dp_group_ids[group_rank]


def assign_param_owners(params, world_size: int) -> dict:
    """Greedily assign each parameter to the replica rank with the least load (balanced by numel).

    Deterministic given the parameter order (identical on every rank), so all ranks agree on ownership.
    """
    loads = [0] * world_size
    owners = {}
    # Largest-first greedy gives a reasonable balance; ties break on the stable parameter order.
    for _, p in sorted(enumerate(params), key=lambda ip: (-ip[1].numel(), ip[0])):
        r = min(range(world_size), key=lambda i: loads[i])
        owners[p] = r
        loads[r] += p.numel()
    return owners


class DataParallelZero(nn.Module):
    """ZeRO stage 1/2/3 wrapper. Pair with :class:`ShardedOptimizer`, which steps only this rank's
    owned parameters.

    stage 1/2 keep full parameters on every rank; stage 3 keeps only this rank's owned parameters and
    all-gathers each parameter just-in-time around the forward/backward of the module that owns it
    (FSDP-style), freeing it again right after.
    """

    def __init__(self, module: nn.Module, stage: int = 1):
        super().__init__()
        assert stage in (1, 2, 3), "ZeRO stage must be 1, 2, or 3"
        self.module = module
        self.stage = stage
        self.group = _group()
        self.world_size = _world_size()
        self.rank = dist.get_rank(group=self.group)
        self.require_backward_grad_sync = True

        params = [p for p in module.parameters() if p.requires_grad]
        self.owners = assign_param_owners(params, self.world_size)  # param -> group rank
        self.owned_params = [p for p in params if self.owners[p] == self.rank]

        if stage in (1, 2):
            self._register_grad_hooks()
        else:
            self._setup_param_sharding()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def owns(self, p) -> bool:
        return self.owners[p] == self.rank

    def owner_global_rank(self, p) -> int:
        return _global_rank(self.owners[p])

    # --- stage 1 / 2: gradient reduction via post-accumulate hooks --------------------------------

    def _register_grad_hooks(self):
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._grad_hook)

    def _grad_hook(self, p: torch.nn.Parameter):
        if not self.require_backward_grad_sync:
            return  # gradient accumulation: keep accumulating locally, reduce on the final micro-batch
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=self.group)
        p.grad /= self.world_size
        # stage 2 additionally drops the gradient on non-owners: only the owner needs it for its step.
        if self.stage == 2 and not self.owns(p):
            p.grad = None

    # --- stage 3 (FSDP): parameter sharding + just-in-time all-gather ------------------------------

    def _setup_param_sharding(self):
        # Each parameter keeps its identity and full shape/strides; on non-owners we drop the *storage*
        # (resize to 0 bytes) so only the owner physically holds the data. Storage resize keeps the
        # tensor object (and thus the autograd-saved reference) valid across free/gather, and unlike
        # ``p.data = new_tensor`` it does not bump the autograd version counter. The discarded non-owner
        # values are identical to the owner's anyway (same seed), so no initial gather is needed.
        self._meta = {}  # p -> owner global rank
        for p in self.module.parameters():
            if not p.requires_grad:
                continue
            self._meta[p] = self.owner_global_rank(p)
            if not self.owns(p):
                p.data.untyped_storage().resize_(0)

        # An FSDP "unit" is any module with directly-owned parameters; its direct params are gathered
        # together. Leaf param-modules (Linear/Embedding/Norm/...) partition all parameters exactly.
        #   forward:  gather before the module runs, free right after.
        #   backward: gather before the module's backward runs (its weight is needed for grad_input);
        #             the grad reduction + free happens per-parameter in a post-accumulate hook, which
        #             fires exactly when ``p.grad`` is ready (a module full_backward_hook can fire
        #             *before* the parameter's AccumulateGrad node, reducing a stale gradient).
        for m in self.module.modules():
            if any(q.requires_grad for q in m.parameters(recurse=False)):
                m.register_forward_pre_hook(lambda mod, args: self._gather(mod))
                m.register_forward_hook(lambda mod, args, out: self._free(mod))
                m.register_full_backward_pre_hook(lambda mod, gout: self._gather(mod))
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._reduce_and_free)

    @torch.no_grad()
    def _restore_storage(self, p: torch.nn.Parameter):
        p.data.untyped_storage().resize_(p.numel() * p.element_size())

    @torch.no_grad()
    def _gather(self, module: nn.Module):
        for p in module.parameters(recurse=False):
            if p not in self._meta:
                continue
            src = self._meta[p]
            if dist.get_rank() != src:
                self._restore_storage(p)  # re-materialize the (full-shaped) tensor in place
            dist.broadcast(p.data, src=src, group=self.group)

    @torch.no_grad()
    def _free(self, module: nn.Module):
        for p in module.parameters(recurse=False):
            if p in self._meta and not self.owns(p):
                p.data.untyped_storage().resize_(0)

    @torch.no_grad()
    def _reduce_and_free(self, p: torch.nn.Parameter):
        if self.require_backward_grad_sync and p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=self.group)
            p.grad /= self.world_size
            if not self.owns(p):
                p.grad = None
        if not self.owns(p):
            p.data.untyped_storage().resize_(0)  # free this rank's copy now that its grad is reduced

    # --- utilities --------------------------------------------------------------------------------

    @contextlib.contextmanager
    def no_sync(self):
        """Disable gradient synchronization (for gradient accumulation across micro-batches)."""
        self.require_backward_grad_sync = False
        try:
            yield
        finally:
            self.require_backward_grad_sync = True

    @torch.no_grad()
    def full_parameters(self):
        """Return every parameter as a full (un-sharded) tensor on this rank, in ``module.parameters()``
        order. For stage 1/2 this is just the local copy; for stage 3 it all-gathers each shard."""
        out = []
        for p in self.module.parameters():
            if self.stage == 3 and p in self._meta:
                src = self._meta[p]
                buf = p.data.clone() if dist.get_rank() == src else torch.empty_like(p)
                dist.broadcast(buf, src=src, group=self.group)
                out.append(buf)
            else:
                out.append(p.data.clone())
        return out


class ShardedOptimizer:
    """Wraps a base optimizer so it only owns this rank's shard of the parameters (ZeRO optimizer-state
    sharding). After each step, stage 1/2 broadcast the freshly-updated parameters from their owners so
    every rank holds the full, updated model; stage 3 leaves them sharded (the module re-gathers them on
    the next forward).
    """

    def __init__(self, dp_zero: DataParallelZero, optimizer_cls=torch.optim.AdamW, **opt_kwargs):
        self.dp = dp_zero
        self.stage = dp_zero.stage
        self.optimizer = optimizer_cls(dp_zero.owned_params, **opt_kwargs)

    def zero_grad(self, set_to_none: bool = True):
        # Clear *all* parameters' grads, not just the owned ones: non-owned params still receive a grad
        # from autograd (stage 1) that must not accumulate across steps.
        for p in self.dp.module.parameters():
            if p.grad is not None:
                p.grad = None if set_to_none else p.grad.zero_()

    @torch.no_grad()
    def step(self, *args, **kwargs):
        loss = self.optimizer.step(*args, **kwargs)
        if self.stage in (1, 2):
            self._broadcast_params()
        return loss

    @torch.no_grad()
    def _broadcast_params(self):
        # Reconstruct the full, updated model on every rank: each parameter is broadcast from its owner.
        for p in self.dp.module.parameters():
            if p.requires_grad:
                dist.broadcast(p.data, src=self.dp.owner_global_rank(p), group=self.dp.group)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
