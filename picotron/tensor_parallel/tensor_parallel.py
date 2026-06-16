import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import picotron.process_group_manager as pgm
from picotron.sequence_parallel.overlap import (
    overlapped_column_parallel_linear_sp,
    overlapped_row_parallel_linear_sp,
)
from picotron.sequence_parallel.sequence_parallel import (
    ReduceScatterToSequenceParallelRegion,
    column_parallel_linear_sp,
)
from picotron.tensor_parallel.tp_communications import (
    GatherFromModelParallelRegion,
    ReduceFromModelParallelRegion,
    linear_with_all_reduce,
    linear_with_async_all_reduce,
)


def apply_tensor_parallel(
    model,
    async_tp: bool = False,
    sequence_parallel: bool = False,
    vocab_parallel_ce: bool = False,
    overlap_comm: bool = False,
):
    """Replace the dense Linear/Embedding layers of `model` with their tensor-parallel versions.

    Args:
        async_tp: overlap the column-parallel input-gradient all-reduce with the weight-gradient GEMM
            (needs ``CUDA_DEVICE_MAX_CONNECTIONS=1``). Ignored when ``sequence_parallel`` is set.
        sequence_parallel: shard the norm/residual regions along the sequence dimension (Megatron SP).
        vocab_parallel_ce: keep the output logits vocab-sharded (no ``gather_output``) so the loss can
            use ``vocab_parallel_cross_entropy``. When False, the final projection gathers the full
            logits (the original behavior).

    The parallel-transformer-block architecture is selected via ``config.parallel_block`` (read by the
    DecoderLayer). Under sequence parallelism it fuses each layer down to a single shared all-gather +
    single combined reduce-scatter (MegaScale Fig 3b): the per-layer projections run in "raw" mode and
    the DecoderLayer owns the collectives.
    """
    # async overlap targets the all-reduce backward path, which sequence parallelism removes.
    async_tp = async_tp and not sequence_parallel
    # the chunked comm/compute overlap (MegaScale Fig 3c) pipelines the SP all-gather / reduce-scatter,
    # so it only applies when sequence parallelism is on.
    overlap_comm = overlap_comm and sequence_parallel

    def _replace_module(_module, _linear_proj_name, _style, args=None):
        if args is None:
            args = {}
        assert _style in ["column", "row", "vocab"]
        linear_layer = getattr(_module, _linear_proj_name)

        if _style == "column":
            new_linear_layer = ColumnParallelLinear(
                in_features=linear_layer.in_features,
                out_features=linear_layer.out_features,
                bias=linear_layer.bias is not None,
                gather_output=args.get("gather_output", False),
                async_all_reduce=async_tp,
                sequence_parallel=sequence_parallel,
                overlap_comm=overlap_comm,
                ptb=args.get("ptb", False),
            )
        elif _style == "row":
            new_linear_layer = RowParallelLinear(
                in_features=linear_layer.in_features,
                out_features=linear_layer.out_features,
                bias=linear_layer.bias is not None,
                sequence_parallel=sequence_parallel,
                overlap_comm=overlap_comm,
                ptb=args.get("ptb", False),
            )
        else:
            new_linear_layer = VocabParallelEmbedding(
                num_embeddings=linear_layer.num_embeddings,
                embedding_dim=linear_layer.embedding_dim,
                sequence_parallel=sequence_parallel,
            )
        setattr(_module, _linear_proj_name, new_linear_layer)

    module_linear_name_stype_mapping_list = [
        ("attention", "q_proj", "column"),
        ("attention", "k_proj", "column"),
        ("attention", "v_proj", "column"),
        ("attention", "out_proj", "row"),
        ("mlp", "up_proj", "column"),
        ("mlp", "gate_proj", "column"),
        ("mlp", "down_proj", "row"),
    ]

    for layer in model.decoder_layers:
        # PTB collective fusion needs the per-layer projections in raw mode and the DecoderLayer to own
        # the shared gather/scatter; it is wired only for the dense-MLP sequence-parallel path (MoE
        # routes its FFN through expert parallelism). The architecture itself comes from config.
        layer_ptb = getattr(layer, "parallel_block", False) and sequence_parallel and hasattr(layer.mlp, "down_proj")
        layer.parallel_block_fused = layer_ptb
        for module_name, linear_proj_name, style in module_linear_name_stype_mapping_list:
            module = getattr(layer, module_name)
            # MoE layers distribute their FFN via expert parallelism, not tensor parallelism,
            # so the plain up/gate/down projections only exist on dense MLP layers.
            if not hasattr(module, linear_proj_name):
                continue
            _replace_module(module, linear_proj_name, style, args={"ptb": layer_ptb})

    _replace_module(model, "embedding", "vocab")
    # With vocab-parallel cross-entropy the logits stay vocab-sharded (the loss handles the reduction);
    # otherwise gather the full logits so the standard dense cross-entropy works unchanged.
    _replace_module(model, "final_proj", "column", args={"gather_output": not vocab_parallel_ce})

    return model


class ColumnParallelLinear(torch.nn.Module):
    """Column Parallel Linear layer
    Y = XW + b, where weight matrix W is parallelized along its second dimension. W = [W_1, ..., W_p]
    This module returns the results of Y_i = XW_i + b_i in the forward method, Y_i is parallelized in the second dimension.
    Arguments:
        in_features: first dimension of weight matrix W.
        out_features: second dimension of weight matrix W.
        bias: If true, add bias
        init_method: method to initialize weights
        gather_output: If true, gather the output from all the partitions. This is used for the last linear layer
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        gather_output: bool = False,
        async_all_reduce: bool = False,
        sequence_parallel: bool = False,
        overlap_comm: bool = False,
        ptb: bool = False,
    ) -> None:
        super().__init__()

        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.tp_rank = pgm.process_group_manager.tp_rank

        self.in_features = in_features
        self.out_features = out_features
        assert out_features % self.tp_world_size == 0, (
            "Hidden dimension must be divisible by the tensor parallel world size"
        )
        self.output_size_per_partition = out_features // self.tp_world_size
        self.gather_output = gather_output
        self.sequence_parallel = sequence_parallel
        # async overlap and sequence parallelism are alternative `f`-operator strategies.
        self.async_all_reduce = async_all_reduce and not sequence_parallel
        # chunked all-gather/GEMM pipeline (MegaScale Fig 3c); only meaningful under sequence parallelism.
        self.overlap_comm = overlap_comm and sequence_parallel
        # parallel-transformer-block "raw" mode: the DecoderLayer all-gathers the shared input once, so
        # this projection is a plain local matmul (no per-linear `f` collective). See model.py PTB path.
        self.ptb = ptb
        # Allocate space for the weight and bias
        # Note: torch.nn.functional.linear performs XW^T + b so we exchange the order of dimensions
        self.weight = nn.Parameter(torch.Tensor(self.output_size_per_partition, self.in_features))  # W_i
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_size_per_partition))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weight tensor with the default initialization method used for nn.Linear in PyTorch
        master_weight = torch.empty(
            self.out_features, self.in_features, dtype=self.weight.dtype, device=self.weight.device, requires_grad=False
        )

        # Calculate bound based on master weight's input dimension
        k = 1 / master_weight.size(1)
        bound = math.sqrt(k)
        torch.nn.init.uniform_(master_weight, -bound, bound)

        # Split the model into size of self.output_size_per_partition
        weight_list = torch.split(master_weight, self.output_size_per_partition, dim=0)
        self.weight.data = weight_list[self.tp_rank].contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ptb:
            # Input was already all-gathered once by the parallel transformer block; just matmul.
            output = F.linear(x, self.weight, self.bias)
        elif self.sequence_parallel and self.overlap_comm:
            # Chunked, pipelined all-gather + matmul (MegaScale Fig 3c): the all-gather is rotated in
            # shard-by-shard and each shard's GEMM overlaps the next rotation.
            output = overlapped_column_parallel_linear_sp(x, self.weight, self.bias)
        elif self.sequence_parallel:
            # Fused all-gather + matmul (the seq-parallel `f`). Only the sequence-sharded input is
            # checkpointed; the gathered tensor is recomputed in backward, which is what actually
            # shrinks activation memory.
            output = column_parallel_linear_sp(x, self.weight, self.bias)
        elif self.async_all_reduce:
            output = linear_with_async_all_reduce(x, self.weight, self.bias)
        else:
            output = linear_with_all_reduce(x, self.weight, self.bias)
        if self.gather_output:
            output = GatherFromModelParallelRegion.apply(output)
        return output


class RowParallelLinear(nn.Module):
    """Linear layer with row parallelism.
    Y = XW + b. W is parallelized along its first dimension and X along its second dimension as:
               -   -
              | W_1 |
              | .   |
          W = | .   |        X = [X_1, ..., X_p]
              | .   |
              | W_p |
               -   -
    We assume that X is already parallelized. This is the case after ColumnParallelLinear.
    This module returns the results of Y = sum(X_i * W_i + b_i) in the forward method.
    Arguments:
        in_features: first dimension of matrix W.
        out_features: second dimension of matrix W.
        bias: If true, add bias
        init_method: method to initialize weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        sequence_parallel: bool = False,
        overlap_comm: bool = False,
        ptb: bool = False,
    ):
        super().__init__()

        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.tp_rank = pgm.process_group_manager.tp_rank
        self.sequence_parallel = sequence_parallel
        # chunked GEMM/reduce-scatter pipeline (MegaScale Fig 3c); only meaningful under sequence parallelism.
        self.overlap_comm = overlap_comm and sequence_parallel
        # parallel-transformer-block "raw" mode: return the un-reduced local partial; the DecoderLayer
        # sums the attention and MLP partials and does a single combined reduce-scatter. See model.py.
        self.ptb = ptb

        self.in_features = in_features
        self.out_features = out_features
        assert in_features % self.tp_world_size == 0, (
            "Hidden dimension must be divisible by the tensor parallel world size"
        )
        self.input_size_per_partition = in_features // self.tp_world_size

        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.input_size_per_partition))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weight tensor with same dtype and device as self.weight
        master_weight = torch.empty(
            self.out_features, self.in_features, dtype=self.weight.dtype, device=self.weight.device, requires_grad=False
        )

        # Calculate bound based on master weight's input dimension
        k = 1 / master_weight.size(1)
        bound = math.sqrt(k)
        torch.nn.init.uniform_(master_weight, -bound, bound)

        # Split the model into size of self.input_size_per_partition
        weight_list = torch.split(master_weight, self.input_size_per_partition, dim=1)
        self.weight.data = weight_list[self.tp_rank].contiguous()

    def forward(self, x):
        if self.ptb:
            # Return the un-reduced partial; the parallel transformer block reduces it (combined with the
            # other branch) in a single reduce-scatter.
            output = F.linear(x, self.weight)
            return output if self.bias is None else output + self.bias
        if self.sequence_parallel and self.overlap_comm:
            # Chunked, pipelined matmul + reduce-scatter (MegaScale Fig 3c): each output sequence chunk
            # launches its reduce while the next chunk's GEMM runs.
            output = overlapped_row_parallel_linear_sp(x, self.weight)
        elif self.sequence_parallel:
            # The seq-parallel `g`: reduce across TP partitions and scatter back to a sequence shard.
            output = ReduceScatterToSequenceParallelRegion.apply(F.linear(x, self.weight))
        else:
            # All-reduce across all the partitions.
            output = ReduceFromModelParallelRegion.apply(F.linear(x, self.weight))
        return output if self.bias is None else output + self.bias


class VocabParallelEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        sequence_parallel: bool = False,
    ):
        super().__init__()

        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.tp_rank = pgm.process_group_manager.tp_rank
        self.sequence_parallel = sequence_parallel

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index, self.vocab_end_index = self._vocab_range_from_global_vocab_size(
            self.num_embeddings, pgm.process_group_manager.tp_rank, pgm.process_group_manager.tp_world_size
        )
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index

        self.weight = nn.Parameter(torch.Tensor(self.num_embeddings_per_partition, self.embedding_dim))

        self.reset_parameters()

    def _vocab_range_from_global_vocab_size(self, global_vocab_size: int, rank: int, world_size: int):
        # TODO: do some padding for the vocab size
        assert global_vocab_size % world_size == 0, f"{global_vocab_size} is not divisible by {world_size}"
        per_partition_vocab_size = global_vocab_size // world_size
        # vocab_range_from_per_partition_vocab_size
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    def reset_parameters(self):
        master_weight = torch.empty(
            self.num_embeddings,
            self.embedding_dim,
            dtype=self.weight.dtype,
            device=self.weight.device,
            requires_grad=False,
        )
        torch.nn.init.normal_(master_weight, mean=0.0, std=1.0)
        # Split the model into size of self.num_embeddings_per_partition
        weight_list = torch.split(master_weight, self.num_embeddings_per_partition, dim=0)
        self.weight.data = weight_list[self.tp_rank].contiguous()

    def forward(self, x):
        """
        Performs an embedding lookup for input tokens in the parallelized embedding layer
        1. Masks tokens that fall outside the specified vocabulary range and adjusts the input
        2. Performs embedding lookups for valid tokens, setting embeddings of out-of-vocabulary tokens to zero
        3. Reduces the embeddings across model parallel GPUs using all-reduce for synchronization
        """
        # Build the mask for out-of-vocabulary tokens.
        input_mask = (x < self.vocab_start_index) | (x >= self.vocab_end_index)
        # Mask the input.
        masked_input = x.clone() - self.vocab_start_index
        masked_input[input_mask] = 0
        # Get the embeddings for the valid tokens.
        output_parallel = F.embedding(
            masked_input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        # Embedding of out-of-vocabulary tokens is set to 0.
        output_parallel[input_mask, :] = 0.0
        if self.sequence_parallel:
            # Reduce the per-rank lookups and scatter to a sequence shard, so the rest of the network
            # runs on sequence-parallel activations.
            output = ReduceScatterToSequenceParallelRegion.apply(output_parallel)
        else:
            output = ReduceFromModelParallelRegion.apply(output_parallel)
        return output
