#!/usr/bin/env bash
# EP weak-scaling study --------------------------------------------------------------------------
# The whole EP module rests on one claim: overlap / FP8 / DeepEP pay off only when the all-to-all
# cost fraction rho = t_comm / t_compute is non-trivial, and rho GROWS with the EP degree (more
# ranks => each token's dispatch fans out to more peers). This sweep makes that visible.
#
# Weak scaling: per-GPU work is held FIXED (LOCAL_E experts/GPU, TOKENS tokens/GPU) and only the EP
# degree grows, so num_experts = LOCAL_E * EP. We time one MoE layer (plain / overlap / tiled) for
# each EP degree, fwd and fwd+bwd, in two link regimes:
#   * nvlink   : intra-node NVSwitch (compute-bound; rho tiny -> overlap ~ no-op)
#   * slowlink : NCCL_P2P_DISABLE=1 forces dispatch/combine OFF the NVLink fabric, emulating the
#                cross-node bandwidth regime where production MoE actually lives (rho large).
#
# Usage:
#   bash tests/bench_ep_scaling.sh "2 4 8"     # sweep these EP degrees
#   bash tests/bench_ep_scaling.sh             # default: 2,4,...,#visible GPUs
set -u

LOCAL_E=${LOCAL_E:-4}        # experts per GPU, kept constant  -> weak scaling
TOKENS=${TOKENS:-8192}       # tokens per GPU
HIDDEN=${HIDDEN:-4096}
INTER=${INTER:-4096}
TILES=${TILES:-"2 4"}

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
EPS=${1:-$(seq 2 2 "$NGPU" | tr '\n' ' ')}

echo "######################################################################"
echo "# EP weak-scaling: LOCAL_E=$LOCAL_E tokens/GPU=$TOKENS hidden=$HIDDEN"
echo "# visible GPUs=$NGPU  sweeping EP in: $EPS"
echo "######################################################################"

for EP in $EPS; do
  if (( NGPU < EP )); then echo "[skip] EP=$EP (only $NGPU GPUs)"; continue; fi
  EXPERTS=$((LOCAL_E * EP))
  for LINK in nvlink slowlink; do
    P2P=0; [ "$LINK" = slowlink ] && P2P=1
    for MODE in "" "--backward"; do
      echo ""
      echo ">>>>>> EP=$EP experts=$EXPERTS link=$LINK${MODE:+ (fwd+bwd)} (NCCL_P2P_DISABLE=$P2P) <<<<<<"
      NCCL_P2P_DISABLE=$P2P torchrun --nproc_per_node "$EP" tests/bench_ep_overlap.py \
        --hidden "$HIDDEN" --inter "$INTER" --experts "$EXPERTS" --tokens "$TOKENS" \
        --topk 2 --shared 1 --tiles $TILES $MODE
    done
  done
done
