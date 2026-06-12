#!/usr/bin/env bash
# Capacity (static-memory) vs uncapped dispatch, swept over routing imbalance, on real GPUs.
#
# Each (skew, path) runs in its OWN timeout-bounded torchrun: the uncapped path can pile a huge
# imbalanced buffer on the rank owning the hot experts and stall an all-to-all (the memory-wall
# pathology capacity dispatch prevents) -- isolating each run means that stall is REPORTED as the
# result instead of hanging the whole sweep / burning GPU hours.
#
#   bash tests/bench_capacity_moe.sh                 # ep = #visible GPUs
#   EP=4 SKEWS="0.0 1.0 2.0" bash tests/bench_capacity_moe.sh
set -u

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
EP=${EP:-$NGPU}
SKEWS=${SKEWS:-"0.0 1.0 2.0"}
HIDDEN=${HIDDEN:-4096}; INTER=${INTER:-4096}; TOKENS=${TOKENS:-8192}; TOPK=${TOPK:-2}; FACTOR=${FACTOR:-1.0}
EXPERTS=${EXPERTS:-$((4 * EP))}      # 4 experts/GPU
PER_RUN_TIMEOUT=${PER_RUN_TIMEOUT:-150}

echo "######################################################################"
echo "# capacity vs uncapped | ep=$EP experts=$EXPERTS tokens/GPU=$TOKENS hidden=$HIDDEN factor=$FACTOR"
echo "# per-run timeout=${PER_RUN_TIMEOUT}s ; a TIMEOUT row = the uncapped memory-wall stall itself"
echo "######################################################################"

for SKEW in $SKEWS; do
  for PATH_ in naive dropless drop; do
    if timeout "$PER_RUN_TIMEOUT" torchrun --nproc_per_node "$EP" tests/bench_capacity_moe.py \
        --path "$PATH_" --skew "$SKEW" --hidden "$HIDDEN" --inter "$INTER" --experts "$EXPERTS" \
        --tokens "$TOKENS" --topk "$TOPK" --factor "$FACTOR" 2>/tmp/cap_err.log | grep RESULT; then
      :
    else
      rc=$?
      if [ "$rc" -eq 124 ]; then
        echo "RESULT skew=$SKEW path=$PATH_ -> TIMEOUT/STALL (uncapped pile-up: the failure capacity prevents)"
      else
        echo "RESULT skew=$SKEW path=$PATH_ -> FAILED rc=$rc ($(tail -1 /tmp/cap_err.log))"
      fi
    fi
  done
done
