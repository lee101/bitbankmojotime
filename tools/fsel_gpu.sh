#!/bin/bash
# FSEL wide forecasts under the repo GPU wrapper; retried after a failure (out of memory)
source /nvme0n1-disk/code/bitbankmojotime-hz-wt/tools/fsel_env.sh
for i in $(seq 1 60); do
  nice -n 10 python3 -m train.rankh wide --seeds 0,1 && { echo "FSEL_GPU_OK"; exit 0; }
  echo "FSEL_GPU_RETRY $i $(date -u +%FT%TZ)"; sleep 90
done
echo "FSEL_GPU_FAIL"; exit 1
