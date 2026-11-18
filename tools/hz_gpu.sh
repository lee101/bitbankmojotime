#!/bin/bash
# usage: hz_gpu.sh train|pred [args]   GPU step under the repo wrapper (bmtbench.gpu waits for head-room, caps VRAM); retried after a failure (out of memory)
source /nvme0n1-disk/code/bitbankmojotime-hz-wt/tools/hz_env.sh
for i in $(seq 1 60); do
  nice -n 10 python3 -m train.rankh "$@" && { echo "HZ_GPU_OK $1"; exit 0; }
  echo "HZ_GPU_RETRY $i $(date -u +%FT%TZ)"; sleep 90
done
echo "HZ_GPU_FAIL $1"; exit 1
