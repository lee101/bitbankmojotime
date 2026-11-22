#!/usr/bin/env bash
# build the Mojo inference binaries into build/. MOJO_ENV = a pixi env with mojo (and max for the GPU path).
set -euo pipefail
cd "$(dirname "$0")"
E=${MOJO_ENV:-/nvme0n1-disk/code/mojo-p2m-toolchain/.pixi/envs/default}
export MODULAR_HOME=$E/share/max
export MODULAR_NVPTX_COMPILER_PATH=${MODULAR_NVPTX_COMPILER_PATH:-/usr/local/cuda-12.9/bin/ptxas}
mkdir -p build
nice -n 10 "$E/bin/mojo" build -O3 src/bmt.mojo -o build/bmt
if [ "${1:-}" = gpu ]; then
  nice -n 10 "$E/bin/mojo" build -O3 --target-accelerator=nvidia:sm_120 src/bmt_gpu.mojo -o build/bmt_gpu_sm120
fi
if [ "${1:-}" = gpu ]; then   # size-t build (model dimensions are compile-time)
  nice -n 10 "$E/bin/mojo" build -O3 -D D=64 -D NL=2 -D NH=4 --target-accelerator=nvidia:sm_120 src/bmt_gpu.mojo -o build/bmt_gpu_t_sm120
fi
nice -n 10 "$E/bin/mojo" build -O3 src/bmt2.mojo -o build/bmt2
if [ "${1:-}" = gpu ]; then   # bmt2 GPU builds: dimensions from the weight file (tools/gpu_defs.py); t = the 0.11M promoted size, s = 0.82M
  for w in tests/vectors/bmt2-t-random/bmt2.bin:t ${BMT2_S_WEIGHTS:-}; do
    f=${w%:*}; tag=${w##*:}
    nice -n 10 "$E/bin/mojo" build -O3 $(python3 tools/gpu_defs.py "$f") --target-accelerator=nvidia:sm_120 src/bmt2_gpu.mojo -o build/bmt2_gpu_${tag}_sm120
  done
fi
