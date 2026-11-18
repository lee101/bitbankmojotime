# PREREG-HORIZON environment: code and outputs in this worktree, inputs (kline cache, benchmark rows, frozen mkt+r24 checkpoints) read only
HZ=/nvme0n1-disk/code/bitbankmojotime-hz-wt
cd $HZ
export BMT_RUNS=/nvme0n1-disk/code/bitbankmojotime-agent-wt/runs-agent BMT_DATA=/nvme0n1-disk/code/bitbankmojotime/data/cache PYTHONPATH=$HZ/bench:$HZ PYTHONUNBUFFERED=1
export BMT_HZ_OUT=${BMT_HZ_OUT:-$HZ/runs-agent/hz} BMT_GPU_BLACKOUT=15:20-16:15 BMT_GPU_TZ=America/New_York OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
