# PREREG-FSEL harness environment: bgm-univ-wt tools and binary, our own SHIFT_D (read-only links to the UNIV panels and M11), our own outputs
source /nvme0n1-disk/code/bgm-univ-wt/tools/univ/env.sh
HZ=/nvme0n1-disk/code/bitbankmojotime-hz-wt; UV=/sdb-disk/tradingbots/binance/univ/shift; SD=$HZ/runs-agent/fsel/shift
mkdir -p $SD/u $HZ/runs-agent/fsel/cv
for x in arm bv hetzner; do [ -e $UV/$x ] && ln -sfn $(readlink -f $UV/$x) $SD/$x; done
for x in M11 X30 X50; do ln -sfn $(readlink -f $UV/u/$x) $SD/u/$x; done
export SHIFT_D=$SD CORRVOL_D=$HZ/runs-agent/fsel/cv SHIFT_TH=${FSEL_TH:-4} ALGOS_DIR=/nvme0n1-disk/code/bgm-univ-wt/tools/algos
export BMT_FSEL_OUT=$HZ/runs-agent/fsel FSEL_BOTPRED=$HZ/runs-agent/hz/pred_B.npz
cd $HZ/runs-agent/fsel
