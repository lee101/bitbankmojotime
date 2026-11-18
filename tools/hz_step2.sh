#!/bin/bash
# PREREG-HORIZON step 2 (only after a form is picked). usage: hz_step2.sh <H*> <picked form> [forms list override]
# builds R25-R40 in our own SHIFT_D (links to the archive, prep.py unchanged), then the frozen form + risk-matched arm, 20 placebos.
source /nvme0n1-disk/code/bgm-signal-wt/tools/algos/env.sh
HZ=/nvme0n1-disk/code/bitbankmojotime-hz-wt; SD=$HZ/runs-agent/hz/shift; SRC=/sdb-disk/tradingbots/binance/shift
mkdir -p $SD/u; for x in arm bv hetzner; do ln -sfn $SRC/$x $SD/$x; done; ln -sfn $SRC/u/M11 $SD/u/M11
export SHIFT_D=$SD CORRVOL_D=$HZ/runs-agent/hz/confirm HZ_UNIS=NEW SHIFT_TH=12
mkdir -p $CORRVOL_D; cd $HZ/runs-agent/hz
nice -n 10 python3 $HZ/tools/hz_build.py > build.log 2>&1 || exit 1
P=$HZ/runs-agent/hz/pred_B.npz; F=${3:-$2,${2}r}
export SHIFT_TH=3
G1=R25,R28,R31,R34,R37,R40; G2=R26,R29,R32,R35,R38; G3=R27,R30,R33,R36,R39
for i in 1 2 3; do eval g=\$G$i; nice -n 10 python3 $HZ/tools/hzgate.py run $P $1 $F 20 $g > confirm_run$i.log 2>&1 & done
nice -n 10 python3 $HZ/tools/hzgate.py ruin $P $1 $F 0 confirm > confirm_ruin.log 2>&1 &
wait
nice -n 10 python3 $HZ/tools/hzgate.py report $P $1 $F 20 step2 > step2.md 2>&1
echo "HZ_STEP2_EXIT $?"
