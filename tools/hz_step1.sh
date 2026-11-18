#!/bin/bash
# PREREG-HORIZON step 1. usage: hz_step1.sh <H*> [other horizon for the descriptive gtX arm]
# 3 harness processes x 3 sim threads + 3 single-thread ruin processes (<= 12 threads), then the pooled report.
source /nvme0n1-disk/code/bgm-signal-wt/tools/algos/env.sh
HZ=/nvme0n1-disk/code/bitbankmojotime-hz-wt
export CORRVOL_D=$HZ/runs-agent/hz/gate SHIFT_TH=3
[ -n "$2" ] && export HZ_X=$2
P=$HZ/runs-agent/hz/pred_B.npz; F=gtH,gtB,dtH,dbH${2:+,gtX}
mkdir -p $CORRVOL_D; cd $HZ/runs-agent/hz
G1=M11,R1,R4,R7,M3; G2=M15,M10,R2,R5,R8; G3=M20,M5,R3,R6
for i in 1 2 3; do eval g=\$G$i; nice -n 10 python3 $HZ/tools/hzgate.py run $P $1 $F 10 $g > gate_run$i.log 2>&1 & done
nice -n 10 python3 $HZ/tools/hzgate.py ruin $P $1 gtH 0 a > gate_ruin_a.log 2>&1 &
nice -n 10 python3 $HZ/tools/hzgate.py ruin $P $1 gtB 0 b > gate_ruin_b.log 2>&1 &
nice -n 10 python3 $HZ/tools/hzgate.py ruin $P $1 dtH,dbH 0 c > gate_ruin_c.log 2>&1 &
wait
nice -n 10 python3 $HZ/tools/hzgate.py report $P $1 $F 10 step1 > step1.md 2>&1
echo "HZ_STEP1_EXIT $?"
