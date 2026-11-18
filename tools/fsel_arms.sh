#!/bin/bash
# usage: fsel_arms.sh <lead|full> <list spec>   form-2 arms in 3 shards x 2 sim threads (resumable: finished arms are skipped), then merge
export FSEL_TH=2
source /nvme0n1-disk/code/bitbankmojotime-hz-wt/tools/fsel_harness.sh
for i in 0 1 2; do nice -n 10 python3 $HZ/tools/fsel_trend.py arms $1 "$2" $i/3 > arms_$1_$i.log 2>&1 & done; wait
nice -n 10 python3 $HZ/tools/fsel_trend.py merge $1 > merge_$1.log 2>&1
echo "FSEL_ARMS_EXIT $? $(rg -c '^arm ' arms_$1_0.log arms_$1_1.log arms_$1_2.log | tr '\n' ' ') $(rg -c 'Traceback' arms_$1_0.log arms_$1_1.log arms_$1_2.log | tr '\n' ' ')"
