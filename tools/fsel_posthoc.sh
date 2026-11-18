#!/bin/bash
# FSEL post hoc information (verdict fixed before): form-2 placebo arms for all four (N, K) cells on the lead segment, then the lead report again.
# Resumable; markers in runs-agent/fsel/posthoc.log. Launch detached: setsid nohup tools/fsel_posthoc.sh &
HZ=/nvme0n1-disk/code/bitbankmojotime-hz-wt; O=$HZ/runs-agent/fsel
for s in pick:30:5 pick:50:5 pick:30:10 pick:50:10; do $HZ/tools/fsel_arms.sh lead $s > $O/arms_lead_$s.log 2>&1; echo "POSTHOC $s $(tail -n 1 $O/arms_lead_$s.log)"; done
export BMT_FSEL_OUT=$O OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
nice -n 10 python3 $HZ/tools/fsel.py report lead > $O/report_lead_posthoc.md 2>&1
cp $O/report_lead.json $O/report_lead_posthoc.json
echo "FSEL_POSTHOC_DONE $?"
