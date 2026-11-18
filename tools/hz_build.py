"""PREREG-HORIZON step 2: R25-R40 panels + SEL books in our own SHIFT_D (bgm-signal-wt tools/shift/prep.py unchanged).
env: algos env.sh, then SHIFT_D=<own dir: links to arm / bv / hetzner and u/M11>, CORRVOL_D, SHIFT_TH.  usage: hz_build.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hzgate as HG
import slib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(slib.__file__))))
import prep
for u in HG.NEW:
    if os.path.exists(f"{slib.D}/u/{u}/k20/SEL.deq"):
        continue
    try:
        prep.build(u)
    except Exception as e:
        print("BUILD FAIL", u, repr(e)[:300], flush=True)
print("HZ_BUILD_DONE", flush=True)
