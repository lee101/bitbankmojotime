#!/usr/bin/env python3
"""Print the -D defines src/bmt2_gpu.mojo needs for a bmt2.bin weight file."""
import struct, sys
h = struct.unpack("<64i", open(sys.argv[1], "rb").read(256))
nf = lambda k: 2 if k in (0, 4) else 1
print(f"-D D={h[5]} -D NL={h[6]} -D NH={h[7]} -D NFC={h[10]} -D NCC={h[11]} -D FF={sum(nf(k) for k in h[20:20 + h[10]])} "
      f"-D FC={sum(nf(k) for k in h[36:36 + h[11]]) or 2} -D NS={h[12] + h[13]} -D HAR={h[12]} -D CAL={h[9]}")
