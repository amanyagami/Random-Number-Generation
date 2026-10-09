"""Measure the output of the legacy algorithm (legacy/final_TRNG.py) fed with ideal random input.

Run: uv run --with numpy python tools/measure_legacy.py
"""
import numpy as np, copy, math
from collections import Counter
rng = np.random.default_rng(1)
L, cc, epoch, N = 8, 0.05, 20, 4000
r = rng.integers(-128, 128, size=N * L + 5000).astype(np.int64) + 127   # ideal-noise stand-in for mic bytes
x = [[0.0]*epoch for _ in range(L)]
for i, v in enumerate([0.141592,0.653589,0.793238,0.462643,0.383279,0.502884,0.197169,0.399375]): x[i][0] = v
z=[0]*L; o=[]; c=0
while len(o) <= N:
    for i in range(L):
        x[i][0] = float(((0.071428571*(int(r[c])%8)) + x[i][0]) * 0.666666667); c += 1
    for t in range(epoch):
        for i in range(L):
            x[i][(t+1)%epoch] = (1-cc)*x[i][t] + (cc/2)*(x[(i+1)%L][t] + x[(i-1)%L][t])
    for i in range(L):
        z[i] = int(x[i][4]*(10**8)) % (2**64); x[i][0] = x[i][4]
    for i in range(L//2):
        k = z[i+4]; sw = ((k << 32) | k >> 32); z[i] = (z[i] ^ sw) % (2**64)
    o.append(((z[0] << 64 | z[1]) << 64 | z[2]) << 64 | z[3])
bits = np.array([[ (n >> b) & 1 for b in range(256)] for n in o])
p1 = bits.mean(0)
print("outputs:", len(o), " max bit-length seen:", max(n.bit_length() for n in o))
print("bit positions never 1:", int((p1 == 0).sum()), "of 256;  stuck (p<0.01 or >0.99):", int(((p1<.01)|(p1>.99)).sum()))
print("bias of live bit positions (mean |p-0.5|):", round(float(np.abs(p1[(p1>0)]-0.5).mean()),4))
# Shannon entropy per 8-bit symbol over the whole stream (byte-wise), and min-entropy
by = np.packbits(bits.reshape(-1), bitorder='little')
cnt = np.bincount(by, minlength=256)/len(by)
print("byte Shannon entropy: %.3f / 8   min-entropy: %.3f / 8" % (-(cnt[cnt>0]*np.log2(cnt[cnt>0])).sum(), -math.log2(cnt.max())))
print("distinct 256-bit outputs:", len(set(o)), "of", len(o))
