"""DEBUG: inspect BASE files for the 3-model final_trials.csv.

Source: ghost-p-p (5).ipynb, code cell 50 of 71.
Output file(s): see the cell body and the README in data/processed/ or data/tests/.
"""

import os
import pandas as pd
from math import sqrt

BASE = '/kaggle/input/datasets/mdhamidborkottulla/exp-results/results_v2'
def wilson(k, n, z=1.959964):
    if n == 0: return (float('nan'), float('nan'))
    p = k/n; den = 1+z*z/n; c = (p+z*z/(2*n))/den
    h = z*sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return (100*(c-h), 100*(c+h))

d = pd.read_csv(f'{BASE}/precision_trials_corrected.csv')
U = d[d.union_deny == True].copy()      # union label would deny these
print('DISCOVERY, corrected, union-deny rows:', len(U))
M = {}
for c in ['single','pair','brief']:
    g = U[U.cls == c]; n = len(g)
    dj = int(g.js.sum()); lo = int((g.cat == 'reviewlow').sum())
    w = wilson(lo, n)
    M[f'Disc{c.capitalize()}N'] = n
    M[f'Disc{c.capitalize()}Join'] = dj
    M[f'Disc{c.capitalize()}Neg'] = lo
    M[f'Disc{c.capitalize()}NegPct'] = f'{100*lo/n:.1f}'
    print(f'{c:7s} n={n} join={dj} no-evidence={lo} ({100*lo/n:.1f}% [{w[0]:.1f},{w[1]:.1f}])')
n = len(U); dj = int(U.js.sum()); lo = int((U.cat == 'reviewlow').sum())
M['DiscoveryN'] = n; M['DiscoveryPositive'] = dj
M['DiscoveryNegative'] = lo; M['DiscoveryNegativePct'] = f'{100*lo/n:.1f}'
print(f'ALL     n={n} join={dj} no-evidence={lo} ({100*lo/n:.1f}%)')

N = d[d.cls.isin(['single','pair'])].copy()
creep = N['creep'].astype(bool) if 'creep' in N.columns else None
if creep is None:
    import json
    for col in ['s2','strict_set']:
        if col in N.columns:
            N['s2set'] = N[col].apply(lambda s: set(json.loads(s.replace("'",'\"'))) if isinstance(s,str) and s else set())
            break
    needed = None
else:
    M['CreepK'] = int(creep.sum()); M['CreepN'] = len(N); M['CreepPct'] = f'{100*creep.mean():.1f}'
    print(f'scope creep narrow tasks {int(creep.sum())}/{len(N)} ({100*creep.mean():.1f}%)')

out = '/kaggle/working/numbers_discovery.tex'
with open(out, 'w') as f:
    for k in sorted(M):
        f.write(f'\\newcommand{{\\{k}}}{{{M[k]}}}\n')
print('\nwrote', out)
for k in sorted(M):
    print(f'{k} = {M[k]}')
