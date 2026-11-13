"""Cross-check: cat==review_low must equal (not strict_join).

Source: ghost-p-p (5).ipynb, code cell 51 of 71.
Output file(s): see the cell body and the README in data/processed/ or data/tests/.
"""

import pandas as pd
from math import sqrt

BASE = '/kaggle/input/datasets/mdhamidborkottulla/exp-results/results_v2'
def wilson(k, n, z=1.959964):
    if n == 0: return (float('nan'), float('nan'))
    p = k/n; den = 1+z*z/n; c = (p+z*z/(2*n))/den
    h = z*sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return (100*(c-h), 100*(c+h))

d = pd.read_csv(f'{BASE}/precision_trials_corrected.csv')
print('--- sanity ---')
print('rows:', len(d), '| cat labels:', sorted(d.cat.dropna().unique()))
print('js dtype:', d['js'].dtype, '| union_deny True:', int(d['union_deny'].sum()))

U = d[d['union_deny'] == True].copy()
U['strict_join'] = U['js'].astype(bool)
# cross-check: cat==review_low must equal (not strict_join)
mismatch = int((U['cat'] == 'review_low').sum()) - int((~U['strict_join']).sum())
print('cat==review_low count:', int((U['cat'] == 'review_low').sum()),
      '| computed no-evidence:', int((~U['strict_join']).sum()),
      '| diff:', mismatch)

EXPECT = {'single': 168, 'pair': 127, 'brief': 70}
M = {}
print('\n--- discovery, union-deny rows ---')
for c in ['single','pair','brief']:
    g = U[U['cls'] == c]; n = len(g)
    dj = int(g['strict_join'].sum()); lo = n - dj
    w = wilson(lo, n)
    M[f'Disc{c.capitalize()}N'] = n
    M[f'Disc{c.capitalize()}Join'] = dj
    M[f'Disc{c.capitalize()}Neg'] = lo
    M[f'Disc{c.capitalize()}NegPct'] = f'{100*lo/n:.1f}'
    flag = 'OK' if lo == EXPECT[c] else f'CHECK (expected {EXPECT[c]})'
    print(f'{c:7s} n={n} join={dj} no-evidence={lo} ({100*lo/n:.1f}% [{w[0]:.1f},{w[1]:.1f}])  {flag}')
n = len(U); dj = int(U['strict_join'].sum()); lo = n - dj
M['DiscoveryN'] = n; M['DiscoveryPositive'] = dj
M['DiscoveryNegative'] = lo; M['DiscoveryNegativePct'] = f'{100*lo/n:.1f}'
tag = ('matches Cell E 363' if lo == 363 else
       'matches v3 365' if lo == 365 else
       'REPORT THIS NUMBER — it is neither 363 nor 365')
print(f'ALL     n={n} join={dj} no-evidence={lo} ({100*lo/n:.1f}%)   {tag}')

N = d[d['cls'].isin(['single','pair'])].copy()
M['CreepK'] = int(N['creep'].astype(bool).sum()); M['CreepN'] = len(N)
M['CreepPct'] = f'{100*N["creep"].astype(bool).mean():.1f}'
print(f'scope creep {M["CreepK"]}/{M["CreepN"]} ({M["CreepPct"]}%)')

with open('/kaggle/working/numbers_discovery.tex','w') as f:
    for k in sorted(M):
        f.write(f'\\newcommand{{\\{k}}}{{{M[k]}}}\n')
print('\nwrote /kaggle/working/numbers_discovery.tex')
print({k: M[k] for k in sorted(M)})
