"""DEBUG: inspect BASE folders for available CSVs.

Source: ghost-p-p (5).ipynb, code cell 44 of 71.
Output file(s): see the cell body and the README in data/processed/ or data/tests/.
"""

import os, pandas as pd

ROOTS = ['/kaggle/input/datasets/mdhamidborkottulla',
         '/kaggle/working']
for base in ROOTS:
    for root, dirs, files in os.walk(base):
        for f in sorted(files):
            if f.endswith('.csv'):
                p = os.path.join(root, f)
                try:
                    d = pd.read_csv(p, nrows=None)
                    conds = sorted(d['cond'].unique()) if 'cond' in d.columns else '-'
                    models = sorted({m.split('/')[-1] for m in d['model'].unique()}) if 'model' in d.columns else '-'
                    print(f'{p}\n  rows={len(d)}  conds={conds}\n  models={models}')
                except Exception as e:
                    print(p, 'ERR', repr(e)[:120])
