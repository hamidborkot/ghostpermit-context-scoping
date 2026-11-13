"""DEBUG: assert /kaggle/working/results_v2/defense_larger_trials.csv exists.

Source: ghost-p-p (5).ipynb, code cell 47 of 71.
Output file(s): see the cell body and the README in data/processed/ or data/tests/.
"""

import os
import pandas as pd

P = '/kaggle/working/results_v2/defense_larger_trials.csv'
assert os.path.exists(P), 'run Cell 4 (GPU) first'
d = pd.read_csv(P)
print('rows:', len(d))
print('models:', sorted(m.split('/')[-1] for m in d.model.unique()))
print('conds:', sorted(d.cond.unique()))
print(d.groupby(['model','cond']).size().unstack(fill_value=0).to_string())
need = {'Qwen/Qwen3-8B','Qwen/Qwen2.5-7B-Instruct'}
assert need <= set(d.model.unique())
for cond in ['full_inj','full_def_inj','router_inj']:
    for m in need:
        got = int(((d.model == m) & (d.cond == cond)).sum())
        assert got == 128, f'{m} {cond} = {got}'
print('OK: 768 rows, 2 larger models x 3 arms x 2 phrasings, matched')
