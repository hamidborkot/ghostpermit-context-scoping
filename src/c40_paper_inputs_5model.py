"""Read final_trials.csv (5-model version) and verify the 1920 answer rows.

Source: ghost-p-p (5).ipynb, code cell 41 of 71.
Output file(s): see the cell body and the README in data/processed/ or data/tests/.
"""

import os, json
import pandas as pd

PATH5 = '/kaggle/input/datasets/mdhamidborkottulla/exp-results/results_v2/final_trials.csv'
assert os.path.exists(PATH5), 'final_trials.csv missing'

d = pd.read_csv(PATH5)
print('rows:', len(d))
print('models:', sorted(d.model.unique()))
print('columns:', sorted(d.columns))
print('\ncond counts per model:')
print(d.groupby(['model','cond']).size().to_string())

EXPECTED = {
    'microsoft/Phi-3.5-mini-instruct',
    'Qwen/Qwen3-4B-Instruct-2507',
    'ibm-granite/granite-3.3-2b-instruct',
    'Qwen/Qwen3-8B',
    'Qwen/Qwen2.5-7B-Instruct',
}
assert set(d.model.unique()) == EXPECTED
ARM_COUNTS = {'fullinj':128,'fulldefinj':128,'routerinj':128,
              'routeexplicit':160,'routeimplicit':160}
for m, g in d.groupby('model'):
    for cond, n in ARM_COUNTS.items():
        got = int((g.cond == cond).sum())
        assert got == n, f'{m} {cond}: {got} != {n}'
answers = d[d.cond.isin(['fullinj','fulldefinj','routerinj'])]
print('\nanswer rows:', len(answers))
print('ALL CHECKS PASSED')
