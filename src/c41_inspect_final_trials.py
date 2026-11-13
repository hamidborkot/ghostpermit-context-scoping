"""Inspect final_trials.csv: row counts and condition totals per model.

Source: ghost-p-p (5).ipynb, code cell 42 of 71.
Output file(s): see the cell body and the README in data/processed/ or data/tests/.
"""

import os, json
import pandas as pd

PATH = '/kaggle/input/datasets/mdhamidborkottulla/exp-results/results_v2/final_trials.csv'
d = pd.read_csv(PATH)
print('rows:', len(d))
print('conds:', sorted(d.cond.unique()))
print('columns ok:', {'used','scope','needed','covers_needed','outside_scope','answered'} <= set(d.columns))
for m, g in d.groupby('model'):
    print(m.split('/')[-1], dict(g.groupby('cond').size()))
