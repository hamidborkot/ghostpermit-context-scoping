"""DEBUG: inspect precision_trials_corrected.csv schema.

Source: ghost-p-p (5).ipynb, code cell 46 of 71.
Output file(s): see the cell body and the README in data/processed/ or data/tests/.
"""

import pandas as pd
BASE = '/kaggle/input/datasets/mdhamidborkottulla/exp-results/results_v2'
d = pd.read_csv(f'{BASE}/precision_trials_corrected.csv', nrows=3)
print(sorted(d.columns))
print(d.head(1).to_dict('records'))
