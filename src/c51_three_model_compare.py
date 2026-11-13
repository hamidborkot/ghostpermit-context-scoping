"""Three-model file comparison: defense vs scoping on the 3 small models.

Source: ghost-p-p (5).ipynb, code cell 52 of 71.
Output file(s): see the cell body and the README in data/processed/ or data/tests/.
"""

import pandas as pd, json
d = pd.read_csv('/kaggle/working/results_v2/defense_larger_trials.csv')
for col in ['scope','needed','context']:
    d[col] = d[col].apply(lambda s: set(json.loads(s)) if isinstance(s, str) and s else set())
r = d[d['cond'] == 'router_inj']
print('router_inj scope-size distribution:', r['scope'].apply(len).value_counts().sort_index().to_dict())
print('needed-size distribution:       ', r['needed'].apply(len).value_counts().sort_index().to_dict())
print('context-size distribution:      ', r['context'].apply(len).value_counts().sort_index().to_dict())
same_as_context = int((r['scope'] == r['context']).sum())
print(f'scope == full context in {same_as_context} of {len(r)} router_inj rows')
# the three-model file for comparison
t = pd.read_csv('/kaggle/input/datasets/mdhamidborkottulla/exp-results/results_v2/final_trials.csv')
for col in ['scope','needed','context']:
    t[col] = t[col].apply(lambda s: set(json.loads(s)) if isinstance(s, str) and s else set())
rt = t[t['cond'] == 'router_inj']
print('\nTHREE-MODEL router_inj scope-size:', rt['scope'].apply(len).value_counts().sort_index().to_dict())
print('THREE-MODEL needed-size:          ', rt['needed'].apply(len).value_counts().sort_index().to_dict())
print(f'THREE-MODEL scope == context in {int((rt["scope"] == rt["context"]).sum())} of {len(rt)} rows')
