"""Scoped-context experiment on larger models. Writes scoped_larger_trials.csv.

Source: ghost-p-p (5).ipynb, code cell 54 of 71.
Output file(s): see the cell body and the README in data/processed/ or data/tests/.
"""

import pandas as pd, json

SMALL = '/kaggle/input/datasets/mdhamidborkottulla/exp-results/results_v2/final_trials.csv'
LARGE = '/kaggle/working/results_v2/defense_larger_trials.csv'
OUT   = '/kaggle/working/results_v2/scoped_larger_trials.csv'

s = pd.read_csv(SMALL)
r = s[s['cond'] == 'router_inj'].copy()
# frozen routing decision per identical task
r['scope_set'] = r['scope'].apply(lambda x: tuple(sorted(json.loads(x))))
scope_map = {}
for _, row in r.iterrows():
    scope_map[(row['combo'], row['cls'], row['trial'], row['phr'])] = row['scope_set']
print('frozen scope decisions available:', len(scope_map))

d = pd.read_csv(LARGE)
inj = d[d['cond'] == 'router_inj'].copy()   # same tasks, scope was wrongly full
inj['scope_set'] = inj.apply(
    lambda x: scope_map.get((x['combo'], x['cls'], x['trial'], x['phr'])), axis=1)
missing = inj['scope_set'].isna().sum()
print('larger-model injection tasks:', len(inj), '| without a frozen scope:', missing)
assert missing == 0, 'some tasks lack a frozen scope — report before continuing'
inj['scope_new'] = inj['scope_set'].apply(list)
inj[['model','combo','cls','trial','phr','task','needed','scope_new']].to_json(
    '/kaggle/working/results_v2/scoped_larger_jobs.json', orient='records')
print('wrote scoped_larger_jobs.json')
print('scope sizes:', inj['scope_new'].apply(len).value_counts().sort_index().to_dict())
