"""CPU-only structural audit of confirm_trials.csv: 3520 rows, 5 models, paired arms, JSON consistency. Writes completion_audit.json.

Source: ghost-p-p (5).ipynb, code cell 29 of 71.
Output file(s): see the cell body and the README in data/processed/ or data/tests/.
"""

"""CPU-only structural audit. Does not load models or modify research results."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path('/kaggle/working/final_v3')
p=ROOT/'confirm_trials.csv'
if not p.exists():
    raise FileNotFoundError(f'{p} missing. Restore the saved CSV; do not regenerate model outputs.')
d=pd.read_csv(p,keep_default_na=False)
models=['microsoft/Phi-3.5-mini-instruct','Qwen/Qwen3-4B-Instruct-2507','ibm-granite/granite-3.3-2b-instruct','Qwen/Qwen3-8B','Qwen/Qwen2.5-7B-Instruct']
counts={'full_inj':128,'full_def_inj':128,'router_inj':128,'route_explicit':160,'route_implicit':160}
issues=[]
def check(ok,message):
    print(('PASS: ' if ok else 'CHECK: ')+message)
    if not ok: issues.append(message)
required={'uid','model','combo','cls','trial','cond','phr','context','needed','scope','used','outside_scope','answered','response'}
check(required<=set(d.columns),'required columns present')
if not required<=set(d.columns): raise ValueError(sorted(required-set(d.columns)))
check(len(d)==3520,'3520 rows')
check(set(d.model)==set(models),'five expected models')
check(not d.uid.duplicated().any(),'unique trial IDs')
key=['model','combo','cls','trial','cond','phr']
check(not d.duplicated(key).any(),'unique logical case keys')
for m in models:
    got=d[d.model==m].groupby('cond').size().to_dict()
    check(got==counts,f'{m}: condition totals {got}')
# Count records by cell, not just by model.
for cond in ['route_explicit','route_implicit','full_inj','full_def_inj','router_inj']:
    g=d[d.cond==cond]
    if cond.startswith('route_'):
        ok=(set(g.combo)=={f'C{i}' for i in range(1,11)} and len(g.groupby(['model','combo','cls']))==100 and (g.groupby(['model','combo','cls']).size()==8).all())
    else:
        ok=(set(g.combo)=={'C6','C8','C9','C10'} and set(g.phr)=={'A','B'} and len(g.groupby(['model','combo','cls','phr']))==80 and (g.groupby(['model','combo','cls','phr']).size()==8).all())
    check(bool(ok),f'{cond}: eight trials per planned cell')
answer=d[~d.cond.str.startswith('route_')]
check(len(answer)==1920,'1920 answer rows; the other 1600 records are routing evaluations')
check(answer.response.str.strip().ne('').all(),'answer rows have nonempty saved responses')
check(pd.to_numeric(answer.answered,errors='coerce').eq(1).all(),'answer flags indicate completed responses')
valid={'calendar','contacts','location','notes','email'}
bad=[]
for i,row in d.iterrows():
    try:
        v={c:set(json.loads(row[c])) for c in ['context','needed','scope','used','outside_scope']}
        assert all(s<=valid for s in v.values())
        assert v['needed']<=v['context']
        assert v['scope']<=v['context']
        assert v['outside_scope']==v['used']-v['scope']
    except Exception as e: bad.append({'row':int(i),'uid':row.uid,'error':str(e)})
check(not bad,f'JSON sets and saved out-of-scope sets consistent ({len(bad)} flagged)')
# Confirm all three arms can be matched one-to-one.
a=answer[answer.cond=='full_inj']
for arm in ['full_def_inj','router_inj']:
    b=answer[answer.cond==arm]
    keys=['model','combo','cls','trial','phr']
    check(set(map(tuple,a[keys].values))==set(map(tuple,b[keys].values)),f'full_inj and {arm} have identical case keys')
    if 'task' in d.columns:
        joined=a[keys+['task']].merge(b[keys+['task']],on=keys,validate='one_to_one',suffixes=('_a','_b'))
        check(joined.task_a.eq(joined.task_b).all(),f'{arm}: matched cases use identical base task text')
leaks=answer[answer.outside_scope.apply(lambda s:len(json.loads(s))>0)]
print('\nOUT-OF-SCOPE MARKER MATCHES (do not automatically classify as real leaks):',len(leaks))
for _,r in leaks.iterrows():
    print('\n',r.uid,r.model,r.cond,'scope=',r.scope,'outside=',r.outside_scope)
    print(r.response)
for name in ['human_sample.csv','generated/human_sample.csv']:
    hp=ROOT/name
    if hp.exists():
        h=pd.read_csv(hp)
        print(f'\nExisting human sample {name}: {len(h)} rows. A nine-row sample is a pilot, not the planned validation.')
report={'rows':len(d),'answer_rows':len(answer),'routing_records':len(d)-len(answer),'issues':issues,'invalid_set_rows':bad,'out_of_scope_marker_rows':len(leaks),'status':'PASS' if not issues else 'NEEDS_REVIEW'}
(ROOT/'completion_audit.json').write_text(json.dumps(report,indent=2))
print('\nAUDIT STATUS:',report['status'])
print('Saved completion_audit.json. No models were loaded and no input file was overwritten.')
