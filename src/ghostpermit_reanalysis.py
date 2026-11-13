"""Recompute reported cell-count analyses; no API, GPU, or model requests.
Requires numpy, pandas, scipy, statsmodels. The ledger is a reconstruction
from supplied summaries, not a response-level raw dataset.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f, spearmanr, fisher_exact
ROOT=Path(__file__).resolve().parent
d=pd.read_csv(ROOT/'ghostpermit_cell_counts.csv')
def fit(frame):
 y=np.c_[frame.positive,frame.valid-frame.positive]
 x0=sm.add_constant(pd.get_dummies(frame.deployment,drop_first=True).astype(float))
 x1=pd.concat([x0,pd.get_dummies(frame.combination,drop_first=True).astype(float)],axis=1)
 null=sm.GLM(y,np.ones((len(frame),1)),family=sm.families.Binomial()).fit()
 red=sm.GLM(y,x0,family=sm.families.Binomial()).fit()
 full=sm.GLM(y,x1,family=sm.families.Binomial()).fit()
 lr=2*(full.llf-red.llf)
 F=(lr/9)/(full.deviance/full.df_resid)
 gate=frame.combination.isin(['C6','C8','C9','C10']).astype(float).rename('gate')
 mg=sm.GLM(y,pd.concat([x0,gate],axis=1),family=sm.families.Binomial()).fit()
 fg=2*(mg.llf-red.llf)/(mg.deviance/mg.df_resid)
 return dict(lr=lr,F=F,p=f.sf(F,9,full.df_resid),gate_F=fg,gate_p=f.sf(fg,1,mg.df_resid),
  deployment_lr=2*(red.llf-null.llf),deviance=full.deviance,residual_df=full.df_resid,
  dispersion=full.deviance/full.df_resid,Pearson_p=f.sf(lr/9/(full.pearson_chi2/full.df_resid),9,full.df_resid))
a=d[d.protocol=='A'].copy()
A=fit(a);all_fit=fit(d)
cs=a.groupby('combination')[['positive','valid']].sum().reindex(['C'+str(i) for i in range(1,11)])
r=cs.positive/cs.valid
P=[.02875,.0825,.03375,.03375,.0975,.22,.1075,.4425,.2025,.71]
rescue=fisher_exact([[18,43],[5,55]])
out={'Protocol_A':A,'All_seven':all_fit,'Protocol_A_totals':[int(a.positive.sum()),int(a.valid.sum())],
 'All_totals':[int(d.positive.sum()),int(d.valid.sum())],
 'PARS_rho':spearmanr(P,r).statistic,'PARS_p':spearmanr(P,r).pvalue,
 'Permission_count_rho':spearmanr([2,2,2,2,3,3,3,4,4,5],r).statistic,
 'Permission_count_p':spearmanr([2,2,2,2,3,3,3,4,4,5],r).pvalue,
 'Recovery_OR':rescue.statistic,'Recovery_p':rescue.pvalue}
assert out['Protocol_A_totals']==[545,1435]
assert out['All_totals']==[636,2035]
assert abs(A['F']-3.455051187)<1e-7
assert abs(A['gate_p']-.13602656)<1e-7
(ROOT/'ghostpermit_reanalysis_results.json').write_text(json.dumps(out,indent=2,default=float))
print(json.dumps(out,indent=2,default=float))
