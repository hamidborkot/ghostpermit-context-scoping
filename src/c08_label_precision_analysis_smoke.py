"""CELL B: analysis of the smoke run. Writes precision_summary.csv and precision_table.tex.

Source: ghost-p-p (5).ipynb, code cell 9 of 71.
Output file(s): see the cell body and the README in data/processed/ or data/tests/.
"""


# ==================== CELL B: analysis (run after Cell A has finished) ====================
import os, json, numpy as np, pandas as pd
from math import sqrt
OUT_DIR = "/kaggle/working/results_v2" if os.path.exists("/kaggle") else "./results_v2"
OUT_CSV = f"{OUT_DIR}/precision_trials.csv"
F_V1_BRIEFING = [{"calendar","contacts","email"}, {"calendar","contacts","location"}, {"contacts","email","location"}]

def wilson(k, n, z=1.959964):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d
    h = z * sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (100*(c-h), 100*(c+h))
def pct(k, n):
    lo, hi = wilson(k, n); return f"{k}/{n} ({100*k/n:.1f}%, {lo:.1f}-{hi:.1f})" if n else "0/0"

d = pd.read_csv(OUT_CSV)
d = d[d["refused"] == 0].copy() if False else d.copy()            # refusals kept; reported separately
for c in ("context", "strict_set", "relaxed_set"):
    d[c] = d[c].apply(lambda s: set(json.loads(s)))
has_F = lambda S: any(F <= S for F in F_V1_BRIEFING)
d["union_deny"] = d["context"].apply(has_F)                       # a union label would deny (v1, briefing, owner)
d["join_strict"]  = d["strict_set"].apply(has_F)                  # certain join (verbatim markers)
d["join_relaxed"] = d["relaxed_set"].apply(has_F)                 # possible join (generous detector)
def cat(r):
    if not r.union_deny: return "n/a"
    if r.join_strict:    return "deny"
    if r.join_relaxed:   return "review_high"
    return "review_low"
d["interval"] = d.apply(cat, axis=1)

U = d[d["union_deny"]].copy()
print("rows:", len(d), "| rows whose context contains a forbidden set:", len(U))
print("models:", sorted(d["model"].unique()))

rows = []
for (m, c), g in list(U.groupby(["model", "cls"])) + [(("ALL MODELS", c), g) for c, g in U.groupby("cls")]:
    n = len(g); k_lo = int((g.interval == "review_low").sum())
    rows.append(dict(model=m, cls=c, n_union_deny=n,
        interval_deny=int((g.interval == "deny").sum()),
        review_high=int((g.interval == "review_high").sum()),
        review_low=k_lo, review_low_pct=round(100*k_lo/n, 1) if n else np.nan,
        review_low_ci=f"{wilson(k_lo, n)[0]:.1f}-{wilson(k_lo, n)[1]:.1f}" if n else "",
        join_strict=int(g.join_strict.sum()), join_relaxed=int(g.join_relaxed.sum()),
        mean_strict=round(g.n_strict.mean(), 2), mean_relaxed=round(g.n_relaxed.mean(), 2),
        mean_context=round(g.context.apply(len).mean(), 2), refused=int(g.refused.sum())))
S = pd.DataFrame(rows).sort_values(["cls", "model"])
S.to_csv(f"{OUT_DIR}/precision_summary.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(S.to_string(index=False))

# ---- headline numbers for the paper ----
print("\n=== HEADLINE (all models, outputs a union label would deny) ===")
for c in ("single", "pair", "brief"):
    g = U[U.cls == c]; n = len(g)
    if n == 0: continue
    print(f"{c:6s} n={n:4d} | low-risk review {pct(int((g.interval=='review_low').sum()), n)}"
          f" | certain join {pct(int(g.join_strict.sum()), n)} | possible join (relaxed) {pct(int(g.join_relaxed.sum()), n)}")
print("\nOver-broad outputs on tasks that did not require a forbidden join (single + pair):")
g = U[U.cls.isin(["single", "pair"])]; n = len(g)
print("  certain join :", pct(int(g.join_strict.sum()), n)); print("  possible join:", pct(int(g.join_relaxed.sum()), n))
print("\nDetector sensitivity: share of outputs where the generous detector finds more sources than the strict one:",
      f"{100*(d.n_relaxed > d.n_strict).mean():.1f}%")
print("Refusal rate by model:\n", d.groupby("model")["refused"].mean().round(3).to_string())

# ---- LaTeX table (plain tabular, no \tblwidth) ----
def tex_row(r):
    return (f"{r.model.split('/')[-1] if r.model!='ALL MODELS' else 'All models'} & {r.cls} & {r.n_union_deny} & "
            f"{r.interval_deny} & {r.review_high} & {r.review_low} ({r.review_low_pct}\\%) & {r.review_low_ci} \\\\")
L = ["\\begin{table}[width=\\linewidth,pos=t]",
     "\\caption{Label precision for outputs that a union label would deny in briefings (v1, owner). Single: one-source task. Pair: two-source task. Brief: daily briefing over all sources. Deny: forbidden set among verbatim markers. Review high: forbidden set only under the generous detector. Review low: no forbidden set under either detector. Intervals are Wilson 95\\%.}\\label{tab:precision2}",
     "\\small", "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{@{}llccccc@{}}", "\\toprule",
     "Model & Task & Union denies & Deny & Review high & Review low & 95\\% interval \\\\", "\\midrule"]
for _, r in S.iterrows(): L.append(tex_row(r))
L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(f"{OUT_DIR}/precision_table.tex", "w").write("\n".join(L))

# ---- figure ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
P = S.copy(); P["label"] = P["model"].str.split("/").str[-1].str.replace("ALL MODELS", "all models") + "\n" + P["cls"]
fig, ax = plt.subplots(figsize=(max(7, 0.7*len(P)), 3.8)); x = np.arange(len(P))
a = P.interval_deny / P.n_union_deny * 100; b = P.review_high / P.n_union_deny * 100; c = P.review_low / P.n_union_deny * 100
ax.bar(x, a, label="deny (certain join)", color="#C62828"); ax.bar(x, b, bottom=a, label="review, high risk", color="#EF9A9A")
ax.bar(x, c, bottom=a+b, label="review, low risk", color="#A5D6A7")
ax.set_xticks(x); ax.set_xticklabels(P["label"], rotation=90, fontsize=7); ax.set_ylabel("% of outputs a union label would deny")
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_precision.pdf", bbox_inches="tight"); print("\nSaved:", os.listdir(OUT_DIR))
