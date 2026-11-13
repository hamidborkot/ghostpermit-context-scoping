"""CELL B2: full analysis with visible outputs and LaTeX table.

Source: ghost-p-p (5).ipynb, code cell 12 of 71.
Output file(s): see the cell body and the README in data/processed/ or data/tests/.
"""

# ==================== CELL B2: full analysis with visible outputs ====================
# Run after Cell A has finished the full campaign (SMOKE=False). Prints everything and shows the figure.
import os, json, numpy as np, pandas as pd
from math import sqrt
from IPython.display import display
OUT_DIR = "/kaggle/working/results_v2" if os.path.exists("/kaggle") else "./results_v2"
OUT_CSV = f"{OUT_DIR}/precision_trials.csv"
F_V1_BRIEFING = [{"calendar","contacts","email"}, {"calendar","contacts","location"}, {"contacts","email","location"}]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 80)

def wilson(k, n, z=1.959964):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d
    h = z * sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (100*(c-h), 100*(c+h))
def pct(k, n):
    lo, hi = wilson(k, n); return f"{k}/{n} ({100*k/n:.1f}%, {lo:.1f}-{hi:.1f})" if n else "0/0"

d = pd.read_csv(OUT_CSV).copy()
for c in ("context", "strict_set", "relaxed_set"):
    d[c] = d[c].apply(lambda s: set(json.loads(s)))
d["short"] = d["model"].str.split("/").str[-1]
has_F = lambda S: any(F <= S for F in F_V1_BRIEFING)
d["union_deny"]   = d["context"].apply(has_F)
d["join_strict"]  = d["strict_set"].apply(has_F)
d["join_relaxed"] = d["relaxed_set"].apply(has_F)
def cat(r):
    if not r.union_deny: return "n/a"
    if r.join_strict:    return "deny"
    if r.join_relaxed:   return "review_high"
    return "review_low"
d["interval"] = d.apply(cat, axis=1)
d["n_ctx"] = d["context"].apply(len)
d["tight_strict"] = d["n_strict"] / d["n_ctx"]          # share of context sources visibly used

print("="*78); print("1. COMPLETENESS (rows per model and task class; expected 150 each)"); print("="*78)
display(d.groupby(["short", "cls"]).size().unstack())

print("\n" + "="*78); print("2. LABEL TIGHTNESS over ALL combinations: share of context sources a response visibly uses"); print("="*78)
T = d.groupby(["short", "cls"]).agg(n=("uid", "size"), mean_context=("n_ctx", "mean"), mean_strict=("n_strict", "mean"),
        mean_relaxed=("n_relaxed", "mean"), tightness=("tight_strict", "mean"), refused=("refused", "mean")).round(3)
display(T)
print("Pooled tightness by class:", d.groupby("cls")["tight_strict"].mean().round(3).to_dict())

U = d[d["union_deny"]].copy()
print("\n" + "="*78)
print(f"3. OUTPUTS A UNION LABEL WOULD DENY (v1 briefing, owner): {len(U)} rows from combos C6, C8, C9, C10")
print("="*78)
rows = []
groups = list(U.groupby(["short", "cls"])) + [(("ALL MODELS", c), g) for c, g in U.groupby("cls")]
for (m, c), g in groups:
    n = len(g); k_lo = int((g.interval == "review_low").sum())
    rows.append(dict(model=m, cls=c, n_union_deny=n, interval_deny=int((g.interval == "deny").sum()),
        review_high=int((g.interval == "review_high").sum()), review_low=k_lo,
        review_low_pct=round(100*k_lo/n, 1), review_low_ci=f"{wilson(k_lo, n)[0]:.1f}-{wilson(k_lo, n)[1]:.1f}",
        join_strict=int(g.join_strict.sum()), join_relaxed=int(g.join_relaxed.sum())))
S = pd.DataFrame(rows).sort_values(["cls", "model"]); S.to_csv(f"{OUT_DIR}/precision_summary.csv", index=False)
display(S)

print("\n" + "="*78); print("4. HEADLINE NUMBERS (all models pooled)"); print("="*78)
for c in ("single", "pair", "brief"):
    g = U[U.cls == c]; n = len(g)
    if n:
        print(f"{c:6s} n={n:4d} | low-risk review {pct(int((g.interval=='review_low').sum()), n)}"
              f" | certain join {pct(int(g.join_strict.sum()), n)} | possible join {pct(int(g.join_relaxed.sum()), n)}")
g = U[U.cls.isin(["single", "pair"])]; n = len(g)
print("Over-broad outputs on tasks that did NOT require a forbidden join (single + pair):")
print("   certain join :", pct(int(g.join_strict.sum()), n)); print("   possible join:", pct(int(g.join_relaxed.sum()), n))
print("\nPer model, tasks that did not require a join: certain-join rate")
for m, gm in g.groupby("short"): print(f"   {m:34s} {pct(int(gm.join_strict.sum()), len(gm))}")

print("\n" + "="*78); print("5. DETECTOR SENSITIVITY (strict vs relaxed)"); print("="*78)
print(d.assign(relaxed_finds_more=(d.n_relaxed > d.n_strict)).groupby("short")["relaxed_finds_more"].mean().round(3).to_string())

print("\n" + "="*78); print("6. EXAMPLES (first 600 characters) to eyeball the detectors"); print("="*78)
for c in ("single", "pair", "brief"):
    for _, r in d[d.cls == c].sample(min(2, (d.cls == c).sum()), random_state=3).iterrows():
        print(f"[{r.short} | {r.combo} | {c}] context={sorted(r.context)} strict={sorted(r.strict_set)} relaxed={sorted(r.relaxed_set)}")
        print("   ", (r.response or "").replace("\n", " ")[:600]); print()

def tex_row(r):
    return (f"{r.model} & {r.cls} & {r.n_union_deny} & {r.interval_deny} & {r.review_high} & "
            f"{r.review_low} ({r.review_low_pct}\\%) & {r.review_low_ci} \\\\")
L = ["\\begin{table}[width=\\linewidth,pos=t]",
     "\\caption{Label precision for outputs that a union label would deny in briefings (v1, owner). Single: one-source task. Pair: two-source task. Brief: daily briefing over all sources. Deny: forbidden set among verbatim markers. Review high: forbidden set only under the generous detector. Review low: no forbidden set under either detector. Intervals are Wilson 95\\%.}\\label{tab:precision2}",
     "\\small", "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{@{}llccccc@{}}", "\\toprule",
     "Model & Task & Union denies & Deny & Review high & Review low & 95\\% interval \\\\", "\\midrule"]
L += [tex_row(r) for _, r in S.iterrows()] + ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(f"{OUT_DIR}/precision_table.tex", "w").write("\n".join(L))
print("="*78); print("7. LATEX TABLE (also saved to precision_table.tex)"); print("="*78); print("\n".join(L))

import matplotlib.pyplot as plt
P = S.copy(); P["label"] = P["model"].str.replace("ALL MODELS", "all models") + "\n" + P["cls"]
fig, ax = plt.subplots(figsize=(max(7, 0.7*len(P)), 4)); x = np.arange(len(P))
a = P.interval_deny / P.n_union_deny * 100; b = P.review_high / P.n_union_deny * 100; c = P.review_low / P.n_union_deny * 100
ax.bar(x, a, label="deny (certain join)", color="#C62828"); ax.bar(x, b, bottom=a, label="review, high risk", color="#EF9A9A")
ax.bar(x, c, bottom=a+b, label="review, low risk", color="#A5D6A7")
ax.set_xticks(x); ax.set_xticklabels(P["label"], rotation=90, fontsize=7); ax.set_ylabel("% of outputs a union label would deny")
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=False); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_precision.pdf", bbox_inches="tight"); plt.show()

print("\n" + "#"*78); print("# COPY EVERYTHING BELOW THIS LINE AND SEND IT TO ME"); print("#"*78)
print("MODELS:", sorted(d["short"].unique()), "| ROWS:", len(d), "| UNION-DENY ROWS:", len(U))
print(S.to_string(index=False))
print(T.to_string())
for c in ("single", "pair", "brief"):
    g = U[U.cls == c]
    if len(g): print(f"HEADLINE {c}: n={len(g)} low-risk {pct(int((g.interval=='review_low').sum()), len(g))} certain-join {pct(int(g.join_strict.sum()), len(g))} possible-join {pct(int(g.join_relaxed.sum()), len(g))}")
g = U[U.cls.isin(["single", "pair"])]
print("OVERBROAD (no join required):", "certain", pct(int(g.join_strict.sum()), len(g)), "| possible", pct(int(g.join_relaxed.sum()), len(g)))
print("DETECTOR relaxed-finds-more share:", round(float((d.n_relaxed > d.n_strict).mean()), 3))
print("Saved files:", sorted(os.listdir(OUT_DIR)))
