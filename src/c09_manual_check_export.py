"""CELL C: export 60 random outputs to manual_check.csv for hand labelling.

Source: ghost-p-p (5).ipynb, code cell 10 of 71.
Output file(s): see the cell body and the README in data/processed/ or data/tests/.
"""


# ==================== CELL C (optional, 15 minutes by hand): validate the detectors ====================
# Part 1: export 60 random outputs. Open manual_check.csv, and in column true_sources write the sources
# whose information the response actually uses, separated by commas (calendar,contacts,location,notes,email).
import os, json, pandas as pd
OUT_DIR = "/kaggle/working/results_v2" if os.path.exists("/kaggle") else "./results_v2"
d = pd.read_csv(f"{OUT_DIR}/precision_trials.csv")
s = d.groupby("cls", group_keys=False).apply(lambda g: g.sample(min(len(g), 20), random_state=7))
s = s[["uid","model","cls","context","strict_set","relaxed_set","response"]].copy()
s["response"] = s["response"].str.slice(0, 1500); s["true_sources"] = ""   # write "none" if the response uses no source
s.to_csv(f"{OUT_DIR}/manual_check.csv", index=False); print("wrote", len(s), "rows to manual_check.csv")

# Part 2: after you filled true_sources, run this to get detector precision and recall.
def score_detectors(path):
    m = pd.read_csv(path).dropna(subset=["true_sources"])
    tp = {"strict": 0, "relaxed": 0}; fp = dict(tp); fn = dict(tp)
    for _, r in m.iterrows():
        truth = {x.strip() for x in str(r.true_sources).split(",") if x.strip() and x.strip().lower() != "none"}
        for name, col in (("strict", "strict_set"), ("relaxed", "relaxed_set")):
            det = set(json.loads(r[col]))
            tp[name] += len(det & truth); fp[name] += len(det - truth); fn[name] += len(truth - det)
    for name in ("strict", "relaxed"):
        p = tp[name] / max(1, tp[name] + fp[name]); rc = tp[name] / max(1, tp[name] + fn[name])
        print(f"{name:8s} precision {p:.3f}  recall {rc:.3f}  (TP {tp[name]}, FP {fp[name]}, FN {fn[name]}) over {len(m)} outputs")
# score_detectors(f"{OUT_DIR}/manual_check.csv")      # uncomment after filling the column
