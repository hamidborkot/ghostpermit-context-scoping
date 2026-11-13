"""CELL C2: 30-item in-notebook detector validation.

Source: ghost-p-p (5).ipynb, code cell 13 of 71.
Output file(s): see the cell body and the README in data/processed/ or data/tests/.
"""

# ==================== CELL C2 (optional, 10 minutes): validate the detectors inside the notebook ====================
# Step 1: run this cell. It prints 30 responses. Read each one and decide which sources it actually uses.
# Step 2: type your answers into TRUE below (same order), then run the cell again to get precision and recall.
import os, json, pandas as pd
OUT_DIR = "/kaggle/working/results_v2" if os.path.exists("/kaggle") else "./results_v2"
d = pd.read_csv(f"{OUT_DIR}/precision_trials.csv")
s = d.groupby("cls", group_keys=False).apply(lambda g: g.sample(min(len(g), 10), random_state=11)).reset_index(drop=True)
for i, r in s.iterrows():
    print(f"--- [{i}] class={r.cls} context={json.loads(r.context)}")
    print("    strict detector :", json.loads(r.strict_set), "| relaxed detector:", json.loads(r.relaxed_set))
    print("    RESPONSE:", str(r.response).replace("\n", " ")[:700])

# Fill one string per response, in order, listing the sources whose information the response uses.
# Use "none" if it uses no source. Example: TRUE = ["calendar,contacts", "none", "email", ...]  (30 entries)
TRUE = []

if len(TRUE) == len(s):
    tp = {"strict": 0, "relaxed": 0}; fp = dict(tp); fn = dict(tp)
    for t, (_, r) in zip(TRUE, s.iterrows()):
        truth = {x.strip() for x in t.split(",") if x.strip() and x.strip().lower() != "none"}
        for name, col in (("strict", "strict_set"), ("relaxed", "relaxed_set")):
            det = set(json.loads(r[col]))
            tp[name] += len(det & truth); fp[name] += len(det - truth); fn[name] += len(truth - det)
    for name in ("strict", "relaxed"):
        p = tp[name] / max(1, tp[name] + fp[name]); rc = tp[name] / max(1, tp[name] + fn[name])
        print(f"{name:8s} precision {p:.3f}  recall {rc:.3f}  (TP {tp[name]}, FP {fp[name]}, FN {fn[name]}) over {len(s)} responses")
else:
    print(f"\nFill TRUE with {len(s)} entries (currently {len(TRUE)}), then run again.")
