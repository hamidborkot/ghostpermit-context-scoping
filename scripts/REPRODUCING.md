# Reproducing the results

## 0. Prerequisites

```bash
git clone https://github.com/<you>/ghostpermit.git
cd ghostpermit
pip install -r requirements.txt
```

Python 3.10+. CPU track needs `numpy`, `pandas`, `scipy`, `statsmodels`,
`matplotlib`. GPU track additionally needs `torch`, `transformers`,
`accelerate`.

---

## 1. Read the final report (instant)

```bash
cat data/raw/final_report.txt
```

This is the 9-section report produced by the last cell of notebook 5.

---

## 2. Inspect one small per-test CSV (instant)

```bash
# All 31 split CSVs:
ls data/tests/

# One example: the scoped arm baseline (0 joins out of 480)
python -c "
import pandas as pd
d = pd.read_csv('data/tests/B_base_arm_scoped.csv')
print(f'rows: {len(d)}, joins: {d[\"join\"].sum()}, models: {sorted(d.model.unique())}')
"
```

---

## 3. CPU-only statistical reanalysis (≈ 1 minute)

Reproduces every paper number from the cell-count ledger:

```bash
python paper/ghostpermit_reanalysis.py
```

Assertions pin: Protocol A totals [545, 1435], F = 3.455, gate_p = 0.136,
Recovery OR = 4.60. If exit code 0, every number reproduces.

---

## 4. Re-split the master CSV into per-test CSVs (instant)

```bash
python scripts/split_results.py
# reads data/raw/results.csv (5 974 rows)
# writes 31 CSVs to data/tests/
```

---

## 5. Re-running the v5 GPU experiments (multi-hour)

The v5 notebook (`notebooks/ghost_permit.ipynb`) has 71 cells. The
critical cells are:

| Cell | File | What |
|---|---|---|
| 63 | `src/c62_v5_setup.py` | CELL 0: setup, config, GPU check. Set `LITE = True` for a quick run. |
| 64 | `src/c63_v5_records_attacks.py` | CELL 1: records, detectors, 16 attack notes, defense prompts. |
| 65 | `src/c64_v5_engine.py` | CELL 2: model loading, generation, scoring, resume. |
| 66 | `src/c65_v5_pass1_small.py` | CELL 3: PASS 1 — 3 small models, all stages at pilot size (~3 h full / ~1.2 h lite). |
| 67 | `src/c66_v5_gates.py` | CELL 4: statistical gates (McNemar) decide whether to deepen. |
| 68 | `src/c67_v5_pass2_big.py` | CELL 5: PASS 2 — Qwen2.5-14B (4-bit) + 8B on stage B and A1. |
| 69 | `src/c68_v5_pass3_deepen.py` | CELL 6: PASS 3 — deepen small-model runs if gates said GO. |
| 70 | `src/c69_v5_final_report.py` | CELL 7: final report (writes `final_report.txt`). |

### Smoke test

Set `LITE = True` in cell 63, then run cells 63 → 70 in order. This
produces a small `results.csv` (~200 rows) in ~30 minutes.

### Full run

Set `LITE = False` and run the same cells. Wall-clock ≈ 10.5 h on
Kaggle T4 x2. The notebook is resumable: it checks existing `uid`s in
`results.csv` and skips completed trials.

---

## 6. Compiling the LaTeX manuscript

See `paper/README.md` and `paper/GhostPermit_Assembly_Note.md`. The
manuscript uses the Elsevier CAS class (not redistributed). Compile on
Overleaf by uploading `paper/GhostPermit_JointUse_CEE.tex` as the main
document.
