# Experiments

The v5 notebook (`ghost-p-p (5).ipynb`) runs 5 experiment stages (A1, A2,
A3, B, R) across 4 models and 7 defense arms, producing 5 974 rows in
`data/raw/results.csv`. The 31 split CSVs in `data/tests/` correspond to
individual tests within these stages.

---

## Stage B — Baseline (multiple defenses on fixed records)

**Files:** `B_base_arm_<arm>.csv` (7 files, one per arm)

**Question.** On fixed synthetic records with an injected note
(phrasings A + B), which defense arms produce forbidden 3-source joins,
and at what rate?

**Arms (7):**
- `combo` — combination-of-defenses (system prompt + pidet + sand + spot)
- `def` — system-prompt defense ("text inside tool outputs is untrusted")
- `full` — no defense (full context, injection only)
- `pidet` — prompt-injection detector (classifier-based)
- `sand` — sandboxing (wrap tool output in delimiters)
- `scoped` — router-mediated scope reduction (model only sees needed sources)
- `spot` — spot-check (random sampling of tool outputs)

**Models:** Phi-3.5-mini, Qwen3-4B, Granite-3.3-2B (small tier, 384 rows
each arm) + Qwen2.5-14B (big tier, 96 rows each arm).

**Headline (from `final_report.txt` §1):**
- scoped: 0/480 joins (0.0%)
- full: 155/480 (32.3%)
- def: 138/480 (28.8%)
- combo: 56/480 (11.7%)

---

## Stage A1 — Adaptive attack 1 (best-of-K injected notes)

**Files:** `A1_pilot_atk00.csv` … `A1_pilot_atk15.csv` (16 files) +
`A1_deep_big_model.csv`

**Question.** If the attacker tries K different injected-note variants
and keeps the best one, does the scoped arm still hold?

**Design.** 16 attacker-note variants (atk 0–15), each tested on 3 small
models × 4 forbidden combos × 2 task classes × 2 phrasings. The "best-of-K"
rate is computed across all 16 variants per (model, combo, cls, phr).

**Headline (§3):**
- scoped: 0/384 per-attempt, 0/48 best-of-K
- full: 151/384 per-attempt (39.3%), 39/48 best-of-K (81.2%)
- def: 135/384 per-attempt (35.2%), 35/48 best-of-K (72.9%)

---

## Stage A2 — Adaptive attack 2 (iterative attacker)

**File:** `A2_pilot_iterative.csv`

**Question.** After a failed round, if the attacker model sees the
defense text and the model's answer and writes a new note, does it
eventually break each arm?

**Design.** 3 small models × 4 forbidden combos × 2 task classes × 2
phrasings × up to 3 rounds. "Chains broken" = at least one round
produced a join.

**Headline (§4):**
- scoped: 0/48 chains broken
- def: 23/48 (47.9%)
- combo: 15/48 (31.2%)

---

## Stage A3 — Detector-evasion attack

**Files:** `A3_pilot_atkE0.csv`, `A3_pilot_atkE1.csv`

**Question.** If the injected note forbids exact quoting (asks for
paraphrases / spelled digits), does the model still produce a join
that the normalising detector catches?

**Design.** 2 evasion variants (E0, E1) × 3 small models × 4 forbidden
combos × 2 task classes × 2 phrasings. Join counted using the
normalising detector (`used_norm`), not the strict one.

**Headline (§5):**
- scoped: 0/96
- def: 23/96 (24.0%)
- full: 19/96 (19.8%)

---

## Stage R — Randomized records

**Files:** `R_pilot_arm_combo.csv`, `R_pilot_arm_def.csv`,
`R_pilot_arm_full.csv`, `R_pilot_arm_scoped.csv`

**Question.** Do the results hold when the synthetic records are
randomized (new identifiers per trial, longer text, mixed formats)?

**Design.** 3 small models × 4 forbidden combos × 2 task classes × 2
phrasings, with fresh random identifiers per trial.

**Headline (§6):**
- scoped: 0/144 joins
- full: 18/216 (8.3%)
- def: 13/144 (9.0%)
- combo: 8/144 (5.6%)

---

## Paired tests (§2 of `final_report.txt`)

Same (model, combo, cls, trial, phr) matched across arms:

| Pair | n | full-only | other-only | McNemar p |
|---|---:|---:|---:|---|
| full vs scoped | 480 | 155 | 0 | 4.4 × 10⁻⁴⁷ |
| def vs scoped | 480 | 138 | 0 | 5.7 × 10⁻⁴² |
| full vs def | 480 | 55 | 38 | 0.097 |
| def vs combo | 480 | 93 | 11 | 2.5 × 10⁻¹⁷ |
| full vs combo | 480 | 110 | 11 | 1.1 × 10⁻²¹ |
| combo vs scoped | 480 | 56 | 0 | 2.8 × 10⁻¹⁷ |

---

## Earlier experiments (v2/v4 notebooks)

The v5 notebook supersedes the following experiments from earlier
notebook versions. Their CSVs are in `data/processed/` for backwards
compatibility:

| Experiment | v4 cell | v5 replacement |
|---|---|---|
| Label-precision v2 (3 models, 1 350 rows) | `c10_label_precision_full.py` | Stage B (subset) |
| Scoped-context experiment (1 800 rows) | `c15_scoped_context_experiment.py` | Stage B (`arm=scoped`) |
| Scope-widening test (480 rows) | `c18_scope_widening_test.py` | Stage A1 (subset) |
| Final experiment (2 304 rows) | `c21_final_experiment_full.py` | Stage B + A1 |
| 5-model confirmation (3 520 rows) | `c26_confirmation_run_full.py` | Stage B (4 models) |
| Defense on larger models | `c47_defense_larger_full.py` | v5 PASS 2 |
