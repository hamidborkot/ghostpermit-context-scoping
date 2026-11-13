# Raw model outputs NOT retained (historical, superseded by v5)

The v5 notebook (`ghost-p-p (5).ipynb`) replaced all of the following
CSVs with a single consolidated `data/raw/results.csv` (5 974 rows).
They are documented here for historical reference.

| Missing file | v4 producer cell | v5 replacement |
|---|---|---|
| `raw_trials.csv` | `c01_main_harness_qwen3_8b.py` | v5 stage B (`arm = combo / def / full / scoped`) |
| `utility_trials.csv` | `c02_utility_baseline.py` | v5 stage B (`arm = combo` = baseline) |
| `agentworm_trials.csv` | `c06_agent_worm.py` | (not in v5; results reported from cell-output logs) |
| `confirm_trials.csv` | `c26_confirmation_run_full.py` | v5 stage B (4 models including Qwen2.5-14B) |
| `defense_larger_trials.csv` | `c47_defense_larger_full.py` | v5 PASS 2 (`c67_v5_pass2_big.py`) |
| `scoped_larger_trials.csv` | `c54_scoped_larger_variant.py` | v5 PASS 2 |

The v5 `results.csv` is the authoritative source for all paper numbers.
The older v2/v4 processed CSVs in `data/processed/` are kept for
backwards compatibility with the earlier analysis cells.
