# GhostPermit: context scoping for joint-use release authorization

Code, notebooks, synthetic data, and raw results for the manuscript "GhostPermit: Context Scoping for Joint-Use Release Authorization in Multi-Source AI Assistants" (in preparation).

AI assistants that read calendars, contacts, and email can leak by combining sources that are each authorized. GhostPermit is a release monitor that bounds the sources behind each generated answer from above and below, decides exactly what the two bounds settle (the interval rule), and tightens the upper bound by context scoping: a router that reads only the request chooses the sources the model sees.

## Key results

All records are synthetic. Models are open-weight (2 to 14 billion parameters).

| Result | Value |
|---|---|
| Union-label denials with no strict evidence of a forbidden join | 67.6% (365 of 540) |
| Forbidden joins under an injected instruction, five models (full context / prompt defense / scoping) | 33.8% / 22.3% / 0.2% (640 answers per arm) |
| Second campaign, small models: best prompt-level defense (combined) | 13.0% of answers with a join (384 per arm) |
| Second campaign: scoped answers with a join under injection | 0 of 1,269 |
| Monitor against a policy-derived oracle | 0 mismatches in 864,000 random-policy comparisons; 6 of 6 seeded faults detected |

## Repository layout

```text
.
|-- README.md
|-- REPRODUCING.md          step-by-step reproduction guide
|-- CITATION.cff
|-- requirements.txt        CPU analysis dependencies
|-- requirements-gpu.txt    extra dependencies for the GPU experiments
|-- src/                    experiment code, in notebook-cell order
|-- scripts/                data utilities
|-- notebooks/              Kaggle notebooks, in run order
|-- data/
|   |-- raw/                master results CSV, final report, result bundle
|   `-- tests/              per-test CSVs split from the master CSV
|-- paper/                  manuscript source, bibliography, highlights
|-- figures/                paper figures
`-- docs/                   map from paper numbers to data
```

## Reproducing the results

See REPRODUCING.md. The analysis runs on a CPU in about a minute; the GPU experiments take about 10.5 hours on Kaggle with two T4 GPUs and are resumable.

## Data and ethics

Every record is synthetic. The repository contains no personal data. Model outputs are stored as produced.

## Status and license

The manuscript is in preparation. A license has not been chosen yet; until one is added, all rights are reserved.

## Citation

See CITATION.cff.
