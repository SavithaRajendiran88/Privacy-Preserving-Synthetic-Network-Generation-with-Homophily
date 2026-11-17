# Stage 3: comparison of MOO algorithms

Ten optimisers tune the Stage 2 private generator (DP-stcm with the clustering step on) on the same five datasets.

| | |
|---|---|
| Decision variables | ε (0.05–10, log scale), ρ (0.05–0.95), p clustering probability (0–1) |
| Objectives (minimise) | f1 homophily, f2 motif, f3 structure, f4 leakage (edge-overlap rate) |
| Algorithms | NSGA-II, MOEA/D, SMS-EMOA, MOPSO, NSGA-III, RVEA, AGE-MOEA, ParEGO, qNEHVI, random search |
| Main study | 5 datasets × 10 algorithms × 10 runs × 300 evaluations (tag `main`) |
| Budget check | Texas and Cora × 9 algorithms (no qNEHVI) × 10 runs × 1,000 evaluations (tag `b1000`) |

## Setup

From the repository root:

```bash
pip install -r requirements.txt -r requirements-stage3.txt
```

The scripts import `dp_stcm.py` from `stage2_dp_stcm/` and the shared modules from `common/` on their own. Run them from the repository root so every stage uses the same `data/` folder.

## Quick check (about 15 minutes)

```bash
python stage3_moo_comparison/moo_study.py run --datasets texas cora --runs 2 --budget 40 --tag check
python stage3_moo_comparison/moo_study.py analyze --tag check
python stage3_moo_comparison/moo_study.py status --tag check
```

`status` should show 2 runs for every algorithm and dataset, and `runs with errors: 0`.

## Full study

```bash
python stage3_moo_comparison/moo_study.py all --workers N
```

Set N to the number of CPU cores minus 1. This runs the main study, the 1,000-evaluation check, the analysis and the privacy audit.

The run is resumable: if it stops, run the same command again and finished runs are skipped. To check progress from another terminal:

```bash
python stage3_moo_comparison/moo_study.py status --tag main
```

You can also run it in parts:

```bash
python stage3_moo_comparison/moo_study.py run --tag main --workers N
python stage3_moo_comparison/moo_study.py run --tag b1000 --datasets texas cora --budget 1000 \
    --algos nsga2 moead smsemoa mopso nsga3 rvea agemoea parego random --workers N
python stage3_moo_comparison/moo_study.py analyze --tag main
python stage3_moo_comparison/moo_study.py audit   --tag main --workers N
```

Expected run times are in `RUN_TIME_ESTIMATE.md`. The full study is about 175 CPU-core hours.

## Files

| File | Role |
|---|---|
| `moo_study.py` | the problem, the ten algorithms, and the resumable parallel runner |
| `moo_analysis.py` | indicators (HV, IGD+, ε-indicator, spacing, coverage, anytime HV, worst-case HV, noise gap, overhead), statistical tests, decision rule, privacy audit |

## Output

| Path | Contents |
|---|---|
| `moo_results/<tag>/<dataset>/<algorithm>/run<r>_evals.csv` | every evaluation: settings, objectives, metrics, time |
| `moo_results/<tag>/<dataset>/<algorithm>/run<r>_front.csv` | the run's front, re-scored on fresh noise |
| `moo_results/analysis_<tag>/` | indicators per run, summaries, ranks, verdict, audit |
