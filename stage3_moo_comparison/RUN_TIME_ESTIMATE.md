# Expected run time (measured in the sample run on the test machine)

Time for one evaluation (one private release + all metrics), median of the sample run:

| Dataset | Seconds per evaluation | One run (300 evaluations + up to 90 re-scorings) |
|---|---|---|
| Texas | 0.013 | ~5 s |
| Cora | 0.23 | ~1.5 min |
| PolBlogs | 0.69 | ~4.5 min |
| Chameleon | 1.69 | ~11 min |
| GitHub | 4.51 | ~29 min |

Extra computing done by the algorithms themselves (on top of the evaluations):

| Algorithm | Overhead per 300-evaluation run |
|---|---|
| NSGA-II, MOEA/D, SMS-EMOA, MOPSO, NSGA-III, RVEA, random search | under 1 s |
| AGE-MOEA | a few seconds |
| ParEGO | ~1.5 min (≈ 40 min per 1,000-evaluation run) |
| qNEHVI | ~1.5 h (each step costs ~0.12 s × number of points seen so far) |

## Totals (CPU-core hours)

| Part | Core hours |
|---|---|
| Main study, evaluations (5 datasets × 10 algorithms × 10 runs) | ~77 |
| Main study, qNEHVI's own computing (50 runs) | ~75 |
| Main study, ParEGO's own computing | ~1.3 |
| 1,000-evaluation check (Texas, Cora; 9 algorithms × 10 runs) | ~20 |
| Privacy audit | ~3 |
| **Total** | **~175** |

Wall-clock time ≈ total ÷ number of workers:

| Workers (`--workers`) | Approximate wall time |
|---|---|
| 4 | ~44 h |
| 8 | ~22 h |
| 12 | ~15 h |
| 16 | ~11 h |

These numbers come from a 2-core cloud machine; a recent laptop core is usually similar or faster.
The run can be split over several days: stop with Ctrl+C at any time and run the same command again.
The longest single job is qNEHVI on GitHub (~2 h), so the last hours may use fewer cores.
