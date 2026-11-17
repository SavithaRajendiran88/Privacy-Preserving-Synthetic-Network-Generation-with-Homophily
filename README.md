# Privacy-Preserving-Synthetic-Network-Generation-with-Homophily
A multi-objective evolutionary approach to generating differentially private synthetic networks that keep homophily and motif structure.

This repository contains the code for my final year project. It generates synthetic versions of a real network with group labels (for example, papers labelled by topic or blogs labelled by political leaning). The synthetic network should:

keep homophily, meaning how often nodes link to nodes of the same group,
keep motifs (triangles and clustering) and the overall structure (degree distribution, path lengths),
leak as little as possible about the real edges, with a formal edge-level ε-differential privacy guarantee.

These goals conflict, so the generator's settings are tuned by multi-objective optimisation (MOO). Several MOO algorithms are compared, and the best one is used to pick the final settings.

Real network G + group labels y
        │
        ▼
Statistics: B (edges between each pair of groups), κ (each node's links to each group)
        │
        │   ┌──────────── MOO loop (300 evaluations per run) ────────────┐
        │   │ optimizer proposes θ = (ρ, p, w)                           │
        └──►│ DP: Laplace noise on B and κ (total ε = 1, split by ρ)     │
            │ post-processing: clip, smooth with w, rescale and round    │
            │ stcm: build a synthetic network, closing triangles with p  │
            │ score it against the real network → (f1, f2, f3, f4)       │
            └────────────────────────────────────────────────────────────┘
        ▼
Pareto front of θ → balanced setting → final ε = 1 private synthetic network
Repository layout
Folder	What it contains
common/	Shared modules: homophily_benchmark.py (the 10 metrics, objectives f1–f4, and the 9 generators) and dataset_loaders.py (downloads, caching and the GitHub subsample).
stage1_generators/	Stage 1: comparison of homophily-aware network generators (run_study.py).
stage2_dp_stcm/	Stage 2: edge-level differential privacy added to the stratified configuration model (DP-STCM): ε sweep, ρ sweep, reports and sample graphs.
stage3_moo_comparison/	Stage 3: comparison of 10 MOO algorithms tuning DP-STCM over (ε, ρ, p). It has its own README.
final_pipeline/	Final pipeline: fyp_pipeline.py, one self-contained file with ε = 1 fixed, θ = (ρ, p, w), 6 optimizers, analysis, final networks and figures.

Run every command from the repository root. All stages then share the same data/ cache, and each stage writes its results to its own folder (study/, dp_results/, moo_results/, fyp_results/).

Installation

Tested with Python 3.13; Python 3.10 or newer should work.

bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

Stage 3 also needs PyTorch, BoTorch and numba:

bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-stage3.txt
Datasets

The datasets are downloaded automatically on first use and cached in data/<name>/. If a download fails, the error message gives the URL and the file name; download it yourself and put it in that folder.

Dataset	Nodes	Groups	Character	Source
Cora	2,708	7 topics	citation network, homophilous	Planetoid (kimiyoung/planetoid)
PolBlogs	1,490	2 (liberal / conservative)	strongly homophilous	danielzuegner/gnn-meta-attack
Chameleon	2,277	5 traffic quintiles	heterophilous	MUSAE (benedekrozemberczki)
Texas	183	5 page types	small, very heterophilous	Geom-GCN (WebKB)
GitHub	37,700 → 5,000 sample	2 (ML / web developers)	large, heavy-tailed	MUSAE (benedekrozemberczki)
SocioPatterns*	~330	9 classes	face-to-face contacts	sociopatterns.org

*Stage 1 only. GitHub is reduced to a 5,000-node class-stratified forest-fire sample, the best of 8 by closeness of class shares, edge homophily, clustering and degree Gini. Every graph is made undirected and simple, and only its largest connected component is kept.

bash
python common/dataset_loaders.py --all     # download everything and print a check table
How to run
Stage 1 – generator comparison

Generators: ER and configuration model (null baselines), BA-Homophily, DPAH, PATCH, STCM, GenCAT, SBM with triadic closure, and the canonical Poisson ensemble.

bash
python stage1_generators/run_study.py \
    --datasets cora polblogs chameleon texas sociopatterns github \
    --models er cm bah dpah patch stcm gencat sbm_tc canonical \
    --reps 5 --out study
python stage1_generators/run_study.py --report-only --out study   # rebuild the tables and figures only

The run is resumable and has a time limit per (dataset, model) pair (--timeout, default 900 s). Output: study/report.md, metric, objective, rank and Pareto tables, and figures.

Stage 2 – differentially private STCM
bash
python stage2_dp_stcm/dp_stcm.py verify                                        # sensitivity and noise checks
python stage2_dp_stcm/dp_stcm.py sweep  --reps 5                               # ε ∈ {0.05 … 10, ∞}
python stage2_dp_stcm/dp_stcm.py rho    --datasets texas cora polblogs --reps 3
python stage2_dp_stcm/dp_stcm.py report                                        # tables, figures, dp_report.md
python stage2_dp_stcm/dp_stcm.py samples                                       # original vs synthetic drawings
python stage2_dp_stcm/build_dp_pdf.py                                          # Stage 2 PDF
Stage 3 – MOO algorithm comparison

Algorithms: NSGA-II, MOEA/D, SMS-EMOA, MOPSO, NSGA-III, RVEA, AGE-MOEA, ParEGO, qNEHVI and random search, all with population 20. Decision variables: ε (0.05–10, log scale), ρ and p.

bash
python stage3_moo_comparison/moo_study.py run --datasets texas cora --runs 2 --budget 40 --tag check   # quick check
python stage3_moo_comparison/moo_study.py all --workers N                                             # full study

Full instructions and expected run times are in stage3_moo_comparison/README.md and RUN_TIME_ESTIMATE.md. The full study is about 175 CPU-core hours, roughly 75 of them qNEHVI's own computing.

Final pipeline
bash
python final_pipeline/fyp_pipeline.py check --workers N          # 10–20 min smoke test (Texas + Cora)
python final_pipeline/fyp_pipeline.py all   --workers N --runs 5 # full study, as reported
python final_pipeline/fyp_pipeline.py status                     # progress

Each step can also be run on its own: data | verify | run | analyze | final | figures | pack. All of them are resumable. Use N = number of CPU cores minus 1. Results go to fyp_results/ (all evaluations, Pareto fronts, indicators, statistics, final networks with edge lists, and figures).

The default is --runs 10. The reported results use 5 runs × 300 evaluations per optimizer and dataset: 150 runs and 45,000 evaluations in total.

Method summary

Generator. Stratified configuration model (STCM): each node needs κ[v, a] links to group a. Nodes are wired in descending degree order. An optional clustering step joins pairs of a node's neighbours with probability p.

Privacy model. Pure ε-DP (δ = 0) at the edge level: neighbouring graphs differ in one edge, and the node set and labels are public.

Released statistic	L1 sensitivity	Noise
Block counts B	1	Lap(1 / (ρ ε))
Per-node group counts κ	2	Lap(2 / ((1 − ρ) ε))

The total budget is ε by sequential composition. Clipping, smoothing (w), rescaling to the noisy block totals, rounding, generation and triangle closing all use only the noisy counts, so they are post-processing and cost no extra budget. Both sensitivities are checked by brute-force edge removal (verify).

Objectives (all minimised; relative errors clipped to [0, 1]):

	Objective	Built from
f1	homophily error	edge homophily, Newman assortativity, mixing-matrix distance
f2	motif error	global clustering, share of homophilous triangles
f3	structure error	degree KS statistic, degree Gini, average shortest path
f4	leakage	edge-overlap rate after node alignment

Further privacy measures: exact edge recall and precision, edge membership-inference AUC (against a degree-only baseline), and the share of isolated nodes.

Fair comparison of optimizers.

Common random numbers: evaluation i of run r uses the same noise seed for every optimizer.
Population-based methods start from the same initial design.
Each run's result is its non-dominated archive, capped at 30 points and re-scored on 3 fresh noise draws.

Indicators and statistics.

Indicators: hypervolume (primary), IGD+, additive ε-indicator, spacing, coverage, anytime HV, worst-case HV, noise gap and optimizer overhead.
Within each dataset: Mann-Whitney U tests with Holm correction.
Across datasets: Friedman test with Nemenyi critical difference.
The recommended optimizer has the best average HV rank and must also hold on IGD+ and the ε-indicator.
Main results (final pipeline, ε = 1)
Recommended optimizer: ParEGO. Average HV rank 1.60 (critical difference 1.51, Friedman p = 4.9 × 10⁻⁹). It is also best on IGD+ and the ε-indicator. NSGA-II, SMS-EMOA and NSGA-III form a middle group, and MOEA/D ranks below random search.
Compared with the default private setting (ρ = 0.3, p = 0, w = 0), the balanced MOO setting lowers motif error and leakage on every dataset and keeps homophily. Example, Chameleon: f2 0.659 → 0.019 and edge recall 0.152 → 0.034. The cost is higher structure error and more isolated nodes.
Membership-inference AUC on edges is within 0.007 of a degree-only attacker, so the remaining leakage comes almost entirely from degrees.
Limitations
The optimizer scores candidates against the real network, so choosing θ is not itself private. Each released network is ε-DP for its θ.
The balanced ρ sits at the upper bound 0.95, so a wider ρ range is worth testing.
Settings tuned at ε = 1 do not carry over directly to other budgets: at ε = 2, motif error rises.

Acknowledgements

Generator implementations follow the original papers (BA-Homophily, DPAH, PATCH, STCM, GenCAT, SBM/TC, canonical ensemble) and were cross-checked with the authors' code where it is public. Optimizers use pymoo and BoTorch. Datasets belong to their original authors (see the sources above).
