"""Stage 3: compare MOO algorithms tuning DP-stcm.

Decision variables: eps (0.05-10, log scale), rho, p.  Objectives: f1-f4.

"""
import argparse
import json
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
for _d in ("stage2_dp_stcm", "common"):                                    # shared modules
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import dp_stcm as D                      # noqa: E402
import homophily_benchmark as hb         # noqa: E402

OUT = os.environ.get("MOO_OUT", "moo_results")
DATASETS = ["cora", "polblogs", "chameleon", "texas", "github"]
ALGOS = ["nsga2", "moead", "smsemoa", "mopso", "nsga3", "rvea", "agemoea", "parego", "qnehvi", "random"]
ALGO_NAME = {"nsga2": "NSGA-II", "moead": "MOEA/D", "smsemoa": "SMS-EMOA", "mopso": "MOPSO", "nsga3": "NSGA-III",
             "rvea": "RVEA", "agemoea": "AGE-MOEA", "parego": "ParEGO", "qnehvi": "qNEHVI", "random": "Random search"}
OBJ = ["f1_homophily", "f2_motif", "f3_structure", "f4_privacy"]
METRIC_COLS = ["edge_h", "newman_r", "mix_dist", "clustering", "hom_tri", "deg_ks", "deg_gini", "aspl", "eor", "eor_lift"]
EPS_LO, EPS_HI = np.log10(0.05), 1.0
POP = 20
N_VAR, N_OBJ = 3, 4
B1000_ALGOS = [a for a in ALGOS if a != "qnehvi"]
REEVAL_SEEDS = 3
REEVAL_CAP = 30
STAGE2_EPS = {"cora": 5.0, "polblogs": 0.1, "chameleon": 0.25, "texas": 1.0, "github": 0.1}


# the optimisation problem

def decode(x):
    x = np.clip(np.asarray(x, float), 0, 1)
    return float(10 ** (EPS_LO + x[0] * (EPS_HI - EPS_LO))), float(0.05 + 0.90 * x[1]), float(x[2])


def encode(eps, rho, p):
    return np.array([(np.log10(eps) - EPS_LO) / (EPS_HI - EPS_LO), (rho - 0.05) / 0.90, p])


_DATA = {}


def get_data(ds):
    """Real graph + reference metrics, cached on disk (GitHub's stratified sample takes ~1 min)."""
    if ds not in _DATA:
        path = os.path.join(OUT, "_cache", f"{ds}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as fh:
                _DATA[ds] = pickle.load(fh)
        else:
            G, y, k = D.load_ref(ds)
            d = dict(G=G, y=y, k=k, ref=hb.reference_metrics(G))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + ".tmp", "wb") as fh:
                pickle.dump(d, fh)
            os.replace(path + ".tmp", path)
            _DATA[ds] = d
    return _DATA[ds]


def score(ds, x, seed):
    """One evaluation: private release with settings x, then the four objectives against the real graph."""
    d = get_data(ds)
    eps, rho, p = decode(x)
    Gs, _, _, _, _ = D.release(d["G"], d["y"], d["k"], eps, rho, seed=int(seed), p=p)
    Gk = hb.preprocess_keep(Gs)
    for v in Gk:
        Gk.nodes[v]["y"] = int(d["y"][v])
    m = hb.compute_metrics(Gk, d["G"])
    o = hb.objectives(m, d["ref"])
    out = {c: float(m[c]) for c in METRIC_COLS}
    out.update({c: float(o[c]) for c in OBJ})
    return out


class Evaluator:
    """Counts evaluations, fixes the noise seed per evaluation index and logs everything."""

    def __init__(self, ds, run, budget):
        self.ds, self.run, self.budget = ds, run, budget
        self.rows, self.t_eval = [], 0.0

    def __call__(self, x):
        i = len(self.rows)
        t = time.time()
        r = score(self.ds, x, seed=100000 * (self.run + 1) + i)
        dt = time.time() - t
        self.t_eval += dt
        eps, rho, p = decode(x)
        r.update(eval=i, x0=float(x[0]), x1=float(x[1]), x2=float(x[2]), eps=eps, rho=rho, p=p, eval_time=dt)
        self.rows.append(r)
        return np.array([r[c] for c in OBJ])

    def batch(self, X):
        return np.array([self(x) for x in np.atleast_2d(X)])

    @property
    def n(self):
        return len(self.rows)

    @property
    def left(self):
        return self.budget - len(self.rows)


def initial_design(run, n=POP):
    return np.random.default_rng(50_000 + run).random((n, N_VAR))


def lhs(n, d, rng):
    u = (rng.permuted(np.tile(np.arange(n), (d, 1)), axis=1).T + rng.random((n, d))) / n
    return u


# algorithms

def run_pymoo(name, ev, run):
    from pymoo.algorithms.moo.age import AGEMOEA
    from pymoo.algorithms.moo.moead import MOEAD
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.algorithms.moo.rvea import RVEA
    from pymoo.algorithms.moo.sms import SMSEMOA
    from pymoo.core.problem import Problem
    from pymoo.decomposition.tchebicheff import Tchebicheff
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs import get_reference_directions

    class P(Problem):
        def __init__(self):
            super().__init__(n_var=N_VAR, n_obj=N_OBJ, xl=0.0, xu=1.0)

        def _evaluate(self, X, out, *a, **k):
            out["F"] = ev.batch(X)

    X0 = initial_design(run)
    ref = get_reference_directions("das-dennis", N_OBJ, n_partitions=3)       # 20 directions for 4 objectives
    ga = dict(crossover=SBX(prob=0.9, eta=20), mutation=PM(eta=20))           # NSGA-II / MOEA/D / SMS-EMOA papers
    ma = dict(crossover=SBX(prob=1.0, eta=30), mutation=PM(eta=20))           # NSGA-III / RVEA / AGE-MOEA papers
    if name == "nsga2":
        alg = NSGA2(pop_size=POP, sampling=X0, **ga)
    elif name == "moead":
        alg = MOEAD(ref_dirs=ref, n_neighbors=5, decomposition=Tchebicheff(), prob_neighbor_mating=0.9,
                    sampling=X0, **ga)
    elif name == "smsemoa":    # steady state (mu + 1) as in the paper
        alg = SMSEMOA(pop_size=POP, n_offsprings=1, sampling=X0, **ga)
    elif name == "nsga3":
        alg = NSGA3(ref_dirs=ref, pop_size=POP, sampling=X0, **ma)
    elif name == "rvea":       # alpha = 2, fr = 0.1 (paper)
        alg = RVEA(ref_dirs=ref, pop_size=POP, alpha=2.0, adapt_freq=0.1, sampling=X0, **ma)
    elif name == "agemoea":
        alg = AGEMOEA(pop_size=POP, sampling=X0, **ma)
    else:
        raise ValueError(name)
    # RVEA needs a generation limit (pymoo's n_evals conversion drops one generation)
    term = ("n_gen", int(np.ceil(ev.budget / POP))) if name == "rvea" else ("n_evals", ev.budget)
    minimize(P(), alg, termination=term, seed=run, verbose=False)


def run_random(ev, run):
    X = np.vstack([initial_design(run), np.random.default_rng(60_000 + run).random((ev.budget, N_VAR))])
    for x in X[:ev.budget]:
        ev(x)


def _dominates(a, b):
    return np.all(a <= b) and np.any(a < b)


def nondominated_mask(F):
    F = np.asarray(F)
    n = len(F)
    keep = np.ones(n, bool)
    for i in range(n):
        if keep[i]:
            dom = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
            if dom.any():
                keep[i] = False
    return keep


def run_mopso(ev, run, rep_size=100, ndiv=30, w=0.4, mut_rate=0.5, grid_x=10.0):
    """MOPSO (Coello Coello et al. 2004): repository, adaptive grid and mutation."""
    rng = np.random.default_rng(70_000 + run)
    X = initial_design(run).copy()
    V = np.zeros_like(X)
    F = ev.batch(X)
    PB, PF = X.copy(), F.copy()
    RX, RF = X[nondominated_mask(F)], F[nondominated_mask(F)]
    max_it = int(np.ceil((ev.budget - POP) / POP))
    it = 0

    def hypercube(RF):
        lo, hi = RF.min(0), RF.max(0)
        span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
        lo, span = lo - 0.05 * span, span * 1.1                     # adaptive grid with a small margin
        cell = np.clip(((RF - lo) / span * ndiv).astype(int), 0, ndiv - 1)
        keys = [tuple(c) for c in cell]
        return keys

    def update_repository(RX, RF, X, F):
        AX, AF = np.vstack([RX, X]), np.vstack([RF, F])
        _, idx = np.unique(np.round(AX, 12), axis=0, return_index=True)
        AX, AF = AX[np.sort(idx)], AF[np.sort(idx)]
        m = nondominated_mask(AF)
        AX, AF = AX[m], AF[m]
        while len(AX) > rep_size:                                   # remove from the most crowded hypercube
            keys = hypercube(AF)
            cnt = pd.Series(keys).value_counts()
            crowded = cnt.index[0]
            cand = [i for i, k in enumerate(keys) if k == crowded]
            j = rng.choice(cand)
            AX, AF = np.delete(AX, j, 0), np.delete(AF, j, 0)
        return AX, AF

    RX, RF = update_repository(RX[:0], RF[:0], RX, RF)
    while ev.left > 0:
        keys = hypercube(RF)
        cnt = pd.Series(keys).value_counts()
        cubes = list(cnt.index)
        fit = np.array([grid_x / cnt[c] for c in cubes])
        n_move = min(POP, ev.left)
        newX = np.empty((n_move, N_VAR))
        for i in range(n_move):
            cube = cubes[rng.choice(len(cubes), p=fit / fit.sum())]
            members = [j for j, k in enumerate(keys) if k == cube]
            leader = RX[rng.choice(members)]
            r1, r2 = rng.random(N_VAR), rng.random(N_VAR)
            V[i] = w * V[i] + r1 * (PB[i] - X[i]) + r2 * (leader - X[i])
            x = X[i] + V[i]
            out = (x < 0) | (x > 1)
            V[i][out] *= -1                                         # bounce back from the boundary
            x = np.clip(x, 0, 1)
            # mutation that shrinks over the iterations, as in the paper
            pm = (1 - it / max(max_it, 1)) ** (5.0 / mut_rate)
            if rng.random() < pm:
                dim = rng.integers(N_VAR)
                rg = 1.0 * pm
                x[dim] = rng.uniform(max(0, x[dim] - rg), min(1, x[dim] + rg))
            newX[i] = x
        newF = ev.batch(newX)
        X[:n_move] = newX
        for i in range(n_move):
            if _dominates(newF[i], PF[i]) or (not _dominates(PF[i], newF[i]) and rng.random() < 0.5):
                PB[i], PF[i] = newX[i], newF[i]
        RX, RF = update_repository(RX, RF, newX, newF)
        it += 1


def _torch():
    import torch
    torch.set_num_threads(1)
    return torch


def run_parego(ev, run, s=4, rho_aug=0.05):
    torch = _torch()
    from botorch.acquisition import LogExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from pymoo.util.ref_dirs import get_reference_directions

    rng = np.random.default_rng(80_000 + run)
    torch.manual_seed(80_000 + run)
    lam_set = get_reference_directions("das-dennis", N_OBJ, n_partitions=s)
    n0 = min(11 * N_VAR - 1, ev.budget)
    X = lhs(n0, N_VAR, rng)
    F = ev.batch(X)
    bounds = torch.tensor([[0.0] * N_VAR, [1.0] * N_VAR], dtype=torch.double)
    while ev.left > 0:
        lo, hi = F.min(0), F.max(0)
        Fn = (F - lo) / np.where(hi - lo > 1e-12, hi - lo, 1.0)
        lam = lam_set[rng.integers(len(lam_set))]
        g = np.max(Fn * lam, axis=1) + rho_aug * np.sum(Fn * lam, axis=1)
        tx = torch.tensor(X, dtype=torch.double)
        ty = torch.tensor(-g, dtype=torch.double).unsqueeze(-1)
        try:
            gp = SingleTaskGP(tx, ty, outcome_transform=Standardize(m=1))
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
            acq = LogExpectedImprovement(gp, best_f=ty.max())
            cand, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=10, raw_samples=256)
            x = cand.detach().numpy()[0]
        except Exception:                                           # numerical failure -> random point
            x = rng.random(N_VAR)
        X = np.vstack([X, x])
        F = np.vstack([F, ev(x)])


def run_qnehvi(ev, run, mc=32, restarts=5, raw=128):
    torch = _torch()
    from botorch.acquisition.multi_objective import qNoisyExpectedHypervolumeImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import ModelListGP, SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim import optimize_acqf
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.utils.sampling import draw_sobol_samples
    from gpytorch.mlls import SumMarginalLogLikelihood

    torch.manual_seed(90_000 + run)
    bounds = torch.tensor([[0.0] * N_VAR, [1.0] * N_VAR], dtype=torch.double)
    n0 = min(2 * (N_VAR + 1), ev.budget)
    X = draw_sobol_samples(bounds=bounds, n=n0, q=1, seed=90_000 + run).squeeze(1).numpy()
    F = ev.batch(X)
    ref_point = torch.tensor([-1.2] * N_OBJ, dtype=torch.double)     # objectives lie in [0, 1.15]
    rng = np.random.default_rng(90_000 + run)
    while ev.left > 0:
        tx = torch.tensor(X, dtype=torch.double)
        ty = torch.tensor(-F, dtype=torch.double)                    # BoTorch maximises
        try:
            models = [SingleTaskGP(tx, ty[:, i:i + 1], outcome_transform=Standardize(m=1)) for i in range(N_OBJ)]
            model = ModelListGP(*models)
            fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
            acq = qNoisyExpectedHypervolumeImprovement(
                model=model, ref_point=ref_point, X_baseline=tx, prune_baseline=True,
                sampler=SobolQMCNormalSampler(sample_shape=torch.Size([mc])))
            cand, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=restarts, raw_samples=raw,
                                    options={"batch_limit": 5, "maxiter": 100})
            x = cand.detach().numpy()[0]
        except Exception:
            x = rng.random(N_VAR)
        X = np.vstack([X, x])
        F = np.vstack([F, ev(x)])


def run_algorithm(name, ev, run):
    if name in ("nsga2", "moead", "smsemoa", "nsga3", "rvea", "agemoea"):
        run_pymoo(name, ev, run)
    elif name == "mopso":
        run_mopso(ev, run)
    elif name == "parego":
        run_parego(ev, run)
    elif name == "qnehvi":
        run_qnehvi(ev, run)
    elif name == "random":
        run_random(ev, run)
    else:
        raise ValueError(name)


# runner

def maxmin_subset(F, k):
    """Greedy max-min (farthest point) subset in normalised objective space; same rule for every method."""
    if len(F) <= k:
        return np.arange(len(F))
    lo, hi = F.min(0), F.max(0)
    Z = (F - lo) / np.where(hi - lo > 1e-12, hi - lo, 1.0)
    chosen = [int(np.argmin(Z.sum(1)))]
    d = np.linalg.norm(Z - Z[chosen[0]], axis=1)
    while len(chosen) < k:
        j = int(np.argmax(d))
        chosen.append(j)
        d = np.minimum(d, np.linalg.norm(Z - Z[j], axis=1))
    return np.array(sorted(chosen))


def job_paths(tag, ds, alg, run):
    base = os.path.join(OUT, tag, ds, alg)
    return os.path.join(base, f"run{run}_evals.csv"), os.path.join(base, f"run{run}_front.csv"), \
        os.path.join(base, f"run{run}_info.json")


def run_job(args):
    tag, ds, alg, run, budget = args
    pe, pf, pi = job_paths(tag, ds, alg, run)
    if os.path.exists(pi):
        return ds, alg, run, "skip", 0.0
    os.makedirs(os.path.dirname(pe), exist_ok=True)
    get_data(ds)
    t0 = time.time()
    ev = Evaluator(ds, run, budget)
    err = ""
    try:
        run_algorithm(alg, ev, run)
    except Exception as e:                                          # keep what was evaluated, record the error
        err = repr(e)
    t_search = time.time() - t0
    ev_df = pd.DataFrame(ev.rows)
    ev_df = ev_df[ev_df["eval"] < budget]                           # hard budget cap (identical for all)
    # result = non-dominated set of everything evaluated, re-scored on fresh common seeds
    F = ev_df[OBJ].to_numpy()
    nd = np.where(nondominated_mask(F))[0]
    nd = nd[maxmin_subset(F[nd], REEVAL_CAP)]
    t1 = time.time()
    rows = []
    for i in nd:
        x = ev_df.iloc[i][["x0", "x1", "x2"]].to_numpy(float)
        reps = [score(ds, x, seed=9_000_000 + 1000 * run + j) for j in range(REEVAL_SEEDS)]
        r = {c: float(np.mean([q[c] for q in reps])) for c in OBJ + METRIC_COLS}
        r.update({c + "_sd": float(np.std([q[c] for q in reps])) for c in OBJ})
        r.update({c + "_inrun": float(ev_df.iloc[i][c]) for c in OBJ})
        r.update(eval=int(ev_df.iloc[i]["eval"]), x0=x[0], x1=x[1], x2=x[2],
                 eps=ev_df.iloc[i]["eps"], rho=ev_df.iloc[i]["rho"], p=ev_df.iloc[i]["p"])
        rows.append(r)
    t_reeval = time.time() - t1
    ev_df.to_csv(pe, index=False)
    pd.DataFrame(rows).to_csv(pf, index=False)
    info = dict(dataset=ds, algo=alg, run=run, budget=budget, n_evals=int(len(ev_df)),
                t_search=t_search, t_eval=float(ev_df["eval_time"].sum()),
                t_overhead=t_search - float(ev_df["eval_time"].sum()), t_reeval=t_reeval,
                n_front=len(rows), error=err, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(pi + ".tmp", "w") as fh:
        json.dump(info, fh, indent=1)
    os.replace(pi + ".tmp", pi)
    return ds, alg, run, ("ERROR " + err if err else "ok"), time.time() - t0


SIZE_ORDER = {"github": 0, "chameleon": 1, "polblogs": 2, "cora": 3, "texas": 4}


def cmd_run(datasets, algos, runs, budget, workers, tag, small_first=False):
    for ds in datasets:                      # build the cache once, before forking workers
        get_data(ds)
    jobs = [(tag, ds, a, r, budget) for ds in sorted(datasets, key=lambda d: SIZE_ORDER.get(d, 9), reverse=small_first)
            for r in range(runs) for a in algos]
    todo = [j for j in jobs if not os.path.exists(job_paths(*j[:4])[2])]
    print(f"[{tag}] {len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run, "
          f"{workers} worker(s)", flush=True)
    t0 = time.time()
    done = 0
    log = open(os.path.join(OUT, f"run_{tag}.log"), "a", encoding="utf-8")
    if workers > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(workers, maxtasksperchild=20) as pool:
            for res in pool.imap_unordered(run_job, todo):
                done += 1
                _report(res, done, len(todo), t0, log)
    else:
        for j in todo:
            done += 1
            _report(run_job(j), done, len(todo), t0, log)
    log.close()


def _report(res, done, total, t0, log):
    ds, alg, run, status, dt = res
    el = time.time() - t0
    eta = el / done * (total - done)
    msg = (f"{time.strftime('%H:%M:%S')}  [{done}/{total}] {ds:9s} {alg:8s} run {run}: {status} "
           f"({dt:.0f}s)  elapsed {el / 3600:.2f} h, ETA {eta / 3600:.2f} h")
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()


def cmd_status(tag):
    base = os.path.join(OUT, tag)
    if not os.path.exists(base):
        print("nothing yet")
        return
    rows = []
    for ds in sorted(os.listdir(base)):
        for a in sorted(os.listdir(os.path.join(base, ds))):
            fs = [f for f in os.listdir(os.path.join(base, ds, a)) if f.endswith("_info.json")]
            errs = 0
            for f in fs:
                with open(os.path.join(base, ds, a, f)) as fh:
                    errs += bool(json.load(fh)["error"])
            rows.append(dict(dataset=ds, algo=a, runs_done=len(fs), errors=errs))
    print(pd.DataFrame(rows).pivot(index="algo", columns="dataset", values="runs_done").fillna(0).astype(int))
    tot = sum(r["errors"] for r in rows)
    print(f"runs with errors: {tot}")


# main
if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "analyze", "audit", "status", "all"])
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--algos", nargs="+", default=ALGOS)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--small-first", action="store_true", help="run small datasets first (quick checks)")
    a = ap.parse_args()
    OUT = a.out
    os.environ["MOO_OUT"] = OUT
    os.makedirs(OUT, exist_ok=True)
    if a.cmd == "run":
        cmd_run(a.datasets, a.algos, a.runs, a.budget, a.workers, a.tag, a.small_first)
    elif a.cmd == "status":
        cmd_status(a.tag)
    elif a.cmd == "analyze":
        import moo_analysis as MA
        MA.analyze(OUT, a.tag)
    elif a.cmd == "audit":
        import moo_analysis as MA
        MA.audit(OUT, a.tag, a.workers)
    elif a.cmd == "all":
        import moo_analysis as MA
        cmd_run(DATASETS, ALGOS, 10, 300, a.workers, "main")
        cmd_run(["texas", "cora"], B1000_ALGOS, 10, 1000, a.workers, "b1000")
        for t in ("main", "b1000"):
            MA.analyze(OUT, t)
            MA.audit(OUT, t, a.workers)
