"""Stage 3 analysis: quality indicators, statistical tests, decision rule and privacy audit."""

import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, mannwhitneyu, rankdata

warnings.filterwarnings("ignore")
import moo_study as M  # noqa: E402

REF = 1.1
# Nemenyi critical values q_0.05 (Demsar 2006), index = number of algorithms
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
       11: 3.219, 12: 3.268}
# direction: +1 higher is better, -1 lower is better

METRICS = {"hv": +1, "igd_plus": -1, "eps_add": -1, "spacing": -1, "auc": +1, "noise_gap": -1,
           "overhead_s": -1, "n_front": +1}
LABEL = {"hv": "Hypervolume (% of reference)", "igd_plus": "IGD+", "eps_add": "Additive ε-indicator",
         "spacing": "Spacing", "auc": "Anytime HV (AUC, %)", "noise_gap": "Noise gap (% HV)",
         "overhead_s": "Algorithm overhead (s)", "n_front": "Front size", "worst_hv": "Worst-case HV (%)"}


def hv(F):
    from pymoo.indicators.hv import HV
    F = np.asarray(F, float)
    if len(F) == 0:
        return 0.0
    F = F[np.all(F < REF, axis=1)]
    return float(HV(ref_point=np.full(F.shape[1], REF))(F)) if len(F) else 0.0


def igd_plus(S, R):
    d = np.maximum(S[None, :, :] - R[:, None, :], 0)
    return float(np.sqrt((d ** 2).sum(-1)).min(1).mean())


def eps_add(S, R):
    return float((S[None, :, :] - R[:, None, :]).max(-1).min(1).max())


def spacing(S):
    if len(S) < 3:
        return np.nan
    D = np.abs(S[:, None, :] - S[None, :, :]).sum(-1)
    np.fill_diagonal(D, np.inf)
    d = D.min(1)
    return float(np.sqrt(((d - d.mean()) ** 2).sum() / (len(d) - 1)))


def coverage(A, B):
    """C(A, B): share of B weakly dominated by some point of A."""
    if len(B) == 0:
        return np.nan
    return float(np.mean([np.any(np.all(A <= b, axis=1)) for b in B]))


def nd(F):
    F = np.asarray(F)
    return F[M.nondominated_mask(F)] if len(F) else F


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (len(p) - i) * p[j]))
        adj[j] = run
    return adj


def load_runs(out, tag):
    base = os.path.join(out, tag)
    runs = []
    for ds in sorted(os.listdir(base)):
        for alg in sorted(os.listdir(os.path.join(base, ds))):
            d = os.path.join(base, ds, alg)
            for f in sorted(os.listdir(d)):
                if f.endswith("_info.json"):
                    r = int(f[3:].split("_")[0])
                    with open(os.path.join(d, f)) as fh:
                        info = json.load(fh)
                    front = pd.read_csv(os.path.join(d, f"run{r}_front.csv"))
                    evals = pd.read_csv(os.path.join(d, f"run{r}_evals.csv"))
                    runs.append(dict(dataset=ds, algo=alg, run=r, info=info, front=front, evals=evals))
    return runs


def analyze(out, tag):
    runs = load_runs(out, tag)
    if not runs:
        print("no finished runs")
        return
    A = os.path.join(out, f"analysis_{tag}")
    os.makedirs(A, exist_ok=True)
    O = M.OBJ
    ind, curves, picks, norm_rows, cover_rows, reffronts = [], [], [], [], [], []
    for ds in sorted({r["dataset"] for r in runs}):
        rs = [r for r in runs if r["dataset"] == ds]
        pool = np.vstack([r["front"][O].to_numpy() for r in rs if len(r["front"])])
        ideal = pool.min(0)
        nadir = nd(pool).max(0)
        span = np.where(nadir - ideal > 1e-12, nadir - ideal, 1.0)
        sc = lambda F: (np.asarray(F, float) - ideal) / span   # noqa: E731
        R = nd(sc(pool))
        hvR = hv(R)
        norm_rows.append(dict(dataset=ds, **{f"ideal_{c}": v for c, v in zip(O, ideal)},
                              **{f"nadir_{c}": v for c, v in zip(O, nadir)}, hv_ref_front=hvR, n_ref_front=len(R)))
        rf = pd.DataFrame(R * span + ideal, columns=O)
        rf["dataset"] = ds
        reffronts.append(rf)
        # pooled scale for the compromise choice
        fid_pool = pool[:, :3].mean(1)
        fl, fh = fid_pool.min(), fid_pool.max()
        ll, lh = pool[:, 3].min(), pool[:, 3].max()
        budget = max(r["info"]["budget"] for r in rs)
        checkpoints = np.arange(10, budget + 1, 10)
        for r in rs:
            fr = r["front"]
            S_all = sc(fr[O].to_numpy()) if len(fr) else np.zeros((0, 4))
            S = nd(S_all) if len(S_all) else S_all
            S_in = nd(sc(fr[[c + "_inrun" for c in O]].to_numpy())) if len(fr) else S_all
            ev = r["evals"].sort_values("eval")
            Fin = sc(ev[O].to_numpy())
            hv_t = [hv(nd(Fin[:t])) / hvR * 100 if t <= len(Fin) else np.nan for t in checkpoints]
            for t, h in zip(checkpoints, hv_t):
                curves.append(dict(dataset=ds, algo=r["algo"], run=r["run"], evals=int(t), hv=h))
            info = r["info"]
            ind.append(dict(dataset=ds, algo=r["algo"], run=r["run"],
                            hv=hv(S) / hvR * 100 if len(S) else 0.0,
                            igd_plus=igd_plus(S, R) if len(S) else np.nan,
                            eps_add=eps_add(S, R) if len(S) else np.nan,
                            spacing=spacing(S), n_front=len(S),
                            auc=float(np.nanmean(hv_t)),
                            noise_gap=(hv(S_in) - hv(S)) / hvR * 100 if len(S) else np.nan,
                            overhead_s=info["t_overhead"], search_s=info["t_search"], eval_s=info["t_eval"],
                            n_evals=info["n_evals"], error=info["error"]))
            if len(fr):
                fid = fr[O[:3]].to_numpy().mean(1)
                J = 0.5 * ((fid - fl) / max(fh - fl, 1e-12)) ** 2 + 0.5 * ((fr["f4_privacy"] - ll) / max(lh - ll, 1e-12)) ** 2
                j = int(np.argmin(J.to_numpy()))
                row = fr.iloc[j]
                picks.append(dict(dataset=ds, algo=r["algo"], run=r["run"], J=float(J.iloc[j]), fidelity=float(fid[j]),
                                  **{c: float(row[c]) for c in O + ["eps", "rho", "p", "x0", "x1", "x2"]}))
        # coverage matrix, averaged over run index
        algs = sorted({r["algo"] for r in rs})
        byk = {(r["algo"], r["run"]): nd(sc(r["front"][O].to_numpy())) for r in rs if len(r["front"])}
        for a in algs:
            for b in algs:
                vals = [coverage(byk[(a, k)], byk[(b, k)]) for (aa, k) in byk if aa == a and (b, k) in byk]
                cover_rows.append(dict(dataset=ds, A=a, B=b, C=float(np.nanmean(vals)) if vals else np.nan))
    ind = pd.DataFrame(ind)
    ind.to_csv(os.path.join(A, "indicators_per_run.csv"), index=False)
    pd.DataFrame(curves).to_csv(os.path.join(A, "anytime_curves.csv"), index=False)
    pd.DataFrame(picks).to_csv(os.path.join(A, "compromise_picks.csv"), index=False)
    pd.DataFrame(norm_rows).to_csv(os.path.join(A, "normalisation.csv"), index=False)
    pd.DataFrame(cover_rows).to_csv(os.path.join(A, "coverage.csv"), index=False)
    pd.concat(reffronts).to_csv(os.path.join(A, "reference_fronts.csv"), index=False)
    summary, verdict = statistics(ind, A)
    print(f"[analyze {tag}] {len(ind)} runs analysed -> {A}")
    print(verdict["text"])
    return summary, verdict


def statistics(ind, A):
    rows = []
    for ds, sub in ind.groupby("dataset"):
        for met, sgn in list(METRICS.items()) + [("worst_hv", +1)]:
            if met == "worst_hv":
                agg = sub.groupby("algo")["hv"].min()
                for a, v in agg.items():
                    rows.append(dict(dataset=ds, metric=met, algo=a, mean=v, sd=np.nan, median=v, mark="", p_holm=np.nan))
                continue
            g = sub.groupby("algo")[met]
            means = g.mean()
            best = (means.idxmax() if sgn > 0 else means.idxmin())
            others = [a for a in means.index if a != best]
            pv = []
            for a in others:
                x, y = sub.loc[sub.algo == a, met].dropna(), sub.loc[sub.algo == best, met].dropna()
                pv.append(mannwhitneyu(x, y, alternative="two-sided").pvalue if len(x) > 1 and len(y) > 1 and
                          not (np.allclose(x.to_numpy(), x.iloc[0]) and np.allclose(y.to_numpy(), x.iloc[0])) else 1.0)
            padj = dict(zip(others, holm(pv))) if pv else {}
            for a in means.index:
                mark = "best" if a == best else ("≈" if padj[a] >= 0.05 else "−")
                rows.append(dict(dataset=ds, metric=met, algo=a, mean=means[a], sd=g.std()[a], median=g.median()[a],
                                 mark=mark, p_holm=padj.get(a, np.nan)))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(A, "summary_per_dataset.csv"), index=False)

    # average rank of per-dataset means
    rk = []
    for (ds, met), s in summ.groupby(["dataset", "metric"]):
        sgn = METRICS.get(met, +1)
        r = rankdata(-sgn * s["mean"].to_numpy(), method="average")
        for a, v in zip(s["algo"], r):
            rk.append(dict(dataset=ds, metric=met, algo=a, rank=v))
    rk = pd.DataFrame(rk)
    avg_rank = rk.pivot_table(index="algo", columns="metric", values="rank", aggfunc="mean")
    main5 = ["hv", "igd_plus", "eps_add", "auc", "worst_hv"]
    avg_rank["overall"] = avg_rank[main5].mean(1)
    avg_rank = avg_rank.sort_values("overall")
    avg_rank.to_csv(os.path.join(A, "average_rank_by_dataset.csv"))

    # Friedman + Nemenyi on (dataset, run) blocks
    blk = {}
    algs = sorted(ind["algo"].unique())
    k = len(algs)
    for met in ["hv", "igd_plus", "eps_add", "auc"]:
        sgn = METRICS[met]
        piv = ind.pivot_table(index=["dataset", "run"], columns="algo", values=met).dropna()
        piv = piv[algs]
        R = np.vstack([rankdata(-sgn * row, method="average") for row in piv.to_numpy()])
        N = len(piv)
        cd = Q05.get(k, 3.164) * np.sqrt(k * (k + 1) / (6.0 * N)) if N else np.nan
        try:
            stat, p = friedmanchisquare(*[piv[a].to_numpy() for a in algs]) if N > 1 and k > 2 else (np.nan, np.nan)
        except ValueError:
            stat, p = np.nan, np.nan
        blk[met] = dict(avg_rank=dict(zip(algs, R.mean(0))), n_blocks=int(N), cd=float(cd),
                        friedman_chi2=float(stat), friedman_p=float(p))
    with open(os.path.join(A, "block_ranks.json"), "w") as fh:
        json.dump(blk, fh, indent=1)

    # worst-case HV and overhead ranks across datasets (for tie-breaking)
    w = summ[summ.metric == "worst_hv"].pivot_table(index="dataset", columns="algo", values="mean")
    wr = w.rank(axis=1, ascending=False).mean().to_dict()
    oh = ind.groupby("algo")["overhead_s"].mean().to_dict()

    # decision rule
    hvr = blk["hv"]["avg_rank"]
    cd = blk["hv"]["cd"]
    order = sorted(hvr, key=hvr.get)
    win = order[0]
    tied = [a for a in order if hvr[a] - hvr[win] < cd]
    checks = {}
    for met in ["igd_plus", "eps_add"]:
        r = blk[met]["avg_rank"]
        bestm = min(r, key=r.get)
        checks[met] = dict(best=bestm, winner_rank=r[win], best_rank=r[bestm],
                           within_cd=bool(r[win] - r[bestm] < blk[met]["cd"]))
    if len(tied) > 1:
        aucr = blk["auc"]["avg_rank"]
        rec = sorted(tied, key=lambda a: (round(aucr[a], 6), round(wr.get(a, 99), 6), oh.get(a, 1e9)))[0]
    else:
        rec = win
    name = M.ALGO_NAME
    txt = [f"Best average hypervolume rank: {name[win]} ({hvr[win]:.2f}; critical difference {cd:.2f}, "
           f"{blk['hv']['n_blocks']} dataset-run blocks; Friedman p = {blk['hv']['friedman_p']:.3g})."]
    if len(tied) > 1:
        txt.append("Statistically tied with it on hypervolume: " + ", ".join(name[a] for a in tied[1:]) + ".")
    else:
        txt.append("No other algorithm is within the critical difference on hypervolume.")
    for met, c in checks.items():
        txt.append(f"{LABEL[met]}: best is {name[c['best']]}; {name[win]} is "
                   f"{'within' if c['within_cd'] else 'NOT within'} the critical difference.")
    txt.append(f"Recommended optimiser: {name[rec]}" +
               (" (chosen inside the tied group by anytime HV, then worst-case HV, then overhead)." if rec != win or len(tied) > 1 else "."))
    verdict = dict(winner_hv=win, tied_group=tied, recommended=rec, checks=checks, text="\n".join(txt),
                   hv_avg_rank=hvr, cd=cd, worst_hv_rank=wr, overhead_mean_s=oh)
    with open(os.path.join(A, "verdict.json"), "w") as fh:
        json.dump(verdict, fh, indent=1, default=float)
    return summ, verdict


AUDIT_SEEDS = 3


def _audit_job(args):
    out, ds, label, run, eps, rho, p = args
    import dp_stcm as D
    import homophily_benchmark as hb
    os.environ["MOO_OUT"] = out
    M.OUT = out
    d = M.get_data(ds)
    hr = D.node_homophily(d["G"], d["y"])
    rows = []
    for j in range(AUDIT_SEEDS):
        seed = 7_000_000 + 100 * run + j
        Gs, Bp, Kp, B, K = D.release(d["G"], d["y"], d["k"], eps, rho, seed=seed, p=p)
        m = D.evaluate(d["G"], Gs, d["y"], d["ref"], hr, B, K, Bp, Kp, np.random.default_rng(seed))
        rows.append({c: float(m[c]) for c in M.OBJ + ["recall", "mia_auc_edge", "mia_auc_degree", "isolated_frac",
                                                     "node_hom_err", "edge_h", "newman_r", "clustering"]})
    r = {c: float(np.mean([q[c] for q in rows])) for c in rows[0]}
    r.update(dataset=ds, algo=label, run=run, eps=eps, rho=rho, p=p)
    return r


def audit(out, tag, workers=1):
    A = os.path.join(out, f"analysis_{tag}")
    pk = os.path.join(A, "compromise_picks.csv")
    if not os.path.exists(pk):
        analyze(out, tag)
    picks = pd.read_csv(pk)
    jobs = [(out, r.dataset, r.algo, int(r.run), float(r.eps), float(r.rho), float(r.p)) for r in picks.itertuples()]
    for ds in sorted(picks["dataset"].unique()):
        for run in range(int(picks["run"].max()) + 1):
            jobs.append((out, ds, "stage2_grid", run, M.STAGE2_EPS[ds], 0.3, 0.0))
    fp = os.path.join(A, "audit.csv")
    done = pd.read_csv(fp) if os.path.exists(fp) else pd.DataFrame(columns=["dataset", "algo", "run"])
    have = set(zip(done["dataset"], done["algo"], done["run"].astype(int)))
    todo = [j for j in jobs if (j[1], j[2], j[3]) not in have]
    print(f"[audit {tag}] {len(jobs)} solutions, {len(todo)} to audit", flush=True)
    res = []
    if workers > 1 and todo:
        from multiprocessing import get_context
        with get_context("spawn").Pool(workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_audit_job, todo), 1):
                res.append(r)
                if i % 20 == 0:
                    print(f"   {i}/{len(todo)}", flush=True)
                    pd.concat([done, pd.DataFrame(res)]).to_csv(fp, index=False)
    else:
        for i, j in enumerate(todo, 1):
            res.append(_audit_job(j))
            if i % 20 == 0:
                print(f"   {i}/{len(todo)}", flush=True)
    pd.concat([done, pd.DataFrame(res)], ignore_index=True).to_csv(fp, index=False)
    print(f"[audit {tag}] written {fp}")
