"""Stage 1: run every generator on every dataset and compare them """

import argparse
import os
import signal
import sys
import time
import traceback
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
for _d in ("stage2_dp_stcm", "common"):                                    # shared modules
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import homophily_benchmark as hb

warnings.filterwarnings("ignore")

METRICS = ["edge_h", "newman_r", "mix_dist", "clustering", "hom_tri", "deg_ks", "deg_gini",
           "aspl", "eor", "eor_lift", "runtime"]
OBJ = ["f1_homophily", "f2_motif", "f3_structure", "f4_privacy"]
NULLS = ("er", "cm")


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


# sweep

def run_sweep(datasets, models, reps, out, timeout, lcc=True, binarize_mode="ovr"):
    os.makedirs(out, exist_ok=True)
    raw_p, st_p = os.path.join(out, "raw.csv"), os.path.join(out, "status.csv")
    raw = pd.read_csv(raw_p) if os.path.exists(raw_p) else pd.DataFrame()
    status = pd.read_csv(st_p) if os.path.exists(st_p) else pd.DataFrame(
        columns=["dataset", "model", "status", "message"])
    done = set(zip(raw["dataset"], raw["model"])) if len(raw) else set()
    signal.signal(signal.SIGALRM, _alarm)

    def save():
        raw.to_csv(raw_p, index=False)
        status.to_csv(st_p, index=False)

    for ds in datasets:
        print(f"\n=== dataset: {ds}", flush=True)
        G0 = hb.preprocess(hb.load_dataset(ds), lcc=lcc)
        Gbin = hb.binarize(G0, binarize_mode)
        print(f"    n={G0.number_of_nodes()} m={G0.number_of_edges()} classes={hb.labels(G0).max() + 1}"
              f"  (binary view: n={Gbin.number_of_nodes()} m={Gbin.number_of_edges()})", flush=True)
        if (ds, "REAL_BIN") not in done and hb.labels(G0).max() > 1:
            refb = hb.reference_metrics(Gbin)
            raw = pd.concat([raw, pd.DataFrame([dict(dataset=ds, model="REAL_BIN", rep=0, binarized=True,
                                                     n_ref=Gbin.number_of_nodes(), **refb)])], ignore_index=True)
            done.add((ds, "REAL_BIN"))
            save()
        if (ds, "REAL") not in done:
            ref = hb.reference_metrics(G0)
            raw = pd.concat([raw, pd.DataFrame([dict(dataset=ds, model="REAL", rep=0, binarized=False,
                                                     n_ref=G0.number_of_nodes(), **ref)])], ignore_index=True)
            done.add((ds, "REAL"))
            save()
        for name in models:
            if (ds, name) in done:
                continue
            fit, gen, binary_only = hb.MODELS[name]
            ref_g = Gbin if binary_only else G0
            ref_m = hb.reference_metrics(ref_g)
            rows, t0 = [], time.time()
            try:
                signal.alarm(int(timeout))
                params = fit(ref_g)
                fit_t = time.time() - t0
                for r in range(reps):
                    t1 = time.time()
                    Gg = gen(params, seed=r)
                    rt = time.time() - t1
                    m = hb.compute_metrics(hb.preprocess_keep(Gg), ref_g)
                    m.update(hb.objectives(m, ref_m))
                    m.update(dataset=ds, model=name, rep=r, runtime=rt + fit_t / reps,
                             n_ref=ref_g.number_of_nodes(), binarized=bool(binary_only and ref_g is Gbin
                                                                           and hb.labels(G0).max() > 1))
                    rows.append(m)
                signal.alarm(0)
                raw = pd.concat([raw, pd.DataFrame(rows)], ignore_index=True)
                status.loc[len(status)] = [ds, name, "ok", f"{time.time() - t0:.0f}s"]
                print(f"    {name:10s} ok       ({time.time() - t0:.0f}s)", flush=True)
                done.add((ds, name))
            except Timeout:
                signal.alarm(0)
                status.loc[len(status)] = [ds, name, "TIMEOUT", f">{timeout}s"]
                print(f"    {name:10s} TIMEOUT  (> {timeout}s)", flush=True)
            except Exception as e:
                signal.alarm(0)
                status.loc[len(status)] = [ds, name, "FAILED", repr(e)[:200]]
                print(f"    {name:10s} FAILED   {e!r}", flush=True)
                traceback.print_exc()
            save()
    save()
    return raw, status


# analysis helpers

def pareto_flags(M):
    """M: models x objectives (lower is better)."""
    X = M.values
    flags = []
    for i in range(len(X)):
        dom = any(np.all(X[j] <= X[i]) and np.any(X[j] < X[i]) for j in range(len(X)) if j != i)
        flags.append(not dom)
    return pd.Series(flags, index=M.index)


def make_tables(raw, out, models_order):
    gen = raw[~raw.model.isin(["REAL", "REAL_BIN"])].copy()
    gen["fidelity"] = gen[["f1_homophily", "f2_motif", "f3_structure"]].mean(axis=1)
    cols = METRICS + OBJ + ["fidelity"]
    g = gen.groupby(["dataset", "model"])
    mean, std = g[cols].mean(), g[cols].std()
    mean.add_suffix("_mean").join(std.add_suffix("_std")).to_csv(os.path.join(out, "summary_mean_std.csv"))

    # metric table per dataset, real graph first
    for ds in sorted(gen.dataset.unique()):
        ref = raw[(raw.dataset == ds) & raw.model.isin(["REAL", "REAL_BIN"])].set_index("model")[METRICS[:-1]]
        t = pd.concat([ref, mean.xs(ds, level="dataset")[METRICS]])
        t.round(4).to_csv(os.path.join(out, f"table_metrics_{ds}.csv"))

    mean[OBJ + ["fidelity"]].round(4).to_csv(os.path.join(out, "table_objectives.csv"))

    # average rank over datasets (1 = best)
    rk = mean[OBJ + ["fidelity"]].groupby("dataset").rank(method="average")
    rk_avg = rk.groupby("model").mean().round(2)
    rk_avg["n_datasets"] = mean.reset_index().groupby("model").dataset.nunique()
    rk_avg = rk_avg.reindex([m for m in models_order if m in rk_avg.index])
    rk_avg.to_csv(os.path.join(out, "table_ranks.csv"))

    # pareto-optimal models, on all four objectives and on (fidelity, leakage)
    par, par2 = {}, {}
    for ds, sub in mean.groupby(level="dataset"):
        M = sub.droplevel("dataset")
        par[ds] = pareto_flags(M[OBJ].dropna())
        fid = M[["f1_homophily", "f2_motif", "f3_structure"]].mean(axis=1)
        par2[ds] = pareto_flags(M.assign(fid=fid)[["fid", "f4_privacy"]].dropna())
    order = [m for m in models_order if m in set(mean.reset_index().model)]
    pareto = pd.DataFrame(par).reindex(order)
    pareto.to_csv(os.path.join(out, "table_pareto.csv"))
    pd.DataFrame(par2).reindex(order).to_csv(os.path.join(out, "table_pareto_2d.csv"))

    # leakage on top of what the configuration model already gives
    ex = mean["eor"].unstack("model")
    if "cm" in ex.columns:
        ex = ex.sub(ex["cm"], axis=0).drop(columns=["cm"])
    ex.T.reindex([m for m in models_order if m in ex.columns]).round(4).to_csv(
        os.path.join(out, "table_excess_leakage.csv"))
    return mean, std, rk_avg, pareto


# figures

def make_figures(raw, mean, pareto, out, models_order):
    ds_list = sorted(mean.index.get_level_values("dataset").unique())
    mods = [m for m in models_order if m in set(mean.reset_index().model)]
    col = dict(zip(models_order, plt.cm.tab10(np.linspace(0, 1, 10))))

    # objective heat maps
    fig, axes = plt.subplots(1, len(ds_list), figsize=(3.2 * len(ds_list) + 1, 0.55 * len(mods) + 1.6),
                             squeeze=False)
    for ax, ds in zip(axes[0], ds_list):
        M = mean.xs(ds, level="dataset")[OBJ].reindex(mods)
        im = ax.imshow(M.values, cmap="viridis_r", vmin=0, vmax=max(0.8, np.nanmax(M.values)), aspect="auto")
        ax.set_xticks(range(4)); ax.set_xticklabels(["f1\nhomoph.", "f2\nmotif", "f3\nstruct.", "f4\nleak"], fontsize=8)
        ax.set_yticks(range(len(mods))); ax.set_yticklabels(mods if ax is axes[0][0] else [], fontsize=9)
        ax.set_title(ds, fontsize=10)
        for i in range(len(mods)):
            for j in range(4):
                v = M.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if v > 0.4 else "black")
    fig.suptitle("Objectives (lower = better): f1 homophily, f2 motifs, f3 structure, f4 edge-overlap leakage", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_objectives.png"), dpi=150)
    plt.close(fig)

    # fidelity vs leakage, with Pareto fronts
    nc = min(3, len(ds_list)); nr = int(np.ceil(len(ds_list) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.6 * nc, 4.0 * nr), squeeze=False)
    for ax, ds in zip(axes.ravel(), ds_list):
        sub = mean.xs(ds, level="dataset").reindex(mods).dropna(subset=["f4_privacy"])
        sub = sub.assign(fid=sub[["f1_homophily", "f2_motif", "f3_structure"]].mean(axis=1))
        for md, r in sub.iterrows():
            ax.scatter(r["fid"], r["f4_privacy"], s=60, color=col[md], marker="s" if md in NULLS else "o",
                       edgecolor="k", zorder=3)
            ax.annotate(md, (r["fid"], r["f4_privacy"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
        P = sub[["fid", "f4_privacy"]]
        front = P[pareto_flags(P)].sort_values("fid")
        ax.step(front["fid"], front["f4_privacy"], where="post", color="gray", ls="--", lw=1, zorder=1)
        ax.set_xlabel("fidelity error  mean(f1,f2,f3)  (lower = better)")
        ax.set_ylabel("edge-overlap leakage f4")
        ax.set_title(ds, fontsize=10); ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(ds_list):]:
        ax.axis("off")
    fig.suptitle("Fidelity vs privacy leakage (bottom-left is ideal; dashed = Pareto front; squares = null baselines)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_tradeoff.png"), dpi=150)
    plt.close(fig)

    # homophily of each model vs the real graph
    fig, axes = plt.subplots(2, len(ds_list), figsize=(3.2 * len(ds_list) + 0.5, 6), squeeze=False)
    for k, (metric, title) in enumerate([("edge_h", "edge homophily"), ("newman_r", "Newman assortativity")]):
        for ax, ds in zip(axes[k], ds_list):
            g = raw[(raw.dataset == ds) & (~raw.model.isin(["REAL", "REAL_BIN"]))].groupby("model")[metric].agg(["mean", "std"]).reindex(mods)
            ax.bar(range(len(mods)), g["mean"], yerr=g["std"].fillna(0), color=[col[m] for m in mods], capsize=2)
            real = raw[(raw.dataset == ds) & (raw.model == "REAL")][metric]
            if len(real):
                ax.axhline(real.iloc[0], color="k", ls="--", lw=1, label="real")
            rb = raw[(raw.dataset == ds) & (raw.model == "REAL_BIN")][metric]
            if len(rb):
                ax.axhline(rb.iloc[0], color="crimson", ls=":", lw=1.3, label="real, binary view (for bah/dpah/patch)")
            ax.set_xticks(range(len(mods))); ax.set_xticklabels(mods, rotation=70, fontsize=7)
            ax.set_title(f"{ds}: {title}", fontsize=9)
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Homophily reproduced by each model (black dashed = real graph; red dotted = binary view that bah/dpah/patch are scored against)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_homophily.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cora", "polblogs", "chameleon", "texas", "sociopatterns", "github"])
    ap.add_argument("--models", nargs="+", default=["er", "cm", "bah", "dpah", "patch", "stcm", "gencat", "sbm_tc", "canonical"])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="study")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per (dataset, model)")
    ap.add_argument("--tables-only", action="store_true", help="rebuild tables and figures from raw.csv")
    ap.add_argument("--binarize", choices=["ovr", "top2"], default="ovr")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.tables_only:
        raw = pd.read_csv(os.path.join(a.out, "raw.csv"))
    else:
        raw, _ = run_sweep(a.datasets, a.models, a.reps, a.out, a.timeout, binarize_mode=a.binarize)
    order = ["er", "cm", "bah", "dpah", "patch", "stcm", "gencat", "sbm_tc", "canonical"]
    mean, std, rk_avg, pareto = make_tables(raw, a.out, order)
    make_figures(raw, mean, pareto, a.out, order)
    print(f"\nresults written to {a.out}")
    print(mean[OBJ].round(3).to_string())
    print("\nAverage ranks:\n" + rk_avg.to_string())


if __name__ == "__main__":
    main()
