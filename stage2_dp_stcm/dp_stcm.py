"""Stage 2: edge-level differential privacy for the stratified configuration model.

Laplace noise on the block counts B (sensitivity 1) and per-node group counts kappa (sensitivity 2), budget split rho / 1 - rho
"""
import argparse
import os
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
for _d in ("stage2_dp_stcm", "common"):                                    # shared modules
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import homophily_benchmark as hb

OUT = "dp_results"
EPS_GRID = ["0.05", "0.1", "0.25", "0.5", "1", "2", "5", "10", "inf", "inf_cl"]
DATASETS = ["cora", "polblogs", "chameleon", "texas", "github"]
LAMBDA = 0.5                # weight of fidelity in the trade-off objective J


# statistics, mechanism, post-processing

def block_counts(G, y, k):
    B = np.zeros((k, k))
    for u, v in G.edges():
        a, b = sorted((y[u], y[v]))
        B[a, b] += 1
    return B


def kappa_counts(G, y, k):
    K = np.zeros((len(y), k), dtype=int)
    for u, v in G.edges():
        K[u, y[v]] += 1
        K[v, y[u]] += 1
    return K


def privatize(B, K, y, eps, rho, rng):
    """Laplace mechanism on (B, kappa)."""
    if eps == np.inf:
        return B.astype(float).copy(), K.astype(float).copy()
    iu = np.triu_indices(B.shape[0])
    Bn = np.zeros_like(B, dtype=float)
    Bn[iu] = B[iu] + rng.laplace(0.0, 1.0 / (rho * eps), size=len(iu[0]))
    Kn = K + rng.laplace(0.0, 2.0 / ((1.0 - rho) * eps), size=K.shape)
    return Bn, Kn


def _round_to_total(x, total, cap, rng):
    """Non-negative integers close to x (x >= 0) summing to `total`, each <= cap (largest remainder)."""
    total = int(min(total, cap * len(x)))
    if total <= 0 or len(x) == 0:
        return np.zeros(len(x), dtype=int)
    s = x.sum()
    x = x * (total / s) if s > 0 else np.full(len(x), total / len(x))
    out = np.minimum(np.floor(x), cap).astype(int)
    for _ in range(20):                         # redistribute what the cap / flooring removed
        left = total - out.sum()
        if left <= 0:
            break
        room = cap - out
        frac = np.where(room > 0, x - np.floor(x) + 1e-9 * rng.random(len(x)), -1.0)
        idx = np.argsort(-frac)[:left]
        idx = idx[room[idx] > 0]
        if len(idx) == 0:
            break
        out[idx] += 1
        x = np.where(out >= cap, 0.0, x)
    return out


def postprocess(Bn, Kn, y, k, rng):
    """Free post-processing: clip, then make kappa consistent with the noisy block counts."""
    sizes = np.bincount(y, minlength=k)
    Bp = np.maximum(np.round(Bn), 0)
    K = np.zeros(Kn.shape, dtype=int)
    for a in range(k):
        ia = np.where(y == a)[0]
        for b in range(k):
            cap = sizes[b] - (1 if a == b else 0)
            x = np.clip(Kn[ia, b], 0, cap)
            lo, hi = min(a, b), max(a, b)
            total = 2 * Bp[a, a] if a == b else Bp[lo, hi]
            K[ia, b] = _round_to_total(x, total, cap, rng)
    return Bp, K


def release(G, y, k, eps, rho, seed, p=0.0):
    """Noise -> post-processing -> stcm. Returns the graph and the noisy counts."""
    rng = np.random.default_rng(10_000 + seed)
    B, K = block_counts(G, y, k), kappa_counts(G, y, k)
    Bn, Kn = privatize(B, K, y, eps, rho, rng)
    Bp, Kp = postprocess(Bn, Kn, y, k, rng)
    Gs = hb.gen_stcm(dict(y=y, kappa=Kp, k=k, p=p), seed=seed)
    return Gs, Bp, Kp, B, K


# metrics

def node_homophily(G, y):
    h = np.full(len(y), np.nan)
    for v in range(len(y)):
        nb = list(G[v])
        if nb:
            h[v] = np.mean(y[nb] == y[v])
    return h


def edge_keys(G, n):
    return np.array(sorted(min(u, v) * n + max(u, v) for u, v in G.edges()), dtype=np.int64)


def mia_auc(Gr, Gs, y, rng, n_neg=None):
    """Membership inference on edges."""
    n = len(y)
    E = np.array(Gr.edges())
    m = len(E)
    n_neg = n_neg or m
    real = set(map(int, edge_keys(Gr, n)))
    neg = []
    while len(neg) < n_neg:
        u, v = rng.integers(n, size=2)
        if u != v and (min(u, v) * n + max(u, v)) not in real:
            neg.append((u, v))
    pairs = np.vstack([E, np.array(neg)])
    lab = np.r_[np.ones(m), np.zeros(len(neg))]
    ds = np.array([d for _, d in sorted(Gs.degree())], float)
    dp = ds[pairs[:, 0]] * ds[pairs[:, 1]]
    dp = dp / (dp.max() + 1.0)
    syn = set(map(int, edge_keys(Gs, n)))
    ins = np.array([(min(u, v) * n + max(u, v)) in syn for u, v in pairs], float)
    return float(roc_auc_score(lab, ins + 0.5 * dp)), float(roc_auc_score(lab, dp))


def degree_baseline_recall(Gr):
    d = dict(Gr.degree())
    m2 = 2.0 * Gr.number_of_edges()
    return float(np.mean([min(1.0, d[u] * d[v] / m2) for u, v in Gr.edges()]))


def evaluate(Gr, Gs, y, ref_m, hr, B, K, Bp, Kp, rng):
    Gs_k = hb.preprocess_keep(Gs)
    for v in Gs_k:
        Gs_k.nodes[v]["y"] = int(y[v])
    m = hb.compute_metrics(Gs_k, Gr)
    m.update(hb.objectives(m, ref_m))
    n = len(y)
    er, es = edge_keys(Gr, n), edge_keys(Gs_k, n)
    hit = np.isin(es, er).sum()
    m["recall"] = hit / max(len(er), 1)
    m["precision"] = hit / max(len(es), 1)
    m["m_syn"] = len(es)
    hs = node_homophily(Gs_k, y)
    ok = ~np.isnan(hr) & ~np.isnan(hs)
    m["node_hom_err"] = float(np.mean(np.abs(hs[ok] - hr[ok]))) if ok.any() else np.nan
    m["isolated_frac"] = float(np.mean([d == 0 for _, d in Gs_k.degree()]))
    m["mia_auc_edge"], m["mia_auc_degree"] = mia_auc(Gr, Gs_k, y, rng)
    iu = np.triu_indices(B.shape[0])
    m["block_l1"] = float(np.abs(Bp[iu] - B[iu]).sum() / max(B[iu].sum(), 1))
    m["kappa_l1"] = float(np.abs(Kp - K).sum() / max(K.sum(), 1))
    return m


# sweep

def load_ref(ds):
    G = hb.preprocess(hb.load_dataset(ds), lcc=True)
    y = hb.labels(G)
    return G, y, int(y.max()) + 1


def sweep(datasets, reps, rho, eps_list, out, tag="main"):
    os.makedirs(out, exist_ok=True)
    rp = os.path.join(out, "dp_raw.csv" if tag == "main" else "dp_rho_raw.csv")
    raw = pd.read_csv(rp, dtype={"eps": str}) if os.path.exists(rp) else pd.DataFrame()
    done = set(zip(raw["dataset"], raw["eps"], raw["rho"].round(3))) if len(raw) else set()
    for ds in datasets:
        G, y, k = load_ref(ds)
        ref_m = hb.reference_metrics(G)
        hr = node_homophily(G, y)
        p_fit = None
        print(f"=== {ds}: n={len(y)} m={G.number_of_edges()} k={k}", flush=True)
        for e in eps_list:
            if (ds, e, round(rho, 3)) in done:
                continue
            if e == "inf_cl" and p_fit is None:
                p_fit = hb.fit_stcm(G)["p"]
            eps = np.inf if e.startswith("inf") else float(e)
            rows, t0 = [], time.time()
            for r in range(reps):
                t1 = time.time()
                Gs, Bp, Kp, B, K = release(G, y, k, eps, rho, seed=r, p=(p_fit if e == "inf_cl" else 0.0))
                rt = time.time() - t1
                m = evaluate(G, Gs, y, ref_m, hr, B, K, Bp, Kp, np.random.default_rng(777 + r))
                m.update(dataset=ds, eps=e, rho=rho, rep=r, runtime=rt, n=len(y),
                         m_real=G.number_of_edges(), p_cl=(p_fit if e == "inf_cl" else 0.0),
                         deg_baseline_recall=degree_baseline_recall(G))
                rows.append(m)
            raw = pd.concat([raw, pd.DataFrame(rows)], ignore_index=True)
            raw.to_csv(rp, index=False)
            print(f"   eps={e:6s} rho={rho}  f1={np.mean([x['f1_homophily'] for x in rows]):.3f} "
                  f"recall={np.mean([x['recall'] for x in rows]):.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return raw


# verification

def verify(datasets, n_remove=200, seed=0):
    rng = np.random.default_rng(seed)
    lines = []
    for ds in datasets:
        G, y, k = load_ref(ds)
        B, K = block_counts(G, y, k), kappa_counts(G, y, k)
        edges = list(G.edges())
        idx = rng.choice(len(edges), size=min(n_remove, len(edges)), replace=False)
        sB, sK = 0, 0
        for i in idx:
            H = G.copy()
            H.remove_edge(*edges[i])
            sB = max(sB, np.abs(block_counts(H, y, k) - B).sum())
            sK = max(sK, np.abs(kappa_counts(H, y, k) - K).sum())
        # eps = inf reproduces kappa exactly
        _, Bp, Kp, _, _ = release(G, y, k, np.inf, 0.3, seed=0)
        exact = bool((Kp == K).all() and (Bp == B).all())
        # empirical noise std vs theory (eps=1, rho=0.3)
        noise = []
        for s in range(20):
            Bn, Kn = privatize(B, K, y, 1.0, 0.3, np.random.default_rng(s))
            noise.append((Kn - K).ravel())
        sd_emp, sd_th = float(np.std(np.concatenate(noise))), float(np.sqrt(2) * 2 / 0.7)
        lines.append(dict(dataset=ds, edges_removed=len(idx), max_L1_change_B=sB, max_L1_change_kappa=sK,
                          eps_inf_exact=exact, kappa_noise_sd_emp=round(sd_emp, 3), kappa_noise_sd_theory=round(sd_th, 3)))
        print(lines[-1], flush=True)
    df = pd.DataFrame(lines)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "dp_verification.csv"), index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["verify", "sweep", "rho", "samples"])
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--rho", type=float, default=0.3)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.cmd == "verify":
        verify(a.datasets)
    elif a.cmd == "sweep":
        sweep(a.datasets, a.reps, a.rho, EPS_GRID, a.out)
    elif a.cmd == "rho":
        for r in (0.1, 0.5, 0.8):
            sweep(a.datasets, a.reps, r, [e for e in EPS_GRID if not e.startswith("inf")], a.out, tag="rho")
    elif a.cmd == "samples":
        import dp_samples
        dp_samples.main(a.datasets, a.out)
