"""Metrics, objectives and generators for the homophily benchmark.

Metrics: edge homophily, Newman r, mixing distance, clustering, homophilous triangles,
degree KS, degree Gini, avg path length, edge overlap (+ lift).
Generators: er, cm, bah, dpah, patch, stcm, gencat, sbm_tc, canonical.
Each model has fit(G) -> params and gen(params, seed) -> graph.
"""
import argparse
import os
import time
import traceback

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import ks_2samp


# graph preprocessing / utilities

def preprocess(G, lcc=True):
    G = nx.Graph(G)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    G.remove_nodes_from([v for v in G if "y" not in G.nodes[v]])
    if lcc and G.number_of_nodes() > 0:
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G = nx.convert_node_labels_to_integers(G)
    vals, counts = np.unique([G.nodes[v]["y"] for v in G], return_counts=True)
    order = vals[np.argsort(-counts, kind="stable")]
    remap = {v: i for i, v in enumerate(order)}
    for v in G:
        G.nodes[v]["y"] = remap[G.nodes[v]["y"]]
    return G


def labels(G):
    return np.array([G.nodes[v]["y"] for v in range(G.number_of_nodes())], dtype=int)


def adjacency(G):
    # weight=None so weighted graphs (e.g. karate) become 0/1
    return nx.to_scipy_sparse_array(G, nodelist=range(G.number_of_nodes()),
                                    weight=None, format="csr", dtype=float)


def binarize(G, mode="ovr"):
    """Binary-group view for the two-group models (BAh, DPAH, PATCH)."""
    y = labels(G)
    if y.max() <= 1:
        return G.copy()
    H = G.copy()
    if mode == "top2":
        H = H.subgraph([v for v in H if H.nodes[v]["y"] <= 1]).copy()
        return preprocess(H)
    for v in H:
        H.nodes[v]["y"] = 1 if H.nodes[v]["y"] == 1 else 0
    return preprocess(H)


# dataset loaders

def assign_labels_with_homophily(G, f_min=0.2, target_h=0.8, seed=0, max_iter_factor=60):
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    y = {v: 0 for v in nodes}
    for v in rng.choice(nodes, size=int(round(f_min * n)), replace=False):
        y[v] = 1
    m = G.number_of_edges()
    same = sum(y[u] == y[v] for u, v in G.edges())
    mins = [v for v in nodes if y[v] == 1]
    majs = [v for v in nodes if y[v] == 0]
    for _ in range(max_iter_factor * n):
        if abs(same / m - target_h) < 0.005:
            break
        i, j = rng.integers(len(mins)), rng.integers(len(majs))
        u, v = mins[i], majs[j]          # u: 1 -> 0, v: 0 -> 1 (swap)
        du = sum(y[w] == 0 for w in G[u] if w != v) - sum(y[w] == 1 for w in G[u] if w != v)
        dv = sum(y[w] == 1 for w in G[v] if w != u) - sum(y[w] == 0 for w in G[v] if w != u)
        delta = du + dv
        if (same / m < target_h and delta > 0) or (same / m > target_h and delta < 0):
            y[u], y[v] = 0, 1
            mins[i], majs[j] = v, u
            same += delta
    nx.set_node_attributes(G, y, "y")
    return G


def load_synthetic(kind, n=2000, f_min=0.2, target_h=0.8, seed=0):
    if kind == "er":
        G = nx.gnp_random_graph(n, 8.0 / (n - 1), seed=seed)
    elif kind == "ws":
        G = nx.connected_watts_strogatz_graph(n, 10, 0.1, seed=seed)
    elif kind == "ba":
        G = nx.barabasi_albert_graph(n, 4, seed=seed)
    else:
        raise ValueError(f"unknown synthetic kind {kind}")
    return assign_labels_with_homophily(G, f_min, target_h, seed)


def load_pyg(name, root="./data"):
    """Needs torch and torch_geometric installed."""
    from torch_geometric import datasets as D
    key = name.lower().replace(" ", "").replace("_", "")
    if key in ("cora", "citeseer", "pubmed"):
        ds = D.Planetoid(root, name.capitalize() if key != "pubmed" else "PubMed")
    elif key in ("chameleon", "squirrel"):
        ds = D.WikipediaNetwork(root, key)
    elif key in ("texas", "cornell", "wisconsin"):
        ds = D.WebKB(root, key.capitalize())
    elif key == "github":
        ds = D.GitHub(f"{root}/GitHub")
    elif key == "polblogs":
        ds = D.PolBlogs(f"{root}/PolBlogs")
    elif key in ("amazonphoto", "photo"):
        ds = D.Amazon(root, "Photo")
    else:
        raise ValueError(f"unmapped PyG dataset {name}")
    data = ds[0]
    G = nx.Graph()
    y = data.y.numpy()
    G.add_nodes_from((i, {"y": int(y[i])}) for i in range(data.num_nodes))
    G.add_edges_from(data.edge_index.numpy().T.tolist())
    return G


def load_gml(path, label_key="value"):
    G = nx.read_gml(path, label="id")
    for v in G:
        G.nodes[v]["y"] = G.nodes[v][label_key]
    return G


def load_edgelist_with_labels(edge_path, label_path, edge_cols=(0, 1), label_col=1):
    """Edge list + label file, whitespace or comma separated."""
    E = pd.read_csv(edge_path, sep=r"[\s,]+", header=None, engine="python", comment="#")
    L = pd.read_csv(label_path, sep=r"[\s,]+", header=None, engine="python", comment="#")
    G = nx.Graph()
    G.add_edges_from(E.iloc[:, list(edge_cols)].astype(str).itertuples(index=False, name=None))
    lab = dict(zip(L.iloc[:, 0].astype(str), L.iloc[:, label_col].astype(str)))
    for v in list(G):
        if v in lab:
            G.nodes[v]["y"] = lab[v]
    return G


def load_dataset(spec, edge_cols=(0, 1), label_col=1, label_key="value"):
    s = spec.lower()
    base, _, lim = s.partition(":")
    import dataset_loaders as _dl
    if base in _dl.REGISTRY:
        return _dl.load(base, max_nodes=int(lim) if lim else "default")
    if s == "karate":
        G = nx.karate_club_graph()
        for v in G:
            G.nodes[v]["y"] = 0 if G.nodes[v]["club"] == "Mr. Hi" else 1
        return G
    if s.startswith("synthetic:"):
        return load_synthetic(s.split(":")[1])
    if s.startswith("pyg:"):
        return load_pyg(spec.split(":", 1)[1])
    if s.endswith(".gml"):
        return load_gml(spec, label_key)
    if "," in spec:
        e, l = spec.split(",", 1)
        return load_edgelist_with_labels(e, l, edge_cols, label_col)
    raise ValueError(f"Cannot interpret dataset spec '{spec}'")


# metrics

def mixing_matrix(A, y, C):
    """e_ab: fraction of edge endpoints joining class a to class b (sums to 1)."""
    r, c = A.nonzero()
    M = np.zeros((C, C))
    np.add.at(M, (y[r], y[c]), 1.0)
    return M / max(M.sum(), 1.0)


def edge_homophily(e):            # 1
    return float(np.trace(e))


def newman_r(e):                  # 2
    a = e.sum(1)
    b = e.sum(0)
    denom = 1.0 - float(a @ b)
    return float((np.trace(e) - a @ b) / denom) if denom > 1e-12 else np.nan


def triangles(A):
    return (A @ A).multiply(A).sum() / 6.0


def clustering_and_hom_triangles(A, y, C):   # 4, 5
    deg = np.asarray(A.sum(1)).ravel()
    triples = (deg * (deg - 1) / 2).sum()
    tri = triangles(A)
    clust = 3 * tri / triples if triples > 0 else 0.0
    hom = 0.0
    for k in range(C):
        idx = np.where(y == k)[0]
        if len(idx) > 2:
            Ak = A[idx][:, idx]
            hom += triangles(Ak)
    return float(clust), float(hom / tri) if tri > 0 else np.nan


def gini(x):                      # 7
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) @ x) / (n * x.sum()) - (n + 1) / n) if x.sum() > 0 else 0.0


def avg_shortest_path(A, n_src=300, seed=0):   # 8
    k, comp = connected_components(A, directed=False)
    if k > 1:
        big = np.argmax(np.bincount(comp))
        idx = np.where(comp == big)[0]
        A = A[idx][:, idx]
    n = A.shape[0]
    if n < 2:
        return np.nan
    rng = np.random.default_rng(seed)
    src = rng.choice(n, size=min(n_src, n), replace=False)
    D = shortest_path(A, unweighted=True, indices=src)
    D = D[np.isfinite(D) & (D > 0)]
    return float(D.mean()) if D.size else np.nan


def align_nodes(yg, dg, yr, dr):
    """Match generated nodes to real nodes by label and degree rank."""
    mp = -np.ones(len(yg), dtype=int)
    for k in np.unique(yr):
        gi = np.where(yg == k)[0]
        ri = np.where(yr == k)[0]
        if len(gi) == 0 or len(ri) == 0:
            continue
        gi = gi[np.argsort(-dg[gi], kind="stable")]
        ri = ri[np.argsort(-dr[ri], kind="stable")]
        L = min(len(gi), len(ri))
        mp[gi[:L]] = ri[:L]
    return mp


def edge_overlap(Gg, Gr):         # 9
    yg, yr = labels(Gg), labels(Gr)
    dg = np.array([d for _, d in sorted(Gg.degree())])
    dr = np.array([d for _, d in sorted(Gr.degree())])
    mp = align_nodes(yg, dg, yr, dr)
    nr = Gr.number_of_nodes()
    real = np.array([min(u, v) * nr + max(u, v) for u, v in Gr.edges()])
    gen = []
    for u, v in Gg.edges():
        a, b = mp[u], mp[v]
        if a >= 0 and b >= 0 and a != b:
            gen.append(min(a, b) * nr + max(a, b))
    hits = np.isin(np.array(gen, dtype=np.int64), real).sum() if gen else 0
    eor = hits / max(Gr.number_of_edges(), 1)
    chance = Gr.number_of_edges() / (nr * (nr - 1) / 2)
    return float(eor), float(eor / chance)


def compute_metrics(Gg, Gr):
    """All metrics of a generated graph Gg against the real graph Gr."""
    C = int(max(labels(Gg).max(), labels(Gr).max())) + 1
    Ag, Ar = adjacency(Gg), adjacency(Gr)
    yg, yr = labels(Gg), labels(Gr)
    eg, er = mixing_matrix(Ag, yg, C), mixing_matrix(Ar, yr, C)
    cg, hg = clustering_and_hom_triangles(Ag, yg, C)
    dg = np.asarray(Ag.sum(1)).ravel()
    dr = np.asarray(Ar.sum(1)).ravel()
    eor, lift = edge_overlap(Gg, Gr)
    return dict(
        edge_h=edge_homophily(eg), newman_r=newman_r(eg),
        mix_dist=float(np.linalg.norm(eg - er)),
        clustering=cg, hom_tri=hg,
        deg_ks=float(ks_2samp(dg, dr).statistic), deg_gini=gini(dg),
        aspl=avg_shortest_path(Ag), eor=eor, eor_lift=lift,
    )


def reference_metrics(Gr):
    return compute_metrics(Gr, Gr)   # mix_dist = ks = 0, eor = 1


def rel_err(g, r):
    """Relative error |g-r|/|r|, clipped to [0, 1]."""
    if np.isnan(g) or np.isnan(r):
        return np.nan
    e = abs(g - r) / abs(r) if abs(r) > 1e-9 else abs(g - r)
    return float(min(e, 1.0))


def objectives(m, ref):
    """f1 homophily, f2 motif, f3 structure, f4 privacy (all: lower is better)."""
    f1 = np.nanmean([rel_err(m["edge_h"], ref["edge_h"]),
                     rel_err(m["newman_r"], ref["newman_r"]), m["mix_dist"]])
    f2 = np.nanmean([rel_err(m["clustering"], ref["clustering"]),
                     rel_err(m["hom_tri"], ref["hom_tri"])])
    f3 = np.nanmean([m["deg_ks"], rel_err(m["deg_gini"], ref["deg_gini"]),
                     rel_err(m["aspl"], ref["aspl"])])
    return dict(f1_homophily=f1, f2_motif=f2, f3_structure=f3, f4_privacy=m["eor"])


# models

EPS_PA = 1e-5   # keeps zero-degree nodes selectable, like netin


def _with_labels(G, y):
    nx.set_node_attributes(G, {i: int(y[i]) for i in range(len(y))}, "y")
    return G


def _mean_degree_half(G):
    """m = links per new node used by the growth models: mean degree / 2."""
    return max(1, int(round(G.number_of_edges() / G.number_of_nodes())))


# null baselines

def fit_er(G):
    return dict(n=G.number_of_nodes(), m=G.number_of_edges(), y=labels(G))


def gen_er(p, seed):
    rng = np.random.default_rng(seed)
    G = nx.gnm_random_graph(p["n"], p["m"], seed=int(seed))
    return _with_labels(G, rng.permutation(p["y"]))


def fit_cm(G):
    return dict(deg=[d for _, d in sorted(G.degree())], y=labels(G))


def gen_cm(p, seed):
    rng = np.random.default_rng(seed)
    G = nx.Graph(nx.configuration_model(p["deg"], seed=int(seed)))
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    return _with_labels(G, rng.permutation(p["y"]))


# summary statistics used to fit PATCH

def summary_stats(G):
    """EI, Mann-Whitney MW (P[deg_min > deg_maj]), Gini_min, Gini_maj, CCF."""
    from scipy.stats import mannwhitneyu
    y = labels(G)
    deg = np.array([d for _, d in sorted(G.degree())], float)
    I = sum(1 for u, v in G.edges() if y[u] == y[v])
    E = G.number_of_edges() - I
    dmin, dmaj = deg[y == 1], deg[y == 0]
    mw = mannwhitneyu(dmin, dmaj, method="asymptotic").statistic / (len(dmin) * len(dmaj))
    return dict(EI=(E - I) / max(E + I, 1), MW=float(mw), gini_min=gini(dmin),
                gini_maj=gini(dmaj), CCF=nx.average_clustering(G))


def gen_bah(p, seed):
    rng = np.random.default_rng(seed)
    n, m, alpha = p["n"], p["m"], p.get("alpha", 1.0)
    phases = p.get("phases") or [dict(until=n, h=p["h"], f_min=p["f_min"])]
    grp = np.zeros(n, dtype=int)
    deg = np.zeros(n)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    ph = 0
    for t in range(n):
        while t >= phases[ph]["until"] and ph < len(phases) - 1:
            ph += 1
        grp[t] = int(rng.random() < phases[ph]["f_min"])
        if t < m:                                   # initial fully connected core
            for j in range(t):
                G.add_edge(t, j)
                deg[t] += 1
                deg[j] += 1
            continue
        h = float(np.clip(phases[ph]["h"], 0.01, 0.99))
        w = np.where(grp[:t] == grp[t], h, 1.0 - h) * (deg[:t] + EPS_PA) ** alpha
        for j in rng.choice(t, size=min(m, t), replace=False, p=w / w.sum()):
            G.add_edge(t, int(j))
            deg[t] += 1
            deg[j] += 1
    return _with_labels(G, grp)


def fit_bah(G, n_cal=1200, grid=21, reps=2):
    y = labels(G)
    n = G.number_of_nodes()
    m = _mean_degree_half(G)
    f_min = float((y == 1).mean())
    target = edge_homophily(mixing_matrix(adjacency(G), y, int(y.max()) + 1))
    best, best_err = 0.5, 9e9
    for h in np.linspace(0.02, 0.98, grid):
        errs = []
        for r in range(reps):
            g = gen_bah(dict(n=min(n, n_cal), m=m, h=h, f_min=f_min), seed=1000 + r)
            errs.append(abs(edge_homophily(mixing_matrix(adjacency(g), labels(g), 2)) - target))
        if np.mean(errs) < best_err:
            best, best_err = h, float(np.mean(errs))
    return dict(n=n, m=m, h=float(best), f_min=f_min, alpha=1.0)


def gen_patch(p, seed):
    rng = np.random.default_rng(seed)
    n, m, tau, h = p["n"], p["m"], p["tau"], float(np.clip(p["h"], 0.01, 0.99))
    lg, lt = p.get("lfm_global", "PAH"), p.get("lfm_tc", "PAH")
    grp = (rng.random(n) < p["f_min"]).astype(int)
    adj = [set() for _ in range(n)]
    deg = np.zeros(n)

    def add(u, v):
        adj[u].add(v); adj[v].add(u); deg[u] += 1; deg[v] += 1

    def lfm(kind, s):
        if kind == "U":
            return np.ones(s)
        hr = np.where(grp[:s] == grp[s], h, 1.0 - h)
        return hr * (deg[:s] + EPS_PA) if kind == "PAH" else hr

    for i in range(m):
        for j in range(i):
            add(i, j)
    for s in range(m, n):
        for t in range(m):
            free = np.ones(s)
            free[list(adj[s])] = 0.0
            if t > 0 and rng.random() < tau:                       # triadic-closure link
                fof = {w for f in adj[s] for w in adj[f] if w != s and w not in adj[s]}
                pool = np.zeros(s)
                if fof:
                    pool[list(fof)] = 1.0
                else:
                    pool[:] = 1.0
                w_ = pool * free * lfm(lt, s)
            else:                                                   # global link
                w_ = free * lfm(lg, s)
            if w_.sum() <= 0:
                w_ = free
            add(s, int(rng.choice(s, p=w_ / w_.sum())))
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from((u, v) for u in range(n) for v in adj[u] if u < v)
    return _with_labels(G, grp)


def fit_patch(G, variant=("PAH", "PAH"), n_sim=500, reps=2,
              h_grid=(0.1, 0.3, 0.5, 0.7, 0.9), tau_grid=(0.0, 0.25, 0.5, 0.75, 1.0)):
    y = labels(G)
    obs = summary_stats(G)
    keys = list(obs)
    m = max(2, _mean_degree_half(G))                      # paper: m >= 2
    f_min = float((y == 1).mean())
    base = dict(n=n_sim, m=m, f_min=f_min, lfm_global=variant[0], lfm_tc=variant[1])

    def dist(h, tau):
        d = []
        for r in range(reps):
            s = summary_stats(gen_patch(dict(base, h=h, tau=tau), seed=2000 + r))
            d.append(np.sqrt(sum((s[k] - obs[k]) ** 2 for k in keys)))
        return float(np.mean(d))

    best = min(((dist(h, t), h, t) for h in h_grid for t in tau_grid))
    _, h0, t0 = best
    for h in (h0 - 0.1, h0, h0 + 0.1):
        for t in (t0 - 0.125, t0, t0 + 0.125):
            if 0.01 <= h <= 0.99 and 0 <= t <= 1:
                best = min(best, (dist(h, t), h, t))
    return dict(n=G.number_of_nodes(), m=m, f_min=f_min, h=best[1], tau=best[2],
                lfm_global=variant[0], lfm_tc=variant[1], fit_distance=best[0])


def gen_dpah(p, seed):
    rng = np.random.default_rng(seed)
    N, E = p["N"], int(p["E"])
    NM = int(round(N * (1 - p["f_m"])))
    lab = np.array([0] * NM + [1] * (N - NM))
    rng.shuffle(lab)
    act = np.where(lab == 0, rng.zipf(p["gamma_M"], N), rng.zipf(p["gamma_m"], N)).astype(float)
    act = np.minimum(act, N - 1)   # otherwise gamma < 2 can give one node all the activity
    cdf = np.cumsum(act / act.sum())
    eps = 1e-5
    hMM = eps if p["h_MM"] == 0 else 1 - eps if p["h_MM"] == 1 else p["h_MM"]
    hmm = eps if p["h_mm"] == 0 else 1 - eps if p["h_mm"] == 1 else p["h_mm"]
    Hm = np.array([[hMM, 1 - hMM], [1 - hmm, hmm]])
    indeg = np.zeros(N)
    outdeg = np.zeros(N)
    edges = set()
    tries = 0
    while len(edges) < E and tries < 50 * E + 1000:
        tries += 1
        s = min(int(np.searchsorted(cdf, rng.random())), N - 1)
        w = Hm[lab[s]][lab] * (indeg + 1)
        w[s] = 0.0
        if len(edges) > N / 100.0:
            w[outdeg == 0] = 0.0
        if w.sum() <= 0:
            continue
        t = int(np.searchsorted(np.cumsum(w), rng.random() * w.sum()))
        t = min(t, N - 1)
        if (s, t) not in edges:
            edges.add((s, t))
            indeg[t] += 1
            outdeg[s] += 1
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edges)
    return _with_labels(G, lab)


def _fit_gamma(x):
    """Discrete power-law exponent (xmin=1) by the standard approximate MLE."""
    x = np.asarray(x, float)
    x = x[x >= 1]
    return float(np.clip(1 + len(x) / np.sum(np.log(x / 0.5)), 1.2, 6.0))


def fit_dpah(G, n_cal=300):
    y = labels(G)
    N = G.number_of_nodes()
    deg = np.array([d for _, d in sorted(G.degree())], float)
    base = dict(N=min(N, n_cal), f_m=float((y == 1).mean()),
                gamma_M=_fit_gamma(deg[y == 0]), gamma_m=_fit_gamma(deg[y == 1]))
    base["E"] = int(round(G.number_of_edges() * base["N"] / N))
    ed = np.array(G.edges())
    tgt = np.array([np.mean((y[ed[:, 0]] == 0) & (y[ed[:, 1]] == 0)),
                    np.mean((y[ed[:, 0]] == 1) & (y[ed[:, 1]] == 1))])

    def loss(hMM, hmm):
        g = gen_dpah(dict(base, h_MM=hMM, h_mm=hmm), seed=3000)
        yy = np.array([g.nodes[i]["y"] for i in range(base["N"])])
        e = np.array(g.edges())
        if len(e) == 0:
            return 9e9
        f = np.array([np.mean((yy[e[:, 0]] == 0) & (yy[e[:, 1]] == 0)),
                      np.mean((yy[e[:, 0]] == 1) & (yy[e[:, 1]] == 1))])
        return float(np.abs(f - tgt).sum())

    grid = (0.15, 0.5, 0.85)
    best = min((loss(a, b), a, b) for a in grid for b in grid)
    _, a0, b0 = best
    for a in (a0 - 0.1, a0, a0 + 0.1):
        for b in (b0 - 0.1, b0, b0 + 0.1):
            if 0 <= a <= 1 and 0 <= b <= 1:
                best = min(best, (loss(a, b), a, b))
    out = dict(base, h_MM=best[1], h_mm=best[2])
    out.update(N=N, E=G.number_of_edges())
    return out


def _freeze(q, T):
    z = np.log(np.maximum(q, 1e-12)) / T
    z -= z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _f_rev(U, cls):
    """Eq. 10 of the GenCAT paper, for rows of class `cls`."""
    R = 1.0 - U
    others = np.ones(U.shape[1], bool)
    others[cls] = False
    den = R[:, others].sum(axis=1, keepdims=True)
    out = R * U[:, [cls]] / np.maximum(den, 1e-12)
    out[:, cls] = 1.0 - U[:, cls]
    return out


def fit_gencat(G):
    y = labels(G)
    n = G.number_of_nodes()
    k = int(y.max()) + 1
    A = adjacency(G)
    cnt = np.zeros((n, k))
    r, c = A.nonzero()
    np.add.at(cnt, (r, y[c]), 1.0)
    pref = cnt / np.maximum(cnt.sum(1, keepdims=True), 1.0)       # per-node class proportions
    M = np.zeros((k, k))
    D = np.zeros((k, k))
    for l in range(k):
        rows = pref[y == l]
        M[l] = rows.mean(0)
        D[l] = rows.std(0, ddof=1) if len(rows) > 1 else 0.0
    return dict(n=n, k=k, M=M, D=D, com_size=np.bincount(y, minlength=k) / n,
                theta=np.asarray(A.sum(1)).ravel(), r=50)


def gen_gencat(p, seed):
    rng = np.random.default_rng(seed)
    n, k, M, D = p["n"], p["k"], p["M"], p["D"]
    # --- latent factor generation (lines 2-10)
    C = rng.choice(k, size=n, p=p["com_size"])
    U = np.clip(rng.normal(M[C], D[C]), 0.0, 1.0)
    U = U / np.maximum(U.sum(1, keepdims=True), 1e-12)
    U[U.sum(1) == 0] = 1.0 / k
    # --- adjusting proportion phase (lines 13-23)
    Uprime = U.copy()
    for l in range(k):
        idx = np.where(C == l)[0]
        if len(idx) == 0:
            continue
        best_T, best_loss = 1.0, np.inf
        for T in np.arange(0.01, 1, 0.05):
            F = _freeze(U[idx], T)
            est = (F * (F if M[l, l] >= 1.0 / k else _f_rev(F, l))).mean(0)
            loss = np.linalg.norm(M[l] - est)
            if loss < best_loss:
                best_loss, best_T = loss, T
        U[idx] = _freeze(U[idx], best_T)
        Uprime[idx] = U[idx] if M[l, l] >= 1.0 / k else _f_rev(U[idx], l)
    # --- edge generation (Algorithm 2) with inverse transform sampling
    theta = np.sort(np.asarray(p["theta"], float))[::-1]
    cdfs = []
    for l in range(k):
        col = Uprime[:, l]
        s = col.sum()
        cdfs.append(np.cumsum(col / s) if s > 0 else np.cumsum(np.full(n, 1.0 / n)))
    adj = [set() for _ in range(n)]
    cur = np.zeros(n)
    for i in range(n):
        count = 0
        while count < p["r"] and cur[i] < theta[i]:
            need = int(theta[i] - cur[i])
            Z = rng.choice(k, size=need, p=U[i] / U[i].sum())
            for l in Z:
                j = min(int(np.searchsorted(cdfs[l], rng.random())), n - 1)
                if j != i and j not in adj[i] and cur[i] < theta[i] and cur[j] < theta[j]:
                    adj[i].add(j); adj[j].add(i)
                    cur[i] += 1; cur[j] += 1
            count += 1
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from((u, v) for u in range(n) for v in adj[u] if u < v)
    return _with_labels(G, C)


def fit_stcm(G, p_grid=(0.0, 0.25, 0.5, 0.75, 1.0)):
    y = labels(G)
    n = G.number_of_nodes()
    k = int(y.max()) + 1
    kappa = np.zeros((n, k), dtype=int)
    A = adjacency(G)
    r, c = A.nonzero()
    np.add.at(kappa, (r, y[c]), 1)
    base = dict(y=y, kappa=kappa, k=k)
    target = float(nx.transitivity(G))
    best = (9e9, 0.0)
    for pp in p_grid:
        g = gen_stcm(dict(base, p=pp), seed=4000)
        best = min(best, (abs(nx.transitivity(g) - target), pp))
    return dict(base, p=best[1])


def gen_stcm(params, seed):
    rng = np.random.default_rng(seed)
    y, k, p = params["y"], params["k"], params["p"]
    kap = params["kappa"].copy()
    n = len(y)
    members = [np.where(y == a)[0] for a in range(k)]
    # avail[(a, f)]: nodes of category a that still have kappa[., f] > 0 (list + position map)
    avail, pos = {}, {}
    for a in range(k):
        for f in range(k):
            lst = [int(v) for v in members[a] if kap[v, f] > 0]
            avail[(a, f)] = lst
            pos[(a, f)] = {v: i for i, v in enumerate(lst)}

    def dec(v, f):                       # kappa[v,f] -= 1, maintain availability lists
        kap[v, f] -= 1
        if kap[v, f] == 0:
            key = (y[v], f)
            lst, ps = avail[key], pos[key]
            i = ps.pop(v)
            last = lst.pop()
            if i < len(lst):
                lst[i] = last
                ps[last] = i

    adj = [set() for _ in range(n)]
    order = np.lexsort((rng.random(n), -kap.sum(1)))         # descending degree, random ties
    for v in order:
        fv = y[v]
        for a in range(k):
            while kap[v, a] > 0:
                lst = avail[(a, fv)]
                mu = None
                for _ in range(20):
                    if not lst:
                        break
                    cand = lst[rng.integers(len(lst))]
                    if cand != v and cand not in adj[v]:
                        mu = cand
                        break
                if mu is None:
                    valid = [c for c in lst if c != v and c not in adj[v]]
                    if not valid:
                        break
                    mu = valid[rng.integers(len(valid))]
                adj[v].add(mu); adj[mu].add(v)
                dec(v, a)
                dec(mu, fv)
        if p > 0:                                              # STEP 3
            nb = list(adj[v])
            for mu in nb:
                for chi in nb:
                    if chi != mu and chi not in adj[mu] and kap[mu, y[chi]] > 0 and kap[chi, y[mu]] > 0:
                        if rng.random() < p:
                            adj[mu].add(chi); adj[chi].add(mu)
                            dec(chi, y[mu])
                            dec(mu, y[chi])
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from((u, v) for u in range(n) for v in adj[u] if u < v)
    return _with_labels(G, y)


def fit_sbm_tc(G, L=2, s_grid=(1.0, 0.7, 0.5, 0.35, 0.2, 0.1),
               p_grid=(0.01, 0.03, 0.06, 0.1, 0.2, 0.35, 0.5)):
    base = fit_canonical(G)
    base["L"] = L
    m, cc = G.number_of_edges(), nx.transitivity(G)
    best = (9e9, 1.0, 0.0)
    for s in s_grid:
        for pp in p_grid:
            g = gen_sbm_tc(dict(base, s=s, p=pp, max_edges=2 * m), seed=5000)
            if g.number_of_edges() > 1.5 * m:          # far too many edges: skip costly metrics
                continue
            err = abs(g.number_of_edges() - m) / m + abs(nx.transitivity(g) - cc) / max(cc, 1e-9)
            best = min(best, (err, s, pp))
    return dict(base, s=best[1], p=best[2])


def gen_sbm_tc(params, seed):
    rng = np.random.default_rng(seed)
    base = gen_canonical(dict(params, m_ab=params["m_ab"] * params["s"]), seed)
    n = base.number_of_nodes()
    adj = [set(base[v]) for v in range(n)]
    new_prev = {(min(u, v), max(u, v)) for u, v in base.edges()}
    for _ in range(params["L"]):
        snap = [set(a) for a in adj]
        new_cur = set()
        n_edges_now = sum(len(a) for a in adj) / 2
        for u in range(n):
            if n_edges_now + len(new_cur) > params.get("max_edges", np.inf):
                break                                   # abort early (used during calibration)
            nb = list(snap[u])
            if len(nb) < 2:
                continue
            R = [v for v in nb if (min(u, v), max(u, v)) in new_prev]
            Rset = set(R)
            for i in R:
                for j in nb:
                    if j == i or j in snap[i] or (j in Rset and j < i):
                        continue
                    if rng.random() < params["p"]:
                        new_cur.add((min(i, j), max(i, j)))
        for a, b in new_cur:
            adj[a].add(b); adj[b].add(a)
        new_prev = new_cur
        if not new_cur or sum(len(a) for a in adj) / 2 > params.get("max_edges", np.inf):
            break
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from((u, v) for u in range(n) for v in adj[u] if u < v)
    return _with_labels(G, params["y"])


# canonical Poisson ensemble (Paper 15)

def fit_canonical(G):
    y = labels(G)
    C = int(y.max()) + 1
    A = adjacency(G)
    r, c = A.nonzero()
    M = np.zeros((C, C))
    np.add.at(M, (y[r], y[c]), 1.0)
    m_ab = M.copy()
    np.fill_diagonal(m_ab, np.diag(M) / 2.0)   # undirected edge counts
    return dict(y=y, deg=np.asarray(A.sum(1)).ravel(), m_ab=m_ab, C=C)


def choice_homophily_two_group(G):
    """Homophily h for two groups (Eq. 8 of the canonical-ensemble paper)."""
    y = labels(G)
    e = mixing_matrix(adjacency(G), y, 2) * 2 * G.number_of_edges()
    return float(1.0 / (1.0 + np.sqrt(e[0, 1] / (e[0, 0] * e[1, 1]))))


def gen_canonical(p, seed):
    rng = np.random.default_rng(seed)
    y, deg, C = p["y"], p["deg"], p["C"]
    n = len(y)
    members = [np.where(y == k)[0] for k in range(C)]
    probs = [deg[i] / deg[i].sum() if len(i) and deg[i].sum() > 0 else None for i in members]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for a in range(C):
        for b in range(a, C):
            if probs[a] is None or probs[b] is None:
                continue
            E = rng.poisson(p["m_ab"][a, b])
            if E == 0:
                continue
            u = rng.choice(members[a], size=E, p=probs[a])
            v = rng.choice(members[b], size=E, p=probs[b])
            G.add_edges_from((int(i), int(j)) for i, j in zip(u, v) if i != j)
    return _with_labels(G, y)


MODELS = {
    # name: (fit, generate, binary_only)
    "er": (fit_er, gen_er, False),
    "cm": (fit_cm, gen_cm, False),
    "bah": (fit_bah, gen_bah, True),
    "dpah": (fit_dpah, gen_dpah, True),
    "patch": (fit_patch, gen_patch, True),
    "sbm_tc": (fit_sbm_tc, gen_sbm_tc, False),
    "stcm": (fit_stcm, gen_stcm, False),
    "gencat": (fit_gencat, gen_gencat, False),
    "canonical": (fit_canonical, gen_canonical, False),
}


# simple runner (run_study.py is the full version)

def run(datasets, models, reps, out, edge_cols, label_col, label_key, binarize_mode, lcc=True):
    os.makedirs(out, exist_ok=True)
    rows = []
    for ds in datasets:
        print(f"\n=== dataset: {ds}")
        G0 = preprocess(load_dataset(ds, edge_cols, label_col, label_key), lcc=lcc)
        Gbin = binarize(G0, binarize_mode)
        print(f"    n={G0.number_of_nodes()} m={G0.number_of_edges()} classes={labels(G0).max() + 1}")
        for name in models:
            fit, gen, binary_only = MODELS[name]
            ref = Gbin if binary_only else G0
            ref_m = reference_metrics(ref)
            try:
                t0 = time.time()
                params = fit(ref)
                fit_time = time.time() - t0
                for r in range(reps):
                    t0 = time.time()
                    Gg = gen(params, seed=r)
                    rt = time.time() - t0
                    m = compute_metrics(preprocess_keep(Gg), ref)
                    m.update(objectives(m, ref_m))
                    m.update(dataset=ds, model=name, rep=r, runtime=rt + fit_time / reps,
                             n_ref=ref.number_of_nodes(), binarized=binary_only and ref is Gbin)
                    rows.append(m)
                print(f"    {name:10s} ok")
            except NotImplementedError as e:
                print(f"    {name:10s} SKIPPED ({e})")
            except Exception:
                print(f"    {name:10s} FAILED"); traceback.print_exc()
        rows.append(dict(dataset=ds, model="REAL", rep=0, **ref_m_for(G0)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "results_raw.csv"), index=False)
    num = df.select_dtypes("number").columns.drop("rep", errors="ignore")
    summ = df.groupby(["dataset", "model"])[list(num)].agg(["mean", "std"])
    summ.to_csv(os.path.join(out, "results_summary.csv"))
    show = df[df.model != "REAL"].groupby(["dataset", "model"])[
        ["f1_homophily", "f2_motif", "f3_structure", "f4_privacy", "runtime"]].mean()
    print("\n=== mean objectives (lower is better; f4 = edge overlap rate) ===")
    print(show.round(4).to_string())
    return df


def preprocess_keep(G):
    """Generated graphs keep ALL nodes (no LCC) so node alignment stays valid."""
    G = nx.Graph(G)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    return nx.convert_node_labels_to_integers(G)


def ref_m_for(G):
    return reference_metrics(G)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["karate", "synthetic:ws"])
    ap.add_argument("--models", nargs="+", default=["er", "cm", "bah", "canonical"])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="results")
    ap.add_argument("--edge-cols", nargs=2, type=int, default=[0, 1])
    ap.add_argument("--label-col", type=int, default=1)
    ap.add_argument("--label-key", default="value")
    ap.add_argument("--binarize", choices=["ovr", "top2"], default="ovr")
    a = ap.parse_args()
    run(a.datasets, a.models, a.reps, a.out, tuple(a.edge_cols), a.label_col, a.label_key, a.binarize)
