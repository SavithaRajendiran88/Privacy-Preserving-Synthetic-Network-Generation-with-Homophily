"""Final pipeline: eps = 1 DP-stcm tuned by multi-objective optimisation.

theta = (rho, p, w): budget split, triangle-closing probability, smoothing.
Objectives f1-f4, optimizers NSGA-II, NSGA-III, MOEA/D, SMS-EMOA, ParEGO, random search.

"""
import argparse
import json
import os
import pickle
import platform
import sys
import time
import urllib.request
import warnings
from collections import deque

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")        # one thread per worker process
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import networkx as nx                                 # noqa: E402
import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
from scipy.sparse.csgraph import connected_components, shortest_path   # noqa: E402
from scipy.stats import friedmanchisquare, ks_2samp, mannwhitneyu, rankdata, spearmanr  # noqa: E402

# settings

OUT = os.environ.get("FYP_OUT", "fyp_results")       # all results go here
DATA_DIR = os.environ.get("FYP_DATA", "data")        # downloaded datasets are cached here
DATASETS = ["cora", "polblogs", "chameleon", "texas", "github"]
DS_NAME = {"cora": "Cora", "polblogs": "PolBlogs", "chameleon": "Chameleon", "texas": "Texas", "github": "GitHub"}

EPS = 1.0
RHO_LO, RHO_HI = 0.05, 0.95

ALGOS = ["nsga2", "nsga3", "moead", "smsemoa", "parego", "random"]
ALGO_NAME = {"nsga2": "NSGA-II", "nsga3": "NSGA-III", "moead": "MOEA/D", "smsemoa": "SMS-EMOA",
             "parego": "ParEGO", "random": "Random search"}
OBJ = ["f1_homophily", "f2_motif", "f3_structure", "f4_privacy"]
OBJ_NAME = {"f1_homophily": "f1 homophily error", "f2_motif": "f2 motif error",
            "f3_structure": "f3 structure error", "f4_privacy": "f4 leakage (edge-overlap rate)"}
VARS = ["rho", "p", "w"]
METRIC_COLS = ["edge_h", "newman_r", "mix_dist", "clustering", "hom_tri", "deg_ks", "deg_gini", "aspl",
               "eor", "eor_lift"]
N_VAR, N_OBJ = 3, 4
POP = 20
BUDGET = 300                      # evaluations per run
RUNS = 10                         # independent runs per (dataset, optimizer)
REEVAL_SEEDS = 3
REEVAL_CAP = 30
FINAL_SEEDS = 5
HV_REF = 1.1
STAGE2_DEFAULT = dict(rho=0.3, p=0.0, w=0.0)          # Stage 2 setting (clustering off, no smoothing)


def log(msg):
    print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)


# datasets

GH = "https://raw.githubusercontent.com"
URLS = {
    "cora": {f"ind.cora.{s}": f"{GH}/kimiyoung/planetoid/master/data/ind.cora.{s}"
             for s in ("ally", "ty", "graph", "test.index")},
    "polblogs": {"polblogs.npz": f"{GH}/danielzuegner/gnn-meta-attack/master/data/polblogs.npz"},
    "chameleon": {"chameleon_edges.csv": f"{GH}/benedekrozemberczki/MUSAE/master/input/edges/chameleon_edges.csv",
                  "chameleon_target.csv": f"{GH}/benedekrozemberczki/MUSAE/master/input/target/chameleon_target.csv"},
    "texas": {"out1_graph_edges.txt": f"{GH}/graphdml-uiuc-jlu/geom-gcn/master/new_data/texas/out1_graph_edges.txt",
              "out1_node_feature_label.txt": f"{GH}/graphdml-uiuc-jlu/geom-gcn/master/new_data/texas/out1_node_feature_label.txt"},
    "github": {"git_edges.csv": f"{GH}/benedekrozemberczki/MUSAE/master/input/edges/git_edges.csv",
               "git_target.csv": f"{GH}/benedekrozemberczki/MUSAE/master/input/target/git_target.csv"},
}


def _get(name, fname):
    """Local path of a dataset file, downloaded on first use (local files always take precedence)."""
    d = os.path.join(DATA_DIR, name)
    os.makedirs(d, exist_ok=True)
    dest = os.path.join(d, fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    url = URLS[name][fname]
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
            f.write(r.read())
    except Exception as e:
        if os.path.exists(dest):
            os.remove(dest)
        raise RuntimeError(f"Could not download {fname} for {name} ({e}).\nDownload it by hand from\n  {url}\n"
                           f"and save it as\n  {dest}\nthen run the command again.") from e
    return dest


def load_cora():
    """2,708 papers, 7 topics; citations as undirected edges (Planetoid label assembly)."""
    P = {k: _get("cora", f"ind.cora.{k}") for k in ("ally", "ty", "graph", "test.index")}
    with open(P["ally"], "rb") as f:
        ally = pickle.load(f, encoding="latin1")
    with open(P["ty"], "rb") as f:
        ty = pickle.load(f, encoding="latin1")
    with open(P["graph"], "rb") as f:
        graph = pickle.load(f, encoding="latin1")
    with open(P["test.index"]) as f:
        test_idx = [int(l) for l in f]
    lab = np.vstack((np.asarray(ally.todense() if hasattr(ally, "todense") else ally),
                     np.asarray(ty.todense() if hasattr(ty, "todense") else ty)))
    lab[test_idx, :] = lab[sorted(test_idx), :]
    y = lab.argmax(1)
    G = nx.Graph()
    G.add_nodes_from((i, {"y": int(y[i])}) for i in range(len(y)))
    G.add_edges_from((u, v) for u, nb in graph.items() for v in nb if u != v and v < len(y) and u < len(y))
    return G


def load_polblogs():
    """1,490 political blogs, 2 groups; hyperlink direction dropped."""
    import scipy.sparse as sp
    d = np.load(_get("polblogs", "polblogs.npz"), allow_pickle=True)
    A = sp.csr_matrix((d["adj_data"], d["adj_indices"], d["adj_indptr"]), shape=tuple(d["adj_shape"]))
    y = np.asarray(d["labels"]).astype(int)
    G = nx.Graph()
    G.add_nodes_from((i, {"y": int(y[i])}) for i in range(len(y)))
    r, c = A.nonzero()
    G.add_edges_from((int(u), int(v)) for u, v in zip(r, c) if u != v)
    return G


def load_chameleon(n_classes=5):
    """2,277 Wikipedia pages; 5 classes = traffic quintiles (MUSAE target)."""
    e = pd.read_csv(_get("chameleon", "chameleon_edges.csv"))
    t = pd.read_csv(_get("chameleon", "chameleon_target.csv"))
    y = pd.qcut(t["target"].rank(method="first"), n_classes, labels=False).to_numpy()
    G = nx.Graph()
    G.add_nodes_from((int(i), {"y": int(c)}) for i, c in zip(t["id"], y))
    G.add_edges_from((int(u), int(v)) for u, v in zip(e.iloc[:, 0], e.iloc[:, 1]) if u != v)
    return G


def load_texas():
    """183 WebKB pages, 5 classes."""
    E = pd.read_csv(_get("texas", "out1_graph_edges.txt"), sep="\t")
    N = pd.read_csv(_get("texas", "out1_node_feature_label.txt"), sep="\t")
    G = nx.Graph()
    G.add_nodes_from((int(i), {"y": int(l)}) for i, l in zip(N["node_id"], N["label"]))
    G.add_edges_from((int(u), int(v)) for u, v in zip(E.iloc[:, 0], E.iloc[:, 1]) if u != v)
    return G


def load_github():
    """37,700 developers, 2 groups (ML vs web); subsampled to 5,000 nodes (see subsample_graph)."""
    e = pd.read_csv(_get("github", "git_edges.csv"))
    t = pd.read_csv(_get("github", "git_target.csv"))
    G = nx.Graph()
    G.add_nodes_from((int(i), {"y": int(c)}) for i, c in zip(t["id"], t["ml_target"]))
    G.add_edges_from((int(u), int(v)) for u, v in zip(e.iloc[:, 0], e.iloc[:, 1]) if u != v)
    return G


LOADERS = {"cora": load_cora, "polblogs": load_polblogs, "chameleon": load_chameleon,
           "texas": load_texas, "github": load_github}
MAX_NODES = {"github": 5000}


def _stats_for_sampling(G):
    y = {v: G.nodes[v]["y"] for v in G}
    classes = sorted(set(y.values()), key=str)
    share = {c: sum(1 for v in y.values() if v == c) / len(y) for c in classes}
    same = sum(1 for u, v in G.edges() if y[u] == y[v])
    deg = np.sort(np.array([d for _, d in G.degree()], float))
    n = len(deg)
    gini_ = float((2 * np.arange(1, n + 1) @ deg) / (n * deg.sum()) - (n + 1) / n) if deg.sum() else 0.0
    return dict(class_share=share, edge_h=same / max(G.number_of_edges(), 1), clustering=nx.transitivity(G),
                deg_gini=gini_)


def _forest_fire(adj, y, nodes, classes, shares, n_target, rng, p_burn=0.5):
    """Forest-fire crawl with per-class quotas (class shares are kept exactly)."""
    cap = {c: int(round(shares[i] * n_target)) for i, c in enumerate(classes)}
    cnt = {c: 0 for c in classes}
    by_class = {c: [v for v in nodes if y[v] == c] for c in classes}
    burned = set()

    def ok(w):
        return w not in burned and cnt[y[w]] < cap[y[w]]

    guard = 0
    while len(burned) < n_target and guard < 50 * n_target:
        guard += 1
        c = classes[rng.choice(len(classes), p=shares)]
        seed = by_class[c][rng.integers(len(by_class[c]))]
        if not ok(seed):
            continue
        burned.add(seed); cnt[y[seed]] += 1
        frontier = [seed]
        while frontier and len(burned) < n_target:
            u = frontier.pop(0)
            cand = [w for w in adj[u] if ok(w)]
            if not cand:
                continue
            k = min(len(cand), rng.geometric(1 - p_burn))
            for i in rng.choice(len(cand), size=k, replace=False):
                w = cand[i]
                if len(burned) < n_target and ok(w):
                    burned.add(w); cnt[y[w]] += 1
                    frontier.append(w)
    return burned


def subsample_graph(G, n_target, n_tries=8, seed=0):
    rng = np.random.default_rng(seed)
    y = {v: G.nodes[v]["y"] for v in G}
    nodes = list(G.nodes())
    classes = sorted(set(y.values()), key=str)
    shares = np.array([sum(1 for v in nodes if y[v] == c) for c in classes], float)
    shares /= shares.sum()
    adj = {v: list(G[v]) for v in G}
    full = _stats_for_sampling(G)
    best, best_d = None, 9e9
    for _ in range(n_tries):
        S = _forest_fire(adj, y, nodes, classes, shares, n_target, rng)
        H = G.subgraph(S).copy()
        H = H.subgraph(max(nx.connected_components(H), key=len)).copy()
        st = _stats_for_sampling(H)
        d = (sum(abs(st["class_share"].get(c, 0) - full["class_share"][c]) for c in classes)
             + abs(st["edge_h"] - full["edge_h"]) + abs(st["clustering"] - full["clustering"])
             + abs(st["deg_gini"] - full["deg_gini"]))
        if d < best_d:
            best, best_d = H, d
    return best


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


def load_dataset(name):
    G = LOADERS[name]()
    if name in MAX_NODES and G.number_of_nodes() > MAX_NODES[name]:
        G = subsample_graph(G, MAX_NODES[name])
    return preprocess(G, lcc=True)


_DATA = {}


def get_data(ds):
    """Real graph, labels, number of groups, real metric values and fitted non-private p."""
    if ds in _DATA:
        return _DATA[ds]
    path = os.path.join(OUT, "_cache", f"{ds}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            _DATA[ds] = pickle.load(fh)
        return _DATA[ds]
    G = load_dataset(ds)
    y = labels(G)
    d = dict(G=G, y=y, k=int(y.max()) + 1, ref=reference_metrics(G), hr=node_homophily(G, y))
    d["p_fit"] = fit_stcm_p(G, y, d["k"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as fh:
        pickle.dump(d, fh)
    os.replace(path + ".tmp", path)
    _DATA[ds] = d
    return d


# metrics

def labels(G):
    return np.array([G.nodes[v]["y"] for v in range(G.number_of_nodes())], dtype=int)


def adjacency(G):
    return nx.to_scipy_sparse_array(G, nodelist=range(G.number_of_nodes()), weight=None, format="csr", dtype=float)


def mixing_matrix(A, y, C):
    """e_ab = share of edge ends joining group a to group b (sums to 1)."""
    r, c = A.nonzero()
    M = np.zeros((C, C))
    np.add.at(M, (y[r], y[c]), 1.0)
    return M / max(M.sum(), 1.0)


def edge_homophily(e):                       # metric 1: share of same-group edges
    return float(np.trace(e))


def newman_r(e):                             # metric 2: Newman assortativity
    a, b = e.sum(1), e.sum(0)
    den = 1.0 - float(a @ b)
    return float((np.trace(e) - a @ b) / den) if den > 1e-12 else np.nan


def _triangles(A):
    """Triangle count (edges oriented by degree, so hubs stay cheap)."""
    import scipy.sparse as sp
    A = sp.csr_matrix(A)
    n = A.shape[0]
    if A.nnz == 0:
        return 0.0
    deg = np.asarray(A.sum(1)).ravel()
    rank = np.empty(n, dtype=np.int64)
    rank[np.lexsort((np.arange(n), deg))] = np.arange(n)
    r, c = A.nonzero()
    keep = rank[r] < rank[c]
    D = sp.csr_matrix((np.ones(int(keep.sum())), (r[keep], c[keep])), shape=(n, n))
    return float((D @ D).multiply(D).sum())


def clustering_and_hom_triangles(A, y, C):   # metrics 4, 5
    deg = np.asarray(A.sum(1)).ravel()
    triples = (deg * (deg - 1) / 2).sum()
    tri = _triangles(A)
    clust = 3 * tri / triples if triples > 0 else 0.0
    hom = 0.0
    for k in range(C):
        idx = np.where(y == k)[0]
        if len(idx) > 2:
            hom += _triangles(A[idx][:, idx])
    return float(clust), (float(hom / tri) if tri > 0 else np.nan)


def gini(x):                                 # metric 7
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) @ x) / (n * x.sum()) - (n + 1) / n) if x.sum() > 0 else 0.0


def avg_shortest_path(A, n_src=300, seed=0):
    k, comp = connected_components(A, directed=False)
    if k > 1:
        big = np.argmax(np.bincount(comp))
        idx = np.where(comp == big)[0]
        A = A[idx][:, idx]
    n = A.shape[0]
    if n < 2:
        return np.nan
    src = np.random.default_rng(seed).choice(n, size=min(n_src, n), replace=False)
    D = shortest_path(A, unweighted=True, indices=src)
    D = D[np.isfinite(D) & (D > 0)]
    return float(D.mean()) if D.size else np.nan


def _align_nodes(yg, dg, yr, dr):
    """Generated node -> real node with the same label and the same degree rank (Stage 1 alignment)."""
    mp = -np.ones(len(yg), dtype=int)
    for k in np.unique(yr):
        gi, ri = np.where(yg == k)[0], np.where(yr == k)[0]
        if len(gi) == 0 or len(ri) == 0:
            continue
        gi = gi[np.argsort(-dg[gi], kind="stable")]
        ri = ri[np.argsort(-dr[ri], kind="stable")]
        L = min(len(gi), len(ri))
        mp[gi[:L]] = ri[:L]
    return mp


def edge_overlap(Gg, Gr):                    # metric 9 (= f4) and 10 (lift over chance)
    yg, yr = labels(Gg), labels(Gr)
    dg = np.array([d for _, d in sorted(Gg.degree())])
    dr = np.array([d for _, d in sorted(Gr.degree())])
    mp = _align_nodes(yg, dg, yr, dr)
    nr = Gr.number_of_nodes()
    real = np.array([min(u, v) * nr + max(u, v) for u, v in Gr.edges()], dtype=np.int64)
    gen = [min(mp[u], mp[v]) * nr + max(mp[u], mp[v]) for u, v in Gg.edges()
           if mp[u] >= 0 and mp[v] >= 0 and mp[u] != mp[v]]
    hits = np.isin(np.array(gen, dtype=np.int64), real).sum() if gen else 0
    eor = hits / max(Gr.number_of_edges(), 1)
    chance = Gr.number_of_edges() / (nr * (nr - 1) / 2)
    return float(eor), float(eor / chance)


def compute_metrics(Gg, Gr):
    """The 10 Stage-1 metrics of generated graph Gg against real graph Gr."""
    C = int(max(labels(Gg).max(), labels(Gr).max())) + 1
    Ag, Ar = adjacency(Gg), adjacency(Gr)
    yg, yr = labels(Gg), labels(Gr)
    eg, er = mixing_matrix(Ag, yg, C), mixing_matrix(Ar, yr, C)
    cg, hg = clustering_and_hom_triangles(Ag, yg, C)
    dg, dr = np.asarray(Ag.sum(1)).ravel(), np.asarray(Ar.sum(1)).ravel()
    eor, lift = edge_overlap(Gg, Gr)
    return dict(edge_h=edge_homophily(eg), newman_r=newman_r(eg), mix_dist=float(np.linalg.norm(eg - er)),
                clustering=cg, hom_tri=hg, deg_ks=float(ks_2samp(dg, dr).statistic), deg_gini=gini(dg),
                aspl=avg_shortest_path(Ag), eor=eor, eor_lift=lift)


def reference_metrics(Gr):
    return compute_metrics(Gr, Gr)


def rel_err(g, r):
    """|g - r| / |r| clipped to [0, 1]; absolute difference when r ~ 0 (Stage 1 definition)."""
    if np.isnan(g) or np.isnan(r):
        return np.nan
    e = abs(g - r) / abs(r) if abs(r) > 1e-9 else abs(g - r)
    return float(min(e, 1.0))


def objectives(m, ref):
    f1 = np.nanmean([rel_err(m["edge_h"], ref["edge_h"]), rel_err(m["newman_r"], ref["newman_r"]), m["mix_dist"]])
    f2 = np.nanmean([rel_err(m["clustering"], ref["clustering"]), rel_err(m["hom_tri"], ref["hom_tri"])])
    f3 = np.nanmean([m["deg_ks"], rel_err(m["deg_gini"], ref["deg_gini"]), rel_err(m["aspl"], ref["aspl"])])
    return dict(f1_homophily=float(f1), f2_motif=float(f2), f3_structure=float(f3), f4_privacy=float(m["eor"]))


def node_homophily(G, y):
    """h_v = share of v's neighbours in v's own group (NaN for isolated nodes)."""
    h = np.full(len(y), np.nan)
    for v in range(len(y)):
        nb = list(G[v])
        if nb:
            h[v] = np.mean(y[nb] == y[v])
    return h


def edge_keys(G, n):
    return np.array(sorted(min(u, v) * n + max(u, v) for u, v in G.edges()), dtype=np.int64)


def mia_auc(Gr, Gs, y, rng):
    """Edge membership-inference AUC: synthetic-edge attacker and degree-only attacker."""
    from sklearn.metrics import roc_auc_score
    n = len(y)
    E = np.array(Gr.edges())
    m = len(E)
    real = set(map(int, edge_keys(Gr, n)))
    neg = []
    while len(neg) < m:
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


def with_labels(Gs, y):
    """Generated graph with all n nodes and the public labels."""
    G = nx.Graph()
    G.add_nodes_from(range(len(y)))
    G.add_edges_from((u, v) for u, v in Gs.edges() if u != v)
    for v in G:
        G.nodes[v]["y"] = int(y[v])
    return G


def full_evaluation(d, Gs, rel=None, rng=None):
    G, y = d["G"], d["y"]
    Gk = with_labels(Gs, y)
    m = compute_metrics(Gk, G)
    m.update(objectives(m, d["ref"]))
    n = len(y)
    er, es = edge_keys(G, n), edge_keys(Gk, n)
    hit = np.isin(es, er).sum()
    m.update(recall=hit / max(len(er), 1), precision=hit / max(len(es), 1), m_syn=len(es))
    hs = node_homophily(Gk, y)
    ok = ~np.isnan(d["hr"]) & ~np.isnan(hs)
    m["node_hom_err"] = float(np.mean(np.abs(hs[ok] - d["hr"][ok]))) if ok.any() else np.nan
    m["isolated_frac"] = float(np.mean([deg == 0 for _, deg in Gk.degree()]))
    m["mia_auc_edge"], m["mia_auc_degree"] = mia_auc(G, Gk, y, rng or np.random.default_rng(0))
    if rel is not None:
        B, K, Bp, Kp = rel
        iu = np.triu_indices(B.shape[0])
        m["block_l1"] = float(np.abs(Bp[iu] - B[iu]).sum() / max(B[iu].sum(), 1))
        m["kappa_l1"] = float(np.abs(Kp - K).sum() / max(K.sum(), 1))
    return m


# generator

def gen_stcm(y, kappa, k, p, seed):
    """Stratified configuration model, with triangle closing (prob. p) after each node."""
    rng = np.random.default_rng(seed)
    kap = np.asarray(kappa).copy()
    n = len(y)
    members = [np.where(y == a)[0] for a in range(k)]
    avail, pos = {}, {}
    for a in range(k):
        for f in range(k):
            lst = [int(v) for v in members[a] if kap[v, f] > 0]
            avail[(a, f)] = lst
            pos[(a, f)] = {v: i for i, v in enumerate(lst)}

    rem = kap.sum(1)

    def dec(v, f):
        kap[v, f] -= 1
        rem[v] -= 1
        if kap[v, f] == 0:
            key = (y[v], f)
            lst, ps = avail[key], pos[key]
            i = ps.pop(v)
            last = lst.pop()
            if i < len(lst):
                lst[i] = last
                ps[last] = i

    adj = [set() for _ in range(n)]
    order = np.lexsort((rng.random(n), -kap.sum(1)))
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
        if p > 0:
            nb = np.fromiter(adj[v], dtype=np.int64, count=len(adj[v]))
            for mu in nb:
                if rem[mu] == 0:
                    continue
                cand = nb[(rem[nb] > 0) & (nb != mu)]
                if len(cand) == 0:
                    continue
                cand = cand[(kap[cand, y[mu]] > 0) & (kap[mu, y[cand]] > 0)]
                i = -1
                while True:
                    i += int(rng.geometric(p))
                    if i >= len(cand) or rem[mu] == 0:
                        break
                    chi = int(cand[i])
                    if chi not in adj[mu] and kap[mu, y[chi]] > 0 and kap[chi, y[mu]] > 0:
                        adj[mu].add(chi); adj[chi].add(mu)
                        dec(chi, y[mu])
                        dec(mu, y[chi])
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from((u, v) for u in range(n) for v in adj[u] if u < v)
    return G


def fit_stcm_p(G, y, k, p_grid=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Non-private reference only (Stage 1): p calibrated on the real global clustering."""
    K = kappa_counts(G, y, k)
    target = nx.transitivity(G)
    best = min((abs(nx.transitivity(gen_stcm(y, K, k, pp, 4000)) - target), pp) for pp in p_grid)
    return float(best[1])


# privacy

def block_counts(G, y, k):
    """B[a, b] (a <= b): number of edges between group a and group b."""
    B = np.zeros((k, k))
    for u, v in G.edges():
        a, b = sorted((y[u], y[v]))
        B[a, b] += 1
    return B


def kappa_counts(G, y, k):
    """kappa[v, a]: number of neighbours of node v in group a."""
    K = np.zeros((len(y), k), dtype=int)
    for u, v in G.edges():
        K[u, y[v]] += 1
        K[v, y[u]] += 1
    return K


def privatize(B, K, eps, rho, rng):
    """Laplace noise: B gets rho*eps (sensitivity 1), kappa gets (1-rho)*eps (sensitivity 2)."""
    if not np.isfinite(eps):
        return B.astype(float).copy(), K.astype(float).copy()
    iu = np.triu_indices(B.shape[0])
    Bn = np.zeros_like(B, dtype=float)
    Bn[iu] = B[iu] + rng.laplace(0.0, 1.0 / (rho * eps), size=len(iu[0]))
    Kn = K + rng.laplace(0.0, 2.0 / ((1.0 - rho) * eps), size=K.shape)
    return Bn, Kn


def _round_to_total(x, total, cap, rng):
    """Non-negative integers close to x, summing to `total`, each <= cap (largest-remainder rounding)."""
    total = int(min(total, cap * len(x)))
    if total <= 0 or len(x) == 0:
        return np.zeros(len(x), dtype=int)
    s = x.sum()
    x = x * (total / s) if s > 0 else np.full(len(x), total / len(x))
    out = np.minimum(np.floor(x), cap).astype(int)
    for _ in range(20):
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


def postprocess(Bn, Kn, y, k, w, rng):
    """Clip kappa, smooth it with w, then rescale it to the noisy block counts."""
    sizes = np.bincount(y, minlength=k)
    Bp = np.maximum(np.round(Bn), 0)
    caps = np.array([[sizes[b] - (1 if a == b else 0) for b in range(k)] for a in range(k)])
    X = np.clip(Kn, 0, caps[y])
    if w > 0:
        deg = X.sum(1, keepdims=True)
        mix = np.zeros((k, k))
        for a in range(k):
            ia = y == a
            tot = X[ia].sum()
            mix[a] = X[ia].sum(0) / tot if tot > 0 else 1.0 / k
        X = (1 - w) * X + w * deg * mix[y]
        X = np.clip(X, 0, caps[y])
    K = np.zeros(Kn.shape, dtype=int)
    for a in range(k):
        ia = np.where(y == a)[0]
        for b in range(k):
            lo, hi = min(a, b), max(a, b)
            total = 2 * Bp[a, a] if a == b else Bp[lo, hi]
            K[ia, b] = _round_to_total(X[ia, b], total, caps[a, b], rng)
    return Bp, K


def release(d, eps, rho, p, w, seed):
    G, y, k = d["G"], d["y"], d["k"]
    rng = np.random.default_rng(10_000 + int(seed))
    B, K = block_counts(G, y, k), kappa_counts(G, y, k)
    Bn, Kn = privatize(B, K, eps, rho, rng)
    Bp, Kp = postprocess(Bn, Kn, y, k, w, rng)
    Gs = gen_stcm(y, Kp, k, p, seed=int(seed))
    return Gs, (B, K, Bp, Kp)


def verify_dp(datasets, n_remove=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for ds in datasets:
        d = get_data(ds)
        G, y, k = d["G"], d["y"], d["k"]
        B, K = block_counts(G, y, k), kappa_counts(G, y, k)
        edges = list(G.edges())
        sB = sK = 0
        for i in rng.choice(len(edges), size=min(n_remove, len(edges)), replace=False):
            H = G.copy()
            H.remove_edge(*edges[i])
            sB = max(sB, np.abs(block_counts(H, y, k) - B).sum())
            sK = max(sK, np.abs(kappa_counts(H, y, k) - K).sum())
        Bn, Kn = privatize(B, K, np.inf, 0.3, rng)
        Bp, Kp = postprocess(Bn, Kn, y, k, 0.0, rng)
        noise = np.concatenate([(privatize(B, K, EPS, 0.3, np.random.default_rng(s))[1] - K).ravel() for s in range(20)])
        rows.append(dict(dataset=ds, edges_removed=min(n_remove, len(edges)), max_change_B=sB, max_change_kappa=sK,
                         eps_inf_exact=bool((Kp == K).all() and (Bp == B).all()),
                         kappa_noise_sd_measured=float(np.std(noise)),
                         kappa_noise_sd_theory=float(np.sqrt(2) * 2 / (0.7 * EPS))))
        log(f"verify {ds}: {rows[-1]}")
    os.makedirs(os.path.join(OUT, "verify"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "verify", "dp_verification.csv"), index=False)


# moo problem

def decode(x):
    """Unit cube -> (rho, p, w)."""
    x = np.clip(np.asarray(x, float), 0, 1)
    return dict(rho=float(RHO_LO + (RHO_HI - RHO_LO) * x[0]), p=float(x[1]), w=float(x[2]))


def score(ds, x, seed):
    """One evaluation: private release with settings x, then the metrics and f1-f4."""
    d = get_data(ds)
    t = decode(x)
    Gs, _ = release(d, EPS, t["rho"], t["p"], t["w"], seed)
    m = compute_metrics(with_labels(Gs, d["y"]), d["G"])
    o = objectives(m, d["ref"])
    out = {c: float(m[c]) for c in METRIC_COLS}
    out.update(o)
    return out


class Evaluator:

    def __init__(self, ds, run, budget):
        self.ds, self.run, self.budget = ds, run, budget
        self.rows = []

    def __call__(self, x):
        i = len(self.rows)
        t = time.time()
        r = score(self.ds, x, seed=100_000 * (self.run + 1) + i)
        dt = time.time() - t
        r.update(eval=i, x0=float(x[0]), x1=float(x[1]), x2=float(x[2]), **decode(x), eval_time=dt)
        self.rows.append(r)
        return np.array([r[c] for c in OBJ])

    def batch(self, X):
        return np.array([self(x) for x in np.atleast_2d(X)])

    @property
    def left(self):
        return self.budget - len(self.rows)


def initial_design(run, n=POP):
    """The same 20 starting points for every population-based optimizer in run r."""
    return np.random.default_rng(50_000 + run).random((n, N_VAR))


def nondominated_mask(F):
    F = np.asarray(F)
    keep = np.ones(len(F), bool)
    for i in range(len(F)):
        if keep[i] and (np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)).any():
            keep[i] = False
    return keep


# optimizers

def run_pymoo(name, ev, run):
    """NSGA-II, NSGA-III, MOEA/D and SMS-EMOA from pymoo."""
    from pymoo.algorithms.moo.moead import MOEAD
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
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
    ref = get_reference_directions("das-dennis", N_OBJ, n_partitions=3)     # 20 directions for 4 objectives
    ga = dict(crossover=SBX(prob=0.9, eta=20), mutation=PM(eta=20))         # NSGA-II / MOEA/D / SMS-EMOA papers
    if name == "nsga2":
        alg = NSGA2(pop_size=POP, sampling=X0, **ga)
    elif name == "nsga3":
        alg = NSGA3(ref_dirs=ref, pop_size=POP, sampling=X0, crossover=SBX(prob=1.0, eta=30), mutation=PM(eta=20))
    elif name == "moead":
        alg = MOEAD(ref_dirs=ref, n_neighbors=5, decomposition=Tchebicheff(), prob_neighbor_mating=0.9,
                    sampling=X0, **ga)
    elif name == "smsemoa":
        alg = SMSEMOA(pop_size=POP, n_offsprings=1, sampling=X0, **ga)
    else:
        raise ValueError(name)
    minimize(P(), alg, termination=("n_evals", ev.budget), seed=run, verbose=False)


def run_random(ev, run):
    """Uniform random search (floor every optimizer must beat); starts with the shared initial design."""
    X = np.vstack([initial_design(run), np.random.default_rng(60_000 + run).random((ev.budget, N_VAR))])
    for x in X[:ev.budget]:
        ev(x)


def _lhs(n, d, rng):
    return (rng.permuted(np.tile(np.arange(n), (d, 1)), axis=1).T + rng.random((n, d))) / n


def run_parego(ev, run, s=4, rho_aug=0.05, n_cand=4000):
    """ParEGO (Knowles 2006) with a scikit-learn GP and expected improvement."""
    from scipy.optimize import minimize as sp_min
    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    from pymoo.util.ref_dirs import get_reference_directions

    rng = np.random.default_rng(80_000 + run)
    lam_set = get_reference_directions("das-dennis", N_OBJ, n_partitions=s)
    n0 = min(11 * N_VAR - 1, ev.budget)
    X = _lhs(n0, N_VAR, rng)
    F = ev.batch(X)
    while ev.left > 0:
        lo, hi = F.min(0), F.max(0)
        Fn = (F - lo) / np.where(hi - lo > 1e-12, hi - lo, 1.0)
        lam = lam_set[rng.integers(len(lam_set))]
        g = np.max(Fn * lam, axis=1) + rho_aug * np.sum(Fn * lam, axis=1)
        try:
            kern = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.3] * N_VAR, length_scale_bounds=(1e-2, 1e2),
                                                              nu=2.5) + WhiteKernel(1e-3, (1e-8, 1e-1))
            gp = GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=2,
                                          random_state=int(rng.integers(1 << 30)))
            gp.fit(X, g)
            best = g.min()

            def neg_ei(z):
                mu, sd = gp.predict(np.atleast_2d(z), return_std=True)
                sd = np.maximum(sd, 1e-9)
                u = (best - mu) / sd
                return -((best - mu) * norm.cdf(u) + sd * norm.pdf(u))

            top = X[np.argsort(g)[:10]]
            C = np.vstack([rng.random((n_cand // 2, N_VAR)),
                           np.clip(top[rng.integers(len(top), size=n_cand // 2)] +
                                   rng.normal(0, 0.05, (n_cand // 2, N_VAR)), 0, 1)])
            ei = -neg_ei(C)
            x0 = C[int(np.argmax(ei))]
            r = sp_min(lambda z: float(neg_ei(z)[0]), x0, bounds=[(0, 1)] * N_VAR, method="L-BFGS-B",
                       options={"maxiter": 50})
            x = np.clip(r.x if r.success and -r.fun >= ei.max() else x0, 0, 1)
        except Exception:
            x = rng.random(N_VAR)
        X = np.vstack([X, x])
        F = np.vstack([F, ev(x)])


def run_algorithm(name, ev, run):
    if name in ("nsga2", "nsga3", "moead", "smsemoa"):
        run_pymoo(name, ev, run)
    elif name == "parego":
        run_parego(ev, run)
    elif name == "random":
        run_random(ev, run)
    else:
        raise ValueError(name)


# runner

def maxmin_subset(F, k):
    """Greedy max-min spread subset in normalised objective space (same rule for every optimizer)."""
    if len(F) <= k:
        return np.arange(len(F))
    lo, hi = F.min(0), F.max(0)
    Z = (F - lo) / np.where(hi - lo > 1e-12, hi - lo, 1.0)
    chosen = [int(np.argmin(Z.sum(1)))]
    dist = np.linalg.norm(Z - Z[chosen[0]], axis=1)
    while len(chosen) < k:
        j = int(np.argmax(dist))
        chosen.append(j)
        dist = np.minimum(dist, np.linalg.norm(Z - Z[j], axis=1))
    return np.array(sorted(chosen))


def job_paths(ds, alg, run, tag="runs"):
    base = os.path.join(OUT, tag, ds, alg)
    return (os.path.join(base, f"run{run}_evals.csv"), os.path.join(base, f"run{run}_front.csv"),
            os.path.join(base, f"run{run}_info.json"))


def job_done(ds, alg, run, tag="runs"):
    """True if the job finished without error (failed jobs are re-run automatically)."""
    pi = job_paths(ds, alg, run, tag)[2]
    if not os.path.exists(pi):
        return False
    try:
        with open(pi) as fh:
            return not json.load(fh).get("error")
    except Exception:
        return False


def run_job(args):
    """One (dataset, optimizer, run) job; the front is re-scored on fresh noise."""
    ds, alg, run, budget, tag, out = args
    global OUT
    OUT = out
    pe, pf, pi = job_paths(ds, alg, run, tag)
    if job_done(ds, alg, run, tag):
        return ds, alg, run, "skip", 0.0
    os.makedirs(os.path.dirname(pe), exist_ok=True)
    get_data(ds)
    t0 = time.time()
    ev = Evaluator(ds, run, budget)
    err = ""
    try:
        run_algorithm(alg, ev, run)
    except Exception as e:
        import traceback
        err = (repr(e) + " | " + traceback.format_exc().strip().splitlines()[-3])[:400]
    t_search = time.time() - t0
    if not ev.rows:                                             # optimizer failed before evaluating anything
        info = dict(dataset=ds, algo=alg, run=run, budget=budget, n_evals=0, t_search=t_search, t_eval=0.0,
                    t_overhead=t_search, t_reeval=0.0, n_front=0, error=err or "no evaluations",
                    finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(pi, "w") as fh:
            json.dump(info, fh, indent=1)
        return ds, alg, run, "ERROR " + info["error"], time.time() - t0
    ev_df = pd.DataFrame(ev.rows)
    ev_df = ev_df[ev_df["eval"] < budget]                       # hard budget cap, identical for all
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
        r.update(eval=int(ev_df.iloc[i]["eval"]), x0=x[0], x1=x[1], x2=x[2], **decode(x))
        rows.append(r)
    ev_df.to_csv(pe, index=False)
    pd.DataFrame(rows).to_csv(pf, index=False)
    info = dict(dataset=ds, algo=alg, run=run, budget=budget, n_evals=int(len(ev_df)), t_search=t_search,
                t_eval=float(ev_df["eval_time"].sum()), t_overhead=t_search - float(ev_df["eval_time"].sum()),
                t_reeval=time.time() - t1, n_front=len(rows), error=err,
                finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(pi + ".tmp", "w") as fh:
        json.dump(info, fh, indent=1)
    os.replace(pi + ".tmp", pi)
    return ds, alg, run, ("ERROR " + err if err else "ok"), time.time() - t0


SIZE_ORDER = {"github": 0, "chameleon": 1, "polblogs": 2, "cora": 3, "texas": 4}


def cmd_run(datasets, algos, runs, budget, workers, tag="runs"):
    """Run all jobs, skipping finished ones."""
    for ds in datasets:
        get_data(ds)                                            # build caches before starting workers
    jobs = [(ds, a, r, budget, tag, OUT) for ds in sorted(datasets, key=lambda d: SIZE_ORDER.get(d, 9))
            for r in range(runs) for a in algos]
    todo = [j for j in jobs if not job_done(j[0], j[1], j[2], tag)]
    log(f"[{tag}] {len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run, {workers} worker(s)")
    t0, done = time.time(), 0
    with open(os.path.join(OUT, f"log_{tag}.txt"), "a", encoding="utf-8") as lg:
        def report(res):
            nonlocal done
            done += 1
            ds, alg, run, status, dt = res
            el = time.time() - t0
            msg = (f"[{done}/{len(todo)}] {ds:9s} {alg:8s} run {run}: {status} ({dt:.0f}s)  "
                   f"elapsed {el / 3600:.2f} h, about {el / done * (len(todo) - done) / 3600:.2f} h left")
            log(msg)
            lg.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
            lg.flush()

        if workers > 1 and todo:
            from multiprocessing import get_context
            with get_context("spawn").Pool(workers, maxtasksperchild=20) as pool:
                for res in pool.imap_unordered(run_job, todo):
                    report(res)
        else:
            for j in todo:
                report(run_job(j))


def cmd_status(tag="runs"):
    base = os.path.join(OUT, tag)
    if not os.path.exists(base):
        print("nothing has run yet")
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
    df = pd.DataFrame(rows)
    print(df.pivot(index="algo", columns="dataset", values="runs_done").fillna(0).astype(int).to_string())
    print(f"runs finished: {df.runs_done.sum()}   runs with errors: {df.errors.sum()}")


# analysis
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
IND = {"hv": +1, "igd_plus": -1, "eps_add": -1, "spacing": -1, "auc": +1, "noise_gap": -1, "overhead_s": -1,
       "n_front": +1}
IND_NAME = {"hv": "Hypervolume (% of reference)", "igd_plus": "IGD+", "eps_add": "Additive epsilon-indicator",
            "spacing": "Spacing", "auc": "Anytime HV (%)", "noise_gap": "Noise gap (% HV)",
            "overhead_s": "Optimizer overhead (s)", "n_front": "Front size", "worst_hv": "Worst-case HV (%)"}


def hypervolume(F):
    from pymoo.indicators.hv import HV
    F = np.asarray(F, float)
    F = F[np.all(F < HV_REF, axis=1)] if len(F) else F
    return float(HV(ref_point=np.full(N_OBJ, HV_REF))(F)) if len(F) else 0.0


def igd_plus(S, R):
    return float(np.sqrt((np.maximum(S[None] - R[:, None], 0) ** 2).sum(-1)).min(1).mean())


def eps_add(S, R):
    return float((S[None] - R[:, None]).max(-1).min(1).max())


def spacing(S):
    if len(S) < 3:
        return np.nan
    D = np.abs(S[:, None] - S[None]).sum(-1)
    np.fill_diagonal(D, np.inf)
    dmin = D.min(1)
    return float(np.sqrt(((dmin - dmin.mean()) ** 2).sum() / (len(dmin) - 1)))


def coverage(A, B):
    return float(np.mean([np.any(np.all(A <= b, axis=1)) for b in B])) if len(B) else np.nan


def nd(F):
    F = np.asarray(F)
    return F[nondominated_mask(F)] if len(F) else F


def holm(p):
    p = np.asarray(p, float)
    adj, run = np.empty_like(p), 0.0
    for i, j in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - i) * p[j]))
        adj[j] = run
    return adj


def load_runs(tag="runs"):
    base = os.path.join(OUT, tag)
    runs = []
    for ds in sorted(os.listdir(base)):
        for alg in sorted(os.listdir(os.path.join(base, ds))):
            dd = os.path.join(base, ds, alg)
            for f in sorted(os.listdir(dd)):
                if f.endswith("_info.json"):
                    r = int(f[3:].split("_")[0])
                    with open(os.path.join(dd, f)) as fh:
                        info = json.load(fh)
                    if info.get("error"):
                        log(f"skipping failed run {ds} {alg} run {r}: {info['error']}")
                        continue
                    runs.append(dict(dataset=ds, algo=alg, run=r, info=info,
                                     front=pd.read_csv(os.path.join(dd, f"run{r}_front.csv")),
                                     evals=pd.read_csv(os.path.join(dd, f"run{r}_evals.csv"))))
    return runs


def analyze(tag="runs"):
    runs = load_runs(tag)
    if not runs:
        log("no finished runs to analyse")
        return
    A = os.path.join(OUT, "analysis" if tag == "runs" else f"analysis_{tag}")
    os.makedirs(A, exist_ok=True)
    ind, curves, picks, norm_rows, cover_rows, reffronts = [], [], [], [], [], []
    for ds in [x for x in DATASETS if x in {r["dataset"] for r in runs}]:
        rs = [r for r in runs if r["dataset"] == ds]
        pool = np.vstack([r["front"][OBJ].to_numpy() for r in rs if len(r["front"])])
        ideal, nadir = pool.min(0), nd(pool).max(0)
        span = np.where(nadir - ideal > 1e-12, nadir - ideal, 1.0)
        sc = lambda F: (np.asarray(F, float) - ideal) / span    # noqa: E731
        R = nd(sc(pool))
        hvR = hypervolume(R)
        norm_rows.append(dict(dataset=ds, **{f"ideal_{c}": v for c, v in zip(OBJ, ideal)},
                              **{f"nadir_{c}": v for c, v in zip(OBJ, nadir)}, hv_ref_front=hvR, n_ref_front=len(R)))
        allpts = pd.concat([r["front"].assign(algo=r["algo"], run=r["run"]) for r in rs if len(r["front"])])
        refmask = nondominated_mask(allpts[OBJ].to_numpy())
        reffronts.append(allpts[refmask].assign(dataset=ds))
        fid_pool = pool[:, :3].mean(1)
        fl, fh = fid_pool.min(), fid_pool.max()
        ll, lh = pool[:, 3].min(), pool[:, 3].max()
        budget = max(r["info"]["budget"] for r in rs)
        checkpoints = np.arange(10, budget + 1, 10)
        for r in rs:
            fr = r["front"]
            S = nd(sc(fr[OBJ].to_numpy())) if len(fr) else np.zeros((0, N_OBJ))
            S_in = nd(sc(fr[[c + "_inrun" for c in OBJ]].to_numpy())) if len(fr) else S
            Fin = sc(r["evals"].sort_values("eval")[OBJ].to_numpy())
            hv_t = [hypervolume(nd(Fin[:t])) / hvR * 100 if t <= len(Fin) else np.nan for t in checkpoints]
            curves += [dict(dataset=ds, algo=r["algo"], run=r["run"], evals=int(t), hv=h) for t, h in zip(checkpoints, hv_t)]
            info = r["info"]
            ind.append(dict(dataset=ds, algo=r["algo"], run=r["run"],
                            hv=hypervolume(S) / hvR * 100 if len(S) else 0.0,
                            igd_plus=igd_plus(S, R) if len(S) else np.nan, eps_add=eps_add(S, R) if len(S) else np.nan,
                            spacing=spacing(S), n_front=len(S), auc=float(np.nanmean(hv_t)),
                            noise_gap=(hypervolume(S_in) - hypervolume(S)) / hvR * 100 if len(S) else np.nan,
                            overhead_s=info["t_overhead"], search_s=info["t_search"], eval_s=info["t_eval"],
                            n_evals=info["n_evals"], error=info["error"]))
            if len(fr):
                fid = fr[OBJ[:3]].to_numpy().mean(1)
                J = 0.5 * ((fid - fl) / max(fh - fl, 1e-12)) ** 2 + 0.5 * ((fr["f4_privacy"].to_numpy() - ll) / max(lh - ll, 1e-12)) ** 2
                j = int(np.argmin(J))
                row = fr.iloc[j]
                picks.append(dict(dataset=ds, algo=r["algo"], run=r["run"], J=float(J[j]), fidelity=float(fid[j]),
                                  **{c: float(row[c]) for c in OBJ + VARS + ["x0", "x1", "x2"]}))
        algs = sorted({r["algo"] for r in rs})
        byk = {(r["algo"], r["run"]): nd(sc(r["front"][OBJ].to_numpy())) for r in rs if len(r["front"])}
        for a in algs:
            for b in algs:
                vals = [coverage(byk[(a, k)], byk[(b, k)]) for (aa, k) in byk if aa == a and (b, k) in byk]
                cover_rows.append(dict(dataset=ds, A=a, B=b, C=float(np.nanmean(vals)) if vals else np.nan))
    ind = pd.DataFrame(ind)
    ind.to_csv(os.path.join(A, "indicators_per_run.csv"), index=False)
    pd.DataFrame(curves).to_csv(os.path.join(A, "anytime_curves.csv"), index=False)
    pd.DataFrame(picks).to_csv(os.path.join(A, "balanced_settings_per_run.csv"), index=False)
    pd.DataFrame(norm_rows).to_csv(os.path.join(A, "normalisation.csv"), index=False)
    pd.DataFrame(cover_rows).to_csv(os.path.join(A, "coverage.csv"), index=False)
    pd.concat(reffronts).to_csv(os.path.join(A, "reference_fronts.csv"), index=False)
    variable_effects(runs, A)
    verdict = statistics(ind, A)
    log(f"analysis of {len(ind)} runs written to {A}")
    print(verdict["text"], flush=True)
    return verdict


def statistics(ind, A):
    rows = []
    for ds, sub in ind.groupby("dataset"):
        for a, v in sub.groupby("algo")["hv"].min().items():
            rows.append(dict(dataset=ds, metric="worst_hv", algo=a, mean=v, sd=np.nan, median=v, mark="", p_holm=np.nan))
        for met, sgn in IND.items():
            g = sub.groupby("algo")[met]
            means = g.mean()
            best = means.idxmax() if sgn > 0 else means.idxmin()
            others = [a for a in means.index if a != best]
            pv = []
            for a in others:
                x, y_ = sub.loc[sub.algo == a, met].dropna(), sub.loc[sub.algo == best, met].dropna()
                same = len(x) and len(y_) and np.allclose(np.r_[x.to_numpy(), y_.to_numpy()], x.iloc[0])
                pv.append(mannwhitneyu(x, y_, alternative="two-sided").pvalue if len(x) > 1 and len(y_) > 1 and not same else 1.0)
            padj = dict(zip(others, holm(pv))) if pv else {}
            for a in means.index:
                mark = "best" if a == best else ("tie" if padj[a] >= 0.05 else "worse")
                rows.append(dict(dataset=ds, metric=met, algo=a, mean=means[a], sd=g.std()[a], median=g.median()[a],
                                 mark=mark, p_holm=padj.get(a, np.nan)))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(A, "summary_per_dataset.csv"), index=False)
    rk = []
    for (ds, met), s in summ.groupby(["dataset", "metric"]):
        r = rankdata(-IND.get(met, +1) * s["mean"].to_numpy(), method="average")
        rk += [dict(dataset=ds, metric=met, algo=a, rank=v) for a, v in zip(s["algo"], r)]
    avg_rank = pd.DataFrame(rk).pivot_table(index="algo", columns="metric", values="rank", aggfunc="mean")
    avg_rank["overall"] = avg_rank[["hv", "igd_plus", "eps_add", "auc", "worst_hv"]].mean(1)
    avg_rank.sort_values("overall").to_csv(os.path.join(A, "average_rank_by_dataset.csv"))
    algs = sorted(ind["algo"].unique())
    k = len(algs)
    blk = {}
    for met in ["hv", "igd_plus", "eps_add", "auc"]:
        piv = ind.pivot_table(index=["dataset", "run"], columns="algo", values=met).dropna()[algs]
        Rk = np.vstack([rankdata(-IND[met] * row, method="average") for row in piv.to_numpy()]) if len(piv) else np.zeros((0, k))
        N = len(piv)
        cd = Q05.get(k, 3.164) * np.sqrt(k * (k + 1) / (6.0 * N)) if N else np.nan
        try:
            stat, p = friedmanchisquare(*[piv[a].to_numpy() for a in algs]) if N > 1 and k > 2 else (np.nan, np.nan)
        except ValueError:
            stat, p = np.nan, np.nan
        blk[met] = dict(avg_rank=dict(zip(algs, Rk.mean(0) if N else [np.nan] * k)), n_blocks=int(N), cd=float(cd),
                        friedman_chi2=float(stat), friedman_p=float(p))
    with open(os.path.join(A, "friedman_nemenyi.json"), "w") as fh:
        json.dump(blk, fh, indent=1)
    w = summ[summ.metric == "worst_hv"].pivot_table(index="dataset", columns="algo", values="mean")
    wr = w.rank(axis=1, ascending=False).mean().to_dict()
    oh = ind.groupby("algo")["overhead_s"].mean().to_dict()
    hvr, cd = blk["hv"]["avg_rank"], blk["hv"]["cd"]
    win = sorted(hvr, key=hvr.get)[0]
    tied = [a for a in sorted(hvr, key=hvr.get) if hvr[a] - hvr[win] < cd]
    checks = {}
    for met in ["igd_plus", "eps_add"]:
        r = blk[met]["avg_rank"]
        bm = min(r, key=r.get)
        checks[met] = dict(best=bm, winner_rank=r[win], best_rank=r[bm], within_cd=bool(r[win] - r[bm] < blk[met]["cd"]))
    aucr = blk["auc"]["avg_rank"]
    rec = sorted(tied, key=lambda a: (round(aucr[a], 6), round(wr.get(a, 99), 6), oh.get(a, 1e9)))[0] if len(tied) > 1 else win
    txt = [f"Best average hypervolume rank: {ALGO_NAME[win]} ({hvr[win]:.2f}; critical difference {cd:.2f} over "
           f"{blk['hv']['n_blocks']} dataset-run blocks; Friedman p = {blk['hv']['friedman_p']:.3g})."]
    txt.append("Tied with it on hypervolume (within CD): " + ", ".join(ALGO_NAME[a] for a in tied[1:]) + "."
               if len(tied) > 1 else "No other optimizer is within the critical difference on hypervolume.")
    for met, c in checks.items():
        txt.append(f"{IND_NAME[met]}: best is {ALGO_NAME[c['best']]}; {ALGO_NAME[win]} is "
                   f"{'within' if c['within_cd'] else 'NOT within'} the critical difference.")
    txt.append(f"Recommended optimizer: {ALGO_NAME[rec]}" + (" (chosen within the tied group by anytime HV, then "
               "worst-case HV, then overhead)." if len(tied) > 1 else "."))
    verdict = dict(winner_hv=win, tied_group=tied, recommended=rec, checks=checks, text="\n".join(txt),
                   hv_avg_rank=hvr, cd=cd, worst_hv_rank=wr, overhead_mean_s=oh)
    with open(os.path.join(A, "verdict.json"), "w") as fh:
        json.dump(verdict, fh, indent=1, default=float)
    return verdict


def variable_effects(runs, A):
    """How each decision variable affects each objective."""
    rows, bins = [], []
    for ds in DATASETS:
        ev = [r["evals"] for r in runs if r["dataset"] == ds and r["algo"] == "random"]
        if not ev:
            continue
        E = pd.concat(ev)
        for v in VARS:
            for o in OBJ + ["edge_h", "clustering", "deg_ks", "eor"]:
                rho_s, pval = spearmanr(E[v], E[o])
                rows.append(dict(dataset=ds, variable=v, target=o, spearman=float(rho_s), p_value=float(pval), n=len(E)))
            E["_bin"] = pd.cut(E[v], np.linspace(E[v].min() - 1e-9, E[v].max() + 1e-9, 6), labels=False)
            for b, s in E.groupby("_bin"):
                bins.append(dict(dataset=ds, variable=v, bin=int(b), center=float(s[v].mean()), n=len(s),
                                 **{o: float(s[o].mean()) for o in OBJ}))
        corr = E[OBJ].corr(method="spearman")
        for a in OBJ:
            for b in OBJ:
                rows.append(dict(dataset=ds, variable="objective:" + a, target=b, spearman=float(corr.loc[a, b]),
                                 p_value=np.nan, n=len(E)))
    pd.DataFrame(rows).to_csv(os.path.join(A, "variable_effects_spearman.csv"), index=False)
    pd.DataFrame(bins).to_csv(os.path.join(A, "variable_effects_binned.csv"), index=False)


# final networks

def _eval_setting(args):
    """Worker: FINAL_SEEDS releases of one setting, full evaluation of each."""
    ds, label, eps, rho, p, w, out = args
    global OUT
    OUT = out
    d = get_data(ds)
    rows = []
    for j in range(FINAL_SEEDS):
        seed = 7_000_000 + j
        Gs, rel = release(d, eps, rho, p, w, seed)
        m = full_evaluation(d, Gs, rel, np.random.default_rng(seed))
        m.update(dataset=ds, label=label, seed=j, eps=eps, rho=rho, p=p, w=w)
        rows.append(m)
        if j == 0:
            ed = os.path.join(OUT, "final", "edges")
            os.makedirs(ed, exist_ok=True)
            pd.DataFrame(list(Gs.edges()), columns=["source", "target"]).to_csv(
                os.path.join(ed, f"{ds}_{label}.csv"), index=False)
    return rows


def choose_final_settings(A):
    with open(os.path.join(A, "verdict.json")) as fh:
        rec = json.load(fh)["recommended"]
    ind = pd.read_csv(os.path.join(A, "indicators_per_run.csv"))
    picks = pd.read_csv(os.path.join(A, "balanced_settings_per_run.csv"))
    rows = []
    for ds in [x for x in DATASETS if x in set(picks.dataset)]:
        pairs = [(rec, "moo_recommended")] + ([("nsga2", "moo_nsga2")] if rec != "nsga2" else [])
        for alg, lab in pairs:
            sub = ind[(ind.dataset == ds) & (ind.algo == alg)]
            if not len(sub):
                continue
            run = int(sub.iloc[(sub.hv - sub.hv.median()).abs().argsort().iloc[0]]["run"])
            pk = picks[(picks.dataset == ds) & (picks.algo == alg) & (picks.run == run)]
            if len(pk):
                r = pk.iloc[0]
                rows.append(dict(dataset=ds, label=lab, algo=alg, run=run, rho=r.rho, p=r.p, w=r.w,
                                 **{c: r[c] for c in OBJ}))
    df = pd.DataFrame(rows).drop_duplicates(subset=["dataset", "algo"])
    return rec, df


def cmd_final(workers):
    A = os.path.join(OUT, "analysis")
    F = os.path.join(OUT, "final")
    os.makedirs(F, exist_ok=True)
    rec, chosen = choose_final_settings(A)
    chosen.to_csv(os.path.join(F, "chosen_settings.csv"), index=False)
    jobs = []
    for ds in [x for x in DATASETS if x in set(chosen.dataset)]:
        d = get_data(ds)
        os.makedirs(os.path.join(F, "edges"), exist_ok=True)
        pd.DataFrame(list(d["G"].edges()), columns=["source", "target"]).to_csv(
            os.path.join(F, "edges", f"{ds}_real.csv"), index=False)
        pd.DataFrame(dict(node=range(len(d["y"])), group=d["y"])).to_csv(os.path.join(F, "edges", f"{ds}_labels.csv"), index=False)
        jobs.append((ds, "nonprivate_stcm", np.inf, 0.3, d["p_fit"], 0.0, OUT))
        jobs.append((ds, "stage2_default", EPS, STAGE2_DEFAULT["rho"], STAGE2_DEFAULT["p"], STAGE2_DEFAULT["w"], OUT))
        for r in chosen[chosen.dataset == ds].itertuples():
            jobs.append((ds, r.label, EPS, float(r.rho), float(r.p), float(r.w), OUT))
    log(f"final networks: {len(jobs)} settings x {FINAL_SEEDS} seeds")
    rows = []
    if workers > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(workers) as pool:
            for res in pool.imap_unordered(_eval_setting, jobs):
                rows += res
                log(f"   done {res[0]['dataset']} {res[0]['label']}")
    else:
        for j in jobs:
            rows += _eval_setting(j)
            log(f"   done {j[0]} {j[1]}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(F, "final_metrics_all_seeds.csv"), index=False)
    real = []
    for ds in df.dataset.unique():
        d = get_data(ds)
        real.append(dict(dataset=ds, label="REAL", **d["ref"], node_hom_err=0.0, recall=1.0, precision=1.0,
                         m_syn=d["G"].number_of_edges(), isolated_frac=0.0,
                         deg_baseline_recall=degree_baseline_recall(d["G"])))
    num = df.select_dtypes("number").columns.drop(["seed"], errors="ignore")
    mean = df.groupby(["dataset", "label"])[list(num)].mean().reset_index()
    sd = df.groupby(["dataset", "label"])[list(num)].std().reset_index()
    pd.concat([pd.DataFrame(real), mean]).to_csv(os.path.join(F, "final_metrics_mean.csv"), index=False)
    sd.to_csv(os.path.join(F, "final_metrics_sd.csv"), index=False)
    log(f"final networks written to {F} (recommended optimizer: {ALGO_NAME[rec]})")


# figures
BLUE, ORANGE, AQUA, YELLOW, GREY, LIGHT, INK, MUTED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#9a9893",
                                                       "#d3d1cb", "#0b0b0b", "#52514e")
GROUP_COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
LABEL_NAME = {"REAL": "Original graph", "nonprivate_stcm": "Non-private stcm (no privacy)",
              "stage2_default": "Stage 2 default, eps = 1", "moo_recommended": "MOO recommended, eps = 1",
              "moo_nsga2": "MOO NSGA-II, eps = 1"}
LABEL_COL = {"nonprivate_stcm": GREY, "stage2_default": AQUA, "moo_recommended": BLUE, "moo_nsga2": ORANGE}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#b5b3ad", "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#ecebe7",
                         "grid.linewidth": 0.6, "figure.dpi": 150, "axes.titlesize": 9})
    return plt


def _save(fig, name):
    d = os.path.join(OUT, "figures")
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, name), bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _algo_col(a, rec):
    return BLUE if a == rec else (ORANGE if a == "nsga2" else GREY)


def figures_optimizers():
    plt = _plt()
    A = os.path.join(OUT, "analysis")
    ind = pd.read_csv(os.path.join(A, "indicators_per_run.csv"))
    with open(os.path.join(A, "verdict.json")) as fh:
        rec = json.load(fh)["recommended"]
    with open(os.path.join(A, "friedman_nemenyi.json")) as fh:
        blk = json.load(fh)
    dss = [x for x in DATASETS if x in set(ind.dataset)]
    algs = sorted(blk["hv"]["avg_rank"], key=blk["hv"]["avg_rank"].get)

    # per-run indicator strips (one dot per run, bar = mean)
    for met, fname, title, logx in [("hv", "fig_hv.png", "Hypervolume of the re-scored front (% of pooled reference; higher is better)", False),
                                    ("igd_plus", "fig_igd_plus.png", "IGD+ (lower is better)", False),
                                    ("eps_add", "fig_eps_indicator.png", "Additive epsilon-indicator (lower is better)", False),
                                    ("overhead_s", "fig_overhead.png", "Optimizer overhead per run, seconds, log scale", True)]:
        fig, axes = plt.subplots(1, len(dss), figsize=(2.2 * len(dss) + 0.9, 0.3 * len(algs) + 1.2), sharey=True, squeeze=False)
        for ax, ds in zip(axes[0], dss):
            sub = ind[ind.dataset == ds]
            for i, a in enumerate(algs[::-1]):
                v = sub.loc[sub.algo == a, met].dropna()
                c = _algo_col(a, rec)
                ax.scatter(v, np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), s=10, color=c, alpha=0.6, edgecolor="none")
                if len(v):
                    ax.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color=c if c != GREY else INK, lw=2)
            ax.set_title(DS_NAME[ds])
            ax.set_yticks(range(len(algs)))
            ax.set_yticklabels([ALGO_NAME[a] for a in algs[::-1]])
            ax.grid(axis="y", visible=False)
            if logx:
                ax.set_xscale("log")
        fig.supxlabel(title, fontsize=8, color=MUTED)
        fig.tight_layout()
        _save(fig, fname)

    # anytime hypervolume
    cv = pd.read_csv(os.path.join(A, "anytime_curves.csv")).groupby(["dataset", "algo", "evals"])["hv"].median().reset_index()
    fig, axes = plt.subplots(1, len(dss), figsize=(2.4 * len(dss), 2.5), squeeze=False)
    for ax, ds in zip(axes[0], dss):
        s = cv[cv.dataset == ds]
        for a in algs:
            t = s[s.algo == a]
            hl = a in (rec, "nsga2", "random")
            ax.plot(t.evals, t.hv, color=_algo_col(a, rec) if a != "random" else INK, lw=2 if hl else 0.9,
                    ls="--" if a == "random" else "-", alpha=1 if hl else 0.6, label=ALGO_NAME[a] if hl else None)
        ax.set_title(DS_NAME[ds])
        ax.set_xlabel("evaluations")
    axes[0][0].set_ylabel("median HV so far (% of reference)")
    h, l = axes[0][0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=GREY, lw=0.9)); l.append("other optimizers")
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, "fig_anytime_hv.png")

    # critical-difference diagrams
    for met in ["hv", "igd_plus"]:
        b = blk[met]
        r, cd = b["avg_rank"], b["cd"]
        al = sorted(r, key=r.get)
        k = len(al)
        fig, ax = plt.subplots(figsize=(7.2, 0.26 * k + 1.3))
        ax.set_xlim(0.6, k + 0.4); ax.set_ylim(-0.2, 1); ax.axis("off")
        ax.plot([1, k], [0.85, 0.85], color=INK, lw=0.8)
        for t in range(1, k + 1):
            ax.plot([t, t], [0.85, 0.88], color=INK, lw=0.8)
            ax.text(t, 0.91, str(t), ha="center", fontsize=7, color=MUTED)
        for i, a in enumerate(al):
            left = i < (k + 1) // 2
            yy = 0.75 - 0.08 * (i if left else k - 1 - i)
            xe = 0.75 if left else k + 0.25
            c = _algo_col(a, rec)
            ax.plot([r[a], r[a], xe], [0.85, yy, yy], color=c if c != GREY else MUTED, lw=0.8)
            ax.text(xe + (-0.05 if left else 0.05), yy, f"{ALGO_NAME[a]} ({r[a]:.2f})", ha="right" if left else "left",
                    va="center", fontsize=7.2, color=INK)
        groups = []
        for i in range(k):
            j = max(jj for jj in range(i, k) if r[al[jj]] - r[al[i]] < cd)
            if j > i and not any(g[0] <= i and g[1] >= j for g in groups):
                groups.append((i, j))
        for n_, (i, j) in enumerate(groups):
            ax.plot([r[al[i]] - 0.03, r[al[j]] + 0.03], [0.80 - 0.03 * n_] * 2, color=INK, lw=2.2, solid_capstyle="round")
        ax.plot([1, 1 + cd], [0.05, 0.05], color=INK, lw=1)
        ax.text(1 + cd / 2, 0.08, f"CD = {cd:.2f}", ha="center", fontsize=7, color=MUTED)
        ax.text(k + 0.35, 0.02, f"{IND_NAME[met]}: {b['n_blocks']} dataset-run blocks, Friedman p = {b['friedman_p']:.2g}",
                ha="right", fontsize=7, color=MUTED)
        _save(fig, f"fig_cd_{met}.png")

    # average-rank heat map
    rk = pd.read_csv(os.path.join(A, "average_rank_by_dataset.csv"), index_col=0)
    cols = [c for c in ["hv", "igd_plus", "eps_add", "auc", "worst_hv", "spacing", "noise_gap", "overhead_s", "overall"] if c in rk]
    short = {"hv": "HV", "igd_plus": "IGD+", "eps_add": "eps-ind.", "auc": "Anytime", "worst_hv": "Worst HV",
             "spacing": "Spacing", "noise_gap": "Noise gap", "overhead_s": "Overhead", "overall": "Overall"}
    fig, ax = plt.subplots(figsize=(0.8 * len(cols) + 2.2, 0.32 * len(rk) + 1.0))
    ax.imshow(rk[cols].to_numpy(), cmap="Blues_r", aspect="auto", vmin=1, vmax=len(rk))
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([short[c] for c in cols])
    ax.set_yticks(range(len(rk))); ax.set_yticklabels([ALGO_NAME[a] for a in rk.index]); ax.grid(False)
    for i in range(len(rk)):
        for j, c in enumerate(cols):
            v = rk[c].iloc[i]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7, color="white" if v < len(rk) * 0.45 else INK)
    ax.set_title("Average rank over datasets (1 = best)", loc="left", color=MUTED)
    fig.tight_layout()
    _save(fig, "fig_rank_heatmap.png")

    ref = pd.read_csv(os.path.join(A, "reference_fronts.csv"))
    fig, axes = plt.subplots(1, len(dss), figsize=(2.4 * len(dss), 2.6), squeeze=False)
    for ax, ds in zip(axes[0], dss):
        rf = ref[ref.dataset == ds]
        ax.scatter(rf.f4_privacy, rf[OBJ[:3]].mean(1), s=10, color=LIGHT, label="pooled reference front")
        for a in dict.fromkeys(["nsga2", rec]):
            sub = ind[(ind.dataset == ds) & (ind.algo == a)]
            if not len(sub):
                continue
            run = int(sub.iloc[(sub.hv - sub.hv.median()).abs().argsort().iloc[0]]["run"])
            fr = pd.read_csv(job_paths(ds, a, run)[1])
            ax.scatter(fr.f4_privacy, fr[OBJ[:3]].mean(1), s=16, color=_algo_col(a, rec), edgecolor="white", lw=0.5,
                       label=f"{ALGO_NAME[a]} (median run)")
        ax.set_title(DS_NAME[ds]); ax.set_xlabel("leakage f4")
    axes[0][0].set_ylabel("fidelity error mean(f1, f2, f3)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    _save(fig, "fig_fronts_fidelity_vs_leakage.png")

    # balanced settings chosen per optimizer (what theta the optimizers settle on)
    pk = pd.read_csv(os.path.join(A, "balanced_settings_per_run.csv"))
    fig, axes = plt.subplots(len(VARS), len(dss), figsize=(2.3 * len(dss) + 0.9, 2.0 * len(VARS)), sharey="row", squeeze=False)
    for i, v in enumerate(VARS):
        for j, ds in enumerate(dss):
            ax = axes[i][j]
            sub = pk[pk.dataset == ds]
            for t, a in enumerate(algs):
                vals = sub.loc[sub.algo == a, v]
                ax.scatter(np.full(len(vals), t) + np.linspace(-0.15, 0.15, len(vals)), vals, s=9,
                           color=_algo_col(a, rec), alpha=0.7, edgecolor="none")
            ax.set_xticks(range(len(algs)))
            ax.set_xticklabels([ALGO_NAME[a] for a in algs] if i == len(VARS) - 1 else [], rotation=60, fontsize=6.5)
            if i == 0:
                ax.set_title(DS_NAME[ds])
            if j == 0:
                ax.set_ylabel({"rho": "rho (budget split)", "p": "p (clustering)", "w": "w (smoothing)"}[v])
    fig.suptitle("Balanced setting (min J) of every run", fontsize=9, color=MUTED)
    fig.tight_layout()
    _save(fig, "fig_balanced_settings.png")

    # variable effects (random-search samples): correlation heat maps and binned trends
    ve = pd.read_csv(os.path.join(A, "variable_effects_spearman.csv"))
    vb = pd.read_csv(os.path.join(A, "variable_effects_binned.csv"))
    if len(ve):
        dsv = [x for x in dss if x in set(ve.dataset)]
        fig, axes = plt.subplots(1, len(dsv), figsize=(2.5 * len(dsv) + 0.8, 2.0), squeeze=False)
        for ax, ds in zip(axes[0], dsv):
            M = ve[(ve.dataset == ds) & ve.variable.isin(VARS) & ve.target.isin(OBJ)].pivot(index="variable", columns="target", values="spearman").reindex(index=VARS, columns=OBJ)
            ax.imshow(M.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(4)); ax.set_xticklabels(["f1", "f2", "f3", "f4"])
            ax.set_yticks(range(3)); ax.set_yticklabels(VARS if ax is axes[0][0] else []); ax.grid(False)
            for a_ in range(3):
                for b_ in range(4):
                    ax.text(b_, a_, f"{M.iloc[a_, b_]:+.2f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(M.iloc[a_, b_]) > 0.6 else INK)
            ax.set_title(DS_NAME[ds])
        fig.suptitle("Spearman correlation of each decision variable with each objective (random-search samples; "
                     "+ = raising the variable raises the error)", fontsize=8, color=MUTED)
        fig.tight_layout()
        _save(fig, "fig_variable_effects_heatmap.png")

        fig, axes = plt.subplots(1, len(dsv), figsize=(2.5 * len(dsv) + 0.8, 2.0), squeeze=False)
        for ax, ds in zip(axes[0], dsv):
            M = ve[(ve.dataset == ds) & (ve.variable.str.startswith("objective:"))].copy()
            M["variable"] = M.variable.str.replace("objective:", "")
            M = M.pivot(index="variable", columns="target", values="spearman").reindex(index=OBJ, columns=OBJ)
            ax.imshow(M.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(4)); ax.set_xticklabels(["f1", "f2", "f3", "f4"])
            ax.set_yticks(range(4)); ax.set_yticklabels(["f1", "f2", "f3", "f4"] if ax is axes[0][0] else []); ax.grid(False)
            for a_ in range(4):
                for b_ in range(4):
                    ax.text(b_, a_, f"{M.iloc[a_, b_]:+.2f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(M.iloc[a_, b_]) > 0.6 else INK)
            ax.set_title(DS_NAME[ds])
        fig.suptitle("Correlation between objectives (negative = conflicting objectives, i.e. a real trade-off)",
                     fontsize=8, color=MUTED)
        fig.tight_layout()
        _save(fig, "fig_objective_conflicts.png")

        oc = dict(zip(OBJ, [BLUE, ORANGE, AQUA, YELLOW]))
        fig, axes = plt.subplots(len(VARS), len(dsv), figsize=(2.4 * len(dsv), 1.9 * len(VARS)), squeeze=False)
        for i, v in enumerate(VARS):
            for j, ds in enumerate(dsv):
                ax = axes[i][j]
                s = vb[(vb.dataset == ds) & (vb.variable == v)].sort_values("center")
                for o in OBJ:
                    ax.plot(s.center, s[o], color=oc[o], lw=1.6, marker="o", ms=3, label=OBJ_NAME[o])
                if i == 0:
                    ax.set_title(DS_NAME[ds])
                if j == 0:
                    ax.set_ylabel("mean objective")
                ax.set_xlabel(v)
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03))
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        _save(fig, "fig_variable_trends.png")


def _draw(ax, edges, pos, y, nodes, node_size, ew, ealpha):
    from matplotlib.collections import LineCollection
    ns = set(nodes)
    E = [(u, v) for u, v in edges if u in ns and v in ns]
    if E:
        same = np.array([y[u] == y[v] for u, v in E])
        cols = np.where(same[:, None], [[0.45, 0.45, 0.45, ealpha]], [[0.85, 0.25, 0.25, ealpha]])
        ax.add_collection(LineCollection([(pos[u], pos[v]) for u, v in E], colors=cols, linewidths=ew, zorder=1))
    xy = np.array([pos[v] for v in nodes])
    ax.scatter(xy[:, 0], xy[:, 1], s=node_size, c=[GROUP_COL[y[v] % 8] for v in nodes], linewidths=0, zorder=2)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.set_aspect("equal"); ax.autoscale()
    for s in ax.spines.values():
        s.set_visible(False)


def figures_final():
    plt = _plt()
    F = os.path.join(OUT, "final")
    mean = pd.read_csv(os.path.join(F, "final_metrics_mean.csv"))
    dss = [x for x in DATASETS if x in set(mean.dataset)]
    labs = [l for l in ["nonprivate_stcm", "stage2_default", "moo_recommended", "moo_nsga2"] if l in set(mean.label)]
    graphs = {}
    for ds in dss:
        d = get_data(ds)
        G, y = d["G"], d["y"]
        ed = {"REAL": list(G.edges())}
        for l in labs:
            p = os.path.join(F, "edges", f"{ds}_{l}.csv")
            if os.path.exists(p):
                e = pd.read_csv(p)
                ed[l] = list(zip(e.source, e.target))
        graphs[ds] = ed
        # real vs generated drawings: same node positions in every panel; whole graph + zoom
        n = len(y)
        pos = nx.spring_layout(G, seed=0, iterations=60 if n > 3000 else 120, k=1.6 / np.sqrt(n))
        start = max(G.degree(), key=lambda t: t[1])[0]
        ball, q = [start], deque([start])
        while q and len(ball) < (120 if n > 300 else 80):
            u = q.popleft()
            for w_ in G[u]:
                if w_ not in ball and len(ball) < (120 if n > 300 else 80):
                    ball.append(w_); q.append(w_)
        pos_z = nx.spring_layout(G.subgraph(ball), seed=1, iterations=200)
        panels = ["REAL"] + [l for l in labs if l in ed]
        big = n > 1500
        fig, axes = plt.subplots(2, len(panels), figsize=(3.6 * len(panels), 7.6), squeeze=False)
        real_keys = set(map(tuple, map(sorted, G.edges())))
        for j, l in enumerate(panels):
            E = ed[l]
            Ce = np.zeros((d["k"], d["k"]))
            for u, v in E:
                Ce[y[u], y[v]] += 1; Ce[y[v], y[u]] += 1
            h = np.trace(Ce) / max(Ce.sum(), 1)
            sub = f"edges {len(E):,} | edge homophily {h:.3f}"
            if l != "REAL":
                share = np.mean([tuple(sorted(e)) in real_keys for e in E]) if E else 0
                sub += f"\nshare of edges that are real: {share:.1%}"
            _draw(axes[0][j], E, pos, y, list(range(n)), 2 if big else 7, 0.15 if big else 0.4, 0.08 if big else 0.35)
            axes[0][j].set_title(LABEL_NAME[l] + "\n" + sub, fontsize=8.5)
            _draw(axes[1][j], E, pos_z, y, ball, 22, 0.8, 0.7)
        axes[0][0].set_ylabel("whole graph")
        axes[1][0].set_ylabel(f"zoom: {len(ball)} nodes around the largest hub")
        hs = [plt.Line2D([], [], marker="o", ls="", color=GROUP_COL[c % 8], label=f"group {c}") for c in range(d["k"])]
        hs += [plt.Line2D([], [], color="#737373", label="same-group edge"), plt.Line2D([], [], color="#d94040", label="cross-group edge")]
        fig.legend(handles=hs, loc="lower center", ncol=min(d["k"] + 2, 9), frameon=False)
        fig.suptitle(f"{DS_NAME[ds]}: original graph vs generated graphs (same node positions in every panel)", fontsize=11)
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        _save(fig, f"sample_graphs_{ds}.png")

    # mixing matrices
    cols = ["REAL"] + labs
    fig, axes = plt.subplots(len(dss), len(cols), figsize=(2.8 * len(cols), 2.7 * len(dss)), squeeze=False)
    for i, ds in enumerate(dss):
        d = get_data(ds)
        y, k = d["y"], d["k"]
        for j, l in enumerate(cols):
            ax = axes[i][j]
            M = np.zeros((k, k))
            for u, v in graphs[ds].get(l, []):
                M[y[u], y[v]] += 1; M[y[v], y[u]] += 1
            M = M / max(M.sum(), 1)
            ax.imshow(M, cmap="Blues", vmin=0, vmax=mixing_matrix(adjacency(d["G"]), y, k).max())
            ax.grid(False); ax.set_xticks(range(k)); ax.set_yticks(range(k)); ax.tick_params(labelsize=6)
            if k <= 5:
                for a in range(k):
                    for b in range(k):
                        ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=6,
                                color="white" if M[a, b] > 0.6 * M.max() else INK)
            ax.set_title((LABEL_NAME[l] + "\n" if i == 0 else "") + f"{DS_NAME[ds]}  h = {np.trace(M):.3f}", fontsize=7.5)
    fig.suptitle("Group mixing matrices (share of edge ends between group a and group b)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, "fig_final_mixing_matrices.png")

    # degree distributions
    fig, axes = plt.subplots(1, len(dss), figsize=(3.0 * len(dss), 3.0), squeeze=False)
    for ax, ds in zip(axes[0], dss):
        for l in cols:
            E = graphs[ds].get(l)
            if not E:
                continue
            deg = np.bincount(np.array(E).ravel(), minlength=len(get_data(ds)["y"]))
            deg = np.sort(deg[deg > 0])
            ax.loglog(deg, 1.0 - np.arange(len(deg)) / len(deg), color=INK if l == "REAL" else LABEL_COL[l],
                      lw=2 if l in ("REAL", "moo_recommended") else 1.3, label=LABEL_NAME[l])
        ax.set_title(DS_NAME[ds]); ax.set_xlabel("degree k")
    axes[0][0].set_ylabel("share of nodes with degree >= k")
    axes[0][0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    _save(fig, "fig_final_degree_distributions.png")

    # metric comparison of the final networks (means over 5 seeds, error bars = sd)
    sd = pd.read_csv(os.path.join(F, "final_metrics_sd.csv"))
    mets = ["f1_homophily", "f2_motif", "f3_structure", "f4_privacy", "node_hom_err", "recall", "mia_auc_edge", "isolated_frac"]
    nice = {"f1_homophily": "f1 homophily error", "f2_motif": "f2 motif error", "f3_structure": "f3 structure error",
            "f4_privacy": "f4 leakage (EOR)", "node_hom_err": "node-level homophily error", "recall": "exact edge recall",
            "mia_auc_edge": "membership-inference AUC", "isolated_frac": "isolated-node share"}
    fig, axes = plt.subplots(len(mets), len(dss), figsize=(2.3 * len(dss), 1.55 * len(mets)), squeeze=False)
    for i, m in enumerate(mets):
        for j, ds in enumerate(dss):
            ax = axes[i][j]
            for t, l in enumerate(labs):
                r = mean[(mean.dataset == ds) & (mean.label == l)]
                e = sd[(sd.dataset == ds) & (sd.label == l)]
                if len(r):
                    ax.bar(t, r[m].iloc[0], yerr=e[m].iloc[0] if len(e) else 0, color=LABEL_COL[l], width=0.7,
                           capsize=2, edgecolor="white")
            if m == "recall":
                rr = mean[(mean.dataset == ds) & (mean.label == "REAL")]
                if len(rr):
                    ax.axhline(rr["deg_baseline_recall"].iloc[0], color=INK, ls=":", lw=1)
            if m == "mia_auc_edge":
                ax.axhline(0.5, color=INK, ls=":", lw=1)
            ax.set_xticks([])
            if i == 0:
                ax.set_title(DS_NAME[ds])
            if j == 0:
                ax.set_ylabel(nice[m], fontsize=7)
    hs = [plt.Rectangle((0, 0), 1, 1, color=LABEL_COL[l]) for l in labs]
    fig.legend(hs, [LABEL_NAME[l] for l in labs], loc="lower center", ncol=len(labs), frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.5, 0.995, "Final networks: lower is better for every panel (dotted: degree-only recall / AUC 0.5)",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.985))
    _save(fig, "fig_final_metrics.png")

    # homophily, motif and structure values against the real graph
    hm = ["edge_h", "newman_r", "clustering", "hom_tri", "deg_gini", "aspl"]
    hn = {"edge_h": "edge homophily", "newman_r": "Newman assortativity", "clustering": "global clustering",
          "hom_tri": "homophilous-triangle share", "deg_gini": "degree Gini", "aspl": "average path length"}
    fig, axes = plt.subplots(len(hm), len(dss), figsize=(2.3 * len(dss), 1.55 * len(hm)), squeeze=False)
    for i, m in enumerate(hm):
        for j, ds in enumerate(dss):
            ax = axes[i][j]
            for t, l in enumerate(labs):
                r = mean[(mean.dataset == ds) & (mean.label == l)]
                e = sd[(sd.dataset == ds) & (sd.label == l)]
                if len(r):
                    ax.bar(t, r[m].iloc[0], yerr=e[m].iloc[0] if len(e) else 0, color=LABEL_COL[l], width=0.7,
                           capsize=2, edgecolor="white")
            rr = mean[(mean.dataset == ds) & (mean.label == "REAL")]
            if len(rr):
                ax.axhline(rr[m].iloc[0], color=INK, ls="--", lw=1)
            ax.set_xticks([])
            if i == 0:
                ax.set_title(DS_NAME[ds])
            if j == 0:
                ax.set_ylabel(hn[m], fontsize=7)
    fig.legend(hs, [LABEL_NAME[l] for l in labs], loc="lower center", ncol=len(labs), frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.5, 0.995, "Final networks vs the original graph (dashed line = original value)", ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.985))
    _save(fig, "fig_final_homophily_motif_structure.png")


# command line

def preflight():
    import importlib
    missing = []
    for lib, pipname in [("numpy", "numpy"), ("scipy", "scipy"), ("pandas", "pandas"), ("networkx", "networkx"),
                         ("sklearn", "scikit-learn"), ("matplotlib", "matplotlib"), ("pymoo", "pymoo==0.6.2")]:
        try:
            m = importlib.import_module(lib)
            if lib == "pymoo" and not m.__version__.startswith("0.6"):
                missing.append(f"{pipname}   (found pymoo {m.__version__}, need 0.6.x)")
            elif lib == "pymoo":
                from pymoo.algorithms.moo.nsga2 import NSGA2   # noqa: F401
                from pymoo.indicators.hv import HV             # noqa: F401
        except Exception as e:
            missing.append(f"{pipname}   ({e.__class__.__name__}: {e})")
    if missing:
        print("\n" + "=" * 78)
        print("MISSING LIBRARIES in the Python that is running this file:")
        print("   " + sys.executable)
        for m in missing:
            print("   - " + m)
        print("\nFix:")
        print("  * Anaconda Prompt:  conda activate fyp   then   pip install " +
              " ".join(x.split()[0] for x in missing))
        print("  * Jupyter notebook: run this in a cell, then restart the kernel (Kernel > Restart):")
        print("        %pip install " + " ".join(x.split()[0] for x in missing))
        print("=" * 78 + "\n")
        sys.exit(1)


def cmd_data(datasets):
    """Download, preprocess and cache the datasets; save their summary and the run environment."""
    rows = []
    for ds in datasets:
        t = time.time()
        d = get_data(ds)
        G, y = d["G"], d["y"]
        rows.append(dict(dataset=ds, nodes=G.number_of_nodes(), edges=G.number_of_edges(), groups=d["k"],
                         group_shares=";".join(f"{s:.3f}" for s in np.bincount(y) / len(y)),
                         degree_mean=2 * G.number_of_edges() / G.number_of_nodes(), p_fit_nonprivate=d["p_fit"],
                         deg_baseline_recall=degree_baseline_recall(G), **d["ref"]))
        log(f"data {ds}: {rows[-1]['nodes']} nodes, {rows[-1]['edges']} edges, {d['k']} groups ({time.time() - t:.0f}s)")
    os.makedirs(os.path.join(OUT, "datasets"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "datasets", "summary.csv"), index=False)
    import importlib
    env = dict(python=sys.version, platform=platform.platform(), cpu_count=os.cpu_count(),
               settings=dict(EPS=EPS, RHO_RANGE=[RHO_LO, RHO_HI], P_RANGE=[0, 1], W_RANGE=[0, 1], POP=POP,
                             BUDGET=BUDGET, RUNS=RUNS, REEVAL_SEEDS=REEVAL_SEEDS, REEVAL_CAP=REEVAL_CAP,
                             FINAL_SEEDS=FINAL_SEEDS, HV_REF=HV_REF, ALGOS=ALGOS, DATASETS=datasets))
    for lib in ["numpy", "scipy", "pandas", "networkx", "sklearn", "matplotlib", "pymoo"]:
        try:
            env[lib] = importlib.import_module(lib).__version__
        except Exception as e:
            env[lib] = f"missing ({e})"
    with open(os.path.join(OUT, "environment.json"), "w") as fh:
        json.dump(env, fh, indent=1)


def main():
    global OUT
    try:
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "all", "data", "verify", "run", "status", "analyze", "final", "figures"])
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--algos", nargs="+", default=ALGOS, choices=ALGOS)
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=None, help="results folder (default fyp_results; check uses fyp_check)")
    a = ap.parse_args()
    OUT = a.out or ("fyp_check" if a.cmd == "check" else OUT)
    os.environ["FYP_OUT"] = OUT
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    if a.cmd in ("check", "all", "run", "analyze", "final", "figures", "verify", "data"):
        preflight()
    if a.cmd == "check":
        ds = ["texas", "cora"]
        cmd_data(ds)
        verify_dp(["texas"], n_remove=50)
        cmd_run(ds, ALGOS, 2, 40, a.workers)
        analyze()
        cmd_final(a.workers)
        figures_optimizers()
        figures_final()
        cmd_status()
        log(f"CHECK FINISHED in {(time.time() - t0) / 60:.1f} min. Results are in the folder {OUT}. "
            f"If 'runs with errors: 0' is shown above, start the full study.")
    elif a.cmd == "all":
        cmd_data(a.datasets)
        if not os.path.exists(os.path.join(OUT, "verify", "dp_verification.csv")):
            verify_dp(a.datasets)
        cmd_run(a.datasets, a.algos, a.runs, a.budget, a.workers)
        analyze()
        cmd_final(a.workers)
        figures_optimizers()
        figures_final()
        cmd_status()
        log(f"ALL DONE in {(time.time() - t0) / 3600:.2f} h")
    elif a.cmd == "data":
        cmd_data(a.datasets)
    elif a.cmd == "verify":
        verify_dp(a.datasets)
    elif a.cmd == "run":
        cmd_run(a.datasets, a.algos, a.runs, a.budget, a.workers)
    elif a.cmd == "status":
        cmd_status()
    elif a.cmd == "analyze":
        analyze()
    elif a.cmd == "final":
        cmd_final(a.workers)
    elif a.cmd == "figures":
        figures_optimizers()
        figures_final()


if __name__ == "__main__":
    main()
