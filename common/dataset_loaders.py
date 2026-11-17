"""Loaders for the real datasets: cora, polblogs, chameleon, texas, sociopatterns, github.

Files are downloaded on first use into data/<name>/. GitHub is subsampled to 5000 nodes
(class-stratified forest fire).

    python common/dataset_loaders.py --all
"""
import argparse
import warnings
import gzip
import io
import os
import pickle
import sys
import urllib.request

import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*align.*")
DATA_DIR = os.environ.get("HOMOPHILY_DATA_DIR", "data")

GH = "https://raw.githubusercontent.com"
URLS = {
    "cora": {f"ind.cora.{s}": f"{GH}/kimiyoung/planetoid/master/data/ind.cora.{s}"
             for s in ("ally", "ty", "graph", "test.index")},
    "polblogs": {"polblogs.npz": f"{GH}/danielzuegner/gnn-meta-attack/master/data/polblogs.npz"},
    "chameleon": {"chameleon_edges.csv": f"{GH}/benedekrozemberczki/MUSAE/master/input/edges/chameleon_edges.csv",
                  "chameleon_target.csv": f"{GH}/benedekrozemberczki/MUSAE/master/input/target/chameleon_target.csv"},
    "texas": {"out1_graph_edges.txt": f"{GH}/graphdml-uiuc-jlu/geom-gcn/master/new_data/texas/out1_graph_edges.txt",
              "out1_node_feature_label.txt": f"{GH}/graphdml-uiuc-jlu/geom-gcn/master/new_data/texas/out1_node_feature_label.txt"},
    "sociopatterns": {"High-School_data_2013.csv.gz":
                      "http://www.sociopatterns.org/wp-content/uploads/2015/07/High-School_data_2013.csv.gz"},
    "sociopatterns_gender": {
        "High-School_data_2013.csv.gz": "http://www.sociopatterns.org/wp-content/uploads/2015/07/High-School_data_2013.csv.gz",
        "metadata_2013.txt": "http://www.sociopatterns.org/wp-content/uploads/2015/09/metadata_2013.txt"},
    "github": {"git_edges.csv": f"{GH}/benedekrozemberczki/MUSAE/master/input/edges/git_edges.csv",
               "git_target.csv": f"{GH}/benedekrozemberczki/MUSAE/master/input/target/git_target.csv"},
}


# download / cache

def _path(name, fname, data_dir):
    d = os.path.join(data_dir, "sociopatterns" if name.startswith("sociopatterns") else name)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, fname)


def _get(name, fname, data_dir):
    """Return local path of file `fname` of dataset `name`, downloading it if missing."""
    dest = _path(name, fname, data_dir)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    url = URLS[name][fname]
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            f.write(r.read())
    except Exception as e:                                           # pragma: no cover
        if os.path.exists(dest):
            os.remove(dest)
        raise RuntimeError(
            f"Could not download '{fname}' for dataset '{name}' ({e}).\n"
            f"Download it manually from\n   {url}\nand save it as\n   {dest}\nthen run again.") from e
    return dest


# cora

def load_cora(data_dir=DATA_DIR):
    """2708 papers, 7 topic classes; citations treated as undirected edges."""
    P = {k: _get("cora", f"ind.cora.{k}", data_dir) for k in ("ally", "ty", "graph", "test.index")}
    ally = pickle.load(open(P["ally"], "rb"), encoding="latin1")
    ty = pickle.load(open(P["ty"], "rb"), encoding="latin1")
    graph = pickle.load(open(P["graph"], "rb"), encoding="latin1")
    test_idx = [int(l) for l in open(P["test.index"])]
    labels = np.vstack((np.asarray(ally), np.asarray(ty)))
    labels[test_idx, :] = labels[sorted(test_idx), :]
    y = labels.argmax(1)
    G = nx.Graph()
    G.add_nodes_from((i, {"y": int(y[i])}) for i in range(len(y)))
    G.add_edges_from((u, v) for u, nb in graph.items() for v in nb if u != v)
    return G


# polblogs

def load_polblogs(data_dir=DATA_DIR):
    """1490 blogs, 2 groups (liberal / conservative)."""
    import scipy.sparse as sp
    d = np.load(_get("polblogs", "polblogs.npz", data_dir), allow_pickle=True)
    A = sp.csr_matrix((d["adj_data"], d["adj_indices"], d["adj_indptr"]), shape=tuple(d["adj_shape"]))
    y = np.asarray(d["labels"]).astype(int)
    G = nx.Graph()
    G.add_nodes_from((i, {"y": int(y[i])}) for i in range(len(y)))
    r, c = A.nonzero()
    G.add_edges_from((int(u), int(v)) for u, v in zip(r, c) if u != v)
    G.graph["directed_source"] = True
    return G


# chameleon

def load_chameleon(data_dir=DATA_DIR, n_classes=5):
    """2277 pages, 31 421 undirected edges."""
    e = pd.read_csv(_get("chameleon", "chameleon_edges.csv", data_dir))
    t = pd.read_csv(_get("chameleon", "chameleon_target.csv", data_dir))
    y = pd.qcut(t["target"].rank(method="first"), n_classes, labels=False).to_numpy()
    G = nx.Graph()
    G.add_nodes_from((int(i), {"y": int(c)}) for i, c in zip(t["id"], y))
    G.add_edges_from((int(u), int(v)) for u, v in zip(e.iloc[:, 0], e.iloc[:, 1]) if u != v)
    return G


# texas

def load_texas(data_dir=DATA_DIR):
    """183 web pages, 5 classes (student, faculty, course, project, staff)."""
    ep = _get("texas", "out1_graph_edges.txt", data_dir)
    npth = _get("texas", "out1_node_feature_label.txt", data_dir)
    E = pd.read_csv(ep, sep="\t")
    N = pd.read_csv(npth, sep="\t")
    G = nx.Graph()
    G.add_nodes_from((int(i), {"y": int(l)}) for i, l in zip(N["node_id"], N["label"]))
    G.add_edges_from((int(u), int(v)) for u, v in zip(E.iloc[:, 0], E.iloc[:, 1]) if u != v)
    return G


# sociopatterns

def load_sociopatterns(data_dir=DATA_DIR, label="class", min_contacts=1):
    """Contact file columns (space/tab separated): t i j class_i class_j."""
    name = "sociopatterns_gender" if label == "gender" else "sociopatterns"
    p = _get(name, "High-School_data_2013.csv.gz", data_dir)
    opener = gzip.open if p.endswith(".gz") else open
    with opener(p, "rt") as f:
        C = pd.read_csv(f, sep=r"\s+", header=None, names=["t", "i", "j", "ci", "cj"], engine="python")
    cnt = C.groupby(["i", "j"]).size()
    cnt = cnt[cnt >= min_contacts]
    cls = {}
    for a, b, ca, cb in C[["i", "j", "ci", "cj"]].itertuples(index=False):
        cls.setdefault(a, ca)
        cls.setdefault(b, cb)
    if label == "gender":
        M = pd.read_csv(_get(name, "metadata_2013.txt", data_dir), sep=r"\s+", header=None,
                        names=["id", "class", "gender"], engine="python")
        lab = {int(i): g for i, g in zip(M["id"], M["gender"]) if g in ("M", "F")}
    else:
        lab = cls
    G = nx.Graph()
    G.add_edges_from((int(i), int(j)) for (i, j) in cnt.index if i != j)
    for v in list(G):
        if v in lab:
            G.nodes[v]["y"] = lab[v]
    return G


# github

def load_github(data_dir=DATA_DIR):
    e = pd.read_csv(_get("github", "git_edges.csv", data_dir))
    t = pd.read_csv(_get("github", "git_target.csv", data_dir))
    G = nx.Graph()
    G.add_nodes_from((int(i), {"y": int(c)}) for i, c in zip(t["id"], t["ml_target"]))
    G.add_edges_from((int(u), int(v)) for u, v in zip(e.iloc[:, 0], e.iloc[:, 1]) if u != v)
    return G


REGISTRY = {
    "cora": load_cora, "polblogs": load_polblogs, "chameleon": load_chameleon,
    "texas": load_texas, "sociopatterns": load_sociopatterns,
    "sociopatterns_gender": lambda data_dir=DATA_DIR: load_sociopatterns(data_dir, label="gender"),
    "github": load_github,
}
DEFAULT_MAX_NODES = {"github": 5000}


# statistics + fidelity-checked subsampling

def graph_stats(G):
    """n, m, mean degree, class shares, edge homophily, global clustering, degree Gini."""
    y = {v: G.nodes[v]["y"] for v in G}
    classes = sorted(set(y.values()), key=str)
    share = {c: sum(1 for v in y.values() if v == c) / len(y) for c in classes}
    m = G.number_of_edges()
    same = sum(1 for u, v in G.edges() if y[u] == y[v])
    deg = np.sort(np.array([d for _, d in G.degree()], float))
    n = len(deg)
    gini = float((2 * np.arange(1, n + 1) @ deg) / (n * deg.sum()) - (n + 1) / n) if deg.sum() else 0.0
    return dict(n=G.number_of_nodes(), m=m, mean_deg=2 * m / max(G.number_of_nodes(), 1),
                n_classes=len(classes), class_share=share,
                edge_h=same / max(m, 1), clustering=nx.transitivity(G), deg_gini=gini)


def _forest_fire(adj, y, nodes, classes, shares, n_target, rng, p_burn=0.5):
    cap = {c: int(round(shares[i] * n_target)) for i, c in enumerate(classes)}
    cnt = {c: 0 for c in classes}
    by_class = {c: [v for v in nodes if y[v] == c] for c in classes}
    burned = set()

    def ok(w):
        return w not in burned and cnt[y[w]] < cap[y[w]]

    def add(w):
        burned.add(w)
        cnt[y[w]] += 1

    guard = 0
    while len(burned) < n_target and guard < 50 * n_target:
        guard += 1
        c = classes[rng.choice(len(classes), p=shares)]
        seed = by_class[c][rng.integers(len(by_class[c]))]
        if not ok(seed):
            continue
        add(seed)
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
                    add(w)
                    frontier.append(w)
    return burned


def subsample_graph(G, n_target, n_tries=8, seed=0, verbose=True):
    if G.number_of_nodes() <= n_target:
        return G
    rng = np.random.default_rng(seed)
    y = {v: G.nodes[v]["y"] for v in G}
    nodes = list(G.nodes())
    classes = sorted(set(y.values()), key=str)
    shares = np.array([sum(1 for v in nodes if y[v] == c) for c in classes], float)
    shares /= shares.sum()
    adj = {v: list(G[v]) for v in G}
    full = graph_stats(G)
    best, best_d, best_stats = None, 9e9, None
    for t in range(n_tries):
        S = _forest_fire(adj, y, nodes, classes, shares, n_target, rng)
        H = G.subgraph(S).copy()
        H = H.subgraph(max(nx.connected_components(H), key=len)).copy()
        st = graph_stats(H)
        d = (sum(abs(st["class_share"].get(c, 0) - full["class_share"][c]) for c in classes)
             + abs(st["edge_h"] - full["edge_h"]) + abs(st["clustering"] - full["clustering"])
             + abs(st["deg_gini"] - full["deg_gini"]))
        if d < best_d:
            best, best_d, best_stats = H, d, st
    best.graph["subsampled"] = True
    if verbose:
        print(f"  [subsample] {full['n']} -> {best_stats['n']} nodes "
              f"(class-stratified forest fire, best of {n_tries})")
        rows = [("nodes", full["n"], best_stats["n"]), ("edges", full["m"], best_stats["m"]),
                ("mean degree", full["mean_deg"], best_stats["mean_deg"]),
                ("edge homophily", full["edge_h"], best_stats["edge_h"]),
                ("global clustering", full["clustering"], best_stats["clustering"]),
                ("degree Gini", full["deg_gini"], best_stats["deg_gini"])]
        rows += [(f"share of class {c}", full["class_share"][c], best_stats["class_share"].get(c, 0.0))
                 for c in classes]
        print(pd.DataFrame(rows, columns=["statistic", "full graph", "subsample"])
              .to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
        dens = "denser" if best_stats["mean_deg"] > full["mean_deg"] else "sparser"
        print(f"  NOTE: class shares and edge homophily are protected; the sample is {dens} than the full graph "
              f"(mean degree {best_stats['mean_deg']:.1f} vs {full['mean_deg']:.1f}) - crawl samplers "
              f"over-represent the dense core.  The benchmark fits and evaluates every model against THIS "
              f"sample, so all models are compared on the same reference graph.")
    return best


def load(name, data_dir=DATA_DIR, max_nodes="default", verbose=True):
    key = name.lower()
    if key not in REGISTRY:
        raise KeyError(f"unknown dataset '{name}'. Available: {sorted(REGISTRY)}")
    G = REGISTRY[key](data_dir)
    unl = [v for v in G if "y" not in G.nodes[v]]
    if unl:                                              # e.g. students with unknown gender
        G.remove_nodes_from(unl)
        if verbose:
            print(f"  [{key}] dropped {len(unl)} unlabeled node(s)")
    G.graph["name"] = key
    limit = DEFAULT_MAX_NODES.get(key) if max_nodes == "default" else max_nodes
    if limit and G.number_of_nodes() > limit:
        G = subsample_graph(G, limit, verbose=verbose)
        G.graph["name"] = key
    return G


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help=f"datasets: {sorted(REGISTRY)}")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--max-nodes", type=int, default=None, help="override subsampling limit")
    a = ap.parse_args()
    names = sorted(REGISTRY) if a.all else a.names
    rows = []
    for n in names:
        try:
            G = load(n, a.data_dir, max_nodes=a.max_nodes if a.max_nodes else "default")
            s = graph_stats(G)
            rows.append(dict(dataset=n, nodes=s["n"], edges=s["m"], classes=s["n_classes"],
                             edge_h=round(s["edge_h"], 3), clustering=round(s["clustering"], 3),
                             minority_share=round(sorted(s["class_share"].values())[0], 3),
                             subsampled=bool(G.graph.get("subsampled"))))
        except Exception as e:
            print(f"[{n}] FAILED: {e}", file=sys.stderr)
    if rows:
        print("\n" + pd.DataFrame(rows).to_string(index=False))
