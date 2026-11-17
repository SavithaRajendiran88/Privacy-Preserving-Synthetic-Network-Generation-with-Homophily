"""Drawings of original vs DP-synthetic graphs, mixing matrices and degree distributions."""
import os
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import networkx as nx
import numpy as np

import dp_stcm as D
import homophily_benchmark as hb
DS_ORDER = ["cora", "polblogs", "chameleon", "texas", "github"]
DS_NAME = {"cora": "Cora", "polblogs": "PolBlogs", "chameleon": "Chameleon", "texas": "Texas", "github": "GitHub"}
BLUE, ORANGE, AQUA, INK = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b"

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
PANELS = ["real", "inf_cl", "2", "0.5", "0.1"]
TITLE = {"real": "Original graph", "inf_cl": "Non-private stcm (ε = ∞)", "2": "DP-stcm, ε = 2",
         "0.5": "DP-stcm, ε = 0.5", "0.1": "DP-stcm, ε = 0.1", "1": "DP-stcm, ε = 1", "inf": "stcm, ε = ∞"}


def synth(G, y, k, e, seed=0):
    if e == "real":
        return G
    p = hb.fit_stcm(G)["p"] if e == "inf_cl" else 0.0
    eps = np.inf if e.startswith("inf") else float(e)
    return D.release(G, y, k, eps, 0.3, seed=seed, p=p)[0]


def layout(G, seed=0):
    n = G.number_of_nodes()
    it = 60 if n > 3000 else 120
    return nx.spring_layout(G, seed=seed, iterations=it, k=1.6 / np.sqrt(n))


def draw(ax, H, pos, y, nodes=None, node_size=4, ew=0.25, ealpha=0.25):
    nodes = list(H.nodes()) if nodes is None else nodes
    ns = set(nodes)
    segs = [(pos[u], pos[v]) for u, v in H.edges() if u in ns and v in ns]
    same = [y[u] == y[v] for u, v in H.edges() if u in ns and v in ns]
    if segs:
        cols = np.where(np.array(same)[:, None], np.array([[0.45, 0.45, 0.45, ealpha]]),
                        np.array([[0.85, 0.25, 0.25, ealpha]]))
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=ew, zorder=1))
    xy = np.array([pos[v] for v in nodes])
    ax.scatter(xy[:, 0], xy[:, 1], s=node_size, c=[PALETTE[y[v] % 8] for v in nodes], linewidths=0, zorder=2)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)
    ax.set_aspect("equal")
    ax.autoscale()


def bfs_ball(G, size=120):
    start = max(G.degree(), key=lambda t: t[1])[0]
    seen, q = [start], deque([start])
    while q and len(seen) < size:
        u = q.popleft()
        for w in G[u]:
            if w not in seen and len(seen) < size:
                seen.append(w)
                q.append(w)
    return seen


def stats(H, G, y):
    n = len(y)
    er, es = D.edge_keys(G, n), D.edge_keys(H, n)
    e = hb.mixing_matrix(hb.adjacency(H), y, int(y.max()) + 1)
    return hb.edge_homophily(e), np.isin(es, er).mean() if len(es) else 0.0, H.number_of_edges()


def sample_figure(ds, out, graphs, G, y):
    n = len(y)
    pos = layout(G)
    ball = bfs_ball(G, 120 if n > 300 else 80)
    pos_z = nx.spring_layout(G.subgraph(ball), seed=1, iterations=200)
    big = n > 1500
    fig, axes = plt.subplots(2, len(PANELS), figsize=(3.6 * len(PANELS), 7.6))
    for j, e in enumerate(PANELS):
        H = graphs[e]
        h, prec, m = stats(H, G, y)
        draw(axes[0][j], H, pos, y, node_size=2 if big else 7, ew=0.15 if big else 0.4,
             ealpha=0.08 if big else 0.35)
        sub = f"edges {m:,}  ·  edge homophily {h:.3f}"
        if e != "real":
            sub += f"\nshare of edges that are real: {prec:.1%}"
        axes[0][j].set_title(TITLE[e] + "\n" + sub, fontsize=8.5)
        draw(axes[1][j], H, pos_z, y, nodes=ball, node_size=22, ew=0.8, ealpha=0.7)
        if j == 0:
            axes[0][j].set_ylabel("whole graph", fontsize=9)
            axes[1][j].set_ylabel(f"zoom: {len(ball)} nodes around the largest hub", fontsize=9)
    k = int(y.max()) + 1
    hs = [plt.Line2D([], [], marker="o", ls="", color=PALETTE[c % 8], label=f"class {c}") for c in range(k)]
    hs += [plt.Line2D([], [], color="#737373", label="same-class edge"),
           plt.Line2D([], [], color="#d94040", label="cross-class edge")]
    fig.legend(handles=hs, loc="lower center", ncol=min(k + 2, 9), fontsize=8, frameon=False)
    fig.suptitle(f"{DS_NAME[ds]}: original graph vs synthetic graphs (same node positions in every panel)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(os.path.join(out, f"sample_graphs_{ds}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(datasets, out=D.OUT):
    datasets = [d for d in DS_ORDER if d in datasets]
    os.makedirs(out, exist_ok=True)
    allg = {}
    for ds in datasets:
        G, y, k = D.load_ref(ds)
        graphs = {e: synth(G, y, k, e) for e in PANELS}
        allg[ds] = (G, y, k, graphs)
        sample_figure(ds, out, graphs, G, y)
        print("sample graphs:", ds, flush=True)

    # mixing matrices: real | ε=∞ | ε=0.5 | ε=0.1
    cols = ["real", "inf_cl", "0.5", "0.1"]
    fig, axes = plt.subplots(len(datasets), len(cols), figsize=(3.0 * len(cols), 2.8 * len(datasets)), squeeze=False)
    for i, ds in enumerate(datasets):
        G, y, k, graphs = allg[ds]
        for j, e in enumerate(cols):
            M = hb.mixing_matrix(hb.adjacency(graphs[e]), y, k)
            ax = axes[i][j]
            ax.imshow(M, cmap="Blues", vmin=0, vmax=hb.mixing_matrix(hb.adjacency(G), y, k).max())
            ax.grid(False)
            ax.set_xticks(range(k)); ax.set_yticks(range(k))
            ax.tick_params(labelsize=6)
            if k <= 5:
                for a in range(k):
                    for b in range(k):
                        ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=6,
                                color="white" if M[a, b] > 0.6 * M.max() else INK)
            ax.set_title(("" if i else TITLE[e].replace("DP-stcm, ", "") + "\n") + f"{DS_NAME[ds]}  h={np.trace(M):.3f}",
                         fontsize=8)
    fig.suptitle("Class mixing matrices (share of edge ends between class a and class b)", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(out, "fig_dp_mixing_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # degree distributions (complementary CDF)
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.2 * len(datasets), 3.2), squeeze=False)
    sty = {"real": (INK, "-", "original"), "inf_cl": (BLUE, "-", "ε = ∞"), "0.5": (ORANGE, "-", "ε = 0.5"),
           "0.1": (AQUA, "-", "ε = 0.1")}
    for ax, ds in zip(axes[0], datasets):
        G, y, k, graphs = allg[ds]
        for e, (c, ls, lab) in sty.items():
            d = np.sort(np.array([x for _, x in graphs[e].degree()]))
            d = d[d > 0]
            ccdf = 1.0 - np.arange(len(d)) / len(d)
            ax.loglog(d, ccdf, color=c, ls=ls, lw=2 if e == "real" else 1.4, label=lab)
        ax.set_title(DS_NAME[ds])
        ax.set_xlabel("degree k", fontsize=8)
    axes[0][0].set_ylabel("share of nodes with degree ≥ k", fontsize=8)
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle("Degree distributions: original vs DP-stcm", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(out, "fig_dp_degree_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
