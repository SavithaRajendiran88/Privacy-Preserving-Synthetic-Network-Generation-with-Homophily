"""Quick demo: real network vs. default private vs. tuned private network at eps = 1.

Uses the generator and the measures from final_pipeline/fyp_pipeline.py, so the
numbers come from the same code as the full study.

    python demo.py                      # Cora
    python demo.py --dataset texas      # smallest network, runs in seconds
    python demo.py --dataset polblogs --setting privacy

The tuned settings are the operating points chosen from the pooled ParEGO front
(thesis, Table 4.15). The default private setting is rho = 0.3, p = 0, w = 0.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_pipeline"))
import fyp_pipeline as fp  # noqa: E402

import numpy as np  # noqa: E402

# (rho, p, w) from the pooled ParEGO front, eps = 1
TUNED = {
    "cora":      {"fidelity": (0.87, 0.12, 1.00), "balanced": (0.87, 0.12, 1.00), "privacy": (0.89, 0.13, 0.96)},
    "polblogs":  {"fidelity": (0.87, 0.35, 0.98), "balanced": (0.95, 0.23, 1.00), "privacy": (0.95, 0.27, 1.00)},
    "chameleon": {"fidelity": (0.94, 0.78, 1.00), "balanced": (0.94, 0.78, 1.00), "privacy": (0.94, 0.78, 1.00)},
    "texas":     {"fidelity": (0.95, 0.00, 0.64), "balanced": (0.95, 0.00, 0.88), "privacy": (0.95, 0.00, 0.88)},
    "github":    {"fidelity": (0.95, 0.02, 1.00), "balanced": (0.95, 0.02, 1.00), "privacy": (0.95, 0.02, 1.00)},
}
DEFAULT = (0.3, 0.0, 0.0)

ROWS = [  # (label, key, format)
    ("Edges", "m_syn", "{:,.0f}"),
    ("Edge homophily", "edge_h", "{:.3f}"),
    ("Assortativity r", "newman_r", "{:.3f}"),
    ("Clustering", "clustering", "{:.3f}"),
    ("f1 homophily error", "f1_homophily", "{:.3f}"),
    ("f2 motif error", "f2_motif", "{:.3f}"),
    ("f3 structure error", "f3_structure", "{:.3f}"),
    ("f4 leakage (EOR)", "f4_privacy", "{:.3f}"),
    ("Real edges reproduced", "recall", "{:.1%}"),
    ("Membership-inference AUC", "mia_auc_edge", "{:.3f}"),
    ("Isolated nodes", "isolated_frac", "{:.1%}"),
]


def real_row(d):
    G = d["G"]
    r = dict(d["ref"])
    r.update(m_syn=G.number_of_edges(), f1_homophily=0.0, f2_motif=0.0, f3_structure=0.0,
             f4_privacy=float("nan"), recall=1.0, mia_auc_edge=float("nan"), isolated_frac=0.0)
    return r


def print_table(ds, cols):
    names = [c[0] for c in cols]
    w0 = max(len(r[0]) for r in ROWS) + 2
    w = max(16, max(len(n) for n in names) + 2)
    line = "-" * (w0 + w * len(cols))
    print(f"\n{fp.DS_NAME[ds]}: real network vs. private synthetic networks (eps = 1)")
    print(line)
    print("".ljust(w0) + "".join(n.rjust(w) for n in names))
    print(line)
    for label, key, fmt in ROWS:
        cells = []
        for _, m in cols:
            v = m.get(key, float("nan"))
            cells.append(("-" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)).rjust(w))
        print(label.ljust(w0) + "".join(cells))
    print(line)


def draw(ds, d, nets, path, max_nodes=1500):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    G, y = d["G"], d["y"]
    nodes = list(G.nodes())
    if len(nodes) > max_nodes:                         # draw a sample on big graphs, keep it readable
        rng = np.random.default_rng(0)
        nodes = sorted(rng.choice(nodes, size=max_nodes, replace=False).tolist())
    H = G.subgraph(nodes)
    pos = nx.spring_layout(H, seed=1, iterations=60)   # same node positions in every panel
    real = set(map(tuple, map(sorted, G.edges())))

    fig, axes = plt.subplots(1, len(nets), figsize=(5.2 * len(nets), 5.4))
    for ax, (title, Gs, m) in zip(axes, nets):
        fp._draw(ax, list(Gs.edges()), pos, y, nodes, node_size=6 if len(nodes) > 500 else 14,
                 ew=0.4, ealpha=0.25 if len(nodes) > 500 else 0.5)
        ax.set_title(title, fontsize=13)
        if m is None:
            sub = f"h = {d['ref']['edge_h']:.3f}   |   {G.number_of_edges():,} edges"
        else:
            shared = sum(tuple(sorted(e)) in real for e in Gs.edges())
            sub = f"h = {m['edge_h']:.3f}   |   real edges kept {shared / G.number_of_edges():.1%}"
        ax.text(0.5, -0.04, sub, transform=ax.transAxes, ha="center", va="top", fontsize=11)
    fig.suptitle(f"{fp.DS_NAME[ds]}: node colour = group, grey edge = same group, red edge = different groups",
                 fontsize=11, color="#52514e")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="cora", choices=fp.DATASETS)
    ap.add_argument("--setting", default="balanced", choices=["fidelity", "balanced", "privacy"],
                    help="operating point on the ParEGO front (default: balanced)")
    ap.add_argument("--seed", type=int, default=0, help="noise seed of the private release")
    ap.add_argument("--out", default="demo_output")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()

    fp.OUT = a.out                                     # cache and outputs go to the demo folder
    os.makedirs(a.out, exist_ok=True)
    ds = a.dataset
    t0 = time.time()

    print(f"Loading {fp.DS_NAME[ds]} (downloaded on first use) ...")
    d = fp.get_data(ds)
    G = d["G"]
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges, {d['k']} groups")

    rho, p, w = TUNED[ds][a.setting]
    runs = [("Default private", DEFAULT), (f"Tuned ({a.setting})", (rho, p, w))]
    nets, cols = [("Real network", G, None)], [("Real", real_row(d))]
    for name, (r_, p_, w_) in runs:
        print(f"Generating {name.lower()} network: eps = 1, rho = {r_}, p = {p_}, w = {w_} ...")
        seed = 7_000_000 + a.seed
        Gs, rel = fp.release(d, fp.EPS, r_, p_, w_, seed)
        m = fp.full_evaluation(d, Gs, rel, np.random.default_rng(seed))
        nets.append((f"{name}, ε = 1", fp.with_labels(Gs, d["y"]), m))
        cols.append((name, m))

    print_table(ds, cols)

    tag = f"{ds}_{a.setting}"
    import pandas as pd
    for (name, Gs, m), (_, r_) in zip(nets[1:], runs):
        f = os.path.join(a.out, f"{tag}_{name.split(',')[0].split(' (')[0].lower().replace(' ', '_')}_edges.csv")
        pd.DataFrame(list(Gs.edges()), columns=["source", "target"]).to_csv(f, index=False)
    if not a.no_plot:
        path = os.path.join(a.out, f"{tag}_networks.png")
        print("Drawing the three networks ...")
        draw(ds, d, nets, path)
        print(f"Figure saved to {path}")
    print(f"Edge lists saved to {a.out}/   (total time {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
