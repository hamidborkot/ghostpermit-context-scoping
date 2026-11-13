"""PAPER dictionary: all numbers (headline, precision, router, soundness) used by the paper figures.

Source: ghost-p-p (5).ipynb, code cell 57 of 71.
Output file(s): see the cell body and the README in data/processed/ or data/tests/.
"""

import itertools, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Patch
from matplotlib.lines import Line2D
from math import sqrt

BLACK, CRIM, TEAL, AMBER, NAVY = "#000000", "#8E1B1B", "#0B5D5D", "#A86400", "#14213D"
TINT = {"teal": "#CFE6E4", "crim": "#EFCFCF", "amber": "#F6E0AE"}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "text.color": BLACK, "axes.labelcolor": BLACK, "xtick.color": BLACK, "ytick.color": BLACK,
    "axes.edgecolor": BLACK, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.size": 13,
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})

def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)

def make_fig2(out):
    S5 = ["Ca", "Co", "Lo", "No", "Em"]
    subs = [frozenset(c) for r in range(1, 6) for c in itertools.combinations(S5, r)]
    F1 = [{"Ca", "Co", "Em"}, {"Ca", "Co", "Lo"}, {"Co", "Em", "Lo"}]
    F2 = [{"Ca", "Co"}, {"Co", "Em", "Lo"}]
    viol = lambda s, F: any(frozenset(f) <= s for f in F)
    level = {k: [s for s in subs if len(s) == k] for k in range(1, 6)}
    pos = {}
    for k, nodes in level.items():
        for i, s in enumerate(nodes):
            pos[s] = (k, 10 - (i + 0.5) * 10 / len(nodes))
    fig = plt.figure(figsize=(6.8, 6.9)); ax = fig.add_axes([0.0, 0.10, 1.0, 0.86])
    ax.set_xlim(0.42, 5.58); ax.set_ylim(-0.2, 10.9); ax.axis("off")
    for a in subs:
        for b in subs:
            if a < b and len(b) == len(a) + 1:
                (x0, y0), (x1, y1) = pos[a], pos[b]
                ax.plot([x0, x1], [y0, y1], color="#6B6B6B", lw=0.7, zorder=1, alpha=0.75)
    texts, styles, newly = {}, {}, []
    for s in subs:
        d1, d2 = viol(s, F1), viol(s, F2)
        if not d1 and not d2: styles[s] = (TINT["teal"], TEAL)
        elif d1 and d2: styles[s] = (TINT["crim"], CRIM)
        else: styles[s] = (TINT["amber"], AMBER); newly.append(s)
        x, y = pos[s]
        texts[s] = ax.text(x, y, " ".join(e for e in S5 if e in s), ha="center", va="center", fontsize=10, zorder=4)
    assert len(newly) == 2, newly
    fig.canvas.draw(); rend = fig.canvas.get_renderer(); inv = ax.transData.inverted()
    for s in subs:
        bb = texts[s].get_window_extent(rend); (xa, _), (xb, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        w = (xb - xa) + 0.16; x, y = pos[s]
        minimal = any(frozenset(f) == s for f in F2)
        fc, ec = styles[s]
        ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.30), w, 0.60, boxstyle="round,pad=0.0,rounding_size=0.12",
                                    fc=fc, ec=BLACK if minimal else ec, lw=2.8 if minimal else 1.4, zorder=3))
    for k in range(1, 6):
        ax.text(k, 10.55, f"{k} source" + ("s" if k > 1 else ""), ha="center", va="center", fontsize=12.5)
    fig.legend(handles=[Patch(fc=TINT["teal"], ec=TEAL, lw=1.4, label="allowed under v1 and v2"),
                        Patch(fc=TINT["crim"], ec=CRIM, lw=1.4, label="denied under v1 and v2"),
                        Patch(fc=TINT["amber"], ec=AMBER, lw=1.4, label="denied only under v2"),
                        Patch(fc="white", ec=BLACK, lw=2.8, label="minimal forbidden set, v2")],
               loc="lower center", ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.0))
    save(fig, out, "fig2_lattice")

def make_fig3(csv_path, out):
    import pandas as pd
    s = pd.read_csv(csv_path)
    order = [("Phi-3.5-mini-instruct", "Phi-3.5-mini (3.8B)"), ("Qwen3-4B-Instruct-2507", "Qwen3-4B (4B)"),
             ("granite-3.3-2b-instruct", "Granite-3.3 (2B)"), ("Qwen2.5-7B-Instruct", "Qwen2.5 (7B)"),
             ("Qwen3-8B", "Qwen3 (8B)")]
    arms = [("full_inj", "Full context", CRIM, "o"), ("full_def_inj", "Prompt defense", AMBER, "s"),
            ("router_inj", "Context scoping", TEAL, "D")]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ys = [5, 4, 3, 1.7, 0.7]
    for (key, lab), y in zip(order, ys):
        vals = []
        for cond, _, col, mk in arms:
            r = s[(s["model"] == key) & (s["cond"] == cond)]
            assert len(r) == 1, (key, cond, len(r))
            k, n = int(r["sum"].iloc[0]), int(r["count"].iloc[0]); p = 100 * k / n; lo, hi = wilson(k, n)
            vals.append((p, lo, hi, col, mk))
        ax.plot([min(v[0] for v in vals), max(v[0] for v in vals)], [y, y], color="#8A8A8A", lw=2.2, zorder=1)
        for p, lo, hi, col, mk in vals:
            ax.plot([lo, hi], [y, y], color=col, lw=1.6, alpha=0.75, zorder=2)
            ax.scatter([p], [y], s=150, c=col, marker=mk, edgecolors=BLACK, linewidths=0.9, zorder=4)
            ax.text(p, y + 0.30, f"{p:.1f}", ha="center", va="bottom", fontsize=11, zorder=5)
    ax.set_yticks(ys); ax.set_yticklabels([l for _, l in order], fontsize=12.5)
    ax.axhline(2.35, color=BLACK, lw=0.8, ls=(0, (4, 3)))
    ax.text(0.995, 5.62, "2 to 4B models", transform=ax.get_yaxis_transform(), ha="right", fontsize=11.5)
    ax.text(0.995, 1.95, "7 to 8B models", transform=ax.get_yaxis_transform(), ha="right", fontsize=11.5)
    ax.set_xlim(-2, 62); ax.set_ylim(0.1, 6.0)
    ax.set_xlabel("Answers with a forbidden join (%), 128 answers per model and arm", fontsize=12.5)
    ax.tick_params(axis="x", labelsize=12); ax.grid(axis="x", color="#D0D0D0", lw=0.7); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(handles=[Line2D([0], [0], marker=mk, color="w", markerfacecolor=col, markeredgecolor=BLACK, markersize=11, label=lab)
                       for _, lab, col, mk in arms], loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=False, fontsize=12)
    save(fig, out, "fig3_fivemodel")

def make_fig4(out):
    data = [("Single-source", 12, 180, "6.7", "2.8 to 11.1"), ("Pair", 53, 180, "29.4", "20.0 to 39.4"),
            ("Briefing", 110, 180, "61.1", "48.3 to 75.0")]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.9)); ncols = 20
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.40, wspace=0.10)
    for ax, (name, k, n, pj, ci) in zip(axes, data):
        rows = int(np.ceil(n / ncols))
        for i in range(n):
            r, c = divmod(i, ncols)
            ax.add_patch(Rectangle((c, rows - 1 - r), 0.84, 0.84, fc=CRIM if i < k else TEAL, ec="none"))
        ax.set_xlim(-0.2, ncols); ax.set_ylim(-0.2, rows); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(name, fontsize=14, pad=8)
    fig.canvas.draw()
    for ax, (name, k, n, pj, ci) in zip(axes, data):
        bb = ax.get_position(); xc = (bb.x0 + bb.x1) / 2
        fig.text(xc, 0.33, f"{k} of {n} contain a join", ha="center", va="top", fontsize=11.5)
        fig.text(xc, 0.25, f"{pj}% (CI {ci})", ha="center", va="top", fontsize=11.5)
        fig.text(xc, 0.17, f"{n - k} with no join", ha="center", va="top", fontsize=11.5)
    fig.legend(handles=[Patch(fc=CRIM, label="forbidden join found"), Patch(fc=TEAL, label="denied, no join found")],
               loc="lower center", ncol=2, frameon=False, fontsize=12.5, bbox_to_anchor=(0.5, 0.0))
    save(fig, out, "fig4_union_label")

OUT = "/kaggle/working/figures"
make_fig2(OUT)
csv = "/kaggle/working/fivemodel_final_summary.csv"
if os.path.exists(csv): make_fig3(csv, OUT)
else: print("Figure 3 skipped: run the five-model summary cell first:", csv)
make_fig4(OUT)
print("saved to", OUT, sorted(os.listdir(OUT)))
