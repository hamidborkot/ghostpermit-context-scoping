"""Figures 3 & 4: PARS scatter and instruction reinsertion/propagation.

Source: ghost-p-p (5).ipynb, code cell 60 of 71.
Output file(s): see the cell body and the README in data/processed/ or data/tests/.
"""

def box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec="#37474F", lw=0.9))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=8, linespacing=1.25)

def arrow(ax, p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9, lw=0.9, color="#37474F"))

fig, ax = plt.subplots(figsize=(7.4, 3.9)); ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")
T, U = "#DCEBFA", "#FBE0DC"
W, H = 24, 10; X = (4, 38, 72); Y = (41, 24, 7)
box(ax, X[0], Y[0], W, H, "Request $q$ and\ngranted sources $G$", T)
box(ax, X[1], Y[0], W, H, "Router $R$\nreads $q$ and source\nnames only", T)
box(ax, X[2], Y[0], W, H, "Scope\n$\\Sigma=R(q,G)\\cap G$", T)
box(ax, X[2], Y[1], W, H, "Tool outputs of $\\Sigma$\n(may carry injected text)", U)
box(ax, X[1], Y[1], W, H, "Language model\ngenerates answer $y$", U)
box(ax, X[0], Y[1], W, H, "Candidate answer $y$", U)
box(ax, X[0], Y[2], W, H, "Bounds on sources\n$L_{hi}=\\Sigma$,\n$L_{lo}$: identifiers in $y$", T)
box(ax, X[1], Y[2], W, H, "Release monitor\npolicy $v$, context $c$,\nrecipient $r$", T)
box(ax, X[2], Y[2], W, H, "Sink: owner, external,\nor memory\nALLOW / REVIEW / DENY", T)
arrow(ax, (X[0]+W+0.6, Y[0]+H/2), (X[1]-0.6, Y[0]+H/2))
arrow(ax, (X[1]+W+0.6, Y[0]+H/2), (X[2]-0.6, Y[0]+H/2))
arrow(ax, (X[2]+W/2, Y[0]-0.6), (X[2]+W/2, Y[1]+H+0.6))
arrow(ax, (X[2]-0.6, Y[1]+H/2), (X[1]+W+0.6, Y[1]+H/2))
arrow(ax, (X[1]-0.6, Y[1]+H/2), (X[0]+W+0.6, Y[1]+H/2))
arrow(ax, (X[0]+W/2, Y[1]-0.6), (X[0]+W/2, Y[2]+H+0.6))
arrow(ax, (X[0]+W+0.6, Y[2]+H/2), (X[1]-0.6, Y[2]+H/2))
arrow(ax, (X[1]+W+0.6, Y[2]+H/2), (X[2]-0.6, Y[2]+H/2))
ax.legend(handles=[Patch(fc=T, ec="#37474F", label="trusted orchestration layer"),
                   Patch(fc=U, ec="#37474F", label="untrusted: model text and tool data")],
          loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=2, frameon=False, fontsize=7)
fig.savefig(f"{OUT}/fig1_architecture.pdf"); fig.savefig(f"{OUT}/fig1_architecture.png", dpi=200); plt.close(fig)
