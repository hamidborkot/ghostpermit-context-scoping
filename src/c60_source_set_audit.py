"""Audit: enumerate all source subsets and check forbidden-set membership under V1 and V2 rule families.

Source: ghost-p-p (5).ipynb, code cell 61 of 71.
Output file(s): see the cell body and the README in data/processed/ or data/tests/.
"""

SRC = ["Ca", "Co", "Lo", "No", "Em"]
V1 = [{"Ca","Co","Em"}, {"Ca","Co","Lo"}, {"Co","Em","Lo"}]
V2 = [{"Ca","Co"}, {"Co","Em","Lo"}]
viol = lambda S, fam: any(f <= S for f in fam)
sets = [frozenset(c) for r in range(1, 6) for c in itertools.combinations(SRC, r)]
assert len(sets) == 31
status = {}
for S in sets:
    status[S] = "deny1" if viol(S, V1) else ("deny2" if viol(S, V2) else "allow")
thick = {frozenset(f) for f in V1} | {frozenset({"Ca","Co"})}
col = {"allow": "#C8E6C9", "deny1": "#EF9A9A", "deny2": "#FFCC80"}
pos = {}
for r in range(1, 6):
    layer = sorted([S for S in sets if len(S) == r], key=lambda S: [SRC.index(x) for x in sorted(S, key=SRC.index)])
    xs = np.linspace(0.06, 0.94, len(layer)) if len(layer) > 1 else [0.5]
    for S, x in zip(layer, xs): pos[S] = (x, r)
fig, ax = plt.subplots(figsize=(7.4, 4.4)); ax.set_xlim(0, 1); ax.set_ylim(0.5, 5.6); ax.axis("off")
for S in sets:
    for e in SRC:
        if e not in S and len(S) < 5:
            T2 = S | {e}; ax.plot([pos[S][0], pos[T2][0]], [pos[S][1], pos[T2][1]], color="#90A4AE", lw=0.4, alpha=0.5, zorder=1)
for S in sets:
    x, y = pos[S]; lab = "".join(sorted(S, key=SRC.index))
    ax.text(x, y, lab, ha="center", va="center", fontsize=5.6, zorder=3,
            bbox=dict(boxstyle="round,pad=0.25", fc=col[status[S]], ec="black" if S in thick else "#607D8B", lw=1.6 if S in thick else 0.5))
ax.legend(handles=[Patch(fc=col["allow"], ec="#607D8B", label="allowed under v1 and v2"),
                   Patch(fc=col["deny1"], ec="#607D8B", label="denied under v1 (and v2)"),
                   Patch(fc=col["deny2"], ec="#607D8B", label="denied only under v2"),
                   Patch(fc="white", ec="black", lw=1.6, label="minimal forbidden set")],
          loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=4, frameon=False, fontsize=6.8)
fig.savefig(f"{OUT}/fig2_lattice.pdf"); fig.savefig(f"{OUT}/fig2_lattice.png", dpi=200); plt.close(fig)
print({k: sum(1 for v in status.values() if v == k) for k in ["allow","deny1","deny2"]})   # expect 21, 8, 2
