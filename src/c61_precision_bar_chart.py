"""Precision bar chart: forbidden-join rate per task class with cluster CIs.

Source: ghost-p-p (5).ipynb, code cell 62 of 71.
Output file(s): see the cell body and the README in data/processed/ or data/tests/.
"""

P = PAPER["precision"]["classes"]; names = list(P)
join = np.array([P[c]["share"] for c in names]); nojoin = 100 - join
lo = np.array([P[c]["cluster_ci"][0] for c in names]); hi = np.array([P[c]["cluster_ci"][1] for c in names])
fig, ax = plt.subplots(figsize=(5.4, 3.1)); x = np.arange(len(names))
ax.bar(x, join, 0.58, color="#C62828", label="forbidden join found (strict evidence)")
ax.bar(x, nojoin, 0.58, bottom=join, color="#A5D6A7", label="no evidence of a join")
ax.errorbar(x, join, yerr=[join-lo, hi-join], fmt="none", ecolor="black", capsize=3, lw=0.9)
for i, c in enumerate(names):
    ax.text(i, join[i]/2 if join[i] > 12 else hi[i] + 4, f'{P[c]["join"]}/{P[c]["n"]}', ha="center", va="center",
            color="white" if join[i] > 12 else "black", fontsize=7)
    ax.text(i, join[i] + (100-join[i])/2 + (3 if join[i] < 12 else 0), f'{P[c]["no_join"]}/{P[c]["n"]}', ha="center", va="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 100); ax.set_ylabel("Share of denied answers (%)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=False, fontsize=7)
fig.savefig(f"{OUT}/fig3_precision.pdf"); fig.savefig(f"{OUT}/fig3_precision.png", dpi=200); plt.close(fig)
