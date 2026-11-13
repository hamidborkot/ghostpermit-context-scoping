"""CELL 3 (v5): PASS 1 - all three small models on every stage at pilot size. setup_pidet(); A1 (adaptive attack 1, 16 notes), A2 (iterative), A3 (evasion), B (baseline), R (randomized).

Source: ghost-p-p (5).ipynb, code cell 66 of 71.
Output file(s): see the cell body and the README in data/processed/ or data/tests/.
"""


# ============================================================================
# CELL 3 - PASS 1: the three small models, all stages at pilot size
#          (about 3 h in full mode; 1.2 h in lite mode)
# ============================================================================
setup_pidet()
print("PASS 1 | small models | pilot size | time used so far: %.2f h" % time_used_h(), flush=True)
for _mid in CFG["SMALL"]:
    if over_budget():
        break
    with_model(_mid, "small", 16, "pilot", "pilot", False)
print("PASS 1 finished | time used: %.2f h | rows on disk: %d" % (time_used_h(), len(DONE)))
