"""CELL 6 (v5): PASS 3 - deepen small-model runs (only if a gate said GO). Each gate-controlled deepen writes to the same results.csv under level="deep".

Source: ghost-p-p (5).ipynb, code cell 69 of 71.
Output file(s): see the cell body and the README in data/processed/ or data/tests/.
"""

# ============================================================================
# CELL 6 - PASS 3: deepen the small-model runs (only if a gate said GO). Guarded.
# ============================================================================
LV_A = globals().get("LV_A", "deep"); LV_R = globals().get("LV_R", "deep")
try:
    if LV_A == "deep" or LV_R == "deep":
        print("PASS 3 | small models deepened | time used: %.2f h" % time_used_h(), flush=True)
        for _mid in CFG["SMALL"]:
            if over_budget():
                break
            with_model(_mid, "small", 16, LV_A, LV_R, False)
    else:
        print("PASS 3 skipped: both gates said the pilot is enough")
except Exception as _e:
    import traceback; traceback.print_exc()
    print("PASS 3 error was caught, the notebook continues:", repr(_e))
print("ALL RUNS FINISHED | time used: %.2f h | rows on disk: %d" % (time_used_h(), len(DONE)))
