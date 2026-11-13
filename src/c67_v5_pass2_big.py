"""CELL 5 (v5): PASS 2 - big model (Qwen2.5-14B-Instruct, 4-bit) and the 8B model on stage B and stage A1. Guarded so an error here cannot stop later cells.

Source: ghost-p-p (5).ipynb, code cell 68 of 71.
Output file(s): see the cell body and the README in data/processed/ or data/tests/.
"""

# ============================================================================
# CELL 5 - PASS 2: big model (Qwen2.5-14B, 4-bit) and the 8B model. Guarded: an error here cannot stop later cells.
# ============================================================================
LV_A = globals().get("LV_A", "deep"); LV_R = globals().get("LV_R", "deep")
print("PASS 2 | big and mid models | adaptive level:", LV_A, "| randomized level:", LV_R, "| time used: %.2f h" % time_used_h(), flush=True)
try:
    if CFG["BIG"] and not over_budget():
        with_model(CFG["BIG"], "big", 8, LV_A, LV_R, True)
    for _mid in CFG["MID"]:
        if over_budget():
            break
        with_model(_mid, "mid", 8, LV_A, LV_R, True)
except Exception as _e:
    import traceback; traceback.print_exc()
    print("PASS 2 error was caught, the notebook continues:", repr(_e))
print("PASS 2 finished | time used: %.2f h | rows on disk: %d" % (time_used_h(), len(DONE)))
