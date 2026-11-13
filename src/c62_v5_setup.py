"""CELL 0 (v5): setup, config, GPU check. Imports, paths, LITE/FULL mode, time_used_h(), BATCH scaling.

Source: ghost-p-p (5).ipynb, code cell 63 of 71.
Output file(s): see the cell body and the README in data/processed/ or data/tests/.
"""


# ============================================================================
# CELL 0 - SETUP, CONFIG, GPU CHECK  (run first; about 2-5 minutes)
# ============================================================================
import os, sys, re, json, gc, time, math, random, hashlib, shutil, subprocess, warnings, zipfile
from itertools import combinations
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
T_START = time.time()

MOCK = os.environ.get("GP_MOCK", "0") == "1"          # internal test switch, leave alone

CFG = dict(
    MODE="full",        # "full" = about 9-10 h on 2xT4 | "lite" = about 4-5 h
    SMOKE=False,        # True = tiny dry run of EVERY stage (about 15 min); then set False and Run All
    FORCE_DEEP=None,    # None = the gates decide | True = always go deep | False = never go deep
    BUDGET_H=10.5,      # no new work is started after this many hours; the final report always runs
    SMALL=["microsoft/Phi-3.5-mini-instruct", "Qwen/Qwen3-4B-Instruct-2507", "ibm-granite/granite-3.3-2b-instruct"],
    MID=["Qwen/Qwen3-8B"],
    BIG=[("unsloth/Qwen2.5-14B-Instruct-bnb-4bit", False), ("Qwen/Qwen2.5-14B-Instruct", True)],  # (repo, quantize_on_load)
    USE_PIDET=True,     # protectai/deberta-v3-base-prompt-injection-v2 as an extra baseline (skipped if it cannot load)
)
if CFG["MODE"] == "lite":
    CFG["MID"] = []

# per-mode sizes: trials per cell. B = baselines, A = adaptive (pilot / deep), R = randomized records
SIZES = {
    "full": dict(B_T=8, B_T_BIG=6, A_T=(2, 3), A_NT=(8, 12), A_R=(3, 3), R_T=(3, 6)),
    "lite": dict(B_T=4, B_T_BIG=4, A_T=(2, 3), A_NT=(6, 8), A_R=(2, 3), R_T=(2, 4)),
}
SZ = dict(SIZES[CFG["MODE"]])
if CFG["SMOKE"]:
    SZ = dict(B_T=1, B_T_BIG=1, A_T=(1, 1), A_NT=(2, 2), A_R=(2, 2), R_T=(1, 1))

ON_KAGGLE = os.path.exists("/kaggle/working")
OUT = "/kaggle/working/final_v4" if ON_KAGGLE else "./final_v4"
if CFG["SMOKE"]:
    OUT += "_smoke"
os.makedirs(OUT, exist_ok=True)
CSV = f"{OUT}/results.csv"
os.environ.setdefault("HF_HOME", "/kaggle/tmp/hf" if ON_KAGGLE else "./.hf")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def sh(*a):
    return subprocess.run(list(a), capture_output=True, text=True)

if not MOCK:
    try:
        import bitsandbytes  # noqa
    except Exception:
        print("installing bitsandbytes ...", flush=True)
        sh(sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes", "accelerate")
    import torch
    assert torch.cuda.is_available(), "No GPU. In Kaggle: Settings -> Accelerator -> GPU T4 x2, then Run All."
    n = torch.cuda.device_count()
    tot = sum(torch.cuda.get_device_properties(i).total_memory for i in range(n)) / 2**30
    print("GPUs:", n, [torch.cuda.get_device_name(i) for i in range(n)], f"total VRAM {tot:.1f} GiB")
    free = shutil.disk_usage(os.environ["HF_HOME"] if os.path.exists(os.environ["HF_HOME"]) else "/").free / 2**30
    print(f"free disk near HF cache: {free:.0f} GiB (a 7-8B fp16 model needs about 16 GiB; the 14B 4-bit checkpoint about 10 GiB)")
else:
    print("MOCK mode (no GPU): pipeline test only")
print("MODE", CFG["MODE"], "| SMOKE", CFG["SMOKE"], "| sizes", SZ)
print("output folder:", OUT)
