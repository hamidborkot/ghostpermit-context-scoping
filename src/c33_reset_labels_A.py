"""Delete labels_A.json so annotation resumes at item 17 (kept backup).

Source: ghost-p-p (5).ipynb, code cell 34 of 71.
Output file(s): see the cell body and the README in data/processed/ or data/tests/.
"""

from pathlib import Path
p = Path("/kaggle/working/final_v3/validation_final_60_v2/labels_A.json")
if p.exists():
    # Keep backup
    import shutil
    shutil.copy2(p, p.with_suffix(".json.bak_items17_60_wrong"))
    p.unlink()
    print("Deleted labels_A.json (backup saved). Will resume at item 17.")
