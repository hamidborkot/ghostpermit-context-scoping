"""Utility: zip results_v2/ into experiment_results.zip for download.

Source: ghost-p-p (5).ipynb, code cell 24 of 71.
Output file(s): see the cell body and the README in data/processed/ or data/tests/.
"""

import os
import zipfile
from IPython.display import FileLink

# Define the precise list of files and directories to include
files_to_zip = [
    ".virtual_documents",
    "results_v2",
    "fig_precision.pdf",
    "final_trials.csv",
    "manual_check.csv",
    "precision_summary.csv",
    "precision_table.tex",
    "precision_trials.csv",
    "precision_trials_corrected.csv",
    "scoped_trials.csv",
    "widen_trials.csv"
]

zip_name = "experiment_results.zip"
base_dir = "/kaggle/working"

with zipfile.ZipFile(os.path.join(base_dir, zip_name), 'w', zipfile.ZIP_DEFLATED) as zipf:
    for item in files_to_zip:
        item_path = os.path.join(base_dir, item)
        
        if os.path.exists(item_path):
            if os.path.isdir(item_path):
                # Recursively walk through folders like results_v2
                for root, dirs, files in os.walk(item_path):
                    for file in files:
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, base_dir)
                        zipf.write(full_path, rel_path)
            else:
                # Add individual files
                zipf.write(item_path, item)
        else:
            print(f"Warning: {item} not found in {base_dir}")

print(f"Successfully created {zip_name}!")

