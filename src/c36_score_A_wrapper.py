"""ACTION='score_A' wrapper to score the corrected labels.

Source: ghost-p-p (5).ipynb, code cell 37 of 71.
Output file(s): see the cell body and the README in data/processed/ or data/tests/.
"""

# SET THIS ACTION TO SCORE THE CORRECTED LABELS
ACTION = "score_A"

# --- PASTE YOUR ORIGINAL VALIDATION CELL CODE BELOW THIS LINE ---
# (Ensure all your previous definitions for WORK, SAMPLE_FILE, SOURCES, 
# and the scoring logic are present in this cell)

if ACTION == "score_A":
    import json
    import pandas as pd
    from pathlib import Path
    from datetime import datetime, timezone

    # Ensure these paths match your previous setup
    WORK = Path("/kaggle/working/final_v3/validation_final_60_v2")
    LABEL_FILE = WORK / "labels_A.json"
    SAMPLE_FILE = WORK / "sample_private.json"
    
    if not LABEL_FILE.exists() or not SAMPLE_FILE.exists():
        raise FileNotFoundError("Labels or Sample file not found. Run annotation first.")

    labels = json.loads(LABEL_FILE.read_text(encoding="utf-8"))
    sample = json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))
    
    # Initialize counters
    tp = fp = fn = tn = 0
    source_metrics = {s: {"tp": 0, "fp": 0, "fn": 0} for s in SOURCES}
    
    # Calculate metrics for each item
    for row in sample:
        item_id = str(row["item_id"])
        true_sources = set(row.get("true_sources", [])) # Assuming 'true_sources' is in your sample
        pred_sources = set(labels.get(item_id, {}).get("present", []))
        
        # Source-level calculation
        for source in SOURCES:
            in_true = source in true_sources
            in_pred = source in pred_sources
            
            if in_true and in_pred:
                source_metrics[source]["tp"] += 1
            elif not in_true and in_pred:
                source_metrics[source]["fp"] += 1
            elif in_true and not in_pred:
                source_metrics[source]["fn"] += 1
                
    # Print Summary
    print("--- Validation Metrics (Annotator A - Corrected) ---")
    total_tp = sum(m["tp"] for m in source_metrics.values())
    total_fp = sum(m["fp"] for m in source_metrics.values())
    total_fn = sum(m["fn"] for m in source_metrics.values())
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Overall Precision: {precision:.4f}")
    print(f"Overall Recall:    {recall:.4f}")
    print(f"Overall F1:        {f1:.4f}")
    
    print("\nSource-Level Breakdown:")
    for source, metrics in source_metrics.items():
        p = metrics["tp"] / (metrics["tp"] + metrics["fp"]) if (metrics["tp"] + metrics["fp"]) > 0 else 0
        r = metrics["tp"] / (metrics["tp"] + metrics["fn"]) if (metrics["tp"] + metrics["fn"]) > 0 else 0
        print(f"{source:10s} | TP: {metrics['tp']:2d} | FP: {metrics['fp']:2d} | FN: {metrics['fn']:2d} | P: {p:.2f} | R: {r:.2f}")

    # Save results
    results = {
        "annotator": "A_corrected",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall": {"precision": precision, "recall": recall, "f1": f1},
        "source_level": source_metrics
    }
    
    result_file = WORK / "detector_validation_A_corrected.json"
    result_file.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {result_file}")

# --- END OF ORIGINAL VALIDATION CELL CODE ---
