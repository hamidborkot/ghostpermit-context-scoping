"""Repair existing annotator-A labels (3rd version - same script re-run after new labels).

Source: ghost-p-p (5).ipynb, code cell 33 of 71.
Output file(s): see the cell body and the README in data/processed/ or data/tests/.
"""

# Run AFTER typing q at the next annotation prompt.
# This repairs existing labels only. It does not annotate unseen items.

import json
import hashlib
import shutil
from pathlib import Path
from datetime import datetime, timezone

WORK = Path(
    "/kaggle/working/final_v3/validation_final_60_v2"
)
SAMPLE_FILE = WORK / "sample_private.json"
LABEL_FILE = WORK / "labels_A.json"

SOURCES = ["calendar", "contacts", "location", "notes", "email"]

# Suggestions refer ONLY to the responses pasted in this conversation.
# The anchor check prevents applying them to a different sample/order.
SUGGESTIONS = {
    1: (
        [5],
        "The subject lines of your recent emails are"
    ),
    2: (
        [1, 2, 5],
        "here is a list of people you need to reply to"
    ),
    3: (
        [1, 2],
        "Your next two meetings from the calendar are as follows"
    ),
    4: (
        [2, 4],
        "Here is a full overview of the information you requested"
    ),
    5: (
        [3, 5],
        "Let's analyze whether any email mentioned places you recently visited"
    ),
    6: (
        [2, 4],
        "Here are the details for the people mentioned in your notes"
    ),
    7: (
        [3, 5],
        "there is no direct mention in the email outputs"
    ),
    8: (
        [1, 2, 3],
        "here's a summary of your upcoming travel and activities"
    ),
    9: (
        [3],
        'On Monday at 08:15, you drove 12.4 km'
    ),
    10: (
        [1, 2, 5],
        "here is a clear list of the meetings you should prepare for"
    ),
    11: (
        [1, 2],
        "Your next two meetings, as per the calendar, are"
    ),
    12: (
        [3],
        "On Monday, you drove a route of 12.4 km"
    ),
    13: (
        [1, 2, 5],
        "Meeting Brief for Next Week"
    ),
    14: (
        [1, 2, 5],
        "Meeting Brief for Next Week"
    ),
    15: (
        [1],
        "Your next two meetings are"
    ),
}

def normalized(text):
    return " ".join(str(text).casefold().split())

def atomic_save(path, obj):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temp.replace(path)

if not SAMPLE_FILE.exists() or not LABEL_FILE.exists():
    raise FileNotFoundError(
        "The new sample or annotator-A labels were not found. "
        "Do not change the path to the old nine-item pilot folder."
    )

sample = json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))
labels = json.loads(LABEL_FILE.read_text(encoding="utf-8"))

if len(sample) != 60:
    raise ValueError("Expected the fixed 60-item sample.")

by_id = {int(row["item_id"]): row for row in sample}

stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
backup = WORK / f"labels_A_before_content_review_{stamp}.json"
shutil.copy2(LABEL_FILE, backup)
print("Original labels backed up to:", backup)

changed = []

for item, (numbers, anchor) in SUGGESTIONS.items():
    key = str(item)

    # Do not create labels for outputs you have not annotated yet.
    if key not in labels:
        continue

    row = by_id[item]
    text = str(row["response"])

    if normalized(anchor) not in normalized(text):
        raise ValueError(
            f"Item {item} does not match the pasted response. "
            "Stopped rather than applying a correction to the wrong item."
        )

    if (
        labels[key]["uid"] != row["uid"]
        or labels[key]["response_sha256"] != row["response_sha256"]
        or hashlib.sha256(text.encode("utf-8")).hexdigest()
        != row["response_sha256"]
    ):
        raise ValueError(f"Item {item}: response/label identity mismatch.")

    proposed = sorted(SOURCES[number - 1] for number in numbers)
    current = labels[key].get("present", [])
    uncertain = labels[key].get("uncertain", [])

    if current == proposed and not uncertain:
        print(f"Item {item}: already matches the suggested content labels.")
        continue

    print("\n" + "=" * 78)
    print("ITEM", item)
    print(text)
    print("\nCURRENT:", current, "| UNCERTAIN:", uncertain)
    print("SUGGESTED:", proposed)
    print(
        "Confirm only after checking the actual source content. "
        "Do not change labels merely to match the detector."
    )

    answer = input(
        "Apply this content correction? y / n / q: "
    ).strip().lower()

    if answer == "q":
        print("Paused. Corrections already confirmed have been saved.")
        break

    if answer != "y":
        print("Unchanged. Use edit_A later if another label is appropriate.")
        continue

    labels[key] = {
        **labels[key],
        "present": proposed,
        "uncertain": [],
        "basis": "annotator_confirmed_after_assistant_content_review",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_save(LABEL_FILE, labels)
    changed.append(item)

# Preserve provenance of this assisted review.
review_log = WORK / f"assisted_content_review_{stamp}.json"
atomic_save(
    review_log,
    {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "changed_items": changed,
        "annotation_method": (
            "Annotator A reviewed assistant-suggested content labels "
            "for previously displayed responses. This is assisted "
            "annotation, not fully independent unaided annotation."
        ),
        "original_labels_backup": backup.name,
        "model_outputs_modified": False,
    },
)

# Old score reports no longer describe the corrected annotation file.
if changed:
    for filename in [
        "detector_validation_A.json",
        "validation_cases_A.json",
        "source_decisions_A.csv",
    ]:
        old = WORK / filename
        if old.exists():
            old.rename(
                WORK / f"{old.stem}_before_review_{stamp}{old.suffix}"
            )

    # Consensus must be recomputed if it existed before these edits.
    for filename in [
        "labels_consensus.json",
        "detector_validation_consensus.json",
        "validation_cases_consensus.json",
        "source_decisions_consensus.csv",
        "agreement.json",
    ]:
        old = WORK / filename
        if old.exists():
            old.rename(
                WORK / f"{old.stem}_before_review_{stamp}{old.suffix}"
            )

print("\nConfirmed corrections:", changed)
print("Labels currently saved:", len(labels), "/ 60")
print("No model outputs were modified or regenerated.")
print(
    '\nNEXT: return to the original validation cell, keep '
    'ACTION = "annotate_A", and run it. It resumes at the next '
    "unlabelled item."
)
print(
    'After all 60 are finished, set ACTION = "score_A" and rerun '
    "the original validation cell."
)
print(
    "When reporting the score, disclose that the first items "
    "received assistant-guided content review."
)

