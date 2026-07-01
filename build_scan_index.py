from pathlib import Path
import pandas as pd

from adapters.topaneu_adapter import build_topaneu_records
from adapters.totalsegmentator_adapter import build_totalsegmentator_records
from adapters.sega_adapter import build_sega_records

DATASETS = {
    "topaneu": Path(
        r"C:\Users\cd26\OneDrive - King's College London\MT Anatomy Localisation Project\topaneu_deployment"
    ),
    "totalsegmentator": Path(
        r"C:\Users\cd26\OneDrive - King's College London\MT Anatomy Localisation Project\Totalsegmentator_dataset_v201"
    ),
    "sega": Path(
        r"C:\Users\cd26\OneDrive - King's College London\MT Anatomy Localisation Project\SegA"
    )
}


all_records = []

print("\nStarting TopAneu indexing...", flush=True)

topaneu_records = build_topaneu_records(
    DATASETS["topaneu"],
    annotation_type="manual",
    )

all_records.extend(topaneu_records)

print(f"Finished TopAneu: {len(topaneu_records)} records found", flush=True)

print("\nStarting TotalSegmentator indexing...", flush=True)
totalseg_records = build_totalsegmentator_records(
    DATASETS["totalsegmentator"],
    annotation_type="automatic",
)
print(f"Finished TotalSegmentator: {len(totalseg_records)} records found", flush=True)

all_records.extend(totalseg_records)

print("\nStarting SegA indexing...", flush=True)
sega_records = build_sega_records(
    DATASETS["sega"],
    annotation_type="manual",
)
print(f"Finished SegA: {len(sega_records)} records found", flush=True)

all_records.extend(sega_records)

rows = []

for record in all_records:
    rows.append({
        "case_id": record.case_id,
        "dataset": record.dataset,
        "image_path": str(record.image_path),
        "label_path": str(record.label_path) if record.label_path else "",
        "anatomical_region": record.anatomical_region,
        "labels_present": "|".join(record.labels_present),
        "annotation_type": record.annotation_type,
    })

df = pd.DataFrame(rows)

output_path = Path("metadata") / "scan_index.csv"
output_path.parent.mkdir(exist_ok=True)

df.to_csv(output_path, index=False)

print("Saved scan index to:", output_path)
print("Number of scans:", len(df))

df.head()