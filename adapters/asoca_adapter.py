from pathlib import Path
from typing import List

from adapters.base_adapter import ScanRecord

ASOCA_LABEL_NAME = "coronary_vessel_lumen"
ASOCA_REGION = "thorax_chest"

def strip_nrrd_suffix(path: Path) -> str:
    name = path.name

    if name.endswith(".nrrd"):
        return name[:-5]
    
    return path.stem

def build_asoca_records(
        dataset_root: Path,
        annotation_type: str = "manual",
) -> List[ScanRecord]:
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"ASOCA dataset root path does not exist: {dataset_root}")
    records = []

    # This is written generally so it can also handle other ASOCA groups later,
    # for example Diseased/ if you add it.
    for group_dir in sorted(dataset_root.iterdir()):
        if not group_dir.is_dir():
            continue

        ctca_dir = group_dir / "CTCA"
        annotations_dir = group_dir / "Annotations"
        centerlines_dir = group_dir / "Centerlines"

        # Skip folders that do not have the expected ASOCA structure
        if not ctca_dir.exists():
            continue

        if not annotations_dir.exists():
            print(f"Skipping {group_dir.name}: no Annotations folder found")
            continue

        for image_path in sorted(ctca_dir.glob("*.nrrd")):
            case_id = strip_nrrd_suffix(image_path)

            label_path = annotations_dir / f"{case_id}.nrrd"
            centerline_path = centerlines_dir / f"{case_id}.vtp"

            if not label_path.exists():
                print(f"Skipping {case_id}: missing annotation {label_path.name}")
                continue

            # Centerlines are useful, but your current ScanRecord does not store them.
            # This warning is only informational.
            if centerlines_dir.exists() and not centerline_path.exists():
                print(f"Note: {case_id} has no matching centerline file")

            record = ScanRecord(
                case_id=case_id,
                dataset=f"ASOCA_{group_dir.name}",
                image_path=image_path,
                label_path=label_path,
                anatomical_region=ASOCA_REGION,
                labels_present=[ASOCA_LABEL_NAME],
                annotation_type=annotation_type,
            )

            records.append(record)

    print(f"Found {len(records)} ASOCA cases")

    return records