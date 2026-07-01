from pathlib import Path
from typing import List

from adapters.base_adapter import ScanRecord


SEGA_LABEL_NAME = "Aorta_tree"
SEGA_REGION = "thorax_chest"


def build_sega_records(
    dataset_root: Path,
    annotation_type: str = "manual",
) -> List[ScanRecord]:
    """
    Build ScanRecord entries for the SegA dataset.

    Expected folder structure:

    SegA/
        Dongyang/
            D1/
                D1.nrrd
                D1.seg.nrrd
            D2/
                D2.nrrd
                D2.seg.nrrd

        KiTS/
            K1/
                K1.nrrd
                K1.seg.nrrd

        Rider/
            R1/
                R1.nrrd
                R1.seg.nrrd

    Each segmentation is a binary mask of the aorta and its branches.
    Therefore every valid case gets:
        labels_present = ["Aorta_tree"]
        anatomical_region = "thorax_chest"
    """

    dataset_root = Path(dataset_root)

    if not dataset_root.exists():
        raise FileNotFoundError(f"SegA dataset root does not exist: {dataset_root}")

    # Top-level source folders and their expected patient-folder prefixes
    source_folders = {
        "Dongyang": "D",
        "KiTS": "K",
        "Rider": "R",
    }

    records = []

    for source_name, expected_prefix in source_folders.items():
        source_dir = dataset_root / source_name

        if not source_dir.exists():
            print(f"Warning: missing SegA source folder: {source_dir}")
            continue

        for case_dir in sorted(source_dir.iterdir()):
            if not case_dir.is_dir():
                continue

            case_id = case_dir.name

            # Only use folders with the expected prefix:
            # Dongyang -> D...
            # KiTS -> K...
            # Rider -> R...
            if not case_id.startswith(expected_prefix):
                print(f"Skipping unexpected folder in {source_name}: {case_id}")
                continue

            image_path = case_dir / f"{case_id}.nrrd"
            label_path = case_dir / f"{case_id}.seg.nrrd"

            if not image_path.exists():
                print(f"Skipping {case_id}: missing image file {image_path.name}")
                continue

            if not label_path.exists():
                print(f"Skipping {case_id}: missing segmentation file {label_path.name}")
                continue

            record = ScanRecord(
                case_id=case_id,
                dataset=f"SegA_{source_name}",
                image_path=image_path,
                label_path=label_path,
                anatomical_region=SEGA_REGION,
                labels_present=[SEGA_LABEL_NAME],
                annotation_type=annotation_type,
            )

            records.append(record)

    print(f"Found {len(records)} SegA cases")

    return records