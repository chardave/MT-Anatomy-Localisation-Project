from pathlib import Path
import json
import nibabel as nib
import numpy as np

from adapters.base_adapter import ScanRecord


def strip_nii_suffix(path: Path):
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def topaneu_image_to_mask_path(image_path: Path, masks_dir: Path):
    base = strip_nii_suffix(image_path)

    if base.endswith("_0000"):
        base = base[:-5]

    return masks_dir / f"{base}.nii.gz"


def load_topaneu_mapping(mapping_path: Path):
    with open(mapping_path, "r") as f:
        mapping = json.load(f)

    label_to_id = {
        name: int(value)
        for name, value in mapping["labels"].items()
    }

    id_to_label = {
        int(value): name
        for name, value in mapping["labels"].items()
    }

    return label_to_id, id_to_label


def get_present_labels_from_multiclass_mask(mask_path: Path, id_to_label: dict):
    img = nib.load(str(mask_path))
    data = np.asanyarray(img.dataobj)

    unique_values = np.unique(data).astype(int)

    labels_present = [
        id_to_label[label_id]
        for label_id in unique_values
        if label_id != 0 and label_id in id_to_label
    ]

    return labels_present


def build_topaneu_records(
    dataset_root: Path,
    annotation_type: str = "manual",
):
    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "vessel_masks"
    mapping_path = dataset_root / "vessel_mapping.json"

    if not mapping_path.exists():
        mapping_path = dataset_root / "vessel_mapping"

    label_to_id, id_to_label = load_topaneu_mapping(mapping_path)

    records = []

    image_files = sorted(images_dir.glob("*.nii.gz"))

    for i, image_path in enumerate(image_files, start=1):

        print(f"[TopAneu] Checking {i}/{len(image_files)}: {image_path.name}", flush=True)

        name_lower = image_path.name.lower()

        if "_ct_" not in name_lower:
            continue

        if not image_path.name.endswith("_0000.nii.gz"):
            continue

        mask_path = topaneu_image_to_mask_path(image_path, masks_dir)

        if not mask_path.exists():
            print(f"Missing mask for {image_path.name}")
            continue

        case_id = strip_nii_suffix(mask_path)

        labels_present = get_present_labels_from_multiclass_mask(
            mask_path,
            id_to_label,
        )

        record = ScanRecord(
            case_id=case_id,
            dataset="topaneu",
            image_path=image_path,
            label_path=mask_path,
            anatomical_region="head_neck",
            labels_present=labels_present,
            annotation_type=annotation_type,
        )

        records.append(record)

    return records

if __name__ == "__main__":
    dataset_root = Path(
        r"C:\Users\cd26\OneDrive - King's College London\MT Anatomy Localisation Project\topaneu_deployment"
    )

    records = build_topaneu_records(
        dataset_root,
        annotation_type="manual"
    )

    print("Number of TopAneu CT records found:", len(records))

    for record in records[:5]:
        print(record)