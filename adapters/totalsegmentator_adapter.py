from pathlib import Path
import nibabel as nib
import numpy as np

from adapters.base_adapter import ScanRecord


GENERATED_LABELMAP_NAMES = {
    "vascular_labelmap.nii.gz",
    "vascular_labelmap_continuous.nii.gz",
    "all_labels_labelmap.nii.gz",
}


def strip_nii_suffix(path: Path):
    name = path.name

    if name.endswith(".nii.gz"):
        return name[:-7]

    if name.endswith(".nii"):
        return name[:-4]

    return path.stem


def list_segmentation_files(seg_dir: Path):
    """
    Returns all original TotalSegmentator binary mask files.
    Excludes any labelmaps we created ourselves.
    """
    return sorted([
        p for p in seg_dir.glob("*.nii.gz")
        if p.is_file()
        and p.name not in GENERATED_LABELMAP_NAMES
    ])


def build_totalseg_label_mapping(dataset_root: Path):
    """
    Builds a stable label mapping from all segmentation filenames in the dataset.

    Example:
        aorta.nii.gz -> 1
        brain.nii.gz -> 2
        ...
    """
    label_names = set()

    case_dirs = sorted([
        p for p in dataset_root.glob("s*")
        if p.is_dir()
        and (p / "ct.nii.gz").exists()
        and (p / "segmentations").exists()
    ])

    for case_dir in case_dirs:
        seg_dir = case_dir / "segmentations"

        for seg_file in list_segmentation_files(seg_dir):
            label_names.add(strip_nii_suffix(seg_file))

    label_names = sorted(label_names)

    label_to_id = {
        label_name: idx + 1
        for idx, label_name in enumerate(label_names)
    }

    id_to_label = {
        idx: label_name
        for label_name, idx in label_to_id.items()
    }

    return label_to_id, id_to_label


def create_all_labels_labelmap_and_get_present(
    case_dir: Path,
    label_to_id: dict,
    id_to_label: dict,
    min_voxels: int = 10,
):
    """
    For one TotalSegmentator case:

    1. Reads each individual binary segmentation mask.
    2. Checks whether that label is actually present.
    3. Creates one combined multi-class labelmap.
    4. Returns:
        labelmap_path, labels_present
    """

    seg_dir = case_dir / "segmentations"
    output_path = seg_dir / "all_labels_labelmap.nii.gz"

    # If the combined all-label map already exists, use it directly
    if output_path.exists():
        data = np.asanyarray(nib.load(str(output_path)).dataobj)
        unique_values = np.unique(data).astype(int)

        labels_present = [
            id_to_label[label_id]
            for label_id in unique_values
            if label_id != 0 and label_id in id_to_label
        ]

        return output_path, labels_present

    seg_files = list_segmentation_files(seg_dir)

    if len(seg_files) == 0:
        raise FileNotFoundError(f"No segmentation masks found for {case_dir.name}")

    # Use the first segmentation file as the reference for shape/affine/header
    reference_img = nib.load(str(seg_files[0]))
    combined = np.zeros(reference_img.shape, dtype=np.uint8)

    labels_present = []

    for seg_file in seg_files:
        label_name = strip_nii_suffix(seg_file)

        if label_name not in label_to_id:
            continue

        label_id = label_to_id[label_name]

        mask_img = nib.load(str(seg_file))
        mask_data = np.asanyarray(mask_img.dataobj)

        mask = mask_data > 0
        voxel_count = int(mask.sum())

        if voxel_count < min_voxels:
            continue

        combined[mask] = label_id
        labels_present.append(label_name)

    combined_img = nib.Nifti1Image(
        combined,
        reference_img.affine,
        reference_img.header,
    )

    combined_img.set_data_dtype(np.uint8)
    nib.save(combined_img, str(output_path))

    return output_path, labels_present

REGION_LABELS = {
    "head_neck": {
        "brain",
        "skull",
        "spinal_cord",
        "vertebrae_C1",
        "vertebrae_C2",
        "vertebrae_C3",
        "vertebrae_C4",
        "vertebrae_C5",
        "vertebrae_C6",
        "vertebrae_C7",
        "thyroid_gland",
        "common_carotid_artery_left",
        "common_carotid_artery_right",
    },

    "thorax_chest": {
        "heart",
        "lung_upper_lobe_left",
        "lung_lower_lobe_left",
        "lung_upper_lobe_right",
        "lung_middle_lobe_right",
        "lung_lower_lobe_right",
        "sternum",
        "rib_left_1",
        "rib_right_1",
        "rib_left_6",
        "rib_right_6",
        "rib_left_12",
        "rib_right_12",
        "vertebrae_T1",
        "vertebrae_T6",
        "vertebrae_T12",
        "clavicula_left",
        "clavicula_right",
        "scapula_left",
        "scapula_right",
        "superior_vena_cava",
        "brachiocephalic_trunk",
        "brachiocephalic_vein_left",
        "brachiocephalic_vein_right",
        "pulmonary_vein",
    },

    "abdomen": {
        "liver",
        "spleen",
        "kidney_left",
        "kidney_right",
        "kidney_cyst_left",
        "kidney_cyst_right",
        "gallbladder",
        "stomach",
        "pancreas",
        "adrenal_gland_left",
        "adrenal_gland_right",
        "duodenum",
        "small_bowel",
        "colon",
        "portal_vein_and_splenic_vein",
        "inferior_vena_cava",
        "vertebrae_L1",
        "vertebrae_L2",
        "vertebrae_L3",
    },

    "pelvis_lower_body": {
        "urinary_bladder",
        "prostate",
        "sacrum",
        "vertebrae_L4",
        "vertebrae_L5",
        "vertebrae_S1",
        "hip_left",
        "hip_right",
        "femur_left",
        "femur_right",
        "iliac_artery_left",
        "iliac_artery_right",
        "iliac_vena_left",
        "iliac_vena_right",
        "gluteus_maximus_left",
        "gluteus_maximus_right",
        "gluteus_medius_left",
        "gluteus_medius_right",
        "gluteus_minimus_left",
        "gluteus_minimus_right",
    },
}


def infer_body_region(labels_present):
    """
    Roughly infers which body region the scan covers based on which
    TotalSegmentator labels are actually present.
    """

    labels_present = set(labels_present)

    scores = {}

    for region_name, region_labels in REGION_LABELS.items():
        scores[region_name] = len(labels_present.intersection(region_labels))

    active_regions = [
        region
        for region, score in scores.items()
        if score > 0
    ]

    if len(active_regions) == 0:
        return "unknown"

    # If it clearly covers multiple regions, keep that information.
    # This is useful for your future rough full-body combination.
    return "+".join(active_regions)

def build_totalsegmentator_records(
    dataset_root: Path,
    annotation_type: str = "automatic",
):
    records = []

    case_dirs = sorted([
        p for p in dataset_root.glob("s*")
        if p.is_dir()
        and (p / "ct.nii.gz").exists()
        and (p / "segmentations").exists()
    ])

    print(f"[TotalSegmentator] Found {len(case_dirs)} case folders", flush=True)

    print("[TotalSegmentator] Building global label mapping...", flush=True)
    label_to_id, id_to_label = build_totalseg_label_mapping(dataset_root)
    print(f"[TotalSegmentator] Found {len(label_to_id)} possible labels", flush=True)

    for i, case_dir in enumerate(case_dirs, start=1):
        print(
            f"[TotalSegmentator] Processing {i}/{len(case_dirs)}: {case_dir.name}",
            flush=True,
        )

        image_path = case_dir / "ct.nii.gz"

        try:
            label_path, labels_present = create_all_labels_labelmap_and_get_present(
                case_dir=case_dir,
                label_to_id=label_to_id,
                id_to_label=id_to_label,
                min_voxels=10,
            )
        except Exception as e:
            print(f"[TotalSegmentator] Skipping {case_dir.name}: {e}", flush=True)
            continue

        anatomical_region = infer_body_region(labels_present)

        print(
            f"  Labels present: {len(labels_present)} | Region: {anatomical_region}",
            flush=True,
        )

        record = ScanRecord(
            case_id=case_dir.name,
            dataset="totalsegmentator",
            image_path=image_path,
            label_path=label_path,
            anatomical_region=anatomical_region,
            labels_present=labels_present,
            annotation_type=annotation_type,
        )

        records.append(record)

    return records