from __future__ import annotations

import argparse
import os
import csv
from pathlib import Path
from typing import Any

import itk
import numpy as np
import SimpleITK as sitk


# ============================================================================
# SETTINGS
# ============================================================================

# Folder holding one sub-folder per patient scan: pass --kch-root or set this
# environment variable. Kept out of the code so no local paths are committed.
KCH_ROOT_ENV = "KCH_SCANS_ROOT"

# Existing registration outputs for each patient are expected in:
#   <patient>\output\CT_registered_to_CTA.nii.gz
#   <patient>\output\TransformParameters.0.txt
REGISTRATION_FOLDER_NAME = "output"

# Cropped outputs are written inside each patient's registration folder.
CROP_FOLDER_NAME = "cropped_to_CT_coverage"

# If True, patients whose three final cropped files already exist are skipped.
# Change to False to recreate/overwrite all cropped outputs.
SKIP_COMPLETED = True


CTA_FILENAME = "4_CTA_Neck_and_COW_0.625mm.nii.gz"
STANDARD_CT_FILENAME = "3_Brain_0.625mm.nii.gz"
TILTED_CT_FILENAME = "3_Brain_0.625mm_Tilt_1.nii.gz"
REGISTERED_CT_FILENAME = "CT_registered_to_CTA.nii.gz"
TRANSFORM_FILENAME = "TransformParameters.0.txt"

FINAL_OUTPUT_FILENAMES = (
    "CTA_cropped.nii.gz",
    "CT_registered_cropped.nii.gz",
    "CT_coverage_cropped.nii.gz",
)


def geometry_matches(
    image_a: sitk.Image,
    image_b: sitk.Image,
    tolerance: float = 1e-5,
) -> bool:
    """Return True when two images occupy the same physical voxel grid."""
    return (
        image_a.GetSize() == image_b.GetSize()
        and np.allclose(
            image_a.GetSpacing(), image_b.GetSpacing(), atol=tolerance, rtol=0
        )
        and np.allclose(
            image_a.GetOrigin(), image_b.GetOrigin(), atol=tolerance, rtol=0
        )
        and np.allclose(
            image_a.GetDirection(), image_b.GetDirection(), atol=tolerance, rtol=0
        )
    )


def print_geometry(name: str, image: sitk.Image) -> None:
    """Print image geometry for troubleshooting."""
    print(f"\n{name}")
    print(f"  Size:      {image.GetSize()}")
    print(f"  Spacing:   {image.GetSpacing()}")
    print(f"  Origin:    {image.GetOrigin()}")
    print(f"  Direction: {image.GetDirection()}")


def choose_original_ct(ct_folder: Path) -> Path:
    """Prefer Tilt_1 when present; otherwise use the standard CT."""
    tilted_ct = ct_folder / TILTED_CT_FILENAME
    standard_ct = ct_folder / STANDARD_CT_FILENAME

    if tilted_ct.is_file():
        return tilted_ct
    if standard_ct.is_file():
        return standard_ct

    raise FileNotFoundError(
        f"Neither expected CT was found in {ct_folder}: "
        f"{TILTED_CT_FILENAME} or {STANDARD_CT_FILENAME}"
    )


def find_patient_directories(root: Path) -> list[Path]:
    """Find immediate KCH patient folders containing folders 3 or 4."""
    if not root.is_dir():
        raise NotADirectoryError(f"KCH root folder not found:\n{root}")

    patients = [
        path
        for path in root.iterdir()
        if path.is_dir() and ((path / "3").is_dir() or (path / "4").is_dir())
    ]
    return sorted(patients, key=lambda path: path.name.lower())


def transform_ct_coverage(
    original_ct_path: Path,
    transform_path: Path,
    original_coverage_path: Path,
    registered_coverage_path: Path,
) -> None:
    """Transform a full-original-CT coverage mask into the fixed CTA grid."""
    original_ct = sitk.ReadImage(str(original_ct_path))

    # Every original CT voxel is valid coverage, so every voxel is set to 1.
    ct_coverage = sitk.Image(original_ct.GetSize(), sitk.sitkUInt8)
    ct_coverage.CopyInformation(original_ct)
    ct_coverage = ct_coverage + 1
    sitk.WriteImage(ct_coverage, str(original_coverage_path), True)

    coverage_itk = itk.imread(str(original_coverage_path), itk.UC)

    saved_transform = itk.ParameterObject.New()
    saved_transform.ReadParameterFile(str(transform_path))

    coverage_parameter_map = saved_transform.GetParameterMap(0)
    coverage_parameter_map["FinalBSplineInterpolationOrder"] = ["0"]
    coverage_parameter_map["DefaultPixelValue"] = ["0"]
    coverage_parameter_map["ResultImagePixelType"] = ["unsigned char"]

    coverage_transform = itk.ParameterObject.New()
    coverage_transform.AddParameterMap(coverage_parameter_map)

    registered_coverage_itk = itk.transformix_filter(
        coverage_itk,
        transform_parameter_object=coverage_transform,
        log_to_console=False,
    )
    itk.imwrite(
        registered_coverage_itk,
        str(registered_coverage_path),
        compression=True,
    )


def process_patient(patient_dir: Path) -> dict[str, Any]:
    """Crop one patient's CTA and registered CT to valid original-CT coverage."""
    patient_name = patient_dir.name
    ct_folder = patient_dir / "3"
    cta_folder = patient_dir / "4"
    registration_dir = patient_dir / REGISTRATION_FOLDER_NAME
    crop_output_dir = registration_dir / CROP_FOLDER_NAME

    original_ct_path = choose_original_ct(ct_folder)
    cta_path = cta_folder / CTA_FILENAME
    registered_ct_path = registration_dir / REGISTERED_CT_FILENAME
    transform_path = registration_dir / TRANSFORM_FILENAME

    required_files = (cta_path, registered_ct_path, transform_path)
    missing_files = [path for path in required_files if not path.is_file()]
    if missing_files:
        missing_text = "; ".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"Missing required file(s): {missing_text}")

    final_output_paths = [
        crop_output_dir / filename for filename in FINAL_OUTPUT_FILENAMES
    ]
    if SKIP_COMPLETED and all(path.is_file() for path in final_output_paths):
        return {
            "patient": patient_name,
            "status": "skipped_completed",
            "ct_used": original_ct_path.name,
            "crop_size_x": "",
            "crop_size_y": "",
            "crop_size_z": "",
            "valid_voxels": "",
            "message": "All final cropped outputs already exist.",
        }

    crop_output_dir.mkdir(parents=True, exist_ok=True)

    original_coverage_path = crop_output_dir / "CT_coverage_original.nii.gz"
    registered_coverage_path = (
        crop_output_dir / "CT_coverage_registered_to_CTA.nii.gz"
    )

    transform_ct_coverage(
        original_ct_path=original_ct_path,
        transform_path=transform_path,
        original_coverage_path=original_coverage_path,
        registered_coverage_path=registered_coverage_path,
    )

    cta = sitk.ReadImage(str(cta_path), sitk.sitkFloat32)
    registered_ct = sitk.ReadImage(str(registered_ct_path), sitk.sitkFloat32)
    registered_coverage = sitk.ReadImage(str(registered_coverage_path))
    registered_coverage = sitk.Cast(registered_coverage > 0, sitk.sitkUInt8)

    if not geometry_matches(cta, registered_ct):
        print_geometry(f"{patient_name} CTA", cta)
        print_geometry(f"{patient_name} registered CT", registered_ct)
        raise ValueError("CTA and registered CT are not on the same voxel grid.")

    if not geometry_matches(cta, registered_coverage):
        print_geometry(f"{patient_name} CTA", cta)
        print_geometry(f"{patient_name} registered CT coverage", registered_coverage)
        raise ValueError("Registered CT coverage is not on the CTA voxel grid.")

    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.Execute(registered_coverage)
    if not shape_stats.HasLabel(1):
        raise ValueError("The transformed CT-coverage mask is empty.")

    bounding_box = shape_stats.GetBoundingBox(1)
    crop_index = [int(value) for value in bounding_box[:3]]
    crop_size = [int(value) for value in bounding_box[3:]]

    cta_cropped = sitk.RegionOfInterest(
        cta, size=crop_size, index=crop_index
    )
    registered_ct_cropped = sitk.RegionOfInterest(
        registered_ct, size=crop_size, index=crop_index
    )
    coverage_cropped = sitk.RegionOfInterest(
        registered_coverage, size=crop_size, index=crop_index
    )

    if not geometry_matches(cta_cropped, registered_ct_cropped):
        raise RuntimeError("Cropped CTA and CT geometries unexpectedly differ.")
    if not geometry_matches(cta_cropped, coverage_cropped):
        raise RuntimeError("Cropped CTA and coverage-mask geometries differ.")

    sitk.WriteImage(cta_cropped, str(final_output_paths[0]), True)
    sitk.WriteImage(registered_ct_cropped, str(final_output_paths[1]), True)
    sitk.WriteImage(coverage_cropped, str(final_output_paths[2]), True)

    valid_voxels = int(
        np.count_nonzero(sitk.GetArrayViewFromImage(coverage_cropped))
    )

    return {
        "patient": patient_name,
        "status": "processed",
        "ct_used": original_ct_path.name,
        "crop_size_x": crop_size[0],
        "crop_size_y": crop_size[1],
        "crop_size_z": crop_size[2],
        "valid_voxels": valid_voxels,
        "message": "Success",
    }


def write_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write a batch report that can be opened directly in Excel."""
    fieldnames = (
        "patient",
        "status",
        "ct_used",
        "crop_size_x",
        "crop_size_y",
        "crop_size_z",
        "valid_voxels",
        "message",
    )
    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    default_root = os.environ.get(KCH_ROOT_ENV)
    parser.add_argument(
        "--kch-root",
        type=Path,
        default=default_root,
        required=default_root is None,
        help=f"folder with one sub-folder per patient scan (or set {KCH_ROOT_ENV})",
    )
    return parser.parse_args()


def main() -> None:
    kch_root = parse_args().kch_root
    patients = find_patient_directories(kch_root)
    if not patients:
        raise RuntimeError(f"No patient folders were found in:\n{kch_root}")

    print(f"Found {len(patients)} patient folder(s).")
    results: list[dict[str, Any]] = []

    for patient_number, patient_dir in enumerate(patients, start=1):
        print("\n" + "=" * 78)
        print(f"[{patient_number}/{len(patients)}] {patient_dir.name}")

        try:
            result = process_patient(patient_dir)
            if result["status"] == "processed":
                print(
                    "Completed using "
                    f"{result['ct_used']}; crop size = "
                    f"({result['crop_size_x']}, "
                    f"{result['crop_size_y']}, "
                    f"{result['crop_size_z']})"
                )
            else:
                print(f"Skipped: {result['message']}")
        except Exception as error:
            result = {
                "patient": patient_dir.name,
                "status": "failed",
                "ct_used": "",
                "crop_size_x": "",
                "crop_size_y": "",
                "crop_size_z": "",
                "valid_voxels": "",
                "message": f"{type(error).__name__}: {error}",
            }
            print(f"FAILED: {result['message']}")

        results.append(result)

    summary_path = kch_root / "crop_to_CT_coverage_summary.csv"
    write_summary(results, summary_path)

    processed = sum(row["status"] == "processed" for row in results)
    skipped = sum(row["status"] == "skipped_completed" for row in results)
    failed = sum(row["status"] == "failed" for row in results)

    print("\n" + "=" * 78)
    print("Batch complete")
    print(f"  Processed: {processed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")
    print(f"  Summary:   {summary_path}")
    print(
        "\nFor intensity analysis, compare CTA_cropped.nii.gz with "
        "CT_registered_cropped.nii.gz only where "
        "CT_coverage_cropped.nii.gz > 0."
    )


if __name__ == "__main__":
    main()