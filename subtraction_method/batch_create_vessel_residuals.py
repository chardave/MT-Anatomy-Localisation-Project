from __future__ import annotations

import argparse
import os
import csv
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk


# =============================================================================
# SETTINGS
# =============================================================================

# Folder holding one sub-folder per patient scan: pass --kch-root or set this
# environment variable. Kept out of the code so no local paths are committed.
KCH_ROOT_ENV = "KCH_SCANS_ROOT"

# Created by batch_joint_histogram_huber.py.
EQUATIONS_CSV_NAME = "CT_to_CTA_Huber_equations.csv"

# Inputs created by crop_cta_to_registered_ct_coverage.py.
CROP_FOLDER = Path("output") / "cropped_to_CT_coverage"
CTA_CROPPED_FILENAME = "CTA_cropped.nii.gz"
CT_CROPPED_FILENAME = "CT_registered_cropped.nii.gz"
COVERAGE_CROPPED_FILENAME = "CT_coverage_cropped.nii.gz"

# Created by transforming folder 3\brain.nii.gz with the same CT-to-CTA
# Elastix transform used for CT_registered_to_CTA.nii.gz. This mask is on the
# full CTA grid; it is placed onto the cropped CTA grid below.
REGISTERED_CT_BRAIN_MASK = (
    Path("output") / "CT_brain_mask_registered_to_CTA.nii.gz"
)

# Outputs created for each patient.
OUTPUT_FOLDER = Path("output") / "vessel_residual3"
CTA_EXPECTED_FILENAME = "CTA_expected_from_CT.nii.gz"
VESSEL_RESIDUAL_FILENAME = "vessel_residual_raw.nii.gz"
POSITIVE_RESIDUAL_FILENAME = "vessel_residual_positive.nii.gz"
BRAIN_MASK_CROPPED_FILENAME = "CT_brain_mask_cropped.nii.gz"

# CTA_expected is set to this value outside the registered brain mask when it
# is saved for inspection. A separate zero-background copy is used only as the
# subtraction term so that the original CTA is preserved outside the brain.
EXPECTED_OUTSIDE_VALUE_HU = -1024.0

# Use air rather than 0 HU (water) outside positive residual enhancement so a
# DRR/fluoroscopy renderer does not project the surrounding rectangular volume.
FLUOROSCOPY_BACKGROUND_HU = -1024.0

# True is required for the first run of this corrected version so that the old
# zero-background residuals are replaced. Set it to False after that if wanted.
OVERWRITE = True


SUMMARY_FIELDS = (
    "patient",
    "status",
    "slope",
    "intercept_hu",
    "equation",
    "valid_coverage_voxels",
    "valid_brain_voxels",
    "positive_residual_voxels",
    "mean_residual_hu",
    "median_residual_hu",
    "message",
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


def parse_float(value: Any, field_name: str, patient: str) -> float:
    """Parse a CSV number, tolerating an Excel-style leading apostrophe."""
    cleaned = str(value).strip().lstrip("'")
    if not cleaned:
        raise ValueError(f"Missing {field_name} for {patient}.")

    try:
        parsed = float(cleaned)
    except ValueError as error:
        raise ValueError(
            f"Invalid {field_name} for {patient}: {value!r}"
        ) from error

    if not np.isfinite(parsed):
        raise ValueError(f"Non-finite {field_name} for {patient}: {value!r}")
    return parsed


def load_equations(csv_path: Path) -> dict[str, tuple[float, float]]:
    """Load patient-specific Huber slope/intercept values from the summary CSV."""
    if not csv_path.is_file():
        raise FileNotFoundError(
            "Huber-equation CSV not found. Run batch_joint_histogram_huber.py "
            f"first:\n{csv_path}"
        )

    equations: dict[str, tuple[float, float]] = {}

    with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"patient", "slope", "intercept_hu"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                "Equation CSV is missing column(s): "
                + ", ".join(sorted(missing_columns))
            )

        for row in reader:
            patient = (row.get("patient") or "").strip()
            if not patient:
                continue

            # Failed cases from the fitting stage normally have blank values.
            if not (row.get("slope") or "").strip() or not (
                row.get("intercept_hu") or ""
            ).strip():
                continue

            slope = parse_float(row["slope"], "slope", patient)
            intercept = parse_float(row["intercept_hu"], "intercept_hu", patient)
            equations[patient] = (slope, intercept)

    if not equations:
        raise ValueError(f"No usable patient equations were found in:\n{csv_path}")

    return equations


def find_patient_directories(root: Path) -> list[Path]:
    """Find immediate KCH patient folders in a stable order."""
    if not root.is_dir():
        raise NotADirectoryError(f"KCH root folder not found:\n{root}")

    return sorted(
        [
            path
            for path in root.iterdir()
            if path.is_dir() and (path / "3").is_dir() and (path / "4").is_dir()
        ],
        key=lambda path: path.name.lower(),
    )


def process_patient(
    patient_dir: Path,
    slope: float,
    intercept: float,
) -> dict[str, Any]:
    """Create expected-CTA and residual volumes for one patient."""
    patient = patient_dir.name
    cropped_dir = patient_dir / CROP_FOLDER
    cta_path = cropped_dir / CTA_CROPPED_FILENAME
    ct_path = cropped_dir / CT_CROPPED_FILENAME
    coverage_path = cropped_dir / COVERAGE_CROPPED_FILENAME
    registered_brain_mask_path = patient_dir / REGISTERED_CT_BRAIN_MASK

    output_dir = patient_dir / OUTPUT_FOLDER
    expected_path = output_dir / CTA_EXPECTED_FILENAME
    residual_path = output_dir / VESSEL_RESIDUAL_FILENAME
    positive_path = output_dir / POSITIVE_RESIDUAL_FILENAME
    brain_mask_cropped_path = output_dir / BRAIN_MASK_CROPPED_FILENAME
    final_outputs = (
        expected_path,
        residual_path,
        positive_path,
        brain_mask_cropped_path,
    )

    missing = [
        str(path)
        for path in (
            cta_path,
            ct_path,
            coverage_path,
            registered_brain_mask_path,
        )
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing required file(s): " + "; ".join(missing))

    if not OVERWRITE and all(path.is_file() for path in final_outputs):
        return {
            "patient": patient,
            "status": "skipped_completed",
            "slope": slope,
            "intercept_hu": intercept,
            "equation": f"CTA_expected = {slope:.8f} * CT {intercept:+.8f}",
            "valid_coverage_voxels": "",
            "valid_brain_voxels": "",
            "positive_residual_voxels": "",
            "mean_residual_hu": "",
            "median_residual_hu": "",
            "message": "All output images already exist.",
        }

    cta = sitk.ReadImage(str(cta_path), sitk.sitkFloat32)
    ct = sitk.ReadImage(str(ct_path), sitk.sitkFloat32)
    coverage = sitk.Cast(sitk.ReadImage(str(coverage_path)) > 0, sitk.sitkUInt8)
    registered_brain_mask = sitk.Cast(
        sitk.ReadImage(str(registered_brain_mask_path)) > 0,
        sitk.sitkUInt8,
    )

    if not geometry_matches(cta, ct):
        raise ValueError("Cropped CTA and registered CT are not on the same grid.")
    if not geometry_matches(cta, coverage):
        raise ValueError("Cropped CT-coverage mask is not on the CTA grid.")

    # The registered brain mask occupies the full CTA grid, whereas `cta` is
    # the cropped CTA. Resample by physical coordinates onto the cropped grid.
    # Identity + nearest-neighbour preserves the binary segmentation labels.
    brain_mask_cropped = sitk.Resample(
        registered_brain_mask,
        cta,
        sitk.Transform(cta.GetDimension(), sitk.sitkIdentity),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    brain_mask_cropped = sitk.Cast(brain_mask_cropped > 0, sitk.sitkUInt8)

    # Use only voxels that are both inside the registered CT brain
    # segmentation and covered by real (non-padded) CT data.
    analysis_mask = sitk.And(brain_mask_cropped, coverage)

    brain_mask_array = sitk.GetArrayViewFromImage(analysis_mask) > 0
    valid_brain_voxels = int(np.count_nonzero(brain_mask_array))
    if valid_brain_voxels == 0:
        raise ValueError(
            "The registered CT brain mask is empty after placing it on the "
            "cropped CTA grid. Check its registration and geometry."
        )

    cta_float = sitk.Cast(cta, sitk.sitkFloat32)

    # Patient-specific Huber mapping:
    # CTA_expected = slope * CT_registered_cropped + intercept
    cta_expected_unmasked = (
        sitk.Cast(ct, sitk.sitkFloat32) * slope + intercept
    )

    # Inspection copy: show the mapped CT only inside the registered brain and
    # use air outside. This image is NOT used directly in the subtraction.
    cta_expected = sitk.Mask(
        cta_expected_unmasked,
        analysis_mask,
        outsideValue=EXPECTED_OUTSIDE_VALUE_HU,
        maskingValue=0,
    )

    # Subtraction term: mapped CT inside the brain and exactly 0 outside.
    # Therefore:
    #   inside brain  -> CTA - mapped CT
    #   outside brain -> CTA - 0 = original CTA unchanged
    enhanced_ct_brain_for_subtraction = sitk.Mask(
        cta_expected_unmasked,
        analysis_mask,
        outsideValue=0.0,
        maskingValue=0,
    )

    vessel_residual_unmasked = cta_float - cta_expected_unmasked
    vessel_residual = cta_float - enhanced_ct_brain_for_subtraction

    # Within the brain, retain only positive enhancement. Non-positive brain
    # residuals become air. Outside the brain, restore the original CTA exactly.
    positive_inside_brain = sitk.Threshold(
        vessel_residual_unmasked,
        lower=float(np.finfo(np.float32).eps),
        upper=float(np.finfo(np.float32).max),
        outsideValue=FLUOROSCOPY_BACKGROUND_HU,
    )

    mask_float = sitk.Cast(analysis_mask, sitk.sitkFloat32)
    positive_residual = (
        positive_inside_brain * mask_float
        + cta_float * (1.0 - mask_float)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(cta_expected, str(expected_path), True)
    sitk.WriteImage(vessel_residual, str(residual_path), True)
    sitk.WriteImage(positive_residual, str(positive_path), True)
    sitk.WriteImage(analysis_mask, str(brain_mask_cropped_path), True)

    # Calculate QA statistics only inside the registered brain segmentation.
    coverage_array = sitk.GetArrayViewFromImage(coverage) > 0
    residual_array = sitk.GetArrayViewFromImage(vessel_residual)
    valid_residuals = residual_array[brain_mask_array]

    return {
        "patient": patient,
        "status": "processed",
        "slope": slope,
        "intercept_hu": intercept,
        "equation": f"CTA_expected = {slope:.8f} * CT {intercept:+.8f}",
        "valid_coverage_voxels": int(np.count_nonzero(coverage_array)),
        "valid_brain_voxels": valid_brain_voxels,
        "positive_residual_voxels": int(np.count_nonzero(valid_residuals > 0)),
        "mean_residual_hu": float(np.mean(valid_residuals, dtype=np.float64)),
        "median_residual_hu": float(np.median(valid_residuals)),
        "message": "Success",
    }


def failed_row(
    patient: str,
    error: Exception,
    slope: float | str = "",
    intercept: float | str = "",
) -> dict[str, Any]:
    """Build one consistent failure row for the batch summary."""
    row: dict[str, Any] = {field: "" for field in SUMMARY_FIELDS}
    row.update(
        {
            "patient": patient,
            "status": "failed",
            "slope": slope,
            "intercept_hu": intercept,
            "message": f"{type(error).__name__}: {error}",
        }
    )
    return row


def write_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Save an Excel-friendly processing report."""
    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDS)
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
    equations = load_equations(kch_root / EQUATIONS_CSV_NAME)
    patients = find_patient_directories(kch_root)
    if not patients:
        raise RuntimeError(f"No patient folders found in:\n{kch_root}")

    print(f"Found {len(patients)} patient folder(s).")
    print(f"Loaded {len(equations)} usable Huber equation(s).")
    rows: list[dict[str, Any]] = []

    for patient_index, patient_dir in enumerate(patients, start=1):
        patient = patient_dir.name
        print("\n" + "=" * 78)
        print(f"[{patient_index}/{len(patients)}] {patient}")

        if patient not in equations:
            error = KeyError(
                "No usable slope/intercept was found for this patient in the "
                "Huber-equation CSV."
            )
            row = failed_row(patient, error)
            print(f"FAILED: {row['message']}")
            rows.append(row)
            continue

        slope, intercept = equations[patient]
        print(f"Using CTA_expected = {slope:.8f} * CT {intercept:+.8f}")

        try:
            row = process_patient(patient_dir, slope, intercept)
            if row["status"] == "processed":
                print("Saved expected-CTA and vessel-residual images.")
                print(
                    "Residual mean/median within brain mask: "
                    f"{row['mean_residual_hu']:.2f} / "
                    f"{row['median_residual_hu']:.2f} HU"
                )
            else:
                print("Skipped: existing complete outputs retained.")
        except Exception as error:
            row = failed_row(patient, error, slope, intercept)
            print(f"FAILED: {row['message']}")

        rows.append(row)

    summary_path = kch_root / "vessel_residual_summary.csv"
    write_summary(rows, summary_path)

    processed = sum(row["status"] == "processed" for row in rows)
    skipped = sum(row["status"] == "skipped_completed" for row in rows)
    failed = sum(row["status"] == "failed" for row in rows)

    print("\n" + "=" * 78)
    print("Batch complete")
    print(f"  Processed: {processed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")
    print(f"  Summary:   {summary_path}")


if __name__ == "__main__":
    main()