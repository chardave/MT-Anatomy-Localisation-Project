from __future__ import annotations

import argparse
import os
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib.colors import LogNorm
from scipy.optimize import least_squares


# =============================================================================
# SETTINGS
# =============================================================================

# Folder holding one sub-folder per patient scan: pass --kch-root or set this
# environment variable. Kept out of the code so no local paths are committed.
KCH_ROOT_ENV = "KCH_SCANS_ROOT"

# Inputs created by the previous registration/cropping steps.
CROP_FOLDER = Path("output") / "cropped_to_CT_coverage"
CTA_CROPPED_FILENAME = "CTA_cropped.nii.gz"
CT_CROPPED_FILENAME = "CT_registered_cropped.nii.gz"
COVERAGE_CROPPED_FILENAME = "CT_coverage_cropped.nii.gz"

# TotalSegmentator CTA brain mask (on the original CTA grid).
CTA_BRAIN_MASK_RELATIVE_PATH = Path("4") / "brain.nii.gz"

# Outputs for each patient.
RESULTS_FOLDER_NAME = "joint_histogram_huber"
PLOT_FILENAME = "CT_vs_CTA_joint_histogram_huber.png"
FIT_FILENAME = "intensity_mapping_huber.json"

# This reproduces the useful display/fit range in joint_histogram_1.py.
# Only paired voxels within BOTH ranges are used for fitting.
CT_FIT_RANGE = (-50.0, 250.0)
CTA_FIT_RANGE = (-75.0, 400.0)

# Histogram bins are centred on HU values and are 2 HU wide, as in the
# supplied script. Change to 1.0 for one-HU bins.
HISTOGRAM_BIN_WIDTH_HU = 2.0

# Optimisation is capped for speed and memory, but the histogram and reported
# total voxel count still use every eligible voxel.
MAX_FIT_POINTS = 200_000
RANDOM_SEED = 42

# The Huber transition scale is estimated from the initial OLS residuals, then
# constrained to a sensible HU range.
HUBER_SCALE_MULTIPLIER = 1.5
HUBER_F_SCALE_MIN = 5.0
HUBER_F_SCALE_MAX = 40.0

# Recreate existing plots/results when True.
OVERWRITE = False


SUMMARY_FIELDS = (
    "patient",
    "status",
    "eligible_voxels",
    "fit_voxels_used",
    "slope",
    "intercept_hu",
    "equation",
    "huber_f_scale_hu",
    "optimizer_success",
    "optimizer_nfev",
    "mae_before_hu",
    "mae_after_hu",
    "rmse_before_hu",
    "rmse_after_hu",
    "median_residual_after_hu",
    "message",
)


def geometry_matches(
    image_a: sitk.Image,
    image_b: sitk.Image,
    tolerance: float = 1e-5,
) -> bool:
    """Return True when two images use the same physical voxel grid."""
    return (
        image_a.GetSize() == image_b.GetSize()
        and np.allclose(image_a.GetSpacing(), image_b.GetSpacing(), atol=tolerance, rtol=0)
        and np.allclose(image_a.GetOrigin(), image_b.GetOrigin(), atol=tolerance, rtol=0)
        and np.allclose(
            image_a.GetDirection(), image_b.GetDirection(), atol=tolerance, rtol=0
        )
    )


def find_patient_directories(root: Path) -> list[Path]:
    """Return immediate KCH patient folders in a stable order."""
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


def integer_centred_bin_edges(
    limits: tuple[float, float],
    width: float,
) -> np.ndarray:
    """Create HU bin edges centred on integer multiples of ``width``."""
    if width <= 0:
        raise ValueError("HISTOGRAM_BIN_WIDTH_HU must be greater than zero.")

    first_centre = np.floor(limits[0] / width) * width
    last_centre = np.ceil(limits[1] / width) * width
    return np.arange(
        first_centre - width / 2,
        last_centre + 1.5 * width,
        width,
        dtype=np.float64,
    )


def place_mask_on_reference_grid(
    mask: sitk.Image,
    reference: sitk.Image,
) -> sitk.Image:
    """Place a physical-space-aligned binary mask on the cropped CTA grid."""
    if geometry_matches(mask, reference):
        return sitk.Cast(mask > 0, sitk.sitkUInt8)

    return sitk.Resample(
        sitk.Cast(mask > 0, sitk.sitkUInt8),
        reference,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )


def fit_huber_line(x_all: np.ndarray, y_all: np.ndarray) -> dict[str, Any]:
    """Fit CTA = slope * CT + intercept using robust Huber least squares."""
    if x_all.size < 100:
        raise ValueError(f"Too few eligible voxels for fitting ({x_all.size}).")

    if x_all.size > MAX_FIT_POINTS:
        rng = np.random.default_rng(RANDOM_SEED)
        indices = rng.choice(x_all.size, size=MAX_FIT_POINTS, replace=False)
        x_fit = x_all[indices].astype(np.float64, copy=False)
        y_fit = y_all[indices].astype(np.float64, copy=False)
    else:
        x_fit = x_all.astype(np.float64, copy=False)
        y_fit = y_all.astype(np.float64, copy=False)

    slope_ols, intercept_ols = np.polyfit(x_fit, y_fit, 1)
    residuals_ols = y_fit - (slope_ols * x_fit + intercept_ols)

    residual_median = np.median(residuals_ols)
    mad = np.median(np.abs(residuals_ols - residual_median))
    robust_sigma = 1.4826 * mad
    f_scale = float(
        np.clip(
            HUBER_SCALE_MULTIPLIER * robust_sigma,
            HUBER_F_SCALE_MIN,
            HUBER_F_SCALE_MAX,
        )
    )

    def residual_function(parameters: np.ndarray) -> np.ndarray:
        slope, intercept = parameters
        return slope * x_fit + intercept - y_fit

    result = least_squares(
        residual_function,
        x0=np.array([slope_ols, intercept_ols], dtype=np.float64),
        loss="huber",
        f_scale=f_scale,
        max_nfev=2_000,
    )

    slope, intercept = (float(value) for value in result.x)
    before_residual = y_fit - x_fit
    after_residual = y_fit - (slope * x_fit + intercept)

    return {
        "slope": slope,
        "intercept_hu": intercept,
        "f_scale_hu": f_scale,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_nfev": int(result.nfev),
        "fit_voxels_used": int(x_fit.size),
        "ols_slope": float(slope_ols),
        "ols_intercept_hu": float(intercept_ols),
        "mae_before_hu": float(np.mean(np.abs(before_residual))),
        "mae_after_hu": float(np.mean(np.abs(after_residual))),
        "rmse_before_hu": float(np.sqrt(np.mean(before_residual**2))),
        "rmse_after_hu": float(np.sqrt(np.mean(after_residual**2))),
        "median_residual_after_hu": float(np.median(after_residual)),
    }


def save_joint_histogram(
    ct_values: np.ndarray,
    cta_values: np.ndarray,
    fit: dict[str, Any],
    patient_name: str,
    output_path: Path,
) -> None:
    """Save a log-count joint histogram with identity, OLS, and Huber lines."""
    x_edges = integer_centred_bin_edges(CT_FIT_RANGE, HISTOGRAM_BIN_WIDTH_HU)
    y_edges = integer_centred_bin_edges(CTA_FIT_RANGE, HISTOGRAM_BIN_WIDTH_HU)
    counts, _, _ = np.histogram2d(ct_values, cta_values, bins=[x_edges, y_edges])

    positive_counts = counts[counts > 0]
    if positive_counts.size == 0:
        raise ValueError("Joint histogram has no occupied bins.")

    fig, ax = plt.subplots(figsize=(9, 8))
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        counts.T,
        cmap="viridis",
        norm=LogNorm(vmin=1, vmax=max(1.0, float(positive_counts.max()))),
        shading="auto",
    )
    fig.colorbar(mesh, ax=ax, label="Paired voxel count (log scale)")

    diagonal_min = max(CT_FIT_RANGE[0], CTA_FIT_RANGE[0])
    diagonal_max = min(CT_FIT_RANGE[1], CTA_FIT_RANGE[1])
    ax.plot(
        [diagonal_min, diagonal_max],
        [diagonal_min, diagonal_max],
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Identity: CTA = CT",
    )

    x_line = np.array(CT_FIT_RANGE, dtype=np.float64)
    ax.plot(
        x_line,
        fit["ols_slope"] * x_line + fit["ols_intercept_hu"],
        color="deepskyblue",
        linestyle=":",
        linewidth=2,
        label=(
            f"OLS: CTA = {fit['ols_slope']:.3f} x CT "
            f"{fit['ols_intercept_hu']:+.2f}"
        ),
    )
    ax.plot(
        x_line,
        fit["slope"] * x_line + fit["intercept_hu"],
        color="white",
        linestyle="-",
        linewidth=2.2,
        label=(
            f"Huber: CTA = {fit['slope']:.3f} x CT "
            f"{fit['intercept_hu']:+.2f}"
        ),
    )

    ax.set_xlim(CT_FIT_RANGE)
    ax.set_ylim(CTA_FIT_RANGE)
    ax.set_xlabel("Registered CT intensity (HU)")
    ax.set_ylabel("CTA intensity (HU)")
    ax.set_title(
        f"{patient_name}: paired brain CT-CTA intensities\n"
        f"Huber fit using {fit['fit_voxels_used']:,} sampled voxels"
    )
    ax.legend(loc="upper left", framealpha=0.92)
    ax.grid(alpha=0.12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def process_patient(patient_dir: Path) -> dict[str, Any]:
    """Create one patient's joint histogram and robust mapping equation."""
    patient_name = patient_dir.name
    cropped_dir = patient_dir / CROP_FOLDER
    cta_path = cropped_dir / CTA_CROPPED_FILENAME
    ct_path = cropped_dir / CT_CROPPED_FILENAME
    coverage_path = cropped_dir / COVERAGE_CROPPED_FILENAME
    brain_mask_path = patient_dir / CTA_BRAIN_MASK_RELATIVE_PATH
    output_dir = patient_dir / "output" / RESULTS_FOLDER_NAME
    plot_path = output_dir / PLOT_FILENAME
    fit_path = output_dir / FIT_FILENAME

    required_paths = (cta_path, ct_path, coverage_path, brain_mask_path)
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required file(s): " + "; ".join(missing))

    if not OVERWRITE and plot_path.is_file() and fit_path.is_file():
        with fit_path.open("r", encoding="utf-8") as fit_file:
            saved = json.load(fit_file)
        return {
            "patient": patient_name,
            "status": "skipped_completed",
            "eligible_voxels": saved.get("eligible_voxels", ""),
            "fit_voxels_used": saved.get("fit_voxels_used", ""),
            "slope": saved.get("slope", ""),
            "intercept_hu": saved.get("intercept_hu", ""),
            "equation": saved.get("equation", ""),
            "huber_f_scale_hu": saved.get("f_scale_hu", ""),
            "optimizer_success": saved.get("optimizer_success", ""),
            "optimizer_nfev": saved.get("optimizer_nfev", ""),
            "mae_before_hu": saved.get("mae_before_hu", ""),
            "mae_after_hu": saved.get("mae_after_hu", ""),
            "rmse_before_hu": saved.get("rmse_before_hu", ""),
            "rmse_after_hu": saved.get("rmse_after_hu", ""),
            "median_residual_after_hu": saved.get(
                "median_residual_after_hu", ""
            ),
            "message": "Existing result retained.",
        }

    output_dir.mkdir(parents=True, exist_ok=True)

    cta = sitk.ReadImage(str(cta_path), sitk.sitkFloat32)
    ct = sitk.ReadImage(str(ct_path), sitk.sitkFloat32)
    coverage = sitk.ReadImage(str(coverage_path))
    brain_mask_original = sitk.ReadImage(str(brain_mask_path))

    if not geometry_matches(cta, ct):
        raise ValueError("Cropped CTA and registered CT are not on the same grid.")
    if not geometry_matches(cta, coverage):
        raise ValueError("Cropped CT coverage mask is not on the CTA grid.")

    brain_mask = place_mask_on_reference_grid(brain_mask_original, cta)

    ct_array = sitk.GetArrayFromImage(ct)
    cta_array = sitk.GetArrayFromImage(cta)
    coverage_array = sitk.GetArrayFromImage(coverage) > 0
    brain_array = sitk.GetArrayFromImage(brain_mask) > 0

    eligible = (
        coverage_array
        & brain_array
        & np.isfinite(ct_array)
        & np.isfinite(cta_array)
        & (ct_array >= CT_FIT_RANGE[0])
        & (ct_array <= CT_FIT_RANGE[1])
        & (cta_array >= CTA_FIT_RANGE[0])
        & (cta_array <= CTA_FIT_RANGE[1])
    )

    ct_values = ct_array[eligible].astype(np.float32, copy=False)
    cta_values = cta_array[eligible].astype(np.float32, copy=False)
    if ct_values.size < 100:
        raise ValueError(
            f"Only {ct_values.size} eligible paired brain voxels were found."
        )

    fit = fit_huber_line(ct_values, cta_values)
    fit["eligible_voxels"] = int(ct_values.size)
    fit["patient"] = patient_name
    fit["equation"] = (
        f"CTA_expected = {fit['slope']:.8f} * CT "
        f"{fit['intercept_hu']:+.8f}"
    )
    fit["vessel_residual_equation"] = "vessel_residual = CTA - CTA_expected"
    fit["ct_fit_range_hu"] = list(CT_FIT_RANGE)
    fit["cta_fit_range_hu"] = list(CTA_FIT_RANGE)
    fit["histogram_bin_width_hu"] = HISTOGRAM_BIN_WIDTH_HU
    fit["mask_definition"] = (
        "CTA TotalSegmentator brain mask AND transformed CT coverage AND HU ranges"
    )

    save_joint_histogram(
        ct_values=ct_values,
        cta_values=cta_values,
        fit=fit,
        patient_name=patient_name,
        output_path=plot_path,
    )

    with fit_path.open("w", encoding="utf-8") as fit_file:
        json.dump(fit, fit_file, indent=2)

    return {
        "patient": patient_name,
        "status": "processed",
        "eligible_voxels": fit["eligible_voxels"],
        "fit_voxels_used": fit["fit_voxels_used"],
        "slope": fit["slope"],
        "intercept_hu": fit["intercept_hu"],
        "equation": fit["equation"],
        "huber_f_scale_hu": fit["f_scale_hu"],
        "optimizer_success": fit["optimizer_success"],
        "optimizer_nfev": fit["optimizer_nfev"],
        "mae_before_hu": fit["mae_before_hu"],
        "mae_after_hu": fit["mae_after_hu"],
        "rmse_before_hu": fit["rmse_before_hu"],
        "rmse_after_hu": fit["rmse_after_hu"],
        "median_residual_after_hu": fit["median_residual_after_hu"],
        "message": fit["optimizer_message"],
    }


def write_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Save all patient-specific equations in one Excel-friendly CSV."""
    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def failed_row(patient_name: str, error: Exception) -> dict[str, Any]:
    row: dict[str, Any] = {field: "" for field in SUMMARY_FIELDS}
    row.update(
        {
            "patient": patient_name,
            "status": "failed",
            "message": f"{type(error).__name__}: {error}",
        }
    )
    return row


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
        raise RuntimeError(f"No patient folders found in:\n{kch_root}")

    print(f"Found {len(patients)} patient folder(s).")
    rows: list[dict[str, Any]] = []

    for patient_index, patient_dir in enumerate(patients, start=1):
        print("\n" + "=" * 78)
        print(f"[{patient_index}/{len(patients)}] {patient_dir.name}")
        try:
            row = process_patient(patient_dir)
            if row["status"] == "processed":
                print(f"Equation: {row['equation']}")
                print(
                    f"MAE before/after: {row['mae_before_hu']:.2f} / "
                    f"{row['mae_after_hu']:.2f} HU"
                )
            else:
                print("Skipped: existing result retained.")
        except Exception as error:
            row = failed_row(patient_dir.name, error)
            print(f"FAILED: {row['message']}")
        rows.append(row)

    summary_path = kch_root / "CT_to_CTA_Huber_equations.csv"
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
    print(
        "\nFor the next step use:\n"
        "  CTA_expected = slope * CT_registered_cropped + intercept\n"
        "  vessel_residual = CTA_cropped - CTA_expected"
    )


if __name__ == "__main__":
    main()