#!/usr/bin/env python3
"""Create a vessel-enhanced NIfTI for an apples-to-apples DRR comparison.

The original CT/CTA is left unchanged.  Voxels selected by an integer vessel
label map are enhanced either by imposing a minimum HU value (``floor``) or by
adding a fixed HU increment (``add``).  The resulting NIfTI can be loaded by
both nanoDRR and DeepDRR, while the unchanged label map is used for overlays.

CT/CTA and label images are read with SimpleITK, allowing the vessel label map
to be supplied as either NIfTI (``.nii``/``.nii.gz``) or MetaImage (``.mha``).
The optional ``--reorient LPS`` setting reorients both images together before
enhancement so that the vessel labels remain aligned with the CT.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import SimpleITK as sitk


def load_mapping(path: Path) -> dict[str, int]:
    """Load ``{"labels": {name: id}}`` JSON or label_name/label_id CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mapping file not found: {path}")

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        labels = payload.get("labels", payload)
        if not isinstance(labels, dict):
            raise ValueError("JSON mapping must contain an object named 'labels'.")
        mapping = {str(name): int(label_id) for name, label_id in labels.items()}
    elif path.suffix.lower() == ".csv":
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"label_name", "label_id"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(
                    f"CSV mapping must contain {sorted(required)}; found {reader.fieldnames}."
                )
            mapping = {
                str(row["label_name"]).strip(): int(row["label_id"])
                for row in reader
            }
    else:
        raise ValueError("Mapping must be .json or .csv.")

    result = {name: label_id for name, label_id in mapping.items() if label_id > 0}
    if not result:
        raise ValueError("Mapping contains no positive vessel label IDs.")
    return result


def parse_selection(tokens: Iterable[str] | None, mapping: dict[str, int]) -> set[int]:
    """Resolve optional label names/IDs; default to every mapped vessel label."""
    if not tokens:
        return set(mapping.values())

    selected: set[int] = set()
    unknown: list[str] = []
    for raw in tokens:
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            if token in mapping:
                selected.add(mapping[token])
                continue
            try:
                selected.add(int(token))
            except ValueError:
                unknown.append(token)
    if unknown:
        raise ValueError("Unknown vessel names/IDs: " + ", ".join(unknown))
    if not selected:
        raise ValueError("No vessel labels were selected.")
    return selected


def assert_same_grid(ct: sitk.Image, labels: sitk.Image) -> None:
    """Require matching voxel grids before applying label indices to the CT."""
    if ct.GetDimension() != 3 or labels.GetDimension() != 3:
        raise ValueError(
            f"CT and label map must both be 3-D; got {ct.GetDimension()}-D and "
            f"{labels.GetDimension()}-D."
        )

    if ct.GetSize() != labels.GetSize():
        raise ValueError(
            f"CT and label-map sizes differ: {ct.GetSize()} versus {labels.GetSize()}. "
            "Resample the label map onto the CT grid before enhancement."
        )

    checks = (
        ("spacing", ct.GetSpacing(), labels.GetSpacing(), 1e-5),
        ("origin", ct.GetOrigin(), labels.GetOrigin(), 1e-3),
        ("direction", ct.GetDirection(), labels.GetDirection(), 1e-5),
    )
    differences: list[str] = []
    for name, ct_values, label_values, tolerance in checks:
        if not np.allclose(ct_values, label_values, atol=tolerance, rtol=0.0):
            delta = float(
                np.max(np.abs(np.asarray(ct_values) - np.asarray(label_values)))
            )
            differences.append(f"{name} (maximum difference {delta:.6g})")

    if differences:
        raise ValueError(
            "CT and label-map physical grids differ in "
            + ", ".join(differences)
            + ". "
            "Resample the label map onto the CT grid before enhancement."
        )


def orientation_code(image: sitk.Image) -> str:
    """Return the three-letter anatomical orientation represented by an image."""
    return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
        image.GetDirection()
    )


def output_name(ct_path: Path, mode: str, value: float, reorient: str) -> Path:
    name = ct_path.name
    stem = name[:-7] if name.endswith(".nii.gz") else ct_path.stem
    value_text = f"{value:g}".replace(".", "p")
    orientation_text = "_LPS" if reorient == "LPS" else ""
    return ct_path.with_name(
        f"{stem}{orientation_text}_vessel_enhanced_{mode}_{value_text}HU.nii.gz"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enhance vessel-labelled CT/CTA voxels and save a new NIfTI."
    )
    parser.add_argument(
        "--ct", required=True, type=Path, help="Input CT/CTA NIfTI or MetaImage"
    )
    parser.add_argument(
        "--labels",
        required=True,
        type=Path,
        help="Integer vessel label map (.nii, .nii.gz or .mha)",
    )
    parser.add_argument("--mapping", required=True, type=Path, help="vessel_mapping.json or mapping CSV")
    parser.add_argument(
        "--mode",
        choices=("floor", "add"),
        default="floor",
        help="floor: impose a minimum vessel HU; add: add a fixed HU increment",
    )
    parser.add_argument(
        "--value",
        type=float,
        default=1500.0,
        help="Minimum HU for floor mode or HU increment for add mode (default: 1500)",
    )
    parser.add_argument(
        "--include",
        action="append",
        help="Optional comma-separated vessel names or IDs. Default: every mapped vessel label.",
    )
    parser.add_argument(
        "--reorient",
        choices=("LPS", "off"),
        default="off",
        help=(
            "LPS: reorient both the CT and label map to LPS before enhancement; "
            "off: retain their original orientation (default: off)"
        ),
    )
    parser.add_argument("--output", type=Path, help="Output .nii or .nii.gz path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.value <= 0:
        raise ValueError("--value must be positive.")
    if not args.ct.exists():
        raise FileNotFoundError(f"CT/CTA not found: {args.ct}")
    if not args.labels.exists():
        raise FileNotFoundError(f"Vessel label map not found: {args.labels}")

    mapping = load_mapping(args.mapping)
    selected_ids = parse_selection(args.include, mapping)

    ct_image = sitk.ReadImage(str(args.ct))
    label_image = sitk.ReadImage(str(args.labels))
    assert_same_grid(ct_image, label_image)

    ct_orientation_before = orientation_code(ct_image)
    label_orientation_before = orientation_code(label_image)
    if args.reorient == "LPS":
        # DICOMOrient permutes and flips voxel axes while preserving every
        # voxel's physical location. Apply it to both images so their array
        # indices continue to correspond after reorientation.
        ct_image = sitk.DICOMOrient(ct_image, "LPS")
        label_image = sitk.DICOMOrient(label_image, "LPS")
        assert_same_grid(ct_image, label_image)
    ct_orientation_after = orientation_code(ct_image)
    label_orientation_after = orientation_code(label_image)

    # SimpleITK returns arrays in [z, y, x] order for both formats. Reading both
    # images through the same library keeps corresponding voxel indices aligned.
    ct = sitk.GetArrayFromImage(ct_image).astype(np.float32, copy=False)
    labels_float = sitk.GetArrayFromImage(label_image)
    labels_rounded = np.rint(labels_float)
    if not np.allclose(labels_float, labels_rounded, atol=1e-4, rtol=0.0):
        raise ValueError("The segmentation contains non-integer label values.")
    labels = labels_rounded.astype(np.int32, copy=False)

    present_ids = {int(x) for x in np.unique(labels) if int(x) > 0}
    selected_present = selected_ids & present_ids
    if not selected_present:
        raise ValueError(
            "None of the selected vessel IDs occur in the segmentation. "
            f"Selected={sorted(selected_ids)}; present={sorted(present_ids)}"
        )

    vessel_mask = np.isin(labels, np.asarray(sorted(selected_present), dtype=np.int32))
    enhanced = ct.copy()
    before = enhanced[vessel_mask].copy()
    if args.mode == "floor":
        enhanced[vessel_mask] = np.maximum(before, np.float32(args.value))
    else:
        enhanced[vessel_mask] = before + np.float32(args.value)

    destination = args.output or output_name(
        args.ct, args.mode, args.value, args.reorient
    )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    output = sitk.GetImageFromArray(enhanced)
    output.CopyInformation(ct_image)
    sitk.WriteImage(output, str(destination), useCompression=True)

    names_by_id = {label_id: name for name, label_id in mapping.items()}
    selected_names = [names_by_id.get(label_id, str(label_id)) for label_id in sorted(selected_present)]
    print(f"Saved: {destination}")
    print(f"Reorientation: {args.reorient}")
    print(f"CT orientation: {ct_orientation_before} -> {ct_orientation_after}")
    print(
        "Label orientation: "
        f"{label_orientation_before} -> {label_orientation_after}"
    )
    print(f"Enhanced voxels: {int(vessel_mask.sum()):,}")
    print(f"Vessel labels present: {len(selected_present)}")
    print("Selected vessels: " + ", ".join(selected_names))
    print(
        "Vessel HU before: "
        f"min={float(before.min()):.1f}, median={float(np.median(before)):.1f}, max={float(before.max()):.1f}"
    )
    after = enhanced[vessel_mask]
    print(
        "Vessel HU after:  "
        f"min={float(after.min()):.1f}, median={float(np.median(after)):.1f}, max={float(after.max()):.1f}"
    )


if __name__ == "__main__":
    main()
