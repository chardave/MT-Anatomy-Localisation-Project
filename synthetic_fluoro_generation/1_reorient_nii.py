#!/usr/bin/env python3
r"""Reorient a paired CT and CTA to a requested anatomical orientation.

This script changes the voxel axis order and NIfTI affine consistently. It does
not interpolate, resample, register, or crop either image, so CT/CTA intensities
and physical patient coordinates are preserved.

Example (PowerShell):

    python .\1_reorient_nii.py `
      --ct "C:\path\to\1_original_CT.nii.gz" `
      --cta "C:\path\to\2_original_CTA.nii.gz" `
      --output-dir "C:\path\to\prepared"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np


def orientation_code(image: nib.spatialimages.SpatialImage) -> str:
    """Return an image orientation such as ``RAS`` or ``LPS``."""
    axes = nib.aff2axcodes(image.affine)
    if any(axis is None for axis in axes):
        raise ValueError("The NIfTI affine does not define three spatial axes.")
    return "".join(axes)


def validate_target_orientation(code: str) -> str:
    """Validate and normalise a three-letter anatomical orientation code."""
    code = code.strip().upper()
    if len(code) != 3:
        raise ValueError(
            f"Orientation must contain three letters, for example LPS; got {code!r}."
        )

    axes = (
        frozenset(("L", "R")),
        frozenset(("A", "P")),
        frozenset(("I", "S")),
    )
    if any(letter not in "LRAPIS" for letter in code):
        raise ValueError(
            f"Invalid orientation {code!r}; use letters from L/R, A/P, and I/S."
        )
    if {next(i for i, pair in enumerate(axes) if letter in pair) for letter in code} != {
        0,
        1,
        2,
    }:
        raise ValueError(
            f"Invalid orientation {code!r}; specify one left-right, one "
            "anterior-posterior, and one inferior-superior axis."
        )
    return code


def world_corners(image: nib.spatialimages.SpatialImage) -> np.ndarray:
    """Return the eight spatial image corners in physical coordinates."""
    nx, ny, nz = image.shape[:3]
    corners = np.asarray(
        [
            (i, j, k)
            for i in (0, nx - 1)
            for j in (0, ny - 1)
            for k in (0, nz - 1)
        ],
        dtype=np.float64,
    )
    return nib.affines.apply_affine(image.affine, corners)


def sorted_points(points: np.ndarray) -> np.ndarray:
    """Sort 3-D points so two unordered corner sets can be compared."""
    rounded = np.round(np.asarray(points, dtype=np.float64), decimals=7)
    order = np.lexsort((rounded[:, 2], rounded[:, 1], rounded[:, 0]))
    return rounded[order]


def assert_physical_space_preserved(
    original: nib.spatialimages.SpatialImage,
    reoriented: nib.spatialimages.SpatialImage,
    *,
    tolerance_mm: float = 1e-4,
) -> None:
    """Check that reorientation preserved the image's world-space corners."""
    before = sorted_points(world_corners(original))
    after = sorted_points(world_corners(reoriented))
    maximum_error = float(np.max(np.abs(before - after)))
    if maximum_error > tolerance_mm:
        raise RuntimeError(
            "Reorientation changed the physical image extent: maximum corner "
            f"difference is {maximum_error:.6g} mm. Output was not saved."
        )


def reorient_nii(
    image: nib.spatialimages.SpatialImage,
    target: str = "LPS",
) -> nib.spatialimages.SpatialImage:
    """Reorder a NIfTI image to ``target`` without interpolation."""
    target = validate_target_orientation(target)
    current = orientation_code(image)
    if current == target:
        return image

    original_orientation = nib.io_orientation(image.affine)
    target_orientation = nib.orientations.axcodes2ornt(tuple(target))
    transform = nib.orientations.ornt_transform(
        original_orientation,
        target_orientation,
    )
    result = image.as_reoriented(transform)
    assert_physical_space_preserved(image, result)
    return result


def nifti_stem(path: Path) -> str:
    """Return a filename without its .nii or .nii.gz extension."""
    name = path.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    if name.lower().endswith(".nii"):
        return name[:-4]
    raise ValueError(f"Expected a .nii or .nii.gz file: {path}")


def describe_image(
    role: str,
    path: Path,
    image: nib.spatialimages.SpatialImage,
) -> None:
    """Print the geometry needed for a quick preprocessing check."""
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    print(f"{role}: {path}")
    print(f"  orientation: {orientation_code(image)}")
    print(f"  shape:       {tuple(int(value) for value in image.shape[:3])}")
    print(f"  spacing:     {spacing} mm")
    print(f"  dtype:       {image.get_data_dtype()}")


def save_reoriented_image(
    *,
    role: str,
    source: Path,
    destination: Path,
    target: str,
    overwrite: bool,
) -> None:
    """Load, reorient, validate, and save one NIfTI image."""
    if not source.is_file():
        raise FileNotFoundError(f"{role} input was not found: {source}")
    nifti_stem(source)

    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {destination}\n"
            "Use --overwrite to replace existing outputs."
        )

    original = nib.load(str(source))
    if len(original.shape) < 3:
        raise ValueError(f"{role} must be at least three-dimensional: {source}")

    print(f"\n{role} before")
    describe_image(role, source, original)

    result = reorient_nii(original, target)
    if orientation_code(result) != target:
        raise RuntimeError(
            f"Failed to reorient {role} to {target}; got {orientation_code(result)}."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(result, str(destination))

    # Reload the written file so the report checks what is actually on disk.
    saved = nib.load(str(destination))
    assert_physical_space_preserved(original, saved)
    if orientation_code(saved) != target:
        raise RuntimeError(
            f"Saved {role} orientation is {orientation_code(saved)}, not {target}."
        )

    print(f"{role} after")
    describe_image(role, destination, saved)
    print(f"  saved:       {destination}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reorient a paired CT and CTA without interpolation, registration, "
            "resampling, or cropping."
        )
    )
    parser.add_argument("--ct", required=True, type=Path, help="Input CT NIfTI")
    parser.add_argument("--cta", required=True, type=Path, help="Input CTA NIfTI")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory in which the reoriented CT and CTA will be saved",
    )
    parser.add_argument(
        "--target",
        default="LPS",
        help="Three-letter output orientation (default: LPS)",
    )
    parser.add_argument(
        "--ct-output-name",
        help="CT output filename (default: 3_CT_<orientation>.nii.gz)",
    )
    parser.add_argument(
        "--cta-output-name",
        help="CTA output filename (default: 4_CTA_<orientation>.nii.gz)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output files if they already exist",
    )
    return parser.parse_args(argv)


def validate_output_name(name: str, option: str) -> str:
    """Require a filename, rather than an accidental nested or absolute path."""
    path = Path(name)
    if path.name != name or path.is_absolute():
        raise ValueError(f"{option} must be a filename only, not a path: {name}")
    nifti_stem(path)
    return name


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    target = validate_target_orientation(args.target)
    ct_name = validate_output_name(
        args.ct_output_name or f"3_CT_{target}.nii.gz",
        "--ct-output-name",
    )
    cta_name = validate_output_name(
        args.cta_output_name or f"4_CTA_{target}.nii.gz",
        "--cta-output-name",
    )

    ct_destination = args.output_dir / ct_name
    cta_destination = args.output_dir / cta_name
    if ct_destination.resolve() == cta_destination.resolve():
        raise ValueError("The CT and CTA output paths must be different.")

    print("Paired CT/CTA reorientation")
    print(f"Target orientation: {target}")
    print("No interpolation or resampling will be performed.")

    # Validate both inputs before writing either output.
    for role, path in (("CT", args.ct), ("CTA", args.cta)):
        if not path.is_file():
            raise FileNotFoundError(f"{role} input was not found: {path}")
        nifti_stem(path)
    for destination in (ct_destination, cta_destination):
        if destination.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {destination}\n"
                "Use --overwrite to replace existing outputs."
            )

    save_reoriented_image(
        role="CT",
        source=args.ct,
        destination=ct_destination,
        target=target,
        overwrite=args.overwrite,
    )
    save_reoriented_image(
        role="CTA",
        source=args.cta,
        destination=cta_destination,
        target=target,
        overwrite=args.overwrite,
    )

    print("\nCompleted successfully.")
    print(f"CT output:  {ct_destination}")
    print(f"CTA output: {cta_destination}")


if __name__ == "__main__":
    main()
