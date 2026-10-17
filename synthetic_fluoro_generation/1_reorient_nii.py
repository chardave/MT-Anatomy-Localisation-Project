#!/usr/bin/env python3
"""Step 1: reorient CT and CTA to orthogonal LPS NIfTI volumes.

Unlike a simple axis reorder, this performs physical resampling when required,
removing gantry-tilt shear from the voxel geometry. No registration is performed.
The nibabel affine is expressed in RAS world coordinates; LPS describes the
voxel-axis orientation, not a change to the NIfTI world convention.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform


def resample_orthogonal_lps(source: Path, target: Path, spacing_override=None, overwrite=False):
    if target.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {target}. Use --overwrite to replace.")
    image = nib.load(str(source))
    if len(image.shape) != 3:
        raise ValueError(f"Only 3D NIfTI supported: {source}: shape={image.shape}")
    src_affine = np.asarray(image.affine, dtype=np.float64)
    if not np.isfinite(src_affine).all() or abs(np.linalg.det(src_affine[:3, :3])) < 1e-10:
        raise ValueError(f"Invalid or singular NIfTI affine: {source}")
    spacing = np.linalg.norm(src_affine[:3, :3], axis=0) if spacing_override is None else np.asarray(spacing_override, dtype=float)
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("Invalid spacing")

    # Bound the voxel centres in RAS physical space (using all 8 corners).
    nx, ny, nz = image.shape
    corners = np.array([[i, j, k, 1] for i in (0, nx-1) for j in (0, ny-1) for k in (0, nz-1)], dtype=float)
    physical = (src_affine @ corners.T).T[:, :3]
    low, high = physical.min(axis=0), physical.max(axis=0)
    out_size = np.ceil((high - low) / spacing).astype(int) + 1

    # NIfTI world coordinates are RAS; voxel axes here are Left, Posterior, Superior.
    out_affine = np.eye(4, dtype=float)
    out_affine[:3, :3] = np.diag([-spacing[0], -spacing[1], spacing[2]])
    out_affine[:3, 3] = [high[0], high[1], low[2]]
    voxel_map = np.linalg.inv(src_affine) @ out_affine

    print(f"\nInput : {source.name} | shape={image.shape} | axes={nib.aff2axcodes(src_affine)}", flush=True)
    print(f"Output: {target.name} | shape={tuple(out_size)} | axes={nib.aff2axcodes(out_affine)}", flush=True)
    print(f"Spacing: {tuple(round(float(v), 5) for v in spacing)} mm", flush=True)
    print("Resampling onto orthogonal LPS grid (linear interpolation)...", flush=True)

    data = np.asarray(image.dataobj, dtype=np.float32)
    # Preserve HU range inside FOV. Outside samples are assigned air attenuation.
    output = affine_transform(
        data, matrix=voxel_map[:3, :3], offset=voxel_map[:3, 3],
        output_shape=tuple(int(n) for n in out_size), order=1,
        mode="constant", cval=-1024.0, prefilter=False, output=np.float32,
    )
    result = nib.Nifti1Image(output, out_affine)
    result.set_qform(out_affine, code=1)
    result.set_sform(out_affine, code=1)
    result.header.set_data_dtype(np.float32)
    nib.save(result, str(target))
    print(f"Saved: {target}", flush=True)
    del data, output, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument("--cta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for p in (args.ct, args.cta):
        if not p.is_file():
            parser.error(f"Input not found: {p}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ct_out = args.output_dir / "3_CT_LPS.nii.gz"
    cta_out = args.output_dir / "4_CTA_LPS.nii.gz"
    for p in (ct_out, cta_out):
        if p.exists() and not args.overwrite:
            parser.error(f"Output already exists: {p} (use --overwrite to replace)")
    resample_orthogonal_lps(args.ct, ct_out, overwrite=args.overwrite)
    resample_orthogonal_lps(args.cta, cta_out, overwrite=args.overwrite)
    print("\nDone. Next: generate fresh brain masks from these corrected volumes.", flush=True)


if __name__ == "__main__":
    main()
