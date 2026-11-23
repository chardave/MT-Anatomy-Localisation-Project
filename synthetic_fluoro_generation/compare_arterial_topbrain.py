"""Bring arterial vessel masks onto the ready grid and compare them with TopBrain.

Steps:
1. Optionally crop the arterial mask from the full-field run (made on
   7_Aligned_CTA, i.e. the template grid) onto the ready grid, using
   ready_grid.json. The cropped mask is saved.
2. For every arterial mask given, check that it is on the 10_ready_CTA grid, then
   compare it with the TopBrain label map:
     - Dice and voxel counts against TopBrain (all labels, and arteries only)
     - per TopBrain label: fraction of its voxels that arterial also marks as vessel
     - arterial voxels that TopBrain does not label at all
   Results are printed and written to a CSV.

Example (Windows, from synthetic_fluoro_generation, in itk_elastix_env):

    python compare_arterial_topbrain.py ^
        --case-dir "..\\synth_fluoro_gen_test\\MTAL_KCH_43\\template" ^
        --ready-mask 13b_Arterial_vessel_mask_ready.nii.gz ^
        --full-mask 13c_Arterial_vessel_mask_full_template.nii.gz

File names are taken relative to --case-dir unless they are absolute.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

VENOUS_LABELS = {35, 36, 37, 38, 39, 40}  # VoG, StS, ICVs, R/L BVR, SSS


def resolve(case_dir: Path, name):
    if name is None:
        return None
    p = Path(name)
    return p if p.is_absolute() else case_dir / p


def read_info(path: Path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    return reader


def same_grid(img, ref) -> bool:
    return (
        np.allclose(img.GetSize(), ref.GetSize())
        and np.allclose(img.GetSpacing(), ref.GetSpacing(), atol=1e-4)
        and np.allclose(img.GetOrigin(), ref.GetOrigin(), atol=1e-3)
        and np.allclose(img.GetDirection(), ref.GetDirection(), atol=1e-4)
    )


def paste_onto_grid(mask, ref, label: str):
    """arterial writes its mask on a head bounding box cut out of the input CTA.

    If the mask has the reference spacing and direction and its origin sits on a
    whole voxel of the reference grid, paste it back into a zero volume on the
    reference grid (nearest neighbour, no interpolation). Otherwise stop.
    """
    if same_grid(mask, ref):
        return mask
    if not (np.allclose(mask.GetSpacing(), ref.GetSpacing(), atol=1e-4)
            and np.allclose(mask.GetDirection(), ref.GetDirection(), atol=1e-4)):
        raise SystemExit(
            f"{label}: spacing/direction differ from the reference CTA; it needs a real resample, not a paste.\n"
            f"  mask: spacing={mask.GetSpacing()} direction={mask.GetDirection()}\n"
            f"  CTA:  spacing={ref.GetSpacing()} direction={ref.GetDirection()}"
        )
    offset = _continuous_index(ref, mask.GetOrigin())
    if not np.allclose(offset, np.round(offset), atol=1e-2):
        raise SystemExit(f"{label}: mask origin is not on a voxel of the reference grid (offset {offset}).")
    offset = np.round(offset).astype(int)
    end = offset + np.array(mask.GetSize())
    inside = bool(np.all(offset >= 0) and np.all(end <= np.array(ref.GetSize())))
    print(f"{label}: arterial cropped box size {mask.GetSize()} at voxel offset {offset.tolist()} "
          f"of the {ref.GetSize()} CTA -> pasting back onto the full grid"
          + ("" if inside else " (box runs past the CTA edge; the overhang is dropped)"))

    return sitk.Resample(
        sitk.Cast(mask > 0, sitk.sitkUInt8), list(ref.GetSize()), sitk.Transform(), sitk.sitkNearestNeighbor,
        ref.GetOrigin(), ref.GetSpacing(), ref.GetDirection(), 0, sitk.sitkUInt8,
    )


def _continuous_index(ref, point):
    """Physical point -> continuous index for an ImageFileReader (which has no transform helpers)."""
    direction = np.array(ref.GetDirection()).reshape(3, 3)
    spacing = np.array(ref.GetSpacing())
    rel = np.array(point) - np.array(ref.GetOrigin())
    return np.linalg.solve(direction * spacing, rel)


def crop_full_to_ready(full_mask_path: Path, grid_json: Path, aligned_cta: Path, out_path: Path):
    grid = json.loads(grid_json.read_text())
    if grid.get("grid") != "template":
        raise SystemExit(f"ready_grid.json says grid={grid.get('grid')!r}; this crop only handles the template grid.")

    mask = sitk.ReadImage(str(full_mask_path))
    if aligned_cta is not None and aligned_cta.exists():
        mask = paste_onto_grid(mask, read_info(aligned_cta), full_mask_path.name)

    index = [int(v) for v in grid["crop_index_xyz"]]
    size = [int(v) for v in grid["crop_size_xyz"]]
    for axis in range(3):
        if index[axis] + size[axis] > mask.GetSize()[axis]:
            raise SystemExit(f"Crop runs past the mask along axis {axis}: index {index} size {size} vs mask {mask.GetSize()}")

    cropped = sitk.RegionOfInterest(mask, size, index)
    cropped = sitk.Cast(cropped > 0, sitk.sitkUInt8)
    sitk.WriteImage(cropped, str(out_path), True)
    print(f"Cropped full-field mask -> {out_path}")
    return out_path


def compare(name, mask_img, labels_img, mapping, rows):
    m = sitk.GetArrayViewFromImage(mask_img) > 0
    lab = sitk.GetArrayViewFromImage(labels_img)
    tb_all = lab > 0
    tb_art = tb_all & ~np.isin(lab, list(VENOUS_LABELS))

    def dice(a, b):
        denom = a.sum() + b.sum()
        return float(2 * (a & b).sum() / denom) if denom else float("nan")

    voxel_ml = float(np.prod(labels_img.GetSpacing())) / 1000.0
    print(f"\n=== {name} ===")
    print(f"arterial vessel volume:        {m.sum() * voxel_ml:8.2f} ml")
    print(f"TopBrain volume (all labels):  {tb_all.sum() * voxel_ml:8.2f} ml")
    print(f"TopBrain volume (arteries):    {tb_art.sum() * voxel_ml:8.2f} ml")
    print(f"Dice vs TopBrain all labels:   {dice(m, tb_all):.3f}")
    print(f"Dice vs TopBrain arteries:     {dice(m, tb_art):.3f}")
    extra = m & ~tb_all
    print(f"arterial voxels not labelled by TopBrain: {extra.sum() * voxel_ml:.2f} ml "
          f"({100 * extra.sum() / max(m.sum(), 1):.1f}% of arterial mask)")

    rows.append({"mask": name, "label_id": "", "label": "ALL (Dice, arteries only)",
                 "topbrain_voxels": int(tb_art.sum()), "covered_by_arterial": int((m & tb_art).sum()),
                 "fraction_covered": round(dice(m, tb_art), 4)})

    print(f"\n{'id':>3}  {'label':10s} {'TopBrain vox':>12} {'covered':>8}")
    id_to_name = {v: k for k, v in mapping.items() if v != 0}
    for lid in sorted(id_to_name):
        sel = lab == lid
        n = int(sel.sum())
        if n == 0:
            continue
        cov = int((m & sel).sum())
        frac = cov / n
        bar = "#" * int(round(frac * 20))
        vein = " (vein)" if lid in VENOUS_LABELS else ""
        print(f"{lid:>3}  {id_to_name[lid]:10s} {n:>12} {frac:>7.0%}  {bar}{vein}")
        rows.append({"mask": name, "label_id": lid, "label": id_to_name[lid],
                     "topbrain_voxels": n, "covered_by_arterial": cov, "fraction_covered": round(frac, 4)})


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case-dir", required=True, type=Path, help="Folder holding 10_ready_CTA.nii.gz etc.")
    ap.add_argument("--ready-mask", help="arterial mask from RUN=ready (already on the ready grid)")
    ap.add_argument("--full-mask", help="arterial mask from RUN=full on 7_Aligned_CTA (template grid)")
    ap.add_argument("--ready-cta", default="10_ready_CTA.nii.gz")
    ap.add_argument("--aligned-cta", default="7_Aligned_CTA.nii.gz")
    ap.add_argument("--ready-grid", default="ready_grid.json")
    ap.add_argument("--labels", default="13_CTA_vessel_labels.nii.gz", help="TopBrain label map on the ready grid")
    ap.add_argument("--mapping", default=str(here / "vessel_mapping.json"))
    ap.add_argument("--out-csv", default="arterial_vs_topbrain.csv")
    args = ap.parse_args()

    case_dir = args.case_dir
    if not args.ready_mask and not args.full_mask:
        ap.error("give --ready-mask and/or --full-mask")

    ready_cta = resolve(case_dir, args.ready_cta)
    reader = read_info(ready_cta)

    labels = sitk.ReadImage(str(resolve(case_dir, args.labels)))
    mapping = json.loads(Path(args.mapping).read_text())["labels"]

    masks = []
    if args.ready_mask:
        ready = resolve(case_dir, args.ready_mask)
        img = sitk.ReadImage(str(ready))
        if not same_grid(img, reader):
            out = ready.with_name(ready.name.replace(".nii.gz", "") + "_on_ready.nii.gz")
            sitk.WriteImage(paste_onto_grid(img, reader, ready.name), str(out), True)
            print(f"Saved {out}")
            ready = out
        masks.append(("ready run (intracranial)", ready))
    if args.full_mask:
        full = resolve(case_dir, args.full_mask)
        out = full.with_name(full.name.replace(".nii.gz", "").replace("_template", "") + "_cropped_ready.nii.gz")
        crop_full_to_ready(full, resolve(case_dir, args.ready_grid), resolve(case_dir, args.aligned_cta), out)
        masks.append(("full run (extracranial), cropped", out))

    rows = []
    for name, path in masks:
        img = sitk.ReadImage(str(path))
        print(f"\nGrid check for {path.name} against {ready_cta.name}:")
        ok = True
        for key, value in (("Size", reader.GetSize()), ("Spacing", reader.GetSpacing()),
                           ("Origin", reader.GetOrigin()), ("Direction", reader.GetDirection())):
            match = bool(np.allclose(getattr(img, "Get" + key)(), value, atol=1e-3))
            ok &= match
            print(f"  {key:9s} {'OK' if match else 'DIFFERS'}")
        if not ok:
            raise SystemExit(f"{path.name} is not on the ready grid; resample it (nearest neighbour) first.")
        values = np.unique(sitk.GetArrayViewFromImage(img))
        print(f"  values: {values.tolist()}")
        compare(name, img, labels, mapping, rows)

    out_csv = resolve(case_dir, args.out_csv)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
