"""Make one label map showing where TopBrain and arterial agree, for viewing in ITK-SNAP.

Labels in the output (and the colours in the matching ITK-SNAP label file):
  1  both models               yellow
  2  TopBrain artery only      green
  3  arterial only             red
  4  TopBrain vein             blue   (arterial voxels on a vein also stay 4)

Example (from synthetic_fluoro_generation):

    python make_arterial_topbrain_overlap.py ^
        --case-dir "..\\synth_fluoro_gen_test\\MTAL_KCH_43\\template" ^
        --arterial-mask 13b_Arterial_vessel_mask_ready.nii.gz

In ITK-SNAP: File > Open Main Image = 10_ready_CTA.nii.gz,
Segmentation > Open Segmentation = 13d_TopBrain_vs_Arterial_overlap.nii.gz,
Segmentation > Import Label Descriptions = 13d_TopBrain_vs_Arterial_overlap_labels.txt.
"""

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk

VENOUS_LABELS = [35, 36, 37, 38, 39, 40]

LABELS = [
    # id, r, g, b, name
    (1, 255, 255, 0, "Both models"),
    (2, 0, 200, 0, "TopBrain artery only"),
    (3, 255, 0, 0, "arterial only"),
    (4, 0, 120, 255, "TopBrain vein"),
]


def resolve(case_dir: Path, name: str) -> Path:
    p = Path(name)
    return p if p.is_absolute() else case_dir / p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case-dir", required=True, type=Path)
    ap.add_argument("--arterial-mask", default="13b_Arterial_vessel_mask_ready.nii.gz")
    ap.add_argument("--labels", default="13_CTA_vessel_labels.nii.gz", help="TopBrain label map")
    ap.add_argument("--out", default="13d_TopBrain_vs_Arterial_overlap.nii.gz")
    args = ap.parse_args()

    tb_img = sitk.ReadImage(str(resolve(args.case_dir, args.labels)))
    ar_img = sitk.ReadImage(str(resolve(args.case_dir, args.arterial_mask)))
    for key in ("Size", "Spacing", "Origin", "Direction"):
        if not np.allclose(getattr(tb_img, "Get" + key)(), getattr(ar_img, "Get" + key)(), atol=1e-3):
            raise SystemExit(f"{key} differs between the TopBrain labels and the arterial mask; "
                             "run compare_arterial_topbrain.py first and use its *_on_ready mask.")

    tb = sitk.GetArrayFromImage(tb_img)
    ar = sitk.GetArrayFromImage(ar_img) > 0
    vein = np.isin(tb, VENOUS_LABELS)
    art = (tb > 0) & ~vein

    out = np.zeros(tb.shape, np.uint8)
    out[art & ar] = 1
    out[art & ~ar] = 2
    out[ar & (tb == 0)] = 3
    out[vein] = 4

    img = sitk.GetImageFromArray(out)
    img.CopyInformation(tb_img)
    out_path = resolve(args.case_dir, args.out)
    sitk.WriteImage(img, str(out_path), True)

    label_path = out_path.with_name(out_path.name.replace(".nii.gz", "") + "_labels.txt")
    lines = [
        "################################################",
        "# ITK-SnAP Label Description File",
        "# IDX   -R-  -G-  -B-  -A--  VIS MSH  LABEL",
        "################################################",
        '    0     0    0    0     0  0  0    "Clear Label"',
    ]
    lines += [f'{i:5d} {r:5d} {g:4d} {b:4d}     1  1  1    "{name}"' for i, r, g, b, name in LABELS]
    label_path.write_text("\n".join(lines) + "\n")

    vox_ml = float(np.prod(tb_img.GetSpacing())) / 1000
    for i, *_, name in LABELS:
        print(f"{i}  {name:22s} {(out == i).sum() * vox_ml:6.2f} ml")
    print(f"Wrote {out_path}\nWrote {label_path}")


if __name__ == "__main__":
    main()
