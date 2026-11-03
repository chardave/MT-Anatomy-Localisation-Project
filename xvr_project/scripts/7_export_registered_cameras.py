"""Export the registered C-arm cameras from 6_register_biplane.py in scanner coordinates.

Run in the xvr environment. 8_cameras_to_nanodrr.py (fluoro environment) then turns
these into the rotation / translation / SDD / pixel size to type into the nanoDRR
simulator. The two steps are separate because DiffDRR (xvr) and nanoDRR live in
different environments.

For each view this saves, in the CT's scanner (RAS) world coordinates:
  - the X-ray source position,
  - a cloud of 3D points around the head and the pixel where xvr's camera sees each one
    (this captures every convention: axes, image origin and mirroring),
  - the detector size, pixel size and SDD,
plus the registered DRR and the X-ray for checking.

Example:
    python 7_export_registered_cameras.py --poses ..\\outputs\\registration\\case_43\\poses.json ^
        --ap ..\\input\\fluoro_43\\fluoro_AP_non_contrast_masked.dcm ^
        --lat ..\\input\\fluoro_43\\fluoro_LAT_non_contrast_masked.dcm
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform
from torchio import ScalarImage
from xvr.io import read_xray


def resolve(path_text, poses_path):
    # poses.json stores the volume path as given on the command line (relative to where 6_ was run)
    path = Path(path_text)
    for candidate in (path, Path(__file__).parent / path, poses_path.parent / path.name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot find the registration volume {path_text}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--poses", required=True, type=Path, help="poses.json from 6_register_biplane.py")
    parser.add_argument("--ap", required=True, type=Path, help="The same AP DICOM given to 6_register_biplane.py")
    parser.add_argument("--lat", required=True, type=Path, help="The same lateral DICOM")
    parser.add_argument("--use", choices=["joint", "separate"], default="joint",
                        help="Which poses to export (default: joint, if present)")
    parser.add_argument("--output", type=Path, help="Default: cameras.json next to poses.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data = json.loads(args.poses.read_text(encoding="utf-8"))
    volume = resolve(data["volume"], args.poses)
    orientation = data["orientation"]
    flips = data["reverse_x_axis"]
    use = args.use if args.use == "separate" or "joint" in data else "separate"
    poses = {v: (data["joint"][v]["pose"] if use == "joint" else data["views"][v]["final_pose"]) for v in flips}
    output = args.output or args.poses.with_name("cameras.json")

    # DiffDRR works in the scanner frame shifted so the volume centre is the origin
    centre = np.array(ScalarImage(volume).get_center(), dtype=float)
    subject = read(str(volume), orientation=orientation)
    rng = np.random.default_rng(0)
    points_local = rng.uniform(-70, 70, size=(400, 3))  # around the head centre, mm

    cameras, images = {}, {}
    for view, dcm in {"AP": args.ap, "LAT": args.lat}.items():
        gt, intrinsics, _ = read_xray(str(dcm))
        *_, height, width = gt.shape
        drr = DRR(subject=subject, height=height, width=width, reverse_x_axis=flips[view],
                  renderer="trilinear", voxel_shift=0.0, **intrinsics).to(args.device)
        pose = RigidTransform(torch.tensor(poses[view], dtype=torch.float32)).to(args.device)
        with torch.no_grad():
            source, _ = drr.detector(pose, None)
            pts = torch.tensor(points_local, dtype=torch.float32, device=args.device)[None]
            pixels = drr.perspective_projection(pose, pts)[0].cpu().numpy()
            image = drr(pose).squeeze().cpu().numpy()
        source = source.reshape(-1, 3)[0].cpu().numpy().astype(float)
        cameras[view] = {
            "source_ras": (source + centre).tolist(),
            "points_ras": (points_local + centre).tolist(),
            "pixels_uv": pixels.tolist(),
            "height": int(height), "width": int(width),
            "sdd_mm": float(intrinsics.sdd), "pixel_mm": float(intrinsics.delx),
        }
        images[f"{view}_drr"] = image
        images[f"{view}_xray"] = gt.squeeze().cpu().numpy()
        print(f"{view}: source {np.round(source + centre, 1)} (RAS mm), {width} x {height} px, "
              f"SDD {float(intrinsics.sdd):.0f} mm, pixel {float(intrinsics.delx):.4f} mm")
        del drr

    output.write_text(json.dumps({"from": str(args.poses), "poses_used": use, "volume": str(volume),
                                  "cameras": cameras}, indent=2), encoding="utf-8")
    np.savez_compressed(output.with_suffix(".npz"), **images)
    print(f"Saved {output} and {output.with_suffix('.npz').name}")


if __name__ == "__main__":
    main()
