"""Turn the cameras exported by 7_export_registered_cameras.py into nanoDRR simulator settings.

Run in the fluoro (nanoDRR) environment. For each view it fits a nanoDRR camera to the
registered geometry, converts it with nanoDRR's own invert_rt_inv into the numbers the
simulator (8_1_fluoroscopy_sim_nanodrr.py) uses, renders that camera with nanoDRR and
compares it with the real X-ray, so you can see the settings reproduce the registration.

The translation depends on which CT you load in the simulator (it is measured from that
volume's centre), so pass the same --ct you will open there. The rotation does not.

Example:
    python 8_cameras_to_nanodrr.py --cameras ..\\outputs\\registration\\case_43\\cameras.json ^
        --ct "..\\..\\synth_fluoro_gen_test\\MTAL_KCH_43_test2\\9_ready_CT.nii.gz"
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from nanodrr.camera import invert_rt_inv, make_k_inv, make_rt_inv
from nanodrr.data import Subject
from nanodrr.drr import render


def fit_rotation(camera, mirror):
    """Rotation whose camera sees every exported 3D point at its exported pixel (Kabsch)."""
    s = np.array(camera["source_ras"])
    pts = np.array(camera["points_ras"])
    uv = np.array(camera["pixels_uv"], dtype=float)
    if mirror:
        uv[:, 0] = camera["width"] - uv[:, 0]
    f = camera["sdd_mm"] / camera["pixel_mm"]
    # nanoDRR camera: x = image column, y = image row (down), z = towards the detector
    rays_cam = np.stack([(uv[:, 0] - camera["width"] / 2) / f, (uv[:, 1] - camera["height"] / 2) / f,
                         np.ones(len(uv))], axis=1)
    rays_cam /= np.linalg.norm(rays_cam, axis=1, keepdims=True)
    rays_world = pts - s
    rays_world /= np.linalg.norm(rays_world, axis=1, keepdims=True)
    u, _, vt = np.linalg.svd(rays_cam.T @ rays_world)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1, 1, d]) @ u.T
    # reprojection check, in pixels
    cam = (rot.T @ (pts - s).T).T
    reproj = np.stack([f * cam[:, 0] / cam[:, 2] + camera["width"] / 2,
                       f * cam[:, 1] / cam[:, 2] + camera["height"] / 2], axis=1)
    return rot, d > 0, float(np.abs(reproj - uv).max())


def ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def normalise(img):
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cameras", required=True, type=Path, help="cameras.json from 7_export_registered_cameras.py")
    parser.add_argument("--ct", required=True, type=Path,
                        help="The CT you will load in the nanoDRR simulator (same scanner space as the registration)")
    parser.add_argument("--output-dir", type=Path, help="Default: folder of --cameras")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data = json.loads(args.cameras.read_text(encoding="utf-8"))
    images = np.load(args.cameras.with_suffix(".npz"))
    out_dir = args.output_dir or args.cameras.parent
    subject = Subject.from_filepath(str(args.ct)).to(args.device)
    iso = subject.isocenter

    settings, lines = {}, [f"Simulator CT: {args.ct}", f"Its centre (isocentre) in RAS mm: "
                                                       f"{np.round(iso.cpu().numpy(), 2).tolist()}", ""]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for row, (view, camera) in enumerate(data["cameras"].items()):
        # ---- 1. fit a proper (unmirrored) camera; mirror the image instead if the view needs it ----
        rot, proper, err = fit_rotation(camera, mirror=False)
        flip = not proper
        if flip:
            rot, proper, err = fit_rotation(camera, mirror=True)
        rt_inv = torch.eye(4, dtype=torch.float64)
        rt_inv[:3, :3] = torch.from_numpy(rot)
        rt_inv[:3, 3] = torch.tensor(camera["source_ras"], dtype=torch.float64)

        # ---- 2. nanoDRR's own inverse gives the simulator's rotation / translation ----
        rotation, translation = invert_rt_inv(rt_inv[None].float().to(args.device), orientation="AP", isocenter=iso)
        rotation, translation = rotation[0].cpu().numpy(), translation[0].cpu().numpy()

        # ---- 3. render exactly what the simulator would, and compare with the X-ray ----
        h, w, sdd, pix = camera["height"], camera["width"], camera["sdd_mm"], camera["pixel_mm"]
        k_inv = make_k_inv(sdd, pix, pix, 0.0, 0.0, h, w, device=args.device)
        rt = make_rt_inv(torch.tensor([rotation], dtype=torch.float32, device=args.device),
                         torch.tensor([translation], dtype=torch.float32, device=args.device),
                         orientation="AP", isocenter=iso)
        with torch.no_grad():
            line_integral = render(subject, k_inv, rt, torch.tensor([sdd], device=args.device), h, w)
        nano = line_integral.sum(dim=1)[0].cpu().numpy()
        if flip:
            nano = nano[:, ::-1]
        xray, xvr_drr = images[f"{view}_xray"], images[f"{view}_drr"]
        score_xray, score_xvr = ncc(nano, xray), ncc(nano, xvr_drr)

        settings[view] = {"rot_x": float(rotation[0]), "rot_y": float(rotation[1]), "rot_z": float(rotation[2]),
                          "trans_x": float(translation[0]), "trans_y": float(translation[1]),
                          "trans_z": float(translation[2]), "sdd": sdd, "pixel_size": pix,
                          "height": h, "width": w, "flip_horizontal": flip,
                          "check": {"reprojection_error_px": err, "ncc_vs_xray": score_xray,
                                    "ncc_vs_xvr_drr": score_xvr}}
        lines += [f"{view}:",
                  f"  Rotation    rot_x {rotation[0]:9.3f}   rot_y {rotation[1]:9.3f}   rot_z {rotation[2]:9.3f}  (deg)",
                  f"  Translation trans_x {translation[0]:8.2f}   trans_y {translation[1]:8.2f}   "
                  f"trans_z {translation[2]:8.2f}  (mm)",
                  f"  SDD {sdd:.1f} mm | pixel size {pix:.4f} mm | height {h} | width {w}",
                  f"  Flip image left-right: {'YES' if flip else 'no'}",
                  f"  Check: camera fit error {err:.2f} px | nanoDRR vs xvr DRR NCC {score_xvr:.3f} | "
                  f"nanoDRR vs X-ray NCC {score_xray:.3f}", ""]
        overlay = np.stack([normalise(nano), normalise(xray), normalise(xray)], axis=-1)
        for ax, img, name in zip(axes[row], (normalise(xray), normalise(nano), overlay),
                                 (f"{view} X-ray", f"{view} nanoDRR with these settings", "red = nanoDRR, cyan = X-ray")):
            ax.imshow(img, cmap=None if img.ndim == 3 else "gray")
            ax.set_title(name)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "nanodrr_check.png", dpi=110)
    plt.close(fig)

    report = "\n".join(lines)
    print(report)
    (out_dir / "nanodrr_settings.txt").write_text(report, encoding="utf-8")
    (out_dir / "nanodrr_settings.json").write_text(json.dumps({"ct": str(args.ct), "views": settings}, indent=2),
                                                   encoding="utf-8")
    print(f"Saved nanodrr_settings.txt/.json and nanodrr_check.png in {out_dir}")


if __name__ == "__main__":
    main()
