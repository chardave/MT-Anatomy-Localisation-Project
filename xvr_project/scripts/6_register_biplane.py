"""Register the AP and lateral snapshots to the CT, check left-right orientation and biplane consistency.

For every display convention (CT orientation AP/PA x DRR flipped/not flipped) the
script registers BOTH views with xvr, starting from the C-arm angles in the DICOM
files written by 5_fluoro_png_to_dicom.py, after a coarse search over the source
distance and head position (the screen snapshot's pixel size is only approximate).
Then it reports:

1. Left-right orientation: the convention whose registered DRRs best match the
   real images (mean similarity over both views). A wrong mirror cannot match an
   asymmetric skull as well. Each plane may be mirrored differently on screen, so
   the flip is chosen per view; the CT orientation (AP/PA) is shared.
2. Biplane consistency for each convention: the two registered C-arm poses must
   agree with the known geometry of a biplane system:
     - the angle between the two beams should equal the angle between the planes
       on the status panel (e.g. 0 deg vs RAO 90 deg -> 90 deg);
     - both central rays should pass (nearly) through one shared isocentre;
     - both sources should sit about the same distance from that isocentre.

Outputs (in --output-dir): one PNG per view and convention (X-ray | registered
DRR | overlay), report.txt and poses.json (final 4x4 poses for the best convention).

Example:
    python 6_register_biplane.py --ap ..\\input\\fluoro_43\\fluoro_AP_non_contrast_masked.dcm ^
        --lat ..\\input\\fluoro_43\\fluoro_LAT_non_contrast_masked.dcm ^
        --volume "...\\MTAL_KCH_43_test2\\7_Aligned_CTA.nii.gz" --output-dir ..\\outputs\\registration\\case_43
"""

import argparse
import json
import math
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from diffdrr.drr import DRR
from diffdrr.pose import convert, euler_angles_to_matrix, matrix_to_euler_angles
from pydicom import dcmread
from xvr.io import read_xray
from xvr.register import Register
from xvr.register.register import parse_scales
from xvr.utils import XrayTransforms


def central_ray(drr, pose):
    # ask DiffDRR where the source and detector pixels are in world (CT) coordinates;
    # its pose matrix includes a fixed camera reorientation, so don't read the axes off it directly
    with torch.no_grad():
        source, target = drr.detector(pose, None)
    source = source.reshape(-1, 3)[0].cpu().numpy().astype(float)
    centre = target.reshape(-1, 3).mean(0).cpu().numpy().astype(float)
    direction = centre - source
    return {"source": source.tolist(), "direction": (direction / np.linalg.norm(direction)).tolist()}


def biplane_geometry(ray_a, ray_b):
    """Angle between the beams, closest approach of the central rays and source distances."""
    sa, da = np.array(ray_a["source"]), np.array(ray_a["direction"])
    sb, db = np.array(ray_b["source"]), np.array(ray_b["direction"])
    angle = np.degrees(np.arccos(np.clip(abs(da @ db), -1, 1)))
    # closest points between the two (infinite) central rays
    w = sa - sb
    a, b, c, d, e = da @ da, da @ db, db @ db, da @ w, db @ w
    denom = a * c - b * b
    if denom < 1e-9:
        return {"beam_angle_deg": angle, "ray_gap_mm": float("nan"),
                "source_a_to_iso_mm": float("nan"), "source_b_to_iso_mm": float("nan")}
    t = (b * e - c * d) / denom
    s = (a * e - b * d) / denom
    pa, pb = sa + t * da, sb + s * db
    iso = (pa + pb) / 2
    return {
        "beam_angle_deg": float(angle),
        "ray_gap_mm": float(np.linalg.norm(pa - pb)),
        "isocentre_xyz_mm": iso.tolist(),
        "source_a_to_iso_mm": float(np.linalg.norm(sa - iso)),
        "source_b_to_iso_mm": float(np.linalg.norm(sb - iso)),
    }


class SearchedDicomPose:
    """xvr initializer: C-arm angles from the DICOM, then a coarse search over the source
    distance and in-plane shift before the fine registration starts.

    Image size on the detector depends on (pixel size x source distance), which a single
    screen snapshot cannot separate. Starting at the wrong size makes the gradient-based
    registration stall, so the best size/position is found by brute force first."""

    def __init__(self, orientation, reverse_x_axis, device, depths_mm, shifts_mm, scale=8.0):
        self.orientation, self.reverse_x_axis, self.device = orientation, reverse_x_axis, device
        self.depths_mm = [float(d) for d in depths_mm]
        self.shifts_mm = [float(s) for s in shifts_mm]
        self.scale = scale
        self.register = None  # set after the Register is built (needs its CT and similarity metric)
        self.last_search = None

    @torch.no_grad()
    def __call__(self, ctx):
        ds = dcmread(ctx.filename)
        alpha, beta = float(ds.PositionerPrimaryAngle), float(ds.PositionerSecondaryAngle)
        sign = -1 if self.orientation == "PA" else 1
        *_, height, width = ctx.img.shape
        drr = DRR(subject=self.register.subject, height=height, width=width, reverse_x_axis=self.reverse_x_axis,
                  renderer="trilinear", voxel_shift=0.0, **ctx.intrinsics).to(self.device)
        drr.rescale_detector_(1 / self.scale)
        transform = XrayTransforms(drr.detector.height, drr.detector.width)
        true = transform(ctx.img).to(self.device)

        shifts = [(dx, dz) for dx in self.shifts_mm for dz in self.shifts_mm]
        rot = torch.tensor([[alpha, beta, 0.0]] * len(shifts), device=self.device)
        best = (-np.inf, None, None)
        for depth in self.depths_mm:
            xyz = torch.tensor([[dx, sign * depth, dz] for dx, dz in shifts], dtype=torch.float32, device=self.device)
            poses = convert(rot, xyz, parameterization="euler_angles", convention="ZXY", degrees=True)
            pred = transform(drr(poses))
            scores = torch.stack([self.register.imagesim(true, p[None]) for p in pred])
            i = int(scores.argmax())
            if scores[i] > best[0]:
                best = (float(scores[i]), depth, shifts[i])
        score, depth, (dx, dz) = best
        self.last_search = {"similarity": score, "source_distance_mm": float(depth), "shift_mm": [dx, dz]}
        print(f"  start search: source distance {depth:.0f} mm, in-plane shift ({dx:+.0f}, {dz:+.0f}) mm, "
              f"similarity {score:.3f}")
        del drr
        return convert(torch.tensor([[alpha, beta, 0.0]], device=self.device),
                       torch.tensor([[dx, sign * depth, dz]], device=self.device),
                       parameterization="euler_angles", convention="ZXY", degrees=True)


class JointResult:
    """Just what save_figure needs, shaped like an xvr RegistrationResult."""

    def __init__(self, drr, final_pose, gt):
        self.drr, self.final_pose, self.gt = drr, final_pose, gt


def register_jointly(subject, views, orientation, flips, starts, separate_poses, scales, n_itrs, imagesim,
                     device, lr_rot=1e-2, lr_xyz=1.0, patience=5, max_n_plateaus=2):
    """Register both planes at once with ONE shared head rotation.

    Each camera pose = nominal C-arm pose from the panel angles (fixed) + its own source
    distance and in-plane shift (free: these absorb the uncertain pixel size and the screen
    crop), followed by a head rotation that is shared by both planes. Sharing the rotation
    keeps the two beams at the angle on the status panel, which a single view cannot enforce.

    Gradient-based fitting can stall in a poor compromise, so it is started several times:
    from the head rotation found by each separate registration, and from no rotation.
    The start with the best mean similarity wins.
    """
    sign = -1 if orientation == "PA" else 1
    data = {}
    for view, dcm in views.items():
        gt, intrinsics, _ = read_xray(str(dcm))
        ds = dcmread(dcm)
        angles = torch.tensor([[float(ds.PositionerPrimaryAngle), float(ds.PositionerSecondaryAngle), 0.0]],
                              device=device)
        *_, height, width = gt.shape
        drr = DRR(subject=subject, height=height, width=width, reverse_x_axis=flips[view],
                  renderer="trilinear", voxel_shift=0.0, **intrinsics).to(device)
        nominal = euler_angles_to_matrix(torch.deg2rad(angles), "ZXY")[0]
        data[view] = {"gt": gt, "drr": drr, "angles": angles, "nominal_rot": nominal}

    # ---- starting points: (head rotation, per-plane xyz) ----
    inits = []
    dx_dz = {v: starts[v]["shift_mm"] for v in views}
    inits.append(("no rotation", torch.eye(3, device=device),
                  {v: torch.tensor([dx_dz[v][0], sign * starts[v]["source_distance_mm"], dx_dz[v][1]],
                                   device=device) for v in views}))
    # Each separate registration implies a head rotation (pose rotation = R_head R_nominal) and,
    # in its own C-arm frame, a source distance and shift (pose translation = R_head R_nominal xyz).
    # Keep every plane's own xyz, so it stays aimed at the head, and try each plane's head rotation
    # as the shared one: this turns the whole C-arm about the head instead of swinging single beams.
    own = {}
    for v in views:
        m = torch.tensor(separate_poses[v], dtype=torch.float32, device=device)
        head_v = m[:3, :3] @ data[v]["nominal_rot"].T
        own[v] = (head_v, data[v]["nominal_rot"].T @ head_v.T @ m[:3, 3])
    for source_view in views:
        inits.append((f"{source_view} rotation", own[source_view][0], {v: own[v][1] for v in views}))

    factors = {view: parse_scales(scales + [1], 0, d["gt"].shape[2]) for view, d in data.items()}
    best = None
    for init_name, head0, xyz0 in inits:
        head_rot = torch.nn.Parameter(matrix_to_euler_angles(head0[None], "ZXY").detach())
        xyzs = {v: torch.nn.Parameter(xyz0[v][None].detach().clone()) for v in views}

        def poses():
            head = convert(head_rot, torch.zeros(1, 3, device=device), parameterization="euler_angles",
                           convention="ZXY")
            return {v: convert(data[v]["angles"], xyzs[v], parameterization="euler_angles", convention="ZXY",
                               degrees=True).compose(head) for v in views}

        sims = None
        for stage, (scale, n) in enumerate(zip(scales, n_itrs)):
            for view, d in data.items():
                d["drr"].rescale_detector_(factors[view][stage])
                d["transform"] = XrayTransforms(d["drr"].detector.height, d["drr"].detector.width)
                d["true"] = d["transform"](d["gt"]).to(device)
            step_scale = math.prod(2**i for i in range(stage + 1))
            optimizer = torch.optim.Adam([{"params": [head_rot], "lr": lr_rot / step_scale},
                                          {"params": list(xyzs.values()), "lr": lr_xyz / step_scale}],
                                         maximize=True)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", 0.1, patience, 1e-4,
                                                                   threshold_mode="rel")
            current_lr, n_plateaus = np.inf, 0
            for _ in range(n):
                optimizer.zero_grad()
                current = poses()
                per_view = [imagesim(d["true"], d["transform"](d["drr"](current[v]))) for v, d in data.items()]
                loss = sum(per_view) / len(per_view)
                loss.backward()
                optimizer.step()
                scheduler.step(loss.detach())
                sims = [float(s.detach()) for s in per_view]
                lr = min(scheduler.get_last_lr())
                if lr < current_lr:
                    current_lr, n_plateaus = lr, n_plateaus + 1
                if n_plateaus > max_n_plateaus:
                    break
        # back to native resolution after the last stage, ready for the next start
        for view, d in data.items():
            d["drr"].rescale_detector_(factors[view][-1])
        print(f"  joint, start from {init_name}: AP {sims[0]:.3f} | LAT {sims[1]:.3f}")
        if best is None or np.mean(sims) > np.mean(best["sims"]):
            with torch.no_grad():
                best = {"init": init_name, "sims": sims, "poses": poses(),
                        "head_rotation_deg": np.degrees(head_rot.detach().cpu().numpy()[0]).tolist()}

    return {view: {"similarity": best["sims"][i], "drr": d["drr"], "pose": best["poses"][view], "gt": d["gt"],
                   "head_rotation_deg": best["head_rotation_deg"], "start": best["init"]}
            for i, (view, d) in enumerate(data.items())}


def crop_to_head(volume_path, mask_path, margins_mm, out_path):
    """Crop the CT to the brain mask plus margins, so the volume centre (xvr's starting
    isocentre) sits in the head instead of the neck, and anatomy outside the field is removed."""
    vol, mask = nib.load(str(volume_path)), nib.load(str(mask_path))
    if vol.shape != mask.shape or not np.allclose(vol.affine, mask.affine, atol=1e-3):
        raise ValueError("--head-mask must be on exactly the same grid as --volume")
    idx = np.argwhere(np.asarray(mask.dataobj) > 0)
    if idx.size == 0:
        raise ValueError("--head-mask is empty")
    spacing = np.array(vol.header.get_zooms()[:3])
    axcodes = nib.aff2axcodes(vol.affine)
    lo, hi = idx.min(0).astype(float), idx.max(0).astype(float)
    # margins are given per anatomical direction; map them onto the voxel axes
    pad = {"L": margins_mm["side"], "R": margins_mm["side"], "A": margins_mm["front"], "P": margins_mm["back"],
           "S": margins_mm["top"], "I": margins_mm["bottom"]}
    opposite = {"L": "R", "R": "L", "A": "P", "P": "A", "S": "I", "I": "S"}
    for axis, code in enumerate(axcodes):
        # voxel index increases towards `code`, so the high end gets that direction's margin
        hi[axis] += pad[code] / spacing[axis]
        lo[axis] -= pad[opposite[code]] / spacing[axis]
    lo = np.clip(np.floor(lo), 0, None).astype(int)
    hi = np.minimum(np.ceil(hi).astype(int) + 1, vol.shape[:3])
    cropped = vol.slicer[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    nib.save(cropped, str(out_path))
    print(f"Cropped volume to head: {vol.shape[:3]} -> {cropped.shape[:3]} voxels, saved {out_path.name}")
    return out_path


def biplane_check_lines(row):
    angle_err = abs(row["beam_angle_deg"] - row["expected_beam_angle_deg"])
    sod_spread = abs(row["source_a_to_iso_mm"] - row["source_b_to_iso_mm"])
    return [f"  beam angle        {row['beam_angle_deg']:6.1f} deg, expected {row['expected_beam_angle_deg']:.1f} "
            f"({'ok' if angle_err <= 5 else 'CHECK'}, want error <= 5)",
            f"  central-ray gap   {row['ray_gap_mm']:6.1f} mm   ({'ok' if row['ray_gap_mm'] <= 15 else 'CHECK'}, want <= 15)",
            f"  source-iso spread {sod_spread:6.1f} mm   ({'ok' if sod_spread <= 50 else 'CHECK'}, want <= 50)"]


def normalise(img):
    img = np.asarray(img, dtype=float)
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)


def save_figure(result, title, path):
    with torch.no_grad():
        drr = result.drr(result.final_pose)
        # the detector is rescaled between stages, so match the X-ray size exactly before comparing
        drr = torch.nn.functional.interpolate(drr, size=result.gt.shape[-2:], mode="bilinear")
        drr = drr.squeeze().cpu().numpy()
    xray = result.gt.squeeze().cpu().numpy()
    xray_n, drr_n = normalise(xray), normalise(drr)
    overlay = np.stack([drr_n, xray_n, xray_n], axis=-1)  # red where only the DRR has bone
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, img, name in zip(axes, (xray_n, drr_n, overlay),
                             ("X-ray (log, as registered)", "Registered DRR", "Overlay: red = DRR, cyan = X-ray")):
        ax.imshow(img, cmap=None if img.ndim == 3 else "gray")
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ap", required=True, type=Path, help="AP / plane A DICOM from 5_fluoro_png_to_dicom.py")
    parser.add_argument("--lat", required=True, type=Path, help="Lateral / plane B DICOM")
    parser.add_argument("--volume", required=True, type=Path,
                        help="CT/CTA NIfTI of the SAME patient, covering the whole field of view")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--head-mask", type=Path,
                        help="Brain mask on the same grid as --volume (e.g. 8_Aligned_CTA_brain_mask). Recommended: "
                             "crops the CT to the head so registration starts centred on it, not on the neck")
    parser.add_argument("--margins-mm", nargs=5, type=float, default=[45, 60, 45, 35, 90],
                        metavar=("SIDE", "FRONT", "BACK", "TOP", "BOTTOM"),
                        help="How far beyond the brain mask to keep (skull, face, skull base)")
    parser.add_argument("--orientations", nargs="+", choices=["AP", "PA"], default=["AP", "PA"])
    parser.add_argument("--flips", nargs="+", choices=["no", "yes"], default=["no", "yes"],
                        help="Whether to horizontally flip the DRR (reverse_x_axis)")
    parser.add_argument("--depth-range-mm", nargs=3, type=float, default=[550, 1000, 25],
                        metavar=("MIN", "MAX", "STEP"),
                        help="Source-to-head distances tried before registration (absorbs pixel-size error)")
    parser.add_argument("--shift-range-mm", nargs=3, type=float, default=[-40, 40, 20],
                        metavar=("MIN", "MAX", "STEP"), help="In-plane head shifts tried before registration")
    parser.add_argument("--scales", nargs="+", type=float, default=[8.0, 4.0],
                        help="Downsampling per registration stage (coarse to fine)")
    parser.add_argument("--n-itrs", nargs="+", type=int, default=[500, 300])
    parser.add_argument("--no-joint", action="store_true",
                        help="Skip the final joint registration of both planes")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    for p in (args.ap, args.lat, args.volume):
        if not p.is_file():
            parser.error(f"Not found: {p}")
    if len(args.scales) != len(args.n_itrs):
        parser.error("--scales and --n-itrs need the same number of values")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.head_mask:
        side, front, back, top, bottom = args.margins_mm
        args.volume = crop_to_head(args.volume, args.head_mask,
                                   {"side": side, "front": front, "back": back, "top": top, "bottom": bottom},
                                   args.output_dir / "volume_cropped_to_head.nii.gz")
    views = {"AP": args.ap, "LAT": args.lat}
    panel = {k: (float(dcmread(v).PositionerPrimaryAngle), float(dcmread(v).PositionerSecondaryAngle))
             for k, v in views.items()}
    print(f"Device: {args.device} | volume: {args.volume.name}")

    # ---- 1. register both views under every display convention ----
    lo, hi, step = args.depth_range_mm
    depths = np.arange(lo, hi + step / 2, step)
    lo, hi, step = args.shift_range_mm
    shifts = np.arange(lo, hi + step / 2, step).tolist()
    runs, regs = {}, {}
    for orientation in args.orientations:
        reg = None
        for flip in args.flips:
            name = f"{orientation}_{'flipped' if flip == 'yes' else 'unflipped'}"
            initializer = SearchedDicomPose(orientation, flip == "yes", args.device, depths, shifts,
                                            scale=args.scales[0])
            if reg is None:
                # loading the CT is slow, so reuse it for both flip settings
                reg = Register(initializer=initializer, imagepath=str(args.volume), scales=args.scales,
                               n_itrs=args.n_itrs, patience=[5] * len(args.scales), device=args.device)
                regs[orientation] = reg
            reg.initializer = initializer
            initializer.register = reg
            runs[name] = {}
            for view, dcm in views.items():
                print(f"\n=== {name}: {view} ===")
                result = reg(str(dcm))
                if result.log is None:
                    print("Initial DRR was blank; skipping this convention.")
                    runs[name][view] = None
                    continue
                score = float(result.log.losses[-1])
                runs[name][view] = {"similarity": score, "start_search": initializer.last_search,
                                    "init_pose": result.init_pose.matrix.squeeze().cpu().tolist(),
                                    "final_pose": result.final_pose.matrix.squeeze().cpu().tolist(),
                                    "init_ray": central_ray(result.drr, result.init_pose),
                                    "final_ray": central_ray(result.drr, result.final_pose)}
                save_figure(result, f"{view} | {name} | similarity {score:.3f}",
                            args.output_dir / f"{view}_{name}.png")
                del result
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()

    # ---- 2. rank conventions and check biplane geometry ----
    # Each plane can be mirrored on screen independently, so the flip is chosen per view.
    # The CT orientation must be shared: poses from AP- and PA-read volumes live in different frames.
    rows = []
    for orientation in args.orientations:
        for flip_ap, flip_lat in product(args.flips, repeat=2):
            ap = runs.get(f"{orientation}_{'flipped' if flip_ap == 'yes' else 'unflipped'}", {}).get("AP")
            lat = runs.get(f"{orientation}_{'flipped' if flip_lat == 'yes' else 'unflipped'}", {}).get("LAT")
            if not ap or not lat:
                continue
            expected = biplane_geometry(ap["init_ray"], lat["init_ray"])["beam_angle_deg"]
            geo = biplane_geometry(ap["final_ray"], lat["final_ray"])
            rows.append({"convention": f"{orientation} A:{'flip' if flip_ap == 'yes' else 'none'} "
                                       f"B:{'flip' if flip_lat == 'yes' else 'none'}",
                         "orientation": orientation, "flip_AP": flip_ap == "yes", "flip_LAT": flip_lat == "yes",
                         "AP_similarity": ap["similarity"], "LAT_similarity": lat["similarity"],
                         "mean_similarity": (ap["similarity"] + lat["similarity"]) / 2,
                         "expected_beam_angle_deg": expected, **geo})
    if not rows:
        raise RuntimeError("No convention produced a usable registration for both views")
    # ---- 3. joint registration: both planes, one shared head rotation ----
    # A nearly symmetric skull lets a MIRRORED AP fit well on its own, so separate scores can pick
    # the wrong flip. Forcing one head pose to explain both planes exposes this, so when the joint
    # step runs it decides the convention.
    joints = {}
    if not args.no_joint:
        for r in rows:
            print(f"\n=== Joint biplane registration ({r['convention']}) ===")
            reg = regs[r["orientation"]]
            names = {"AP": f"{r['orientation']}_{'flipped' if r['flip_AP'] else 'unflipped'}",
                     "LAT": f"{r['orientation']}_{'flipped' if r['flip_LAT'] else 'unflipped'}"}
            joint = register_jointly(reg.subject, views, r["orientation"],
                                     {"AP": r["flip_AP"], "LAT": r["flip_LAT"]},
                                     {v: runs[names[v]][v]["start_search"] for v in views},
                                     {v: runs[names[v]][v]["final_pose"] for v in views},
                                     args.scales, args.n_itrs, reg.imagesim, args.device)
            rays = {v: central_ray(joint[v]["drr"], joint[v]["pose"]) for v in views}
            joint_geo = biplane_geometry(rays["AP"], rays["LAT"])
            r.update({"joint_AP_similarity": joint["AP"]["similarity"],
                      "joint_LAT_similarity": joint["LAT"]["similarity"],
                      "joint_mean_similarity": (joint["AP"]["similarity"] + joint["LAT"]["similarity"]) / 2,
                      "joint_geometry": joint_geo})
            # keep only the best joint result in memory (each holds DRRs on the GPU)
            if not joints or r["joint_mean_similarity"] > max(x["joint_mean_similarity"] for x in rows
                                                              if "joint_mean_similarity" in x and x is not r):
                joints = {r["convention"]: (joint, rays)}
            del joint
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    key = "joint_mean_similarity" if joints else "mean_similarity"
    rows.sort(key=lambda r: r[key], reverse=True)
    best = rows[0]

    lines = [f"Volume: {args.volume}", f"AP:  {args.ap}  (primary/secondary {panel['AP']})",
             f"LAT: {args.lat}  (primary/secondary {panel['LAT']})", "",
             "Similarity: higher is better (xvr gradient-NCC at the finest registration scale).",
             "A/B = whether the plane A (AP) / plane B (LAT) DRR is mirrored (reverse_x_axis).",
             "'separate' = each view registered alone; 'joint' = one head pose for both planes.", "",
             f"{'convention':<20}{'separate AP':>12}{'LAT':>7}{'mean':>7}{'beam angle':>12}"
             + (f"{'joint AP':>10}{'LAT':>7}{'mean':>7}" if joints else "")]
    for r in rows:
        lines.append(f"{r['convention']:<20}{r['AP_similarity']:>12.3f}{r['LAT_similarity']:>7.3f}"
                     f"{r['mean_similarity']:>7.3f}{r['beam_angle_deg']:>11.1f}°"
                     + (f"{r['joint_AP_similarity']:>10.3f}{r['joint_LAT_similarity']:>7.3f}"
                        f"{r['joint_mean_similarity']:>7.3f}" if joints else ""))
    lines += ["", f"Best convention: {best['convention']} (ranked by {'joint' if joints else 'separate'} similarity)"]
    if len(rows) > 1 and best[key] - rows[1][key] < 0.05:
        lines.append(f"WARNING: runner-up {rows[1]['convention']} is within 0.05 - decide from the PNGs.")
    lines += ["", "Biplane checks, separate registrations:"] + biplane_check_lines(best)

    best_runs = {"AP": f"{best['orientation']}_{'flipped' if best['flip_AP'] else 'unflipped'}",
                 "LAT": f"{best['orientation']}_{'flipped' if best['flip_LAT'] else 'unflipped'}"}
    out = {"note": "DiffDRR camera-to-world 4x4 poses; world = the volume as read by xvr (centred on the volume centre)",
           "volume": str(args.volume), "best_convention": best["convention"], "orientation": best["orientation"],
           "reverse_x_axis": {"AP": best["flip_AP"], "LAT": best["flip_LAT"]},
           "views": {v: runs[best_runs[v]][v] for v in views}, "ranking": rows}
    if joints:
        joint, rays = joints[best["convention"]]
        for v in views:
            save_figure(JointResult(joint[v]["drr"], joint[v]["pose"], joint[v]["gt"]),
                        f"{v} | joint {best['convention']} | similarity {joint[v]['similarity']:.3f}",
                        args.output_dir / f"{v}_joint.png")
        lines += ["", "Joint registration, best convention (use these poses):",
                  f"  similarity AP {joint['AP']['similarity']:.3f} | LAT {joint['LAT']['similarity']:.3f}",
                  f"  best start: {joint['AP']['start']} | head rotation "
                  f"{np.round(joint['AP']['head_rotation_deg'], 1).tolist()} deg (ZXY)"]
        lines += biplane_check_lines({**best["joint_geometry"],
                                      "expected_beam_angle_deg": best["expected_beam_angle_deg"]})
        lines.append("  Check AP_joint.png and LAT_joint.png: both overlays should line up.")
        out["joint"] = {v: {"similarity": joint[v]["similarity"], "head_rotation_deg": joint[v]["head_rotation_deg"],
                            "pose": joint[v]["pose"].matrix.squeeze().cpu().tolist(), "ray": rays[v]} for v in views}
        out["joint_geometry"] = best["joint_geometry"]
    report = "\n".join(lines)
    print("\n" + report)
    (args.output_dir / "report.txt").write_text(report + "\n", encoding="utf-8")
    (args.output_dir / "poses.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved report.txt, poses.json and per-view PNGs in {args.output_dir}")


if __name__ == "__main__":
    main()
