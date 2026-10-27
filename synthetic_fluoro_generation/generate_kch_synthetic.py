import argparse
import importlib.metadata
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from PIL import Image

from diffdrr.data import read
from diffdrr.drr import DRR


def check_inputs(paths):
    images = [nib.load(str(path)) for path in paths]
    reference = images[0]

    for path, image in zip(paths[1:], images[1:]):
        if image.shape != reference.shape:
            raise ValueError(f"Different voxel dimensions: {path}")
        if not np.allclose(image.affine, reference.affine, atol=1e-4):
            raise ValueError(f"Different spatial grid: {path}")

    for path, image in zip(paths[:2], images[:2]):
        values = image.get_fdata(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite values: {path}")
        print(
            path.name,
            "shape:", image.shape,
            "range:", float(values.min()), float(values.max()),
        )

    labels = images[2].get_fdata(dtype=np.float32)
    if not np.isfinite(labels).all():
        raise ValueError("Labels contain nonfinite values")
    if labels.min() < 0 or not np.allclose(labels, np.round(labels)):
        raise ValueError("Labels must be nonnegative integer IDs")

    ids = np.unique(labels.astype(np.int32)).tolist()
    print("Label IDs:", ids)
    return ids


def preview(array, path):
    """Display-only normalization; raw training arrays are saved separately."""
    low, high = np.percentile(array, [1, 99])
    scaled = np.clip((array - low) / max(high - low, 1e-8), 0, 1)
    Image.fromarray(np.round(255 * scaled).astype(np.uint8)).save(path)


def make_renderer(subject, args, device):
    return DRR(
        subject,
        sdd=args.sdd,
        height=args.size,
        width=args.size,
        delx=args.pixel_size,
        dely=args.pixel_size,
        renderer="trilinear",
        patch_size=args.patch_size,
    ).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--views", type=int, default=10,
                        help="Number of AP/lateral pairs")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--pixel-size", type=float, default=1.2,
                        help="Detector pixel spacing in mm")
    parser.add_argument("--sdd", type=float, default=1020,
                        help="Source-to-detector distance in mm")
    parser.add_argument("--distance", type=float, default=850,
                        help="DiffDRR y translation in mm")
    parser.add_argument("--angle-jitter", type=float, default=10)
    parser.add_argument("--translation-jitter", type=float, default=10)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--patch-size", type=int, default=64,
                    help="Detector patch side length in pixels")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.views < 1 or args.size < 1 or args.samples < 2:
        raise ValueError("Invalid views, size or samples")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Choose an empty output folder")

    paths = [
        args.case_dir / "9_ready_CT.nii.gz",
        args.case_dir / "10_ready_CTA.nii.gz",
        args.case_dir / "13_CTA_vessel_labels.nii.gz",
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    label_ids = check_inputs(paths)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Identical input grids give identical centring for these two subjects.
    ct_subject = read(str(paths[0]), orientation="AP")
    cta_subject = read(
        str(paths[1]),
        labelmap=str(paths[2]),
        orientation="AP",
    )

    ct_renderer = make_renderer(ct_subject, args, device)
    cta_renderer = make_renderer(cta_subject, args, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    configuration = vars(args).copy()
    configuration["case_dir"] = str(args.case_dir)
    configuration["output_dir"] = str(args.output_dir)
    configuration["device"] = str(device)
    configuration["label_ids"] = label_ids
    configuration["versions"] = {
        package: importlib.metadata.version(package)
        for package in ["diffdrr", "torch", "nibabel", "numpy"]
    }
    configuration["angle_convention"] = "DiffDRR intrinsic ZXY, degrees"
    configuration["input_affine"] = nib.load(str(paths[0])).affine.tolist()

    (args.output_dir / "configuration.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )

    with (
        torch.inference_mode(),
        (args.output_dir / "manifest.jsonl").open("w", encoding="utf-8") as manifest,
    ):
        for pair_index in range(args.views):
            # Pair zero is the baseline; later pairs have random perturbations.
            if pair_index == 0:
                jitter = np.zeros(3)
                offset = np.zeros(3)
            else:
                jitter = rng.uniform(
                    -args.angle_jitter, args.angle_jitter, 3
                )
                offset = rng.uniform(
                    -args.translation_jitter,
                    args.translation_jitter, 3
                )

            translation = np.array([0.0, args.distance, 0.0]) + offset

            for plane, baseline in [
                ("A", np.array([0.0, 0.0, 0.0])),
                ("B", np.array([0.0, 0.0, 90.0])),
            ]:
                rotation = baseline + jitter
                rotations = torch.tensor(
                    rotation[None], dtype=torch.float32, device=device
                )
                translations = torch.tensor(
                    translation[None], dtype=torch.float32, device=device
                )
                options = dict(
                    parameterization="euler_angles",
                    convention="ZXY",
                    degrees=True,
                    n_points=args.samples,
                )

                ct = ct_renderer(rotations, translations, **options)
                channels = cta_renderer(
                    rotations, translations,
                    mask_to_channels=True, **options
                )

                ct = ct[0, 0].cpu().numpy()
                channels = channels[0].cpu().numpy()

                # Channel index corresponds to the original integer label ID.
                # Channel 0 contains contributions outside labelled vessels.
                cta = channels.sum(axis=0)
                present_ids = [value for value in label_ids if value != 0]
                vessel_channels = channels[present_ids]

                if (
                    not np.isfinite(ct).all()
                    or not np.isfinite(channels).all()
                    or np.ptp(ct) <= 0
                    or np.ptp(cta) <= 0
                ):
                    raise RuntimeError("Empty or invalid projection")

                name = f"pair_{pair_index:05d}_{plane}"
                np.savez_compressed(
                    args.output_dir / f"{name}.npz",
                    ct=ct.astype(np.float32),
                    cta=cta.astype(np.float32),
                    vessel_channels=vessel_channels.astype(np.float32),
                    label_ids=np.array(present_ids, dtype=np.int32),
                    rotation_deg=rotation.astype(np.float32),
                    translation_mm=translation.astype(np.float32),
                )

                preview(ct, args.output_dir / f"{name}_ct.png")
                preview(cta, args.output_dir / f"{name}_cta.png")
                preview(
                    vessel_channels.sum(axis=0),
                    args.output_dir / f"{name}_vessels.png",
                )

                record = {
                    "case_id": args.case_dir.name,
                    "pair_id": pair_index,
                    "plane": plane,
                    "file": f"{name}.npz",
                    "rotation_deg": rotation.tolist(),
                    "translation_mm": translation.tolist(),
                    "sdd_mm": args.sdd,
                    "pixel_spacing_mm": args.pixel_size,
                    "height": args.size,
                    "width": args.size,
                }
                manifest.write(json.dumps(record) + "\n")
                manifest.flush()
                print("Saved:", name)


if __name__ == "__main__":
    main()