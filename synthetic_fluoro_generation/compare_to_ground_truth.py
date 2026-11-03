#!/usr/bin/env python3
"""Compare a real fluoroscopy frame (ground truth) with one or more synthesised images.

Typical use: the real snapshot from extract_fluoroscopy.py and a PNG saved from the nanoDRR
simulator (8_1_fluoroscopy_sim_nanodrr.py, "Save image") for the same registered view.

For each synthesised image it reports, inside the region that is not masked:
  SSIM fine     local structure at full detail (1 = identical)
  SSIM coarse   structure after blurring: overall shapes, shading (1 = identical)
  NCC           correlation of intensities (1 = identical, 0 = unrelated)
  Edge NCC      correlation of edge strength: are bones/vessels in the same place?
  Hist diff     mean grey-level difference after sorting: overall brightness/contrast
                distribution, ignoring position (0 = identical, in 0-1 grey units)
  MAD           mean absolute pixel difference (0 = identical, in 0-1 grey units)
and saves a figure: real | synthetic | colour overlay | difference | checkerboard | edges.
It also warns if the synthetic image matches better mirrored or inverted.

Requirements: numpy, scipy, Pillow, matplotlib (fluoro environment)
Example:
    python compare_to_ground_truth.py ^
        --real ..\\xvr_project\\input\\fluoro_43\\fluoro_AP_non_contrast_masked.png ^
        --mask ..\\xvr_project\\input\\fluoro_43\\fluoro_AP_non_contrast_mask.png ^
        --synth ..\\xvr_project\\outputs\\registration\\case_43\\nanodrr_fluoroscopy_AP_registered.png ^
                ..\\xvr_project\\outputs\\registration\\case_43\\nanodrr_fluoroscopy_AP_registered_updated.png
Add --interactive to open a window with a slider that fades between the two images.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel


def load_grey(path, size=None):
    img = Image.open(path).convert("L")
    if size is not None and img.size != size:
        img = img.resize(size, Image.BILINEAR)
    return np.asarray(img, dtype=np.float64) / 255.0


def ssim_map(a, b, sigma):
    mu_a, mu_b = gaussian_filter(a, sigma), gaussian_filter(b, sigma)
    var_a = gaussian_filter(a * a, sigma) - mu_a ** 2
    var_b = gaussian_filter(b * b, sigma) - mu_b ** 2
    cov = gaussian_filter(a * b, sigma) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))


def ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def edges(img):
    smooth = gaussian_filter(img, 1.0)
    return np.hypot(sobel(smooth, 0), sobel(smooth, 1))


def metrics(real, synth, valid):
    return {
        "ssim_fine": float(ssim_map(real, synth, 2.0)[valid].mean()),
        "ssim_coarse": float(ssim_map(gaussian_filter(real, 3), gaussian_filter(synth, 3), 8.0)[valid].mean()),
        "ncc": ncc(real[valid], synth[valid]),
        "edge_ncc": ncc(edges(real)[valid], edges(synth)[valid]),
        "hist_diff": float(np.abs(np.sort(real[valid]) - np.sort(synth[valid])).mean()),
        "mad": float(np.abs(real - synth)[valid].mean()),
    }


def checkerboard(a, b, tiles=8):
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    use_a = ((yy * tiles // h) + (xx * tiles // w)) % 2 == 0
    return np.where(use_a, a, b)


def edge_overlay(real, synth):
    er, es = edges(real), edges(synth)
    er, es = er / (np.percentile(er, 99) + 1e-9), es / (np.percentile(es, 99) + 1e-9)
    rgb = np.stack([np.clip(es, 0, 1), np.clip(er, 0, 1), np.clip(er, 0, 1)], axis=-1)
    return rgb  # red = synthetic edges, cyan = real edges, white = both


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", required=True, type=Path, help="Ground-truth frame (ideally the masked PNG)")
    parser.add_argument("--synth", required=True, type=Path, nargs="+", help="One or more synthesised PNGs")
    parser.add_argument("--mask", type=Path,
                        help="Overlay mask from 4_mask_fluoro_overlays.py (white = covered text, ignored in scores)")
    parser.add_argument("--output", type=Path, help="Figure path (default: <first synth>_vs_ground_truth.png)")
    parser.add_argument("--interactive", action="store_true", help="Open a window with a fade slider")
    args = parser.parse_args()

    for p in [args.real, *args.synth] + ([args.mask] if args.mask else []):
        if not p.is_file():
            parser.error(f"Not found: {p}")
    real = load_grey(args.real)
    h, w = real.shape
    valid = np.ones_like(real, dtype=bool)
    if args.mask:
        valid = load_grey(args.mask, (w, h)) < 0.5
    # ignore a thin border: screen captures often have a frame line at the edge
    valid[:3, :] = valid[-3:, :] = valid[:, :3] = valid[:, -3:] = False

    rows, report = [], {}
    print(f"Ground truth: {args.real.name} ({w} x {h}), scoring {100 * valid.mean():.0f}% of pixels\n")
    print(f"{'image':<45}{'SSIM fine':>10}{'coarse':>8}{'NCC':>7}{'Edge NCC':>10}{'Hist diff':>11}{'MAD':>7}")
    for path in args.synth:
        raw_size = Image.open(path).size
        synth = load_grey(path, (w, h))
        if abs(raw_size[0] / raw_size[1] - w / h) > 0.02:
            print(f"WARNING: {path.name} is {raw_size[0]} x {raw_size[1]}, a different shape from the ground truth; "
                  "it was stretched to fit, so scores are unreliable. Match height/width in the simulator.")
        m = metrics(real, synth, valid)
        report[path.name] = m
        print(f"{path.name[:44]:<45}{m['ssim_fine']:>10.3f}{m['ssim_coarse']:>8.3f}{m['ncc']:>7.3f}"
              f"{m['edge_ncc']:>10.3f}{m['hist_diff']:>11.3f}{m['mad']:>7.3f}")
        # catch a forgotten flip or inverted display
        mirrored = ncc(edges(real)[valid], edges(synth[:, ::-1])[valid])
        if mirrored > m["edge_ncc"] + 0.05:
            print(f"   WARNING: edges match better mirrored left-right ({mirrored:.3f}); check 'Flip image left-right'.")
        if m["ncc"] < -0.2:
            print("   WARNING: intensities are anti-correlated; the synthetic image looks inverted (bone bright?).")
        rows.append((path.name, synth, m))

    # ---- figure: one row per synthetic image ----
    import matplotlib
    if not args.interactive:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = ["Ground truth", "Synthetic", "Overlay (red = synth, cyan = real)",
            "|Difference|", "Checkerboard", "Edges (red = synth, cyan = real)"]
    fig, axes = plt.subplots(len(rows), 6, figsize=(24, 4.4 * len(rows)), squeeze=False)
    for r, (name, synth, m) in enumerate(rows):
        panels = [real, synth, np.stack([synth, real, real], -1), np.abs(real - synth) * valid,
                  checkerboard(real, synth), edge_overlay(real, synth)]
        for c, img in enumerate(panels):
            ax = axes[r, c]
            if c == 3:
                ax.imshow(img, cmap="magma", vmin=0, vmax=0.5)
            else:
                ax.imshow(img, cmap=None if img.ndim == 3 else "gray", vmin=0, vmax=1)
            ax.set_title(cols[c] if r == 0 else "", fontsize=10)
            ax.axis("off")
        axes[r, 1].set_title(f"{name}\nSSIM {m['ssim_fine']:.3f}/{m['ssim_coarse']:.3f} | NCC {m['ncc']:.3f} | "
                             f"edge NCC {m['edge_ncc']:.3f} | hist {m['hist_diff']:.3f}", fontsize=9)
    fig.tight_layout()
    output = args.output or args.synth[0].with_name(args.synth[0].stem + "_vs_ground_truth.png")
    fig.savefig(output, dpi=100)
    output.with_suffix(".json").write_text(json.dumps({"ground_truth": str(args.real), "metrics": report},
                                                      indent=2), encoding="utf-8")
    print(f"\nSaved {output} and {output.with_suffix('.json').name}")

    # ---- optional fade viewer ----
    if args.interactive:
        from matplotlib.widgets import RadioButtons, Slider
        plt.close(fig)
        fig2, ax = plt.subplots(figsize=(8, 8.8))
        plt.subplots_adjust(bottom=0.16, left=0.22)
        state = {"synth": rows[0][1]}
        shown = ax.imshow(real, cmap="gray", vmin=0, vmax=1)
        ax.set_title("0 = ground truth, 1 = synthetic")
        ax.axis("off")
        slider = Slider(fig2.add_axes([0.25, 0.06, 0.6, 0.03]), "fade", 0.0, 1.0, valinit=0.0)

        def update(_=None):
            a = slider.val
            shown.set_data((1 - a) * real + a * state["synth"])
            fig2.canvas.draw_idle()

        slider.on_changed(update)
        if len(rows) > 1:
            radio = RadioButtons(fig2.add_axes([0.01, 0.4, 0.19, 0.04 * len(rows)]), [n[:22] for n, _, _ in rows])

            def choose(label):
                state["synth"] = next(s for n, s, _ in rows if n[:22] == label)
                update()

            radio.on_clicked(choose)
        plt.show()


if __name__ == "__main__":
    main()
