
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from diffdrr.data import read
from diffdrr.drr import DRR


ROOT = Path(__file__).resolve().parents[1]
CT_PATH = ROOT / "input" / "CT.nii.gz"
FLUORO_PATH = ROOT / "input" / "fluoroscopy_AP.png"
OUTPUT = ROOT / "outputs" / "drr"
OUTPUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Initial exploratory camera settings (not calibrated).
SDD = 1000.0  # mm
SOURCE_TO_CENTRE = 850.0  # mm
HEIGHT = 516    
PIXEL_SIZE = 0.56  # mm/pixel at rendered resolution


def make_display_drr(img, p_low=5.0, p_high=99.7, gamma=0.7):
    """
    Convert a raw DRR into a display-friendly image.
    - percentile clipping = windowing
    - inversion = makes bone dark like X-ray
    - gamma < 1 brightens midtones
    """
    img = np.asarray(img, dtype=np.float32)
    valid = np.isfinite(img)

    if not valid.any():
        raise ValueError("DRR contains no finite pixels")

    lo, hi = np.percentile(img[valid], [p_low, p_high])

    if hi <= lo:
        return np.zeros_like(img)

    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-8)

    # Invert so dense structures appear dark
    display = 1.0 - img

    # Brighten the midtones
    display = np.power(display, gamma)

    return np.clip(display, 0, 1)


def main():
    print("Using device:", DEVICE)
    print("Loading CT:", CT_PATH)

    if not CT_PATH.exists():
        raise FileNotFoundError(CT_PATH)

    if not FLUORO_PATH.exists():
        raise FileNotFoundError(FLUORO_PATH)

    # Keep original fluoroscopy aspect ratio.
    with Image.open(FLUORO_PATH) as im:
        fluoro = np.asarray(im.convert("L"), dtype=np.float32)
        original_width, original_height = im.size

    width = round(HEIGHT * original_width / original_height)

    print("Real fluoroscopy:", fluoro.shape)
    print("Synthetic resolution:", HEIGHT, "x", width)

    rotations = torch.zeros(
        (1, 3), dtype=torch.float32, device=DEVICE
    )
    translations = torch.tensor(
        [[0.0, SOURCE_TO_CENTRE, 0.0]],
        dtype=torch.float32,
        device=DEVICE,
    )

    results = {}

    for orientation in ("AP", "PA"):
        print(f"\nLoading volume for {orientation} projection...")

        subject = read(
            CT_PATH,
            orientation=orientation,
            center_volume=True,
        )

        print("Subject orientation:", subject.volume.orientation)
        print("Centred CT affine:")
        print(subject.volume.affine)

        drr = DRR(
            subject,
            sdd=SDD,
            height=HEIGHT,
            width=width,
            delx=PIXEL_SIZE,
            dely=PIXEL_SIZE,
            reverse_x_axis=True,
            renderer="siddon",
        ).to(DEVICE)

        print("Rendering...")

        with torch.no_grad():
            image = drr(
                rotations,
                translations,
                parameterization="euler_angles",
                convention="ZXY",
            )

        image = image.detach().cpu().numpy().squeeze()

        print("Output shape:", image.shape)
        print("Raw intensity:", image.min(), image.max())

        display = make_display_drr(image, p_low=5.0, p_high=99.7, gamma=0.7)
        results[orientation] = display

        saved = (display * 255).astype(np.uint8)

        # High attenuation appears dark, like conventional X-ray.
        # saved = ((1.0 - normalised) * 255).astype(np.uint8)

        Image.fromarray(saved).save(
            OUTPUT / f"initial_{orientation}_drr.png"
        )

        del drr, subject, image

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    axes[0].imshow(fluoro, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Real fluoroscopy")

    for ax, orientation in zip(axes[1:], ("AP", "PA")):
        ax.imshow(
            results[orientation],
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"Synthetic {orientation} DRR")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    comparison = OUTPUT / "initial_comparison.png"
    plt.savefig(comparison, dpi=200, bbox_inches="tight")
    plt.close()

    print("\nFinished!")
    print("Outputs saved to:", OUTPUT)
    print("Comparison:", comparison)


if __name__ == "__main__":
    main()
