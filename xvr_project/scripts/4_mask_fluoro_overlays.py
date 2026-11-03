"""Hide burned-in text and grey cover bars on a fluoroscopy snapshot before registration.

Draw one rectangle over each overlay (hospital name, date, protocol text, grey bars).
The covered pixels are filled from the surrounding image (inpainting), so the
registration sees no sharp text edges. The rectangles are saved to JSON, so the
same mask can be re-applied without drawing again (--rects).

Requirements: opencv-python, numpy (xvr environment)
Example:
    python 4_mask_fluoro_overlays.py --image ..\\input\\fluoro_43\\fluoro_AP_non_contrast.png
    python 4_mask_fluoro_overlays.py --image ... --rects ..\\input\\fluoro_43\\fluoro_AP_non_contrast_mask_rects.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def fit_for_display(image, max_width=1200, max_height=680):
    # snapshots are ~500 px, so scale them UP to make drawing easier, but never past the screen
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)
    resized = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_NEAREST)
    return resized, scale


def draw_rectangles(gray, max_width, max_height):
    display, scale = fit_for_display(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), max_width, max_height)
    print("Draw a rectangle over each overlay, pressing Enter/Space after each one.")
    print("Press Esc when finished (Esc straight away = no rectangles).")
    rois = cv2.selectROIs("Cover overlays: Enter after each box, Esc to finish", display,
                          showCrosshair=False, fromCenter=False)
    cv2.destroyAllWindows()
    rects = []
    for dx, dy, rw, rh in np.asarray(rois).reshape(-1, 4):
        if rw > 0 and rh > 0:
            # back to original-image pixels, rounding outwards so nothing is missed
            x0, y0 = int(np.floor(dx / scale)), int(np.floor(dy / scale))
            x1, y1 = int(np.ceil((dx + rw) / scale)), int(np.ceil((dy + rh) / scale))
            rects.append([x0, y0, x1, y1])
    return rects


def build_mask(shape, rects, grow):
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for x0, y0, x1, y1 in rects:
        x0, y0 = max(0, x0 - grow), max(0, y0 - grow)
        x1, y1 = min(w, x1 + grow), min(h, y1 + grow)
        mask[y0:y1, x0:x1] = 255
    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, type=Path, help="Snapshot PNG from extract_fluoroscopy.py")
    parser.add_argument("--rects", type=Path,
                        help="Re-use rectangles saved by a previous run instead of drawing them")
    parser.add_argument("--output-dir", type=Path, help="Default: same folder as --image")
    parser.add_argument("--fill", choices=["inpaint", "white"], default="inpaint",
                        help="inpaint = fill from surrounding image (default); white = paint as air")
    parser.add_argument("--grow", type=int, default=2,
                        help="Enlarge every rectangle by this many pixels to catch anti-aliased text edges")
    parser.add_argument("--max-display-width", type=int, default=1200)
    parser.add_argument("--max-display-height", type=int, default=680)
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    gray = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Cannot read image: {args.image}")
    out_dir = args.output_dir or args.image.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem

    # ---- 1. get the rectangles (drawn now, or loaded from an earlier run) ----
    if args.rects:
        rects = json.loads(args.rects.read_text(encoding="utf-8"))["rects"]
        print(f"Loaded {len(rects)} rectangle(s) from {args.rects}")
    else:
        rects = draw_rectangles(gray, args.max_display_width, args.max_display_height)
        if not rects:
            print("No rectangles drawn; nothing saved.")
            return
        rects_path = out_dir / f"{stem}_mask_rects.json"
        rects_path.write_text(json.dumps({"image": args.image.name, "size_wh": [gray.shape[1], gray.shape[0]],
                                          "rects": rects}, indent=2), encoding="utf-8")
        print(f"Saved rectangles: {rects_path}")

    # ---- 2. fill the covered pixels ----
    mask = build_mask(gray.shape, rects, args.grow)
    if args.fill == "inpaint":
        masked = cv2.inpaint(gray, mask, 7, cv2.INPAINT_TELEA)
    else:
        masked = gray.copy()
        masked[mask > 0] = 255

    # ---- 3. save the result, the mask and a before/after check image ----
    masked_path = out_dir / f"{stem}_masked.png"
    cv2.imwrite(str(masked_path), masked)
    cv2.imwrite(str(out_dir / f"{stem}_mask.png"), mask)
    outline = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x0, y0, x1, y1 in rects:
        cv2.rectangle(outline, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 255), 1)
    check = np.hstack([outline, cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)])
    cv2.imwrite(str(out_dir / f"{stem}_mask_check.png"), check)
    print(f"Saved masked image: {masked_path}")
    print(f"Covered {int((mask > 0).sum())} pixels ({100 * (mask > 0).mean():.1f}% of the image)")
    print(f"Open {stem}_mask_check.png to confirm all text is gone (left = boxes, right = result).")


if __name__ == "__main__":
    main()
