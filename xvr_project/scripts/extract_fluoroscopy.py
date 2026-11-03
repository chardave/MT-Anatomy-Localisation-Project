"""Extract fluoroscopy panels from a screen recording of the Siemens biplane display.

The recorded screen (1920 x 1080) shows two rows of three image panels plus a status
panel on the right:

    Live A | Ref1 A | Ref2 A |  status (angles, zoom, SID)
    Live B | Ref1 B | Ref2 B |

Plane A is the frontal (AP) plane, plane B the lateral plane. "Live" is the current
image; "Ref1"/"Ref2" hold stored reference images, which may come from an earlier
moment, so use the Live panels for registration.

Two modes:
  --pair       save Live A, Live B and the status panel from the SAME frame, with no
               window: <prefix>_AP.png, <prefix>_LAT.png, <prefix>_details.png. Read the
               angles, zoom and SID for 5_fluoro_png_to_dicom.py from the details image,
               because the zoom can change during a procedure.
  (default)    show the frame and choose one panel: keys 1-6 as in the layout above,
               M to draw a rectangle yourself, Q/Esc to cancel.

Requirements: opencv-python (xvr environment)
Examples:
    python extract_fluoroscopy.py --video ..\\input\\fluoro_45\\45_fluorovid.mp4 --time 02:11.5 --pair ^
        --output-dir ..\\input\\fluoro_45 --prefix fluoro_non_contrast
    python extract_fluoroscopy.py --video ..\\input\\fluoro_45\\45_fluorovid.mp4 --time 131.5
"""

import argparse
from pathlib import Path

import cv2

# Image areas measured on the 1920 x 1080 recordings (cases 43 and 45 agree to 1 px).
# (x0, y0, x1, y1); other resolutions are scaled proportionally.
REFERENCE_SIZE = (1920, 1080)
PANEL_WIDTH, PANEL_HEIGHT = 513, 507
COLUMN_X = {"Live": 25, "Ref1": 25 + 563, "Ref2": 25 + 1126}
ROW_Y = {"A": 28, "B": 568}
PANELS = {
    1: ("Live", "A"), 2: ("Ref1", "A"), 3: ("Ref2", "A"),
    4: ("Live", "B"), 5: ("Ref1", "B"), 6: ("Ref2", "B"),
}
STATUS_PANEL = (1690, 0, 1920, 1080)


def panel_box(column, row, width, height):
    sx, sy = width / REFERENCE_SIZE[0], height / REFERENCE_SIZE[1]
    x0, y0 = COLUMN_X[column], ROW_Y[row]
    return (round(x0 * sx), round(y0 * sy), round((x0 + PANEL_WIDTH) * sx), round((y0 + PANEL_HEIGHT) * sy))


def scale_box(box, width, height):
    sx, sy = width / REFERENCE_SIZE[0], height / REFERENCE_SIZE[1]
    x0, y0, x1, y1 = box
    return round(x0 * sx), round(y0 * sy), round(x1 * sx), round(y1 * sy)


def parse_timestamp(raw):
    parts = raw.split(":")
    if len(parts) > 3:
        raise argparse.ArgumentTypeError("Use seconds, MM:SS or HH:MM:SS")
    try:
        numbers = [float(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Invalid timestamp") from exc
    if any(x < 0 for x in numbers):
        raise argparse.ArgumentTypeError("Timestamp must be nonnegative")
    if len(parts) > 1 and any(x >= 60 for x in numbers[1:]):
        raise argparse.ArgumentTypeError("Minutes/seconds components must be less than 60")
    seconds = 0.0
    for value in numbers:
        seconds = seconds * 60 + value
    return seconds


def fit_for_display(frame, max_width=1200, max_height=680):
    h, w = frame.shape[:2]
    scale = min(1.0, max_width / w, max_height / h)
    if scale == 1:
        return frame.copy(), scale
    return cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA), scale


def crop(frame, box, inset=0):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box
    x0, y0, x1, y1 = max(0, x0 + inset), max(0, y0 + inset), min(w, x1 - inset), min(h, y1 - inset)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Crop is empty; reduce --inset or select a valid region")
    return frame[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def save(path, image):
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to save {path}")
    print(f"Saved: {path}  ({image.shape[1]} x {image.shape[0]})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--time", required=True, type=parse_timestamp,
                        help="Seconds, MM:SS or HH:MM:SS (decimals supported)")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "input")
    parser.add_argument("--pair", action="store_true",
                        help="Save Live A, Live B and the status panel from this frame, without a window")
    parser.add_argument("--prefix", default=None,
                        help="--pair file name prefix (default: frame_t<time>)")
    parser.add_argument("--inset", type=int, default=0,
                        help="Trim this many ORIGINAL pixels from each side of a panel")
    parser.add_argument("--max-display-width", type=int, default=1200,
                        help="Largest preview window width in screen pixels (saved crops stay full resolution)")
    parser.add_argument("--max-display-height", type=int, default=680,
                        help="Largest preview window height in screen pixels (saved crops stay full resolution)")
    args = parser.parse_args()

    if not args.video.is_file():
        parser.error(f"Video not found: {args.video}")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps <= 0:
        cap.release()
        raise RuntimeError("Could not read video frame rate")
    duration = nframes / fps
    if args.time >= duration:
        cap.release()
        raise ValueError(f"Timestamp {args.time:.2f}s beyond video duration {duration:.2f}s")

    target_frame = min(int(round(args.time * fps)), int(nframes) - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame = cap.read()
    actual_frame = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
    cap.release()
    if not ok:
        raise RuntimeError("Could not read frame at requested timestamp")
    H, W = frame.shape[:2]
    print(f"Video: {args.video.name} | {fps:.2f} fps | duration {duration:.2f}s")
    print(f"Requested time: {args.time:.3f}s | retrieved frame ~{int(actual_frame)}")
    print(f"Full-frame dimensions: {W} x {H}")
    if (W, H) != REFERENCE_SIZE:
        print(f"WARNING: layout was measured on {REFERENCE_SIZE[0]} x {REFERENCE_SIZE[1]} recordings; "
              "boxes are scaled, so check the saved crops (or use M in the window).")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"t{args.time:.3f}s".replace('.', 'p')

    # ---- pair mode: both live planes and the status panel from one frame ----
    if args.pair:
        prefix = args.prefix or f"frame_{stamp}"
        for suffix, (column, row) in (("AP", PANELS[1]), ("LAT", PANELS[4])):
            image, box = crop(frame, panel_box(column, row, W, H), args.inset)
            save(args.output_dir / f"{prefix}_{suffix}.png", image)
        details, _ = crop(frame, scale_box(STATUS_PANEL, W, H))
        save(args.output_dir / f"{prefix}_details.png", details)
        print("Read plane A (AP) and plane B (LAT) angles, zoom (cm) and SID (cm) from the details image.")
        return

    # ---- interactive mode ----
    print("Select panel 1-6; press M for manual rectangle; Q/Esc to quit.")
    display, scale = fit_for_display(frame, args.max_display_width, args.max_display_height)
    preview = display.copy()
    for number, (column, row) in PANELS.items():
        x0, y0, x1, y1 = [round(v * scale) for v in panel_box(column, row, W, H)]
        cv2.rectangle(preview, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 2)
        text = f"{number}: {column} {row}"
        cv2.putText(preview, text, (x0 + 10, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(preview, text, (x0 + 10, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

    win = "Choose panel: 1-6 | M: manual crop | Q: quit"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(win, preview)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow(win)
    if key in (ord('q'), ord('Q'), 27):
        print("Cancelled")
        return

    if ord('1') <= key <= ord('6'):
        column, row = PANELS[key - ord('0')]
        image, box = crop(frame, panel_box(column, row, W, H), args.inset)
        suffix = f"{column}_{row}"
    elif key in (ord('m'), ord('M')):
        roi = cv2.selectROI("Draw ROI then Enter/Space", display, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()
        dx, dy, rw, rh = map(int, roi)
        if rw <= 0 or rh <= 0:
            print("No manual ROI selected")
            return
        image, box = crop(frame, (round(dx / scale), round(dy / scale),
                                  round((dx + rw) / scale), round((dy + rh) / scale)))
        suffix = "manual_crop"
    else:
        print("No valid selection; cancelled")
        return

    save(args.output_dir / f"{suffix}_{stamp}.png", image)
    print(f"Crop boundaries in original pixels: ({box[0]}, {box[1]}) to ({box[2]}, {box[3]})")


if __name__ == "__main__":
    main()
