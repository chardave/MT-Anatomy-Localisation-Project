"""Interactively extract one of six panels from a fluoroscopy screen recording.

Requirements: pip install opencv-python
Example:
    python 2_extract_fluoroscopy.py --video "C:\\path\\recording.mp4" --time 00:01:23.5
"""

import argparse
from pathlib import Path
import cv2

PANEL_NAMES = {
    1: "AP_roadmap",
    2: "AP_fluoroscopy",
    3: "AP_DSA",
    4: "lateral_roadmap",
    5: "lateral_fluoroscopy",
    6: "lateral_DSA",
}


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


def fit_for_display(frame, max_width=1440, max_height=850):
    h, w = frame.shape[:2]
    scale = min(1.0, max_width / w, max_height / h)
    if scale == 1:
        return frame.copy(), scale
    return cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA), scale


def grid_bounds(panel, width, height):
    col = (panel - 1) % 3
    row = (panel - 1) // 3
    xs = [round(width * i / 3) for i in range(4)]
    ys = [0, round(height / 2), height]
    return xs[col], ys[row], xs[col + 1], ys[row + 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--time", required=True, type=parse_timestamp,
                        help="Seconds, MM:SS or HH:MM:SS (decimals supported)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "input")
    parser.add_argument("--inset", type=int, default=0,
                        help="Trim this many ORIGINAL pixels from each side of an automatically selected panel")
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
    print("Select panel 1-6; press M for manual rectangle; Q/Esc to quit.")

    display, scale = fit_for_display(frame)
    dh, dw = display.shape[:2]
    preview = display.copy()
    for i in range(1, 7):
        x0, y0, x1, y1 = grid_bounds(i, dw, dh)
        cv2.rectangle(preview, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 2)
        cv2.putText(preview, f"{i}: {PANEL_NAMES[i]}", (x0 + 10, y0 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(preview, f"{i}: {PANEL_NAMES[i]}", (x0 + 10, y0 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

    win = "Choose panel: 1-6 | M: manual crop | Q: quit"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(win, preview)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow(win)
    if key in (ord('q'), ord('Q'), 27):
        print("Cancelled")
        return

    if ord('1') <= key <= ord('6'):
        panel = key - ord('0')
        x0, y0, x1, y1 = grid_bounds(panel, W, H)
        inset = args.inset
        x0, y0, x1, y1 = x0 + inset, y0 + inset, x1 - inset, y1 - inset
        suffix = PANEL_NAMES[panel]
    elif key in (ord('m'), ord('M')):
        roi = cv2.selectROI("Draw ROI then Enter/Space", display, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()
        dx, dy, rw, rh = map(int, roi)
        if rw <= 0 or rh <= 0:
            print("No manual ROI selected")
            return
        x0, y0 = round(dx / scale), round(dy / scale)
        x1, y1 = round((dx + rw) / scale), round((dy + rh) / scale)
        suffix = "manual_crop"
    else:
        print("No valid selection; cancelled")
        return

    x0, x1 = max(0, x0), min(W, x1)
    y0, y1 = max(0, y0), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Crop is empty; reduce --inset or select a valid region")
    crop = frame[y0:y1, x0:x1].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"t{args.time:.3f}s".replace('.', 'p')
    path = args.output_dir / f"{suffix}_{stamp}.png"
    if not cv2.imwrite(str(path), crop):
        raise RuntimeError(f"Failed to save {path}")
    print(f"Saved: {path}")
    print(f"Crop boundaries in original pixels: ({x0}, {y0}) to ({x1}, {y1})")
    print(f"Saved resolution: {crop.shape[1]} x {crop.shape[0]}")

    if suffix == "AP_fluoroscopy":
        fixed = args.output_dir / "fluoroscopy_AP.png"
        cv2.imwrite(str(fixed), crop)
        print(f"Also updated xvr input: {fixed}")
    elif suffix == "lateral_fluoroscopy":
        fixed = args.output_dir / "fluoroscopy_lateral.png"
        cv2.imwrite(str(fixed), crop)
        print(f"Also updated xvr input: {fixed}")


if __name__ == "__main__":
    main()
