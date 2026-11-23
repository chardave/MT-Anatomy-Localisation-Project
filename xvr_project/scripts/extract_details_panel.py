"""Read the C-arm geometry from the Siemens details panel in fluoroscopy screen recordings
and write one Excel row each time it changes.

Input: the screen recordings of ONE patient, in the order they were recorded. Each video is
read frame by frame (every --step frames). Two parts of the screen are read:

  details pane (right, always visible), per plane A (frontal) and B (lateral):
      rotation (LAO/RAO + degrees), angulation (CRAN/CAUD + degrees), zoom (cm), SID (cm),
      and the time at the top of the pane. That is either the wall clock (with AM/PM) or a
      stopwatch; details_time_type says which. Because the clock only shows minutes and is
      sometimes replaced by the stopwatch, estimated_clock_time gives the wall clock (to a
      few seconds) for every row, fitted from the AM/PM readings. Files named
      VGA.<seconds>.Auto share one time axis (the number is the start time in seconds), so
      the readings of all videos are fitted together and videos showing only the stopwatch
      still get a clock time; other names are fitted per video. It assumes the recording
      runs in real time and is left empty if the readings disagree.
  top panel (only shown while the arms or table move):
      per plane: "C-arm with curved arrow" (deg), "vertical arrow + crosshair" (cm),
                 "C-arm + horizontal arrow" (cm);
      table: tilt, longitudinal, lateral, height, cradle, rotation.
      Once the top panel disappears its last values are carried forward, and
      top_panel_visible says whether it was on screen at that row.

A new row is written when any of these values changes and the new values have been read
on --confirm consecutive samples (so a single misread frame does not create a row).
Numbers are read by OCR (RapidOCR): each number is cut out, its unit/degree sign removed,
tiled three times and recognised; the three copies must agree. Fields that could not be
read with confidence are listed in the uncertain_fields column.

Videos must be 1920 x 1080 screen recordings of the standard biplane layout (other sizes
are resized to it). A second sheet, "videos", lists each video's fps, frame count and the
offset added to get the overall time and frame.

Requirements (xvr environment): opencv-python, rapidocr-onnxruntime, pandas, openpyxl
Examples:
    python extract_details_panel.py --patient 45 ..\\input\\fluoro_45\\45_fluorovid.mp4
    python extract_details_panel.py --patient 12 VGA.1.Auto.mp4 VGA.2.Auto.mp4 VGA.3.Auto.mp4
    python extract_details_panel.py --patient 12 D:\\patient12_videos     (natural name order)
"""

import argparse
import re
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REFERENCE_SIZE = (1920, 1080)
TEXT_THRESHOLD = 115  # max(B, G, R) above this is text; the panel background is ~40

# ---- details pane (right) ----------------------------------------------------------------
# (x0, y0, x1, y1) in 1920 x 1080 pixels; plane B sits 100 px below plane A.
CLOCK_BOX = (1740, 188, 1880, 224)
SIDE_FIELDS = {}
for _plane, _dy in (("A", 0), ("B", 100)):
    SIDE_FIELDS[f"{_plane}_rotation_deg"] = (1698, 626 + _dy, 1766, 652 + _dy)
    SIDE_FIELDS[f"{_plane}_angulation_deg"] = (1764, 626 + _dy, 1834, 652 + _dy)
    SIDE_FIELDS[f"{_plane}_zoom_cm"] = (1872, 592 + _dy, 1919, 616 + _dy)
    SIDE_FIELDS[f"{_plane}_SID_cm"] = (1866, 621 + _dy, 1919, 645 + _dy)
SIDE_LABELS = {}
for _plane, _dy in (("A", 0), ("B", 100)):
    SIDE_LABELS[f"{_plane}_rotation_dir"] = (1700, 584 + _dy, 1764, 606 + _dy)
    SIDE_LABELS[f"{_plane}_angulation_dir"] = (1766, 584 + _dy, 1832, 606 + _dy)

# ---- top panel (shown while the arms/table move) ------------------------------------------
TOP_PANEL_BACKGROUND = np.array([38, 16, 16])            # BGR, opaque navy
TOP_PANEL_PROBES = [(484, 20), (1362, 20), (484, 122), (1362, 122), (924, 120)]
_TOP_COLUMNS = [478 + x // 2 for x in (0, 160, 300, 450, 580, 760, 880, 930, 1060, 1200, 1350, 1500, 1640, 1780)]
_PLANE_COLUMNS = ["top_rotation_deg", "top_angulation_deg", "top_SID_cm",
                  "carm_curved_arrow_deg", "vertical_crosshair_cm", "carm_horizontal_arrow_cm"]
TABLE_COLUMNS = ["table_tilt_deg", "table_longitudinal_cm", "table_lateral_cm",
                 "table_height_cm", "table_cradle_deg", "table_rotation_deg"]
TOP_FIELDS = {}
for _i, _name in enumerate(_PLANE_COLUMNS):
    TOP_FIELDS[f"A_{_name}"] = (_TOP_COLUMNS[_i], 68, _TOP_COLUMNS[_i + 1], 93)
    TOP_FIELDS[f"B_{_name}"] = (_TOP_COLUMNS[_i], 94, _TOP_COLUMNS[_i + 1], 120)
for _i, _name in enumerate(TABLE_COLUMNS):
    TOP_FIELDS[_name] = (_TOP_COLUMNS[_i + 7], 68, _TOP_COLUMNS[_i + 8], 96)
TOP_OUTPUT = [f"{p}_{c}" for p in "AB" for c in _PLANE_COLUMNS[3:]] + TABLE_COLUMNS

SIDE_OUTPUT = [f"{p}_{c}" for p in "AB" for c in
               ("rotation_dir", "rotation_deg", "angulation_dir", "angulation_deg", "zoom_cm", "SID_cm")]
PLAUSIBLE = {"zoom_cm": (5, 60), "SID_cm": (60, 150), "rotation_deg": (0, 180), "angulation_deg": (0, 90)}


# ---- OCR helpers -------------------------------------------------------------------------
def number_only(gray):
    """Cut the number out of a field: drop a trailing degree sign and a unit such as 'cm'.

    Ink is split into runs of columns. Digits reach both the top and the bottom of the text
    line; 'cm' does not reach the top, a degree sign has no ink in the lower half, and a
    leading minus sign sits in the middle only."""
    mask = gray > TEXT_THRESHOLD
    rows = np.where(mask.any(1))[0]
    if not len(rows):
        return None
    top, bottom = rows.min(), rows.max()
    height = bottom - top + 1
    if height < 6:
        return None
    mask = mask[top:bottom + 1]
    cols = np.where(mask.any(0))[0]
    runs, start, prev = [], cols[0], cols[0]
    for c in cols[1:]:
        if c != prev + 1:
            runs.append((start, prev))
            start = c
        prev = c
    runs.append((start, prev))
    upper = mask[:max(1, int(0.25 * height))]
    lower = mask[int(0.5 * height):]
    middle = mask[int(0.3 * height):int(0.75 * height)]
    bottom_band = mask[-max(1, int(0.2 * height)):]
    keep = []
    for a, b in runs:
        if upper[:, a:b + 1].any() and lower[:, a:b + 1].any():
            keep.append((a, b))
        elif not keep and middle[:, a:b + 1].any() and not upper[:, a:b + 1].any() \
                and not bottom_band[:, a:b + 1].any():
            keep.append((a, b))      # leading minus sign
        elif keep:
            break                    # degree sign or unit: the number has ended
    if not keep or not any(upper[:, a:b + 1].any() for a, b in keep):
        return None
    x0, x1 = keep[0][0], keep[-1][1]
    while x1 > x0 and not lower[:, x1].any():   # degree sign touching the last digit
        x1 -= 1
    return gray[top:bottom + 1, max(0, x0 - 1):x1 + 2]


def tiled(gray, scale, copies=3):
    """Black-on-white, enlarged, repeated `copies` times: the OCR model reads a lone '0'
    poorly but reads '000' reliably, and the copies must agree."""
    g = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    g = 255 - cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    h = g.shape[0]
    gap = np.full((h, max(4, h // 2)), 255, np.uint8)
    strip = np.hstack([gap] + [x for _ in range(copies) for x in (g, gap)])
    strip = cv2.copyMakeBorder(strip, h // 2, h // 2, 0, 0, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)


def plain(gray, scale):
    g = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    g = 255 - cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    g = cv2.copyMakeBorder(g, 12, 12, 20, 20, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


_CHAR_FIX = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "l": "1", "I": "1", "|": "1",
                           "i": "1", "S": "5", "B": "8", "—": "-", "–": "-", "一": "-"})


def parse_tiled(text, copies=3):
    """Return (value, confident) from the OCR text of a tiled number."""
    s = re.sub(r"\s", "", text).translate(_CHAR_FIX)
    if s and len(s) % copies == 0:
        n = len(s) // copies
        parts = [s[i * n:(i + 1) * n] for i in range(copies)]
        good = [p for p in parts if re.fullmatch(r"-?\d+", p)]
        if good:
            value, count = Counter(good).most_common(1)[0]
            if count >= 2:
                return int(value), count == copies
    numbers = re.findall(r"-?\d+", s)
    if numbers:
        value, count = Counter(numbers).most_common(1)[0]
        if count >= 2:
            return int(value), False
    return None, False


def parse_direction(text):
    t = re.sub(r"[^A-Z]", "", text.upper())
    for key, out in (("RAO", "RAO"), ("LAO", "LAO"), ("CRA", "CRAN"), ("CAU", "CAUD")):
        if key in t:
            return out
    if t.endswith("AO") and t[:1] in "RL":
        return t[:1] + "AO"
    return None


def parse_clock(text):
    m = re.search(r"(\d{1,2}[:.]\d{2}(?:[:.]\d{2})?)", text.replace(" ", ""))
    if not m:
        return None, None
    value = m.group(1).replace(".", ":")
    ampm = re.search(r"([AP])\.?M", text.upper().replace(" ", ""))
    if ampm:
        return f"{value} {ampm.group(1)}M", "clock"
    return value, "stopwatch"


class DetailsReader:
    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self.ocr = RapidOCR()
        self._cache = {}

    def _recognise(self, images):
        if not images:
            return []
        results, _ = self.ocr.text_rec(images)
        return [text for text, _score in results]

    def read_numbers(self, frame, fields, scale):
        """{field: (value or None, confident)}; empty fields give (None, True)."""
        out, keys, images = {}, [], []
        for key, (x0, y0, x1, y1) in fields.items():
            gray = frame[y0:y1, x0:x1].max(axis=2)
            cut = number_only(gray)
            if cut is None:
                out[key] = (None, True)
                continue
            # identical pixels give identical text: cache on the thresholded cut-out
            sig = (cut.shape, np.packbits(cut > TEXT_THRESHOLD).tobytes())
            if sig in self._cache:
                out[key] = self._cache[sig]
                continue
            keys.append((key, sig))
            images.append(tiled(cut, scale))
        for (key, sig), text in zip(keys, self._recognise(images)):
            out[key] = parse_tiled(text)
            self._cache[sig] = out[key]
        return out

    def read_labels(self, frame, fields):
        out, keys, images = {}, [], []
        for key, (x0, y0, x1, y1) in fields.items():
            gray = frame[y0:y1, x0:x1].max(axis=2)
            if (gray > TEXT_THRESHOLD).sum() < 8:
                out[key] = None
                continue
            keys.append(key)
            images.append(plain(gray, 3))
        for key, text in zip(keys, self._recognise(images)):
            out[key] = parse_direction(text)
        return out

    def read_clock(self, frame):
        x0, y0, x1, y1 = CLOCK_BOX
        gray = frame[y0:y1, x0:x1].max(axis=2)
        if (gray > TEXT_THRESHOLD).sum() < 8:
            return None, None
        (text,) = self._recognise([plain(gray, 2)])
        return parse_clock(text)


def top_panel_visible(frame):
    return all(np.abs(frame[y, x].astype(int) - TOP_PANEL_BACKGROUND).max() <= 12
               for x, y in TOP_PANEL_PROBES)


def side_signature(frame):
    """Thresholded pixels of every field read from the details pane and the top panel; if
    these are unchanged the readings are unchanged and OCR is skipped."""
    parts = []
    boxes = list(SIDE_FIELDS.values()) + list(SIDE_LABELS.values())
    visible = top_panel_visible(frame)
    if visible:
        boxes += list(TOP_FIELDS.values())
    for x0, y0, x1, y1 in boxes:
        parts.append(np.packbits(frame[y0:y1, x0:x1].max(axis=2) > TEXT_THRESHOLD).tobytes())
    return visible, hash(b"".join(parts))


def read_state(reader, frame, visible):
    """Values for one frame. Returns (side, top or None, uncertain field names)."""
    uncertain = []
    numbers = reader.read_numbers(frame, SIDE_FIELDS, scale=3)
    labels = reader.read_labels(frame, SIDE_LABELS)
    side = {}
    for key, (value, confident) in numbers.items():
        kind = key.split("_", 1)[1]
        lo, hi = PLAUSIBLE[kind]
        if value is not None and not lo <= value <= hi:
            value, confident = None, False
        side[key] = value
        if not confident:
            uncertain.append(key)
    for key, value in labels.items():
        side[key] = value
    for plane in "AB":
        for kind in ("rotation", "angulation"):
            # the direction label is blank at 0 degrees; a label with no number is a misread
            if side[f"{plane}_{kind}_deg"] in (None, 0):
                side[f"{plane}_{kind}_dir"] = None
    top = None
    if visible:
        top_numbers = reader.read_numbers(frame, TOP_FIELDS, scale=2)
        top = {}
        for key, (value, confident) in top_numbers.items():
            top[key] = value
            if not confident:
                uncertain.append(key)
        for plane in "AB":   # the top panel repeats the SID: use it as a check
            t, s = top.get(f"{plane}_top_SID_cm"), side.get(f"{plane}_SID_cm")
            if t is not None and s is not None and t != s:
                uncertain.append(f"{plane}_SID_cm (details {s} vs top panel {t})")
    return side, top, uncertain


# ---- main loop ---------------------------------------------------------------------------
def natural_key(path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def collect_videos(paths):
    videos = []
    for p in paths:
        if p.is_dir():
            found = sorted([q for q in p.iterdir() if q.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")],
                           key=natural_key)
            if not found:
                raise SystemExit(f"No videos in {p}")
            videos += found
        elif p.is_file():
            videos.append(p)
        else:
            raise SystemExit(f"Not found: {p}")
    return videos


def video_name(path):
    # VGA.182664526.Auto.mp4 -> VGA.182664526.Auto
    return path.stem


def fmt_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:06.3f}"


CLOCK_SAMPLE_S = 10   # read the details-pane time every 10 s of video


def clock_offset(samples):
    """Seconds-of-day at video time 0, from the AM/PM clock readings.

    The clock only shows minutes, so each reading m at video time t says the offset lies in
    [m*60 - t, m*60 + 60 - t); intersecting all readings narrows it to a few seconds. This
    assumes the recording runs in real time without cuts; if the readings disagree (empty
    intersection) None is returned. Stopwatch readings are not used."""
    lo, hi, n = -np.inf, np.inf, 0
    for t, (text, kind) in samples:
        if kind != "clock" or not text:
            continue
        m = re.fullmatch(r"(\d{1,2}):(\d{2}) ([AP])M", text)
        if not m:
            continue
        hour, minute = int(m.group(1)) % 12, int(m.group(2))
        if m.group(3) == "P":
            hour += 12
        start = (hour * 60 + minute) * 60
        lo, hi, n = max(lo, start - t), min(hi, start + 60 - t), n + 1
    if n == 0 or lo >= hi:
        return None
    return (lo + hi) / 2


def fmt_clock(seconds_of_day):
    s = int(round(seconds_of_day)) % 86400
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def process_video(reader, path, patient, offset_s, offset_frame, args, rows):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        raise RuntimeError(f"Could not read the frame rate of {path}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (w, h) != REFERENCE_SIZE:
        print(f"  WARNING: {w} x {h}; frames are resized to {REFERENCE_SIZE} before reading, check the output")

    committed = None          # values of the last written row
    clock_samples = []        # (video time, (text, type)) for estimating the wall clock
    top_memory = None         # last top-panel values seen in this video
    candidate, candidate_count, candidate_row = None, 0, None
    last_sig, last_read = None, None
    frame_idx, t_start, n_rows_before = -1, time.time(), len(rows)

    while True:
        frame_idx += 1
        if frame_idx % args.step:
            if not cap.grab():
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        if (frame.shape[1], frame.shape[0]) != REFERENCE_SIZE:
            frame = cv2.resize(frame, REFERENCE_SIZE, interpolation=cv2.INTER_AREA)

        visible, sig = side_signature(frame)
        if sig == last_sig:
            side, top, uncertain = last_read
        else:
            side, top, uncertain = read_state(reader, frame, visible)
            last_sig, last_read = sig, (side, top, uncertain)

        if frame_idx % round(fps * 60) < args.step:
            print(f"  {fmt_time(frame_idx / fps)} / {fmt_time(n_frames / fps)}  "
                  f"rows {len(rows) - n_rows_before}  ({time.time() - t_start:.0f}s)")
        if side["A_zoom_cm"] is None and side["A_SID_cm"] is None:
            continue            # details pane covered (e.g. "Automatic run" message): skip frame
        if frame_idx % round(fps * CLOCK_SAMPLE_S) < args.step:
            clock_samples.append((frame_idx / fps, reader.read_clock(frame)))
        if top is not None:
            top_memory = top
        values = [side.get(k) for k in SIDE_OUTPUT] + [(top_memory or {}).get(k) for k in TOP_OUTPUT]
        if committed is not None:
            # an unreadable field (None) is not a change: keep the last written value for it
            values = [c if v is None else v for v, c in zip(values, committed)]
        key = tuple(values)

        if committed is not None and key == committed:
            candidate, candidate_count = None, 0
            continue
        if key != candidate:
            t = frame_idx / fps
            candidate, candidate_count = key, 0
            candidate_row = {
                "patient": patient,
                "video": video_name(path),
                "video_time": fmt_time(t),
                "video_time_s": round(t, 3),
                "video_frame": frame_idx,
                "overall_time_s": round(offset_s + t, 3),
                "overall_frame": offset_frame + frame_idx,
                "_frame": frame,
                "_side": side, "_top": top_memory, "_visible": visible, "_uncertain": uncertain,
            }
        candidate_count += 1
        if candidate_count >= args.confirm:
            row = candidate_row
            clock, clock_type = reader.read_clock(row.pop("_frame"))
            side_r, top_r = row.pop("_side"), row.pop("_top") or {}
            row["details_time"] = clock
            row["details_time_type"] = clock_type
            row.update({k: side_r.get(k) for k in SIDE_OUTPUT})
            row["top_panel_visible"] = row.pop("_visible")
            row.update({k: top_r.get(k) for k in TOP_OUTPUT})
            row["uncertain_fields"] = ", ".join(
                u for u in row.pop("_uncertain") if u.split(" ")[0] in SIDE_OUTPUT + TOP_OUTPUT)
            clock_samples.append((row["video_time_s"], (clock, clock_type)))
            rows.append(row)
            committed, candidate, candidate_count = key, None, 0
            if args.verbose:
                print(f"    {row['video_time']}  A {row['A_rotation_dir'] or ''}{row['A_rotation_deg']}/"
                      f"{row['A_angulation_dir'] or ''}{row['A_angulation_deg']} {row['A_zoom_cm']}cm "
                      f"SID {row['A_SID_cm']} | B {row['B_rotation_dir'] or ''}{row['B_rotation_deg']}/"
                      f"{row['B_angulation_dir'] or ''}{row['B_angulation_deg']} {row['B_zoom_cm']}cm "
                      f"SID {row['B_SID_cm']}")
    cap.release()
    n_read = frame_idx  # frames actually decoded
    if n_read < n_frames - fps:
        print(f"  WARNING: stopped at frame {n_read} of {n_frames} (unreadable frame?)")

    return {"order": None, "video": video_name(path), "file": str(path),
            "name_seconds": name_seconds(path), "fps": fps,
            "frames": n_read, "duration_s": round(n_read / fps, 3),
            "overall_time_offset_s": round(offset_s, 3), "overall_frame_offset": offset_frame,
            "video_start_clock_estimate": None,
            "rows": len(rows) - n_rows_before,
            "_rows": (n_rows_before, len(rows)), "_clock_samples": clock_samples}


def name_seconds(path):
    """VGA.12372690.Auto -> 12372690: the recorder names each file with its start time in
    seconds (consecutive files differ by exactly the previous file's length)."""
    m = re.fullmatch(r"VGA\.(\d+)\.Auto", video_name(path), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def assign_clock(summary, rows):
    """Fill estimated_clock_time. With VGA.<seconds>.Auto names, all videos share one time
    axis, so AM/PM readings from any video fix the clock for all of them (including videos
    that only show the stopwatch). Otherwise each video is fitted on its own."""
    names = [s["name_seconds"] for s in summary]
    shared = all(n is not None for n in names) and all(b > a for a, b in zip(names, names[1:]))
    if shared:
        for a, b in zip(summary, summary[1:]):
            gap = b["name_seconds"] - a["name_seconds"] - a["duration_s"]
            if abs(gap) > 2:
                print(f"  NOTE: {gap:+.0f} s between the end of {a['video']} and the start of {b['video']} "
                      "(recording gap or overlap); overall_time_s ignores it")
        base = names[0]
        samples = [(t + s["name_seconds"] - base, c) for s in summary for t, c in s["_clock_samples"]]
        offset0 = clock_offset(samples)
        offsets = [None if offset0 is None else offset0 + s["name_seconds"] - base for s in summary]
    else:
        offsets = [clock_offset(s["_clock_samples"]) for s in summary]
    for s, offset in zip(summary, offsets):
        a, b = s.pop("_rows")
        s.pop("_clock_samples")
        s["video_start_clock_estimate"] = None if offset is None else fmt_clock(offset)
        for row in rows[a:b]:
            row["estimated_clock_time"] = None if offset is None else fmt_clock(row["video_time_s"] + offset)
        if offset is None:
            print(f"  Could not estimate the wall clock for {s['video']} (no consistent AM/PM readings)")
    if shared and offsets[0] is not None:
        print(f"  Clock fitted across all videos from the file names: {summary[0]['video']} "
              f"starts at {fmt_clock(offsets[0])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+", type=Path,
                        help="Videos of one patient IN RECORDING ORDER, or one folder (sorted by name, numbers numerically)")
    parser.add_argument("--patient", required=True, help="Patient number written in the first column")
    parser.add_argument("--output", type=Path, default=None,
                        help="Excel file (default: details_patient<N>.xlsx next to the first video)")
    parser.add_argument("--step", type=int, default=3,
                        help="Read every Nth frame (default 3 = 10 per second for 30 fps videos)")
    parser.add_argument("--confirm", type=int, default=2,
                        help="Samples a new reading must persist for before a row is written (default 2)")
    parser.add_argument("--verbose", action="store_true", help="Print each row as it is found")
    args = parser.parse_args()

    videos = collect_videos(args.videos)
    print("Video order:")
    for i, v in enumerate(videos, 1):
        print(f"  {i}. {v.name}")
    output = args.output or videos[0].parent / f"details_patient{args.patient}.xlsx"

    reader = DetailsReader()
    rows, summary = [], []
    offset_s, offset_frame = 0.0, 0
    for i, path in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {path.name}")
        info = process_video(reader, path, args.patient, offset_s, offset_frame, args, rows)
        info["order"] = i
        summary.append(info)
        offset_s += info["duration_s"]
        offset_frame += info["frames"]
        print(f"  {info['rows']} rows")
    assign_clock(summary, rows)

    columns = (["patient", "video", "video_time", "video_time_s", "video_frame", "overall_time_s",
                "overall_frame", "details_time", "details_time_type", "estimated_clock_time"] + SIDE_OUTPUT
               + ["top_panel_visible"] + TOP_OUTPUT + ["uncertain_fields"])
    table = pd.DataFrame(rows, columns=columns)
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as xl:
        table.to_excel(xl, sheet_name="details", index=False)
        pd.DataFrame(summary).to_excel(xl, sheet_name="videos", index=False)
        for sheet in xl.sheets.values():
            sheet.freeze_panes = "B2"
            for col in sheet.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                sheet.column_dimensions[col[0].column_letter].width = min(max(8, width + 2), 40)
    n_uncertain = (table["uncertain_fields"] != "").sum() if len(table) else 0
    print(f"Saved {len(table)} rows to {output}  ({n_uncertain} rows with uncertain fields)")


if __name__ == "__main__":
    main()
