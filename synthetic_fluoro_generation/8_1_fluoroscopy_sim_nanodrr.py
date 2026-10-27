import argparse
import csv
import json
from pathlib import Path
import sys
import time
import traceback
import tempfile
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter1d, minimum_filter1d

# imports for GUI
from PySide6.QtCore import Qt, QObject, Signal, QRunnable, QThreadPool
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

#imports for DRR rendering
from nanodrr.data import Subject
from nanodrr.camera import make_k_inv, make_rt_inv
from nanodrr.drr import render

# USER PATHS - starts with default s0224 from total_segmentator
# can be changed in the app
#==================================================================================

IMAGE_PATH = Path(
    r"C:\Users\cd26\OneDrive - King's College London\MT Anatomy Localisation Project\Totalsegmentator_dataset_v201\s0224\ct_fixed_orthogonal.nii.gz"
)

LABEL_PATH = Path(
    r"C:\Users\cd26\OneDrive - King's College London\MT Anatomy Localisation Project\Totalsegmentator_dataset_v201\s0224\segmentations\all_labels_labelmap.nii.gz"
)

# Label mapping assigns labels to label IDs - this can be used to add the labels from 
# other datasets which may have different names to all align to the same label name
LABEL_MAPPING_PATH = Path(
    r"C:\Users\cd26\OneDrive - King's College London\MT Anatomy Localisation Project\MT-Anatomy-Localisation-Project\metadata\totalsegmentator_label_mapping.csv"
)

# ============================================================
# Label mapping helpers
# ============================================================
def load_label_mapping(mapping_path):
    """
    assigned the label to the id and vice versa
    allows multiple datasets to be considered using the same script even if the names are written slightly different
    """
    if mapping_path is None:
        return {}, {}
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        print(f"Warning: label mapping file not found at {mapping_path}.")
        print("segmentations can still be seen but individual label selection will be unavailable")
        return {}, {}

    label_id_to_name = {}

    if mapping_path.suffix.lower() == ".json":
        with mapping_path.open("r", encoding="utf-8-sig") as f:
            payload = json.load(f)
        labels = payload.get("labels", payload)
        if not isinstance(labels, dict):
            raise ValueError("JSON mapping must contain an object named 'labels'.")
        for label_name, label_id in labels.items():
            label_id = int(label_id)
            if label_id > 0:
                label_id_to_name[label_id] = str(label_name).strip()
    elif mapping_path.suffix.lower() == ".csv":
        with mapping_path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            required = {"label_name", "label_id"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(
                    f"Mapping CSV must contain columns {required}."
                    f"Found columns: {reader.fieldnames}"
                )
            for row in reader:
                label_name = str(row["label_name"]).strip()
                label_id = int(row["label_id"])
                if label_id > 0:
                    label_id_to_name[label_id] = label_name
    else:
        raise ValueError("Label mapping must be a .json or .csv file.")
    
    label_name_to_id = {
        label_name: label_id
        for label_id, label_name in label_id_to_name.items()
    }

    return label_id_to_name, label_name_to_id

def choose_starting_visible_label_ids(label_id_to_name):
    """
    Decide which labels are ticked when the app starts.

    - If there is no CSV mapping, return None, meaning show all label channels.
    - Otherwise tick every label in the mapping.
    """
    if not label_id_to_name:
        return None

    return set(label_id_to_name.keys())

def build_label_id_to_channel(label_id_to_name):
    """
    Map real label IDs from the CSV to nanoDRR label channel indices.

    label_channels[0] is the first segmentation channel after the CT channel.
    For normal labelmaps with IDs 1..N, this gives: 1->0, 2->1, 3->2, ...
    For non-consecutive IDs, this uses sorted label IDs from the CSV.

    Best practice: keep/remap each dataset labelmap to consecutive IDs 1..N.
    """
    if not label_id_to_name:
        return None

    return {
        int(label_id): int(label_id) - 1
        for label_id in label_id_to_name
        if int(label_id) > 0
    }

# ============================================================
# nanoDRR helper functions - taken from original script
# ============================================================
def normalise_for_display(img_2d, p_low=5.0, p_high=99.7):
    """Percentile-window raw NanoDRR output (display only)."""
    img_2d = np.asarray(img_2d, dtype=np.float32)
    valid = np.isfinite(img_2d)
    if not np.any(valid):
        return np.zeros_like(img_2d, dtype=np.float32)
    low, high = np.percentile(img_2d[valid], [p_low, p_high])
    if high <= low:
        return np.zeros_like(img_2d, dtype=np.float32)
    clipped = np.nan_to_num(img_2d, nan=low, posinf=high, neginf=low)
    return np.clip((clipped - low) / (high - low), 0, 1)


# required for Qt display - greyscale to RGB uint8 converter
def greyscale_to_rgb_uint8(img_np):
    img_uint8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([img_uint8, img_uint8, img_uint8], axis=-1)
    return np.ascontiguousarray(rgb)

# sets the camera geometry that we can adjust using the gui 
# - this will be updates every time the user changes the 
# settings on the gui and sent to render the image per frame
def make_geometry(
    device,
    subject_isocenter,
    rot_x = 0.0,
    rot_y=0.0,
    rot_z=0.0,
    trans_x=0.0,
    trans_y=850.0,
    trans_z=0.0,
    sdd_value=1020.0,
    height=256,
    width=256,
    pixel_size=2.0,
):
    delx = dely = pixel_size
    x0 = y0 = 0.0

    #nanodrr function to make the inverse intrinsic matrix for a cone beam projector
    k_inv = make_k_inv(
        sdd_value,
        delx,
        dely,
        x0,
        y0,
        height,
        width,
        device=device,
    )

    sdd = torch.tensor([sdd_value], device=device)

    #set rotation and translations - values will come from gui inputs
    rotations = torch.tensor(
        [[rot_x, rot_y, rot_z]],
        dtype=torch.float32,
        device=device,
    )

    translations = torch.tensor(
        [[trans_x, trans_y, trans_z]],
        dtype=torch.float32,
        device=device,
    )

    # construct the camera to world (extrinsic inverse) matrices
    rt_inv = make_rt_inv(
        rotations,
        translations,
        orientation="AP",
        isocenter=subject_isocenter
    )

    return k_inv, rt_inv, sdd

def render_ct_projection(subject_ct, k_inv, rt_inv, sdd, height, width):
    with torch.no_grad():
        return render(subject_ct, k_inv, rt_inv, sdd, height, width)

# render projected segmentation labels as seperate per-label channels
def render_label_channels(
        subject_labels,
        k_inv, 
        rt_inv,
        sdd,
        height,
        width,
):
    
    with torch.no_grad():
        label_img = render(subject_labels, k_inv, rt_inv, sdd, height, width)
    
    label_cpu = label_img.detach().cpu()
    
    if label_cpu.ndim !=4: 
        raise ValueError(f"Expected nanoDRR output [B, C, H, W], got {label_cpu.shape}")
    
    if label_cpu.shape[1] > 1:
        # Channel 0 is the CT/DRR channel. Channels 1..N are label projections.
        label_channels = label_cpu[0, 1:, :, :]
    else:
        # fallback for label-only subjects (no CT channel). Channel 0 is the label projection.
        label_channels = label_cpu[0, 0:1, :, :]

    return label_channels.numpy().astype(np.float32)

# ============================================================
# Segmentation overlay helpers
# ============================================================
PALETTE = np.array(
    [
        [230, 25, 75],     # red
        [60, 180, 75],     # green
        [0, 130, 200],     # blue
        [245, 130, 48],    # orange
        [145, 30, 180],    # purple
        [70, 240, 240],    # cyan
        [240, 50, 230],    # magenta
        [210, 245, 60],    # lime
        [250, 190, 190],   # pink
        [0, 128, 128],     # teal
        [230, 190, 255],   # lavender
        [170, 110, 40],    # brown
        [255, 250, 200],   # beige
        [128, 0, 0],       # maroon
        [170, 255, 195],   # mint
        [0, 0, 128],       # navy
    ],
    dtype=np.float32
)

# assigns a colour to each label id based on the palette, 
# cycling through if there are more labels than colours
def label_to_colour(label_id):
    if label_id <= 0:
        return np.array([0, 0, 0], dtype=np.float32)
    
    return PALETTE[(int(label_id) - 1) % len(PALETTE)]

# 1 pixel mask border - can be adjusted using dilate_mask function
def mask_border(mask):
    mask = mask.astype(bool)

    padded = np.pad(mask, 1, mode="constant", constant_values=False)

    centre = padded[1:-1, 1:-1]
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]

    inner = centre & up & down & left & right
    border = centre & ~inner

    return border

def dilate_mask(mask, iterations=1):
    mask = mask.astype(bool)

    for _ in range(max(0, int(iterations))):
        padded = np.pad(mask, 1, mode="constant", constant_values=False)

        mask = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
            | padded[:-2, :-2]
            | padded[:-2, 2:]
            | padded[2:, :-2]
            | padded[2:, 2:]
        )

    return mask

def overlay_labels_channels_on_rgb(
        rgb,
        label_channels,
        visible_label_ids=None,
        label_id_to_channel=None,
        fill_alpha=0.30,
        border_width=2,
        label_threshold=0.0,
):
    out = rgb.astype(np.float32).copy()

    if label_channels is None or label_channels.size == 0:
        return rgb
    
    n_labels = int(label_channels.shape[0])

    if visible_label_ids is None:
        visible_ids = list(range(1, n_labels + 1))
    else:
        visible_ids = sorted(int(x) for x in visible_label_ids)
    
    masks_to_border = []

    for label_id in visible_ids:

        if label_id_to_channel is None:
            channel_idx = label_id - 1
        else:
            channel_idx = label_id_to_channel.get(label_id, None)
            if channel_idx is None:
                continue

        if channel_idx < 0 or channel_idx >= n_labels:
            continue
        
        mask = label_channels[channel_idx] > label_threshold

        if not np.any(mask):
            continue

        colour = label_to_colour(label_id)

        out[mask] = (1.0 -fill_alpha) * out[mask] + fill_alpha * colour
        masks_to_border.append((label_id, mask))
    
    for label_id, mask in masks_to_border:
        border = mask_border(mask)

        if border_width > 1:
            border = dilate_mask(border, iterations=border_width - 1)
        
        colour = label_to_colour(label_id)
        out[border] = colour
    
    return np.clip(out, 0, 255).astype(np.uint8)

# ---------- HU-derived vessels and optional TopBrain fusion ----------
# All 3D array operations use the NIfTI array axis order, NOT NanoDRR tensor order.
def _box_morph(mask, radius_mm, spacing, operation):
    """Axis-wise box dilation/erosion; radius 0 means no change."""
    result = np.asarray(mask, dtype=np.uint8)
    if radius_mm <= 0:
        return result.astype(bool)
    filt = maximum_filter1d if operation == "dilate" else minimum_filter1d
    for axis, mm in enumerate(spacing):
        r = max(0, int(round(float(radius_mm) / float(mm))))
        if r:
            result = filt(result, size=2 * r + 1, axis=axis, mode="constant", cval=0)
    return result.astype(bool)


def extract_hu_vessels(hu, spacing, p):
    soft = (hu > p["soft_lower"]) & (hu < p["soft_upper"])
    brain = _box_morph(soft, p["opening_mm"], spacing, "erode")
    brain = _box_morph(brain, p["opening_mm"], spacing, "dilate")
    brain = _box_morph(brain, p["closing_mm"], spacing, "dilate")
    brain = _box_morph(brain, p["closing_mm"], spacing, "erode")
    bone = hu > p["bone_threshold"]
    excluded = _box_morph(bone, p["bone_exclusion_mm"], spacing, "dilate")
    candidate = brain & (hu > p["vessel_threshold"]) & ~excluded
    candidate = _box_morph(candidate, p["vessel_dilation_mm"], spacing, "dilate")
    return candidate & brain & ~_box_morph(bone, p["final_bone_exclusion_mm"], spacing, "dilate")


def _prepare_vessel_subject(image_path, label_path, params, device, cache):
    """Rebuild when mask-related settings change. Never modifies the input CTA."""
    keys = ("vessel_threshold", "bone_threshold", "soft_lower", "soft_upper",
            "opening_mm", "closing_mm", "bone_exclusion_mm", "vessel_dilation_mm",
            "final_bone_exclusion_mm", "iodine_gain", "fuse_topbrain")
    signature = (str(image_path), str(label_path), *[params[k] for k in keys])
    if cache.get("key") == signature:
        return cache["subject"], cache["message"]

    original = nib.load(str(image_path))
    hu = np.asarray(original.dataobj, dtype=np.float32)
    if hu.ndim != 3:
        raise ValueError("HU vessel extraction requires a 3D CT/CTA NIfTI image")
    spacing = nib.affines.voxel_sizes(original.affine)[:3]
    mask = extract_hu_vessels(hu, spacing, params)
    message = f"HU mask: {int(mask.sum()):,} voxels"

    if params["fuse_topbrain"]:
        if label_path is None or not Path(label_path).exists():
            raise ValueError("TopBrain fusion enabled but no label map is loaded")
        seg = nib.load(str(label_path))
        if seg.shape != original.shape or not np.allclose(seg.affine, original.affine, atol=1e-3):
            raise ValueError("TopBrain label map must be on exactly the CTA grid/affine for fusion. Resample labels with nearest-neighbour interpolation first.")
        labels = np.asarray(seg.dataobj)
        mask = mask | (labels > 0)
        message += f" | Hybrid: {int(mask.sum()):,} voxels"

    # Adjust attenuation by altering the HU values, so NanoDRR performs its own
    # standard HU -> mu conversion. This is an approximate iodine-boost model.
    gain = float(params["iodine_gain"])
    adjusted = hu.copy()
    adjusted[mask] = 40.0 + gain * np.maximum(hu[mask] - 40.0, 0.0)

    # Preserve original affine and header; never overwrite the input file.
    # An uncompressed temporary NIfTI is faster than repeatedly saving .nii.gz.
    # The mask is attached as a 0/1 labelmap on the same grid, so one render gives
    # channel 0 = non-vessel attenuation and channel 1 = vessel attenuation,
    # both with exactly the DRR camera geometry.
    with tempfile.TemporaryDirectory(prefix="nanodrr_vessel_") as folder:
        out_path = Path(folder) / "enhanced_cta.nii"
        mask_path = Path(folder) / "vessel_mask.nii"
        nib.save(nib.Nifti1Image(adjusted, original.affine, original.header), str(out_path))
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), original.affine), str(mask_path))
        enhanced_subject = Subject.from_filepath(
            str(out_path), str(mask_path), max_label=1
        ).to(device)

    cache.clear()
    cache.update(key=signature, subject=enhanced_subject, message=message)
    return enhanced_subject, message


def vessel_projection_to_rgb(vessel_2d):
    """Display the projected vessel channel: black background, vessels bright."""
    vessel_2d = np.asarray(vessel_2d, dtype=np.float32)
    inside = vessel_2d > 0
    if not np.any(inside):
        return greyscale_to_rgb_uint8(np.zeros_like(vessel_2d))
    high = np.percentile(vessel_2d[inside], 99.5)
    return greyscale_to_rgb_uint8(vessel_2d / max(float(high), 1e-6))


# ---------- enhancement pipeline ----------
def _blur(x):
    k = torch.tensor([1., 4., 6., 4., 1.], device=x.device) / 16.0
    kx = k.view(1, 1, 1, 5)
    ky = k.view(1, 1, 5, 1)
    x = F.conv2d(F.pad(x, (2, 2, 0, 0), mode="reflect"), kx)
    x = F.conv2d(F.pad(x, (0, 0, 2, 2), mode="reflect"), ky)
    return x

def _up(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

def laplacian_pyramid(x, levels):
    details = []
    for _ in range(levels):
        low = _blur(x)[..., ::2, ::2]
        details.append(x - _up(low, x.shape[-2:]))
        x = low
    return details, x

def reconstruct(details, residual):
    x = residual
    for d in reversed(details):
        x = _up(x, d.shape[-2:]) + d
    return x

def add_quantum_noise(L, n0, seed=None):
    """L: [1,1,H,W] line integral. Returns noisy log image (higher = denser)."""
    L = (L - L.min()).clamp(min=0)          # air -> 0
    gen = None
    if seed is not None:
        gen = torch.Generator(device=L.device).manual_seed(int(seed))
    counts = torch.poisson(n0 * torch.exp(-L), generator=gen) / n0
    return -torch.log(counts.clamp(min=0.5 / n0))

def musica(x, levels=7, p=0.5, contrast_scale=0.05, noise_floor=0.003,
           max_gain=10.0, level_gain=(1, 1, 1, 1, 0.6, 0.35, 0.2),
           residual_gain=0.1):
    levels = min(levels, int(np.log2(min(x.shape[-2:]))) - 3)
    details, res = laplacian_pyramid(x, levels)
    out = []
    for i, d in enumerate(details):
        g = (d.abs().clamp(min=noise_floor) / contrast_scale) ** (p - 1.0)
        g = g.clamp(max=max_gain)
        lg = level_gain[i] if i < len(level_gain) else level_gain[-1]
        out.append(d * g * lg)
    mean = res.mean()
    res = mean + residual_gain * (res - mean)   # compress coarse dynamic range
    return reconstruct(out, res)

def window_01(x, lo=0.2, hi=99.8):
    a = x.detach().cpu().numpy()[0, 0]
    l, h = np.percentile(a, [lo, hi])
    if h <= l:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - l) / (h - l), 0, 1).astype(np.float32)

def process_drr(L, params):
    """L: raw render tensor [1,1,H,W]. Returns float image in [0,1], dense = high."""
    if not params.get("enhance", False):
        img = L[0, 0].detach().cpu().numpy()
        return normalise_for_display(img, params["window_low"], params["window_high"])
    n0 = float(params.get("n0", 1e5))
    log_img = add_quantum_noise(L.float(), n0, seed=params.get("noise_seed", 0))
    enh = musica(
        log_img,
        p=float(params.get("musica_p", 0.5)),
        residual_gain=float(params.get("residual_gain", 0.1)),
        noise_floor=float(params.get("noise_floor", 0.003)),
        contrast_scale=float(params.get("contrast_scale", 0.05)),
        max_gain=float(params.get("max_gain", 10.0)),
    )
    return window_01(enh, lo=float(params.get("window_low", 0.2)),
                     hi=float(params.get("window_high", 99.8)))

# render the image optionally with the label overlay if the user has selected to show labels
def render_pose_rgb(
        subject_ct,
        subject_labels,
        subject_isocenter,
        device,
        params,
        image_path=None,
        label_path=None,
        vessel_cache=None,
):
    height = int(params["height"])
    width = int(params["width"])

    k_inv, rt_inv, sdd = make_geometry(
        device=device,
        subject_isocenter=subject_isocenter,
        rot_x=float(params["rot_x"]),
        rot_y=float(params["rot_y"]),
        rot_z=float(params["rot_z"]),
        trans_x=float(params["trans_x"]),
        trans_y=float(params["trans_y"]),
        trans_z=float(params["trans_z"]),
        sdd_value=float(params["sdd"]),
        height=height,
        width=width,
        pixel_size=float(params["pixel_size"]),
    )

    mask_preview = None
    mask_message = ""
    render_subject = subject_ct
    if params.get("enable_hu_vessels", False):
        render_subject, mask_message = _prepare_vessel_subject(
            image_path, label_path, params, device, vessel_cache
        )
    L = render_ct_projection(render_subject, k_inv, rt_inv, sdd, height, width)
    # With a labelmap, NanoDRR splits each ray integral by label; the full DRR
    # is the sum over channels. The vessel subject's channel 1 is the vessel
    # projection, already aligned with the DRR.
    if L.shape[1] > 1:
        if params.get("enable_hu_vessels", False):
            mask_preview = vessel_projection_to_rgb(L[0, 1].detach().cpu().numpy())
        L = L.sum(dim=1, keepdim=True)
    ct_np = process_drr(L, params)

    if bool(params.get("invert", True)):
        ct_np = 1.0 - ct_np

    # Apply gamma on BOTH raw and enhanced paths so it always responds in the GUI.
    ct_np = np.power(np.clip(ct_np, 0, 1), float(params.get("gamma", 0.7)))
    rgb = greyscale_to_rgb_uint8(ct_np)


    if params.get("show_segmentations", False):
        if subject_labels is not None:
            label_channels = render_label_channels(
                subject_labels=subject_labels,
                k_inv=k_inv,
                rt_inv=rt_inv,
                sdd=sdd,
                height=height,
                width=width,
            )

            visible_label_ids = params.get("visible_label_ids", None)

            rgb = overlay_labels_channels_on_rgb(
                rgb=rgb,
                label_channels=label_channels,
                visible_label_ids=visible_label_ids,
                label_id_to_channel=params.get("label_id_to_channel", None),
                fill_alpha=float(params.get("fill_alpha", 0.30)),
                border_width=int(params.get("border_width", 2)),
                label_threshold=float(params.get("label_threshold", 0.0)),
            )

    return rgb, mask_preview, mask_message
    
# ============================================================
# PySide helper widgets
# ============================================================

# QSlider with numeric label for floating-point values

class FloatSlider(QWidget):
    value_changed = Signal(float)

    def __init__(self, name, minimum, maximum, default, step=1.0, decimals=1):
        super().__init__()

        self.name = name
        self.scale = int(round(1.0 / step)) if step < 1.0 else 1
        self.step = step
        self.decimals = decimals

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(int(round(minimum * self.scale)))
        self.slider.setMaximum(int(round(maximum * self.scale)))
        self.slider.setValue(int(round(default * self.scale)))

        # live movement while dragging
        self.slider.setTracking(True)
        
        self.slider.setSingleStep(max(1, int(round(step * self.scale))))
        self.slider.setPageStep(max(1, int(round(step*self.scale * 5))))

        self.label = QLabel()
        self.label.setMinimumWidth(160)
        
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        self.setLayout(layout)

        self.slider.valueChanged.connect(self._emit_value)
        self._update_label(default)
    
    def value(self):
        return self.slider.value() / self.scale
    
    def set_value(self, value):
        self.slider.setValue(int(round(value * self.scale)))
    
    def _update_label(self, value):
        self.label.setText(f"{self.name}: {value:.{self.decimals}f}")
    
    def _emit_value(self, raw_value):
        value = raw_value / self.scale
        self._update_label(value)
        self.value_changed.emit(float(value))


# ============================================================
# Render worker
# ============================================================
 
class RenderSignals(QObject):
    result = Signal(object, float)
    error = Signal(str)
    finished = Signal()

#collects commands from the sliders so that the gui looks smooth
class RenderWorker(QRunnable):
    def __init__(
            self,
            subject_ct,
            subject_labels,
            subject_isocenter,
            device,
            params,
            image_path=None,
            label_path=None,
            vessel_cache=None,
    ):
        super().__init__()

        self.subject_ct = subject_ct
        self.subject_labels = subject_labels
        self.subject_isocenter = subject_isocenter
        self.device = device
        self.params = params
        self.image_path = image_path
        self.label_path = label_path
        self.vessel_cache = vessel_cache

        self.signals = RenderSignals()
    
    def run(self):
        try:
            start = time.time()

            rgb, mask_preview, mask_message = render_pose_rgb(
                subject_ct=self.subject_ct,
                subject_labels=self.subject_labels,
                subject_isocenter=self.subject_isocenter,
                device=self.device,
                params=self.params,
                image_path=self.image_path,
                label_path=self.label_path,
                vessel_cache=self.vessel_cache,
            )
            
            elapsed = time.time() - start

            self.signals.result.emit((rgb, mask_preview, mask_message), elapsed)

        except Exception:
            self.signals.error.emit(traceback.format_exc())
        
        finally:
            self.signals.finished.emit()


# ============================================================
# Main window
# ============================================================

class FluoroControlWindow(QMainWindow):
    def __init__(self, image_path, label_path, mapping_path):
        super().__init__()

        self.setWindowTitle("nanoDRR Fluoroscopy Simulator: C-Arm Control")
        self.resize(1600, 820)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.image_path = image_path
        self.label_path = label_path
        self.mapping_path = mapping_path

        self.subject_ct = None
        self.subject_labels = None
        self.subject_isocenter = None

        self.threadpool = QThreadPool.globalInstance()
        self.render_running = False
        self.render_requested = False

        self.current_rgb = None
        self.current_mask_preview = None
        self.vessel_cache = {}

        self.label_id_to_name, self.label_name_to_id = load_label_mapping(self.mapping_path)
        self.label_id_to_channel = build_label_id_to_channel(self.label_id_to_name)
        self.label_checkboxes = {}

        starting_visible_ids = choose_starting_visible_label_ids(
            self.label_id_to_name
        )
        
        self.params = {
            "rot_x": 0.0,
            "rot_y": 0.0,
            "rot_z": 0.0,
            "trans_x": 0.0,
            "trans_y": 850.0,
            "trans_z": 0.0,
            "sdd": 1020.0,
            # Match the DeepDRR comparison defaults.
            "pixel_size": 0.6,
            "height": 512,
            "width": 512,

            # Segmentation overlay settings
            "show_segmentations": False,
            "fill_alpha": 0.30,
            "border_width": 2,
            "label_threshold": 0.0,
            "visible_label_ids": starting_visible_ids,
            "label_id_to_channel": self.label_id_to_channel,

            "invert": True,
            "window_low": 5.0,
            "window_high": 99.7,
            "gamma": 0.7,

            # Enhancement settings
            "enhance": False,
            "n0": 1e5,
            "musica_p": 0.5,
            "residual_gain": 0.1,
            "noise_seed": 0,
            "noise_floor": 0.003,
            "contrast_scale": 0.05,
            "max_gain": 10.0,
            # Optional 3D HU vessel extraction / anatomical label fusion
            "enable_hu_vessels": False,
            "fuse_topbrain": False,
            "vessel_threshold": 150.0,
            "bone_threshold": 700.0,
            "soft_lower": -20.0,
            "soft_upper": 120.0,
            "opening_mm": 4.0,
            "closing_mm": 3.0,
            "bone_exclusion_mm": 1.0,
            "vessel_dilation_mm": 0.5,
            "final_bone_exclusion_mm": 0.5,
            "iodine_gain": 10.0,
        }

        self._load_subjects()
        self._build_ui()

        self.request_render()
    
    def _load_subjects(self):
        print("Using device: ", self.device)

        if self.image_path is None:
            raise FileNotFoundError("No CT image path specified.")
        
        if not self.image_path.exists():
            raise FileNotFoundError(f"CT image file not found: {self.image_path}")
        
        # Load CT-only subject
        self.subject_ct = Subject.from_filepath(str(self.image_path)).to(self.device)

        # Load CT and segmentation subject when segmentation is selected
        if self.label_path is not None:
            if not self.label_path.exists():
                raise FileNotFoundError(f"Label file does not exist: {self.label_path}")
            
            self.subject_labels = Subject.from_filepath(
                str(self.image_path),
                str(self.label_path),
            ).to(self.device)

            # nanoDRR creates segmentation channels in sorted order of labels
            # that are actually present.  Filtering here prevents a missing
            # vessel ID from shifting every later label/colour association.
            if self.label_path.name.endswith((".nii", ".nii.gz")):
                label_values = np.asarray(nib.load(str(self.label_path)).dataobj)
                rounded = np.rint(label_values)
                if not np.allclose(label_values, rounded, atol=1e-4, rtol=0.0):
                    raise ValueError("The segmentation contains non-integer label values.")
                present_ids = sorted(int(x) for x in np.unique(rounded) if int(x) > 0)
                if self.label_id_to_name:
                    self.label_id_to_name = {
                        label_id: self.label_id_to_name.get(label_id, f"label_{label_id}")
                        for label_id in present_ids
                    }
                    self.label_name_to_id = {
                        name: label_id for label_id, name in self.label_id_to_name.items()
                    }
                else:
                    self.label_id_to_name = {
                        label_id: f"label_{label_id}" for label_id in present_ids
                    }
                    self.label_name_to_id = {
                        name: label_id for label_id, name in self.label_id_to_name.items()
                    }
                self.label_id_to_channel = {
                    label_id: label_id -1
                    for label_id in present_ids
                    }
                self.params["label_id_to_channel"] = self.label_id_to_channel
                visible = self.params.get("visible_label_ids")
                if visible is None:
                    self.params["visible_label_ids"] = set(present_ids)
                else:
                    self.params["visible_label_ids"] = set(visible) & set(present_ids)

            print("Loaded CT subject")
            print("Loaded label subject")

        else:
            self.subject_labels = None
            print("Loaded CT subject only (no labels)")

        # save the isocenter for later use in camera geometry
        self.subject_isocenter = self.subject_ct.isocenter

        print("Subject isocenter:", self.subject_isocenter.detach().cpu().numpy())
    
    def _build_ui(self):
        central = QWidget()
        outer_layout = QVBoxLayout(central)
        self.setCentralWidget(central)

        # Always-visible toolbar: panels can be restored even when hidden.
        toolbar = QHBoxLayout()
        self.toggle_controls_button = QPushButton("Hide controls")
        self.toggle_controls_button.setCheckable(True)
        self.toggle_controls_button.setChecked(True)
        self.toggle_controls_button.toggled.connect(self.set_controls_visible)
        toolbar.addWidget(self.toggle_controls_button)

        self.toggle_mask_button = QPushButton("Hide vessel mask")
        self.toggle_mask_button.setCheckable(True)
        self.toggle_mask_button.setChecked(True)
        self.toggle_mask_button.toggled.connect(self.set_mask_visible)
        toolbar.addWidget(self.toggle_mask_button)

        toolbar.addStretch()
        outer_layout.addLayout(toolbar)

        self.main_row = QHBoxLayout()
        outer_layout.addLayout(self.main_row, stretch=1)

        controls_panel = self._make_controls_panel()
        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setWidget(controls_panel)
        self.controls_scroll.setFixedWidth(430)
        self.main_row.addWidget(self.controls_scroll, stretch=0)

        image_panel = self._make_image_panel()
        self.main_row.addWidget(image_panel, stretch=1)

    def set_controls_visible(self, visible):
        self.controls_scroll.setVisible(bool(visible))
        self.toggle_controls_button.setText("Hide controls" if visible else "Show controls")
        self._refresh_visible_images()

    def set_mask_visible(self, visible):
        self.mask_panel.setVisible(bool(visible))
        self.toggle_mask_button.setText("Hide vessel mask" if visible else "Show vessel mask")
        self._refresh_visible_images()

    def _refresh_visible_images(self):
        # Rescale after Qt has processed the new layout geometry.
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self._rescale_visible_images)

    def _rescale_visible_images(self):
        if self.current_rgb is not None:
            self.update_image_label(self.current_rgb)
        if self.mask_panel.isVisible() and self.current_mask_preview is not None:
            self.update_mask_preview_label()

    def _make_controls_panel(self):
        panel = QWidget()
        panel.setFixedWidth(390)

        layout = QVBoxLayout()
        panel.setLayout(layout)

        device_text = f"Device: {self.device}"
        if self.device.type == "cuda":
            device_text += f" (GPU- {torch.cuda.get_device_name(0)})"

        self.device_label = QLabel(device_text)
        self.device_label.setWordWrap(True)
        layout.addWidget(self.device_label)

        # input CT and segmentation files
        file_group = QGroupBox("CT and Segmentation Input Files")
        file_layout = QVBoxLayout()
        file_group.setLayout(file_layout)

        self.image_path_edit = QLineEdit(str(self.image_path))
        self.label_path_edit = QLineEdit("" if self.label_path is None else str(self.label_path))
        self.mapping_path_edit = QLineEdit(
            "" if self.mapping_path is None else str(self.mapping_path)
        )

        self.image_path_edit.setReadOnly(True)
        self.label_path_edit.setReadOnly(True)
        self.mapping_path_edit.setReadOnly(True)

        browse_image_button = QPushButton("Browse CT Image")
        browse_label_button = QPushButton("Browse Segmentations")
        browse_mapping_button = QPushButton("Browse vessel mapping")
        clear_label_button = QPushButton("Clear Segmentations")
        reload_files_button = QPushButton("Load selected files")

        browse_image_button.clicked.connect(self.browse_image_file)
        browse_label_button.clicked.connect(self.browse_label_file)
        browse_mapping_button.clicked.connect(self.browse_mapping_file)
        clear_label_button.clicked.connect(self.clear_label_file)
        reload_files_button.clicked.connect(self.reload_selected_files)

        file_layout.addWidget(QLabel("CT Image:"))
        file_layout.addWidget(self.image_path_edit)
        file_layout.addWidget(browse_image_button)

        file_layout.addWidget(QLabel("Segmentation:"))
        file_layout.addWidget(self.label_path_edit)
        file_layout.addWidget(browse_label_button)
        file_layout.addWidget(clear_label_button)

        file_layout.addWidget(QLabel("Vessel label mapping JSON/CSV:"))
        file_layout.addWidget(self.mapping_path_edit)
        file_layout.addWidget(browse_mapping_button)

        file_layout.addWidget(reload_files_button)

        layout.addWidget(file_group)

        seg_group = QGroupBox("Segmentation overlay")
        seg_layout = QVBoxLayout()
        seg_group.setLayout(seg_layout)

        self.seg_button = QPushButton("Impose Segmentations")
        self.seg_button.setCheckable(True)

        if self.subject_labels is None:
            self.seg_button.setEnabled(False)
            self.seg_button.setToolTip("No segmentation file loaded.")
        
        # control the transparency of the segmentation overlay - similar to ITKsnap
        self.fill_alpha_box = QDoubleSpinBox()
        self.fill_alpha_box.setRange(0.0, 1.0)
        self.fill_alpha_box.setSingleStep(0.05)
        self.fill_alpha_box.setDecimals(2)
        self.fill_alpha_box.setValue(self.params["fill_alpha"])

        # control the width of the segmentation overlay border
        self.border_width_box = QSpinBox()
        self.border_width_box.setRange(1, 6)
        self.border_width_box.setSingleStep(1)
        self.border_width_box.setValue(self.params["border_width"])

        seg_layout.addWidget(self.seg_button)
        seg_layout.addWidget(QLabel("Fill opacity"))
        seg_layout.addWidget(self.fill_alpha_box)
        seg_layout.addWidget(QLabel("Border width"))
        seg_layout.addWidget(self.border_width_box)

        label_buttons = QHBoxLayout()
        select_all_button = QPushButton("All")
        clear_all_button = QPushButton("None")
        select_all_button.clicked.connect(self.select_all_labels)
        clear_all_button.clicked.connect(self.clear_all_labels)
        label_buttons.addWidget(select_all_button)
        label_buttons.addWidget(clear_all_button)
        seg_layout.addLayout(label_buttons)

        label_scroll = QScrollArea()
        label_scroll.setWidgetResizable(True)
        label_scroll.setFixedHeight(220)
        label_widget = QWidget()
        self.label_list_layout = QVBoxLayout(label_widget)
        label_scroll.setWidget(label_widget)
        seg_layout.addWidget(QLabel("Visible labels"))
        seg_layout.addWidget(label_scroll)
        self._rebuild_label_list()
        
        layout.addWidget(seg_group)

        # Fluoroscopy display settings
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout()
        display_group.setLayout(display_layout)

        self.invert_checkbox = QCheckBox("Invert fluoroscopy-style display")
        self.invert_checkbox.setChecked(bool(self.params["invert"]))
        self.invert_checkbox.toggled.connect(self.set_invert_display)

        display_layout.addWidget(self.invert_checkbox)

        self.window_low_box = QDoubleSpinBox()
        self.window_low_box.setRange(0.0, 95.0)
        self.window_low_box.setDecimals(1)
        self.window_low_box.setSingleStep(1.0)
        self.window_low_box.setValue(self.params["window_low"])
        display_layout.addWidget(QLabel("Lower percentile (window)"))
        display_layout.addWidget(self.window_low_box)

        self.window_high_box = QDoubleSpinBox()
        self.window_high_box.setRange(95.1, 100.0)
        self.window_high_box.setDecimals(1)
        self.window_high_box.setSingleStep(0.1)
        self.window_high_box.setValue(self.params["window_high"])
        display_layout.addWidget(QLabel("Upper percentile (window)"))
        display_layout.addWidget(self.window_high_box)

        self.gamma_box = QDoubleSpinBox()
        self.gamma_box.setRange(0.1, 3.0)
        self.gamma_box.setDecimals(2)
        self.gamma_box.setSingleStep(0.05)
        self.gamma_box.setValue(self.params["gamma"])
        display_layout.addWidget(QLabel("Gamma (< 1 brightens)"))
        display_layout.addWidget(self.gamma_box)

        self.window_low_box.valueChanged.connect(
            lambda v: self.set_param("window_low", float(v)))
        self.window_high_box.valueChanged.connect(
            lambda v: self.set_param("window_high", float(v)))
        self.gamma_box.valueChanged.connect(
            lambda v: self.set_param("gamma", float(v)))

        #Enhancement controls
        self.enhance_checkbox = QCheckBox("Enhancement (noise + MUSICA-like)")
        self.enhance_checkbox.setChecked(self.params["enhance"])
        self.enhance_checkbox.toggled.connect(lambda c: self.set_param("enhance", bool(c)))
        display_layout.addWidget(self.enhance_checkbox)

        self.n0_box = QDoubleSpinBox()
        self.n0_box.setRange(1e3, 1e7)
        self.n0_box.setDecimals(0)
        self.n0_box.setSingleStep(1e4)
        self.n0_box.setValue(self.params["n0"])
        self.n0_box.valueChanged.connect(lambda v: self.set_param("n0", float(v)))
        display_layout.addWidget(QLabel("Photons per pixel (lower = noisier)"))
        display_layout.addWidget(self.n0_box)

        self.p_box = QDoubleSpinBox()
        self.p_box.setRange(0.2, 1.0)
        self.p_box.setSingleStep(0.05)
        self.p_box.setValue(self.params["musica_p"])
        self.p_box.valueChanged.connect(lambda v: self.set_param("musica_p", float(v)))
        display_layout.addWidget(QLabel("Enhancement power p (lower = stronger)"))
        display_layout.addWidget(self.p_box)

        self.res_box = QDoubleSpinBox()
        self.res_box.setRange(0.0, 1.0)
        self.res_box.setSingleStep(0.05)
        self.res_box.setValue(self.params["residual_gain"])
        self.res_box.valueChanged.connect(lambda v: self.set_param("residual_gain", float(v)))
        display_layout.addWidget(QLabel("Low-frequency gain (lower = flatter)"))
        display_layout.addWidget(self.res_box)

        # Additional MUSICA controls. n0, p, residual gain and gamma exist above.
        def add_float_control(group_layout, key, label, low, high, increment, decimals=3):
            box = QDoubleSpinBox()
            box.setRange(low, high)
            box.setDecimals(decimals)
            box.setSingleStep(increment)
            box.setValue(float(self.params[key]))
            box.valueChanged.connect(lambda value, k=key: self.set_param(k, float(value)))
            group_layout.addWidget(QLabel(label))
            group_layout.addWidget(box)
            return box

        self.noise_floor_box = add_float_control(
            display_layout, "noise_floor", "MUSICA noise floor", 0.0001, 0.2, 0.001, 4)
        self.contrast_scale_box = add_float_control(
            display_layout, "contrast_scale", "MUSICA contrast scale", 0.001, 1.0, 0.005, 4)
        self.max_gain_box = add_float_control(
            display_layout, "max_gain", "Maximum MUSICA gain", 1.0, 50.0, 0.5, 2)

        layout.addWidget(display_group)

        vessels_group = QGroupBox("3D HU Vessel Extraction (optional)")
        vessels_layout = QVBoxLayout()
        vessels_group.setLayout(vessels_layout)
        note = QLabel("Extract high-HU vascular candidates, optionally union with loaded labels, and rerender modified CTA attenuation. Changes can take longer than gamma adjustments.")
        note.setWordWrap(True)
        vessels_layout.addWidget(note)
        self.vessels_checkbox = QCheckBox("Enable HU vessel attenuation")
        self.vessels_checkbox.setChecked(self.params["enable_hu_vessels"])
        self.vessels_checkbox.toggled.connect(lambda v: self.set_param("enable_hu_vessels", bool(v)))
        vessels_layout.addWidget(self.vessels_checkbox)
        self.fusion_checkbox = QCheckBox("Include loaded TopBrain/label segmentation in mask")
        self.fusion_checkbox.setChecked(self.params["fuse_topbrain"])
        self.fusion_checkbox.toggled.connect(lambda v: self.set_param("fuse_topbrain", bool(v)))
        vessels_layout.addWidget(self.fusion_checkbox)
        self.vessel_boxes = {}
        for key, label, lo, hi, inc, dec in [
            ("vessel_threshold", "Vessel lower HU", 0, 700, 10, 0),
            ("bone_threshold", "Bone threshold HU", 150, 2000, 25, 0),
            ("soft_lower", "Brain soft-tissue lower HU", -200, 200, 5, 0),
            ("soft_upper", "Brain soft-tissue upper HU", 0, 500, 5, 0),
            ("opening_mm", "Brain opening radius (mm)", 0, 12, 0.5, 2),
            ("closing_mm", "Brain closing radius (mm)", 0, 12, 0.5, 2),
            ("bone_exclusion_mm", "Bone exclusion distance (mm)", 0, 6, 0.25, 2),
            ("vessel_dilation_mm", "Vessel expansion (mm)", 0, 4, 0.25, 2),
            ("final_bone_exclusion_mm", "Final bone exclusion (mm)", 0, 4, 0.25, 2),
            ("iodine_gain", "Iodine attenuation gain (x)", 0, 30, 0.5, 2),
        ]:
            self.vessel_boxes[key] = add_float_control(vessels_layout, key, label, lo, hi, inc, dec)
        layout.addWidget(vessels_group)

        pose_group = QGroupBox("C-Arm Pose Controls")
        pose_layout = QVBoxLayout()
        pose_group.setLayout(pose_layout)

        self.rot_x_slider = FloatSlider("Rotation X (deg)", -90.0, 90.0, self.params["rot_x"], step=1.0)
        self.rot_y_slider = FloatSlider("Rotation Y (deg)", -90.0, 90.0, self.params["rot_y"], step=1.0)
        self.rot_z_slider = FloatSlider("Rotation Z (deg)", -180.0, 180.0, self.params["rot_z"], step=1.0)

        self.trans_x_slider = FloatSlider("Translation X (mm)", -500.0, 500.0, self.params["trans_x"], step=10.0)
        self.trans_y_slider = FloatSlider("Translation Y (mm)", 200.0, 1500.0, self.params["trans_y"], step=10.0)
        self.trans_z_slider = FloatSlider("Translation Z (mm)", -500.0, 500.0, self.params["trans_z"], step=10.0)

        views_row = QHBoxLayout()
        for name, angle in (("AP", 0.0), ("Lateral", -90.0), ("Oblique +30", 30.0), ("Oblique -30", -30.0)):
            button = QPushButton(name)
            button.clicked.connect(lambda checked=False, a=angle: self.set_view_angle(a))
            views_row.addWidget(button)
        pose_layout.addLayout(views_row)
        pose_layout.addWidget(self.rot_x_slider)
        pose_layout.addWidget(self.rot_y_slider)
        pose_layout.addWidget(self.rot_z_slider)
        pose_layout.addWidget(self.trans_x_slider)
        pose_layout.addWidget(self.trans_y_slider)
        pose_layout.addWidget(self.trans_z_slider)

        layout.addWidget(pose_group)

        detector_group = QGroupBox("Detector Controls")
        detector_layout = QVBoxLayout()
        detector_group.setLayout(detector_layout)
        
        self.sdd_box = QDoubleSpinBox()
        self.sdd_box.setRange(500.0, 2000.0)
        self.sdd_box.setSingleStep(10.0)
        self.sdd_box.setValue(self.params["sdd"])
        detector_layout.addWidget(QLabel("SDD"))
        detector_layout.addWidget(self.sdd_box)

        self.pixel_size_box = QDoubleSpinBox()
        self.pixel_size_box.setRange(0.05, 5.0)
        self.pixel_size_box.setSingleStep(0.01)
        self.pixel_size_box.setDecimals(4)
        self.pixel_size_box.setValue(self.params["pixel_size"])
        detector_layout.addWidget(QLabel("Pixel size (mm)"))
        detector_layout.addWidget(self.pixel_size_box)

        self.height_box = QSpinBox()
        self.height_box.setRange(128, 1024)
        self.height_box.setSingleStep(64)
        self.height_box.setValue(self.params["height"])
        detector_layout.addWidget(QLabel("Height"))
        detector_layout.addWidget(self.height_box)

        self.width_box = QSpinBox()
        self.width_box.setRange(128, 1024)
        self.width_box.setSingleStep(64)
        self.width_box.setValue(self.params["width"])
        detector_layout.addWidget(QLabel("Width"))
        detector_layout.addWidget(self.width_box)

        layout.addWidget(detector_group)

        reset_button = QPushButton("Reset pose")
        reset_button.clicked.connect(self.reset_pose)
        layout.addWidget(reset_button)

        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch()

        # connect signals to request render when parameters change
        self.rot_x_slider.value_changed.connect(lambda v: self.set_param("rot_x", v))
        self.rot_y_slider.value_changed.connect(lambda v: self.set_param("rot_y", v))
        self.rot_z_slider.value_changed.connect(lambda v: self.set_param("rot_z", v))
        self.trans_x_slider.value_changed.connect(lambda v: self.set_param("trans_x", v))
        self.trans_y_slider.value_changed.connect(lambda v: self.set_param("trans_y", v))
        self.trans_z_slider.value_changed.connect(lambda v: self.set_param("trans_z", v))   

        self.sdd_box.valueChanged.connect(lambda v: self.set_param("sdd", float(v)))
        self.pixel_size_box.valueChanged.connect(lambda v: self.set_param("pixel_size", float(v)))  
        self.height_box.valueChanged.connect(lambda v: self.set_param("height", int(v)))
        self.width_box.valueChanged.connect(lambda v: self.set_param("width", int(v)))

        self.seg_button.toggled.connect(self.set_segmentation_overlay)
        self.fill_alpha_box.valueChanged.connect(lambda v: self.set_param("fill_alpha", float(v)))
        self.border_width_box.valueChanged.connect(lambda v: self.set_param("border_width", int(v)))

        return panel

    def _make_image_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Two equally sized image areas: simulated DRR on the left and
        # the vessel mask rendered with the same camera geometry on the right.
        images_row = QHBoxLayout()
        images_row.setSpacing(10)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        fluoro_title = QLabel("Live Simulated Fluoroscopy / DRR")
        fluoro_title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(fluoro_title)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(100, 120)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setStyleSheet("background-color: black;")
        left_layout.addWidget(self.image_label, stretch=1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        preview_title = QLabel("Extracted Vessel Mask (projected with DRR geometry)")
        preview_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(preview_title)
        self.mask_preview_label = QLabel("Enable HU vessel attenuation to view mask")
        self.mask_preview_label.setAlignment(Qt.AlignCenter)
        self.mask_preview_label.setMinimumSize(100, 120)
        self.mask_preview_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.mask_preview_label.setStyleSheet("background-color: black; color: white;")
        right_layout.addWidget(self.mask_preview_label, stretch=1)

        images_row.addWidget(left, stretch=1)
        self.mask_panel = right
        images_row.addWidget(self.mask_panel, stretch=1)
        layout.addLayout(images_row, stretch=1)

        save_button = QPushButton("Save displayed image as PNG")
        save_button.clicked.connect(self.save_current_image)
        layout.addWidget(save_button)
        return panel

    def update_mask_preview_label(self):
        if self.current_mask_preview is None:
            self.mask_preview_label.clear()
            self.mask_preview_label.setText("HU vessel extraction disabled")
            return
        preview = np.ascontiguousarray(self.current_mask_preview)
        ph, pw, _ = preview.shape
        qmask = QImage(preview.data, pw, ph, 3 * pw, QImage.Format_RGB888).copy()
        self.mask_preview_label.setPixmap(
            QPixmap.fromImage(qmask).scaled(
                self.mask_preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def save_current_image(self):
        if self.current_rgb is None:
            QMessageBox.information(self, "Nothing to save", "Render an image first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save simulated fluoroscopy", "nanodrr_fluoroscopy.png", "PNG (*.png)"
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        h, w, _ = self.current_rgb.shape
        image = QImage(
            self.current_rgb.data, w, h, 3 * w, QImage.Format_RGB888
        ).copy()
        if not image.save(path, "PNG"):
            QMessageBox.critical(self, "Save failed", f"Could not save {path}")

    def browse_image_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select CT Image File",
            str(self.image_path.parent if self.image_path else Path.home()),
            "Medical images (*.nii *.nii.gz *.nrrd);;All Files (*)",
        )

        if file_path:
            self.image_path = Path(file_path)
            self.image_path_edit.setText(str(self.image_path))

    def browse_label_file(self):
        start_dir = self.image_path.parent if self.image_path else Path.home()

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Segmentation Label File",
            str(start_dir),
            "Medical images (*.nii *.nii.gz *.nrrd);;All Files (*)",
        )

        if file_path:
            self.label_path = Path(file_path)
            self.label_path_edit.setText(str(self.label_path))

    def browse_mapping_file(self):
        start_dir = self.mapping_path.parent if self.mapping_path else Path.home()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select vessel label mapping",
            str(start_dir),
            "Label mappings (*.json *.csv);;JSON (*.json);;CSV (*.csv)",
        )
        if file_path:
            self.mapping_path = Path(file_path)
            self.mapping_path_edit.setText(str(self.mapping_path))

    def _rebuild_label_list(self):
        while self.label_list_layout.count():
            item = self.label_list_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.label_checkboxes.clear()
        visible_ids = self.params.get("visible_label_ids")
        if not self.label_id_to_name:
            message = QLabel(
                "No label mapping is loaded. Load vessel_mapping.json and click "
                "'Load selected files'."
            )
            message.setWordWrap(True)
            self.label_list_layout.addWidget(message)
        else:
            for label_id, label_name in sorted(
                self.label_id_to_name.items(), key=lambda item: item[1].lower()
            ):
                checkbox = QCheckBox(f"{label_name} [ID {label_id}]")
                checkbox.setChecked(visible_ids is None or label_id in visible_ids)
                checkbox.toggled.connect(
                    lambda checked, lid=label_id: self.on_label_checkbox_toggled(lid, checked)
                )
                self.label_checkboxes[label_id] = checkbox
                self.label_list_layout.addWidget(checkbox)
        self.label_list_layout.addStretch()
        

    def clear_label_file(self):
        self.label_path = None
        self.label_path_edit.setText("")

    def reload_selected_files(self):
        try:
            self.status_label.setText("Loading selected files...")
            QApplication.processEvents()

            self.label_id_to_name, self.label_name_to_id = load_label_mapping(
                self.mapping_path
            )
            self.params["visible_label_ids"] = choose_starting_visible_label_ids(
                self.label_id_to_name
            )
            self._load_subjects()
            self.vessel_cache.clear()
            self._rebuild_label_list()

            if self.subject_labels is None:
                self.seg_button.setChecked(False)
                self.seg_button.setEnabled(False)
                self.seg_button.setToolTip("No segmentation file loaded.")
                self.params["show_segmentations"] = False
            
            else:
                self.seg_button.setEnabled(True)
                self.seg_button.setText(
                    "Hide Segmentations"
                    if self.params["show_segmentations"]
                    else "Impose Segmentations")
                
            self.request_render()
        
        except Exception as exc:
            error_text = traceback.format_exc()
            print(error_text)

            QMessageBox.critical(
                self,
                "File Load Error",
                f"Could not load the selected files:\n\n{exc}",
            )

            self.status_label.setText("Error loading files.")

    def set_param(self, key, value):
        self.params[key] = value
        self.request_render()

    def on_label_checkbox_toggled(self, label_id, checked):
        visible = self.params.get("visible_label_ids", set())

        if visible is None:
            visible = set(self.label_id_to_name.keys())
        
        visible = set(visible)

        if checked:
            visible.add(label_id)
        else:
            visible.discard(label_id)
        
        self.params["visible_label_ids"] = visible

        if self.params.get("show_segmentations", False):
            self.request_render()

    def set_visible_labels(self, label_ids):
        if label_ids is None:
            visible = None
        else: 
            visible = set(label_ids)
        
        self.params["visible_label_ids"] = visible

        # update the checkboxes to reflect the new visible labels
        for label_id, checkbox in self.label_checkboxes.items():
            checkbox.blockSignals(True)

            if visible is None:
                checkbox.setChecked(True)
            else:
                checkbox.setChecked(label_id in visible)
            
            checkbox.blockSignals(False)

        if self.params.get("show_segmentations", False):
            self.request_render()

    def select_all_labels(self):
        self.set_visible_labels(set(self.label_id_to_name.keys()))

    def clear_all_labels(self):
        self.set_visible_labels(set())

    def set_segmentation_overlay(self, checked):
        self.params["show_segmentations"] = bool(checked)

        if checked:
            self.seg_button.setText("Hide Segmentations")
        else:
            self.seg_button.setText("Impose Segmentations")
        
        self.request_render()

    def set_invert_display(self, checked):
        self.params["invert"] = bool(checked)
        self.request_render()

    def set_view_angle(self, angle):
        """Convenience presets under NanoDRR's existing AP orientation convention."""
        for widget, key, value in (
            (self.rot_x_slider, "rot_x", angle),
            (self.rot_y_slider, "rot_y", 0.0),
            (self.rot_z_slider, "rot_z", 0.0),
        ):
            widget.blockSignals(True)
            widget.set_value(value)
            widget.blockSignals(False)
            self.params[key] = value
        self.request_render()

    def reset_pose(self):
        defaults = {
            "rot_x": 0.0,
            "rot_y": 0.0,
            "rot_z": 0.0,
            "trans_x": 0.0,
            "trans_y": 850.0,
            "trans_z": 0.0,
        }

        widgets = {
            "rot_x": self.rot_x_slider,
            "rot_y": self.rot_y_slider,
            "rot_z": self.rot_z_slider,
            "trans_x": self.trans_x_slider,
            "trans_y": self.trans_y_slider,
            "trans_z": self.trans_z_slider,
        }

        # block every slider before moving any, so the reset triggers one render, not six
        for w in widgets.values():
            w.blockSignals(True)

        for key, w in widgets.items():
            w.set_value(defaults[key])

        for w in widgets.values():
            w.blockSignals(False)
        
        self.params.update(defaults)
        self.request_render()

    # render immediately if no render is running, otherwise set a flag to request a render after the current one finishes
    def request_render(self):
        if self.render_running:
            self.render_requested = True
            return
        
        self.start_render()

    def start_render(self):
        self.render_running = True
        self.render_requested = False

        params_snapshot = dict(self.params)

        if params_snapshot.get("show_segmentations", False):
            self.status_label.setText("Rendering with segmentations...")
        else:
            self.status_label.setText("Rendering...")
        
        worker = RenderWorker(
            subject_ct=self.subject_ct,
            subject_labels=self.subject_labels,
            subject_isocenter=self.subject_isocenter,
            device=self.device,
            params=params_snapshot,
            image_path=self.image_path,
            label_path=self.label_path,
            vessel_cache=self.vessel_cache,
        )

        worker.signals.result.connect(self.on_render_result)
        worker.signals.error.connect(self.on_render_error)
        worker.signals.finished.connect(self.on_render_finished)

        self.threadpool.start(worker)

    def on_render_result(self, result, elapsed):
        rgb, preview, mask_message = result
        self.current_rgb = rgb
        self.current_mask_preview = preview
        self.update_mask_preview_label()
        self.update_image_label(rgb)

        h, w = rgb.shape[:2]

        overlay_text = "overlay ON" if self.params["show_segmentations"] else "overlay OFF"

        visible_ids = self.params.get("visible_label_ids", None)
        if visible_ids is None:
            selected_text = "all labels"
        else:
            selected_text = f"{len(visible_ids)} selected"
        
        self.status_label.setText(
            f"Rendered on {self.device} | {w} x{h} | "
            f"{elapsed: .3f} s | {overlay_text} | {selected_text} | {mask_message}"
        )

    def on_render_error(self, error_text):
        print(error_text)
        self.status_label.setText("Render failed. see terminal for traceback,")

    def on_render_finished(self):
        self.render_running = False

        if self.render_requested:
            self.start_render()

    def update_image_label(self, rgb):
        h, w, c = rgb.shape

        if c != 3:
            raise ValueError(f"Expected RGB image with 3 channels, got {c} channels.")
        
        bytes_per_line = 3 * w

        q_image = QImage(
            rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888,
        ).copy()
        
        pixmap = QPixmap.fromImage(q_image)

        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if self.current_rgb is not None:
            self.update_image_label(self.current_rgb)
        if self.current_mask_preview is not None:
            self.update_mask_preview_label()

def parse_args():
    parser = argparse.ArgumentParser(description="Interactive nanoDRR vessel comparison simulator")
    parser.add_argument("--ct", type=Path, default=IMAGE_PATH, help="CT/CTA or enhanced NIfTI")
    parser.add_argument("--labels", type=Path, default=LABEL_PATH, help="Vessel label-map NIfTI")
    parser.add_argument(
        "--mapping",
        type=Path,
        default=LABEL_MAPPING_PATH,
        help="vessel_mapping.json or label_name/label_id CSV",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    app = QApplication(sys.argv)

    window = FluoroControlWindow(args.ct, args.labels, args.mapping)
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
