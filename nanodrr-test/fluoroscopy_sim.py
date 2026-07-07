from pathlib import Path
import csv
import sys
import time
import traceback
import numpy as np
import torch

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
LABEL_MAPPING_CSV = Path(
    r"C:\Users\cd26\OneDrive - King's College London\MT Anatomy Localisation Project\MT-Anatomy-Localisation-Project\metadata\topaneu_label_mapping.csv"
)

# Optional preset. This only controls what starts ticked when the app opens.
# If the preset names are not found in the current CSV, the app will simply show all labels.
# For other datasets, you can leave this as "totalsegmentator_vascular" or set it to "all".
DEFAULT_SELECTION_PRESET = "totalsegmentator_vascular"  # options: "totalsegmentator_vascular", "all"

# original will only tick these default labels then others can be be imposed when checked
DEFAULT_VISIBLE_LABEL_NAMES = [
    "aorta",
    "brachiocephalic_trunk",
    "common_carotid_artery_left",
    "common_carotid_artery_right",
    "subclavian_artery_left",
    "subclavian_artery_right",
    "superior_vena_cava",
    "heart",
    "sternum",
    "clavicula_left",
    "clavicula_right",
    "trachea",
]

# ============================================================
# Label mapping helpers
# ============================================================
def load_label_mapping():
    """
    assigned the label to the id and vice versa
    allows multiple datasets to be considered using the same script even if the names are written slightly different
    """
    if not LABEL_MAPPING_CSV.exists():
        print(f"Warning: label mapping CSV file not found at {LABEL_MAPPING_CSV}.")
        print("segmentations can still be seen but individual label selection will be unavailable")
        return {}, {}

    label_id_to_name = {}

    with open(LABEL_MAPPING_CSV, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        required = {"label_name", "label_id"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"Mapping CSV must contain columns {required}."
                f"Found columns: {reader.fieldnames}"
            )
        
        for row in reader:
            label_name = str(row["label_name"])
            label_id = int(row["label_id"])
            label_id_to_name[label_id] = label_name
    
    label_name_to_id = {
        label_name: label_id
        for label_id, label_name in label_id_to_name.items()
    }

    return label_id_to_name, label_name_to_id

# Set default labels to their ids
def default_visible_label_ids(label_name_to_id):
    ids = []
    for label_name in DEFAULT_VISIBLE_LABEL_NAMES:
        if label_name in label_name_to_id:
            ids.append(label_name_to_id[label_name])
    return set(ids)

def choose_starting_visible_label_ids(label_id_to_name, label_name_to_id):
    """
    Decide which labels are ticked when the app starts.

    - If there is no CSV mapping, return None, meaning show all label channels.
    - If DEFAULT_SELECTION_PRESET is "all", tick every label in the CSV.
    - If the TotalSegmentator vascular preset matches this CSV, tick those labels.
    - If the preset does not match this dataset, fall back to all labels instead of crashing.
    """
    if not label_id_to_name:
        return None

    if DEFAULT_SELECTION_PRESET.lower() == "all":
        return set(label_id_to_name.keys())

    preset_ids = default_visible_label_ids(label_name_to_id)

    if preset_ids:
        return preset_ids

    print("No preset/default label names found in this mapping CSV; starting with all labels selected.")
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
        int(label_id): channel_idx
        for channel_idx, label_id in enumerate(sorted(label_id_to_name.keys()))
    }

# ============================================================
# nanoDRR helper functions - taken from original script
# ============================================================
def normalise_for_display(img_2d):
    img_2d = img_2d.astype(np.float32)

    low = np.percentile(img_2d, 1)
    high = np.percentile(img_2d, 99)

    img_2d = (img_2d - low) / (high - low + 1e-8)
    img_2d = np.clip(img_2d, 0, 1)

    return img_2d

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

# renders just the CT DRR (no labels) everytime the user changes the settings on the gui
def render_ct_projection(
        subject_ct,
        k_inv,
        rt_inv,
        sdd,
        height,
        width,
):
    with torch.no_grad():
        img = render(subject_ct, k_inv, rt_inv, sdd, height, width)

    img_np = img[0, 0].detach().cpu().numpy()
    img_np = normalise_for_display(img_np)

    return img_np

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

# render the image optionally with the label overlay if the user has selected to show labels
def render_pose_rgb(
        subject_ct,
        subject_labels,
        subject_isocenter,
        device,
        params,
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

    ct_np = render_ct_projection(
        subject_ct=subject_ct,
        k_inv=k_inv,
        rt_inv=rt_inv,
        sdd=sdd,
        height=height,
        width=width
    )

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

    return rgb
    
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
    ):
        super().__init__()

        self.subject_ct = subject_ct
        self.subject_labels = subject_labels
        self.subject_isocenter = subject_isocenter
        self.device = device
        self.params = params

        self.signals = RenderSignals()
    
    def run(self):
        try:
            start = time.time()

            rgb = render_pose_rgb(
                subject_ct=self.subject_ct,
                subject_labels=self.subject_labels,
                subject_isocenter=self.subject_isocenter,
                device=self.device,
                params=self.params,
            )
            
            elapsed = time.time() - start

            self.signals.result.emit(rgb, elapsed)

        except Exception:
            self.signals.error.emit(traceback.format_exc())
        
        finally:
            self.signals.finished.emit()


# ============================================================
# Main window
# ============================================================

class FluoroControlWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("nanoDRR Fluoroscopy Simulator: C-Arm Control")
        self.resize(1250, 720)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.image_path = IMAGE_PATH
        self.label_path = LABEL_PATH

        self.subject_ct = None
        self.subject_labels = None
        self.subject_isocenter = None

        self.threadpool = QThreadPool.globalInstance()
        self.render_running = False
        self.render_requested = False

        self.current_rgb = None

        self.label_id_to_name, self.label_name_to_id = load_label_mapping()
        self.label_id_to_channel = build_label_id_to_channel(self.label_id_to_name)
        self.label_checkboxes = {}

        starting_visible_ids = choose_starting_visible_label_ids(
            self.label_id_to_name,
            self.label_name_to_id,
        )
        
        self.params = {
            "rot_x": 0.0,
            "rot_y": 0.0,
            "rot_z": 0.0,
            "trans_x": 0.0,
            "trans_y": 850.0,
            "trans_z": 0.0,
            "sdd": 1020.0,
            "pixel_size": 2.0,
            "height": 256,
            "width": 256,

            # Segmentation overlay settings
            "show_segmentations": False,
            "fill_alpha": 0.30,
            "border_width": 2,
            "label_threshold": 0.0,
            "visible_label_ids": starting_visible_ids,
            "label_id_to_channel": self.label_id_to_channel,
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
        main_layout = QHBoxLayout()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

        controls_panel = self._make_controls_panel()
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls_panel)
        controls_scroll.setFixedWidth(430)

        main_layout.addWidget(controls_scroll, stretch=0)

        image_panel = self._make_image_panel()
        main_layout.addWidget(image_panel, stretch=1)

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

        self.image_path_edit.setReadOnly(True)
        self.label_path_edit.setReadOnly(True)

        browse_image_button = QPushButton("Browse CT Image")
        browse_label_button = QPushButton("Browse Segmentations")
        clear_label_button = QPushButton("Clear Segmentations")
        reload_files_button = QPushButton("Load selected files")

        browse_image_button.clicked.connect(self.browse_image_file)
        browse_label_button.clicked.connect(self.browse_label_file)
        clear_label_button.clicked.connect(self.clear_label_file)
        reload_files_button.clicked.connect(self.reload_selected_files)

        file_layout.addWidget(QLabel("CT Image:"))
        file_layout.addWidget(self.image_path_edit)
        file_layout.addWidget(browse_image_button)

        file_layout.addWidget(QLabel("Segmentation:"))
        file_layout.addWidget(self.label_path_edit)
        file_layout.addWidget(browse_label_button)
        file_layout.addWidget(clear_label_button)

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
        self.fill_alpha_box.setSingleStep(0.5)
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

        if self.label_id_to_name:
            label_buttons = QHBoxLayout()

            select_defaults_button = QPushButton("Vascular defaults")
            select_all_button = QPushButton("All")
            clear_all_button = QPushButton("None")

            select_defaults_button.clicked.connect(self.select_default_labels)
            select_all_button.clicked.connect(self.select_all_labels)
            clear_all_button.clicked.connect(self.clear_all_labels)

            label_buttons.addWidget(select_defaults_button)
            label_buttons.addWidget(select_all_button)
            label_buttons.addWidget(clear_all_button)

            seg_layout.addLayout(label_buttons)

            label_scroll = QScrollArea()
            label_scroll.setWidgetResizable(True)
            label_scroll.setFixedHeight(220)

            label_widget = QWidget()
            label_list_layout = QVBoxLayout()
            label_widget.setLayout(label_list_layout)

            visible_ids = self.params.get("visible_label_ids", set())

            for label_id, label_name in sorted(
                self.label_id_to_name.items(),
                key=lambda item: item[1].lower(),
            ):
                checkbox = QCheckBox(f"{label_name} [ID {label_id}]")

                if visible_ids is None:
                    checkbox.setChecked(True)
                else:
                    checkbox.setChecked(label_id in visible_ids)
                
                checkbox.toggled.connect(
                    lambda checked, lid=label_id: self.on_label_checkbox_toggled(lid, checked)
                )

                self.label_checkboxes[label_id] = checkbox
                label_list_layout.addWidget(checkbox)

            label_list_layout.addStretch()
            label_scroll.setWidget(label_widget)

            seg_layout.addWidget(QLabel("Visible labels"))
            seg_layout.addWidget(label_scroll)
        else:
            seg_layout.addWidget(
                QLabel("No label mapping available. Individual label selection is disabled.\n"
                       "Overlay will show all labels if segmentation is imposed.")
            )
        
        layout.addWidget(seg_group)
        pose_group = QGroupBox("C-Arm Pose Controls")
        pose_layout = QVBoxLayout()
        pose_group.setLayout(pose_layout)

        self.rot_x_slider = FloatSlider("Rotation X (deg)", -90.0, 90.0, self.params["rot_x"], step=1.0)
        self.rot_y_slider = FloatSlider("Rotation Y (deg)", -90.0, 90.0, self.params["rot_y"], step=1.0)
        self.rot_z_slider = FloatSlider("Rotation Z (deg)", -180.0, 180.0, self.params["rot_z"], step=1.0)

        self.trans_x_slider = FloatSlider("Translation X (mm)", -500.0, 500.0, self.params["trans_x"], step=10.0)
        self.trans_y_slider = FloatSlider("Translation Y (mm)", 200.0, 1500.0, self.params["trans_y"], step=10.0)
        self.trans_z_slider = FloatSlider("Translation Z (mm)", -500.0, 500.0, self.params["trans_z"], step=10.0)

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
        self.pixel_size_box.setRange(0.5, 5.0)
        self.pixel_size_box.setSingleStep(0.1)
        self.pixel_size_box.setDecimals(1)
        self.pixel_size_box.setValue(self.params["pixel_size"])
        detector_layout.addWidget(QLabel("Pixel size (mm)"))
        detector_layout.addWidget(self.pixel_size_box)

        self.height_box = QSpinBox()
        self.height_box.setRange(128, 512)
        self.height_box.setSingleStep(64)
        self.height_box.setValue(self.params["height"])
        detector_layout.addWidget(QLabel("Height"))
        detector_layout.addWidget(self.height_box)

        self.width_box = QSpinBox()
        self.width_box.setRange(128, 512)
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
        layout = QVBoxLayout()
        panel.setLayout(layout)

        title = QLabel("Live Simulated Fluoroscopy / DRR")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setStyleSheet("background-color: black;")

        layout.addWidget(self.image_label, stretch=1)

        return panel

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
        

    def clear_label_file(self):
        self.label_path = None
        self.label_path_edit.setText("")

    def reload_selected_files(self):
        try:
            self.status_label.setText("Loading selected files...")
            QApplication.processEvents()

            self._load_subjects()

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
                f"Could not load the selected filesL \n\n{exc}",
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

    def select_default_labels(self):
        ids = choose_starting_visible_label_ids(
            self.label_id_to_name,
            self.label_name_to_id,
        )
        self.set_visible_labels(ids)

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

    def reset_pose(self):
        defaults = {
            "rot_x": 0.0,
            "rot_y": 0.0,
            "rot_z": 0.0,
            "trans_x": 0.0,
            "trans_y": 850.0,
            "trans_z": 0.0,
        }

        widgets = [
            self.rot_x_slider,
            self.rot_y_slider,
            self.rot_z_slider,
            self.trans_x_slider,
            self.trans_y_slider,
            self.trans_z_slider,
        ]

        for w in widgets:
            w.blockSignals(True)
            self.rot_x_slider.set_value(defaults["rot_x"])
            self.rot_y_slider.set_value(defaults["rot_y"])
            self.rot_z_slider.set_value(defaults["rot_z"])
            self.trans_x_slider.set_value(defaults["trans_x"])
            self.trans_y_slider.set_value(defaults["trans_y"])
            self.trans_z_slider.set_value(defaults["trans_z"]) 
        
        for w in widgets:
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
        )

        worker.signals.result.connect(self.on_render_result)
        worker.signals.error.connect(self.on_render_error)
        worker.signals.finished.connect(self.on_render_finished)

        self.threadpool.start(worker)

    def on_render_result(self, rgb, elapsed):
        self.current_rgb = rgb
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
            f"{elapsed: .3f} s | {overlay_text} | {selected_text}"
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
            self.update_image_display(self.current_rgb)

def main():
    app = QApplication(sys.argv)

    window = FluoroControlWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()