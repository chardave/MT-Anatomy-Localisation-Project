"""Interactive DeepDRR fluoroscopy simulator.

This is a DeepDRR replacement for the nanoDRR-based ``fluoroscopy_sim.py``.
It keeps the PySide6 GUI, live C-arm controls, selectable label-map overlays,
and image saving, but all CT image formation is performed by DeepDRR.

The C-arm pose has been aligned with the accompanying nanoDRR GUI:

* the zero-angle view is anterior-posterior (beam direction ``-Y``);
* the final detector array is displayed in radiological orientation, with
  superior anatomy at the top;
* the rotation controls reproduce nanoDRR's intrinsic ``ZXY`` convention;
* the physical NIfTI voxel-centre is used as the shared rotation pivot.

Translation intentionally remains a *target/isocentre shift* in DeepDRR.
Consequently, nanoDRR's default camera translation ``(0, 850, 0)`` corresponds
to DeepDRR target shift ``(0, 0, 0)`` with ``SID=850`` rather than to the same
three translation numbers.  Keeping target shift and SID separate gives the
DeepDRR controls an unambiguous physical meaning.

The segmentation overlay is intentionally separate from the attenuation
simulation: DeepDRR's current ``Projector.project_seg`` API projects mesh tags,
not voxel label maps.  This app therefore projects label-map voxel centres with
the *same DeepDRR camera projection matrix* and rasterises their silhouettes.
No nanoDRR code or dependency is used.

Example
-------
python fluoroscopy_sim_deepdrr.py \
    --ct /path/to/ct.nii.gz \
    --labels /path/to/all_labels_labelmap.nii.gz \
    --mapping /path/to/vessel_mapping.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from typing import Iterable, Optional


def configure_conda_cuda() -> None:
    """Expose the CUDA 12.8 Conda layout used by the Blackwell WSL setup.

    NVIDIA's Conda packages place headers below ``targets/x86_64-linux`` and
    the internal compiler ``cicc`` below ``nvvm/bin``.  DeepDRR/CuPy need both
    locations before they are imported.
    """
    prefix_text = os.environ.get("CONDA_PREFIX")
    if not prefix_text:
        return
    prefix = Path(prefix_text)
    target = prefix / "targets" / "x86_64-linux"
    header = target / "include" / "cuda_fp16.h"
    if not header.exists():
        return

    os.environ["CUDA_PATH"] = str(target)
    path_parts = [prefix / "nvvm" / "bin", target / "bin", prefix / "bin"]
    existing_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(
        [str(path) for path in path_parts if path.exists()] + [existing_path]
    )
    library_parts = [target / "lib", prefix / "lib"]
    existing_libraries = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(path) for path in library_parts if path.exists()] + [existing_libraries]
    )
    os.environ.setdefault("CUPY_CACHE_DIR", "/tmp/cupy-sm120-deepdrr")


configure_conda_cuda()

import numpy as np
from scipy import ndimage

try:
    import nibabel as nib
except ImportError:  # handled with a clearer message in main()
    nib = None

DEEPDRR_IMPORT_ERROR: Optional[BaseException] = None
try:
    import deepdrr
except BaseException as exc:  # report the real dependency/CUDA failure in main()
    deepdrr = None
    DEEPDRR_IMPORT_ERROR = exc

try:
    import cupy as cp
except ImportError:
    cp = None

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


# Optional defaults. Leave these as None and browse in the GUI, pass command
# line arguments, or replace them with your own Linux/WSL paths.
DEFAULT_IMAGE_PATH: Optional[Path] = None
DEFAULT_LABEL_PATH: Optional[Path] = None
DEFAULT_MAPPING_CSV: Optional[Path] = None

DEFAULT_SELECTION_PRESET = "vascular"
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

# A very large organ can contain millions of labelled voxels. Regularly
# subsampling it keeps live overlays responsive; a small 2-D splat dilation
# closes the resulting gaps. Vascular labels normally remain below this cap.
MAX_POINTS_PER_LABEL = 1_500_000


PALETTE = np.array(
    [
        [230, 25, 75],
        [60, 180, 75],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
        [250, 190, 190],
        [0, 128, 128],
        [230, 190, 255],
        [170, 110, 40],
        [255, 250, 200],
        [128, 0, 0],
        [170, 255, 195],
        [0, 0, 128],
    ],
    dtype=np.float32,
)


def load_label_mapping(path: Optional[Path]) -> tuple[dict[int, str], dict[str, int]]:
    """Load vessel ``{"labels": {name: id}}`` JSON or a mapping CSV."""
    if path is None:
        return {}, {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Label mapping file not found: {path}")

    id_to_name: dict[int, str] = {}
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        labels = payload.get("labels", payload)
        if not isinstance(labels, dict):
            raise ValueError("JSON mapping must contain an object named 'labels'.")
        for label_name, label_id in labels.items():
            label_id = int(label_id)
            if label_id > 0:
                id_to_name[label_id] = str(label_name).strip() or f"label_{label_id}"
    elif path.suffix.lower() == ".csv":
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"label_name", "label_id"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(
                    f"Mapping CSV must contain {sorted(required)}; found {reader.fieldnames}."
                )
            for row in reader:
                label_id = int(row["label_id"])
                label_name = str(row["label_name"]).strip()
                if label_id > 0:
                    id_to_name[label_id] = label_name or f"label_{label_id}"
    else:
        raise ValueError("Label mapping must be a .json or .csv file.")

    return id_to_name, {name: label_id for label_id, name in id_to_name.items()}


def choose_starting_visible_ids(
    id_to_name: dict[int, str], name_to_id: dict[str, int]
) -> Optional[set[int]]:
    if not id_to_name:
        return None
    if DEFAULT_SELECTION_PRESET.lower() == "all":
        return set(id_to_name)
    selected = {
        name_to_id[name]
        for name in DEFAULT_VISIBLE_LABEL_NAMES
        if name in name_to_id
    }
    return selected if selected else set(id_to_name)


def label_to_colour(label_id: int) -> np.ndarray:
    return PALETTE[(int(label_id) - 1) % len(PALETTE)]


def normalise_for_display(image: np.ndarray, invert: bool = True) -> np.ndarray:
    """Robustly map a DeepDRR float image to [0, 1]."""
    image = np.asarray(image, dtype=np.float32).squeeze()
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D DeepDRR image, got shape {image.shape}.")

    finite = np.isfinite(image)
    if not np.any(finite):
        raise ValueError("DeepDRR returned no finite image values.")
    values = image[finite]
    low, high = np.percentile(values, [1.0, 99.5])
    if high <= low:
        low, high = float(values.min()), float(values.max())
    out = (image - low) / (high - low + 1e-8)
    out = np.clip(out, 0.0, 1.0)
    if invert:
        out = 1.0 - out
    out[~finite] = 0.0
    return out


def greyscale_to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    grey = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    return np.ascontiguousarray(np.repeat(grey[..., None], 3, axis=2))


def mask_border(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3), bool))


def overlay_masks_on_rgb(
    rgb: np.ndarray,
    masks: dict[int, np.ndarray],
    fill_alpha: float,
    border_width: int,
) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    for label_id in sorted(masks):
        mask = masks[label_id].astype(bool)
        if not np.any(mask):
            continue
        colour = label_to_colour(label_id)
        out[mask] = (1.0 - fill_alpha) * out[mask] + fill_alpha * colour

    for label_id in sorted(masks):
        mask = masks[label_id].astype(bool)
        if not np.any(mask):
            continue
        border = mask_border(mask)
        if border_width > 1:
            border = ndimage.binary_dilation(
                border, iterations=int(border_width) - 1
            )
        out[border] = label_to_colour(label_id)

    return np.clip(out, 0, 255).astype(np.uint8)


def rotation_matrix_nanodrr_degrees(
    rot_x_control: float,
    rot_y_control: float,
    rot_z_control: float,
) -> np.ndarray:
    """Reproduce nanoDRR's ``ZXY`` Euler-control convention.

    The nanoDRR GUI passes ``[rot_x, rot_y, rot_z]`` to a ``ZXY`` intrinsic
    Euler conversion.  Therefore its control names do not equal the anatomical
    axes used internally:

    * ``rot_x`` is the Z-axis angle;
    * ``rot_y`` is the X-axis angle;
    * ``rot_z`` is the Y-axis angle.

    The resulting matrix is ``Rz(rot_x) @ Rx(rot_y) @ Ry(rot_z)``.  Retaining
    this convention here makes identical rotation-control values produce the
    same C-arm orientation in both comparison applications.
    """
    angle_z, angle_x, angle_y = np.deg2rad(
        [rot_x_control, rot_y_control, rot_z_control]
    )
    cx, sx = math.cos(angle_x), math.sin(angle_x)
    cy, sy = math.cos(angle_y), math.sin(angle_y)
    cz, sz = math.cos(angle_z), math.sin(angle_z)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rot_z @ rot_x @ rot_y


def camera_projection_matrix(camera_projection: object) -> np.ndarray:
    """Extract the 3 x 4 world-to-pixel matrix from killeengeo safely."""
    candidates = [
        camera_projection,
        getattr(camera_projection, "data", None),
        getattr(camera_projection, "matrix", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            array = np.asarray(candidate, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if array.shape == (3, 4):
            return array

    intrinsic = np.asarray(getattr(camera_projection, "intrinsic"), dtype=np.float64)
    extrinsic = np.asarray(getattr(camera_projection, "extrinsic"), dtype=np.float64)
    if intrinsic.shape == (3, 3) and extrinsic.shape == (4, 4):
        return intrinsic @ extrinsic[:3, :]
    raise ValueError("Could not extract a 3 x 4 matrix from DeepDRR camera projection.")


class LabelPointCloudCache:
    """Load a NIfTI label map and lazily cache label voxels in world space."""

    def __init__(self, path: Path, ct_shape: tuple[int, ...], ct_affine: np.ndarray):
        if nib is None:
            raise RuntimeError("nibabel is required to load the segmentation label map.")
        self.path = Path(path)
        image = nib.load(str(self.path))
        self.affine = np.asarray(image.affine, dtype=np.float64)
        self.labels = np.asanyarray(image.dataobj)
        if self.labels.ndim != 3:
            raise ValueError(f"Label map must be 3-D; got {self.labels.shape}.")
        if tuple(self.labels.shape) != tuple(ct_shape):
            raise ValueError(
                "CT and label-map arrays have different shapes: "
                f"{ct_shape} versus {self.labels.shape}. Resample the label map "
                "onto the CT grid with nearest-neighbour interpolation first."
            )
        if not np.allclose(self.affine, ct_affine, atol=1e-3, rtol=1e-5):
            raise ValueError(
                "CT and label map do not have matching NIfTI affines. Resample the "
                "label map onto the CT grid before using it as an overlay."
            )
        rounded = np.rint(self.labels)
        if not np.allclose(self.labels, rounded, atol=1e-4):
            raise ValueError("The segmentation contains non-integer label values.")
        self.labels = rounded.astype(np.int32, copy=False)
        self.present_ids = {
            int(value) for value in np.unique(self.labels) if int(value) > 0
        }
        self._world_points: dict[int, np.ndarray] = {}

    def ensure(self, label_ids: Iterable[int]) -> None:
        requested = {
            int(label_id)
            for label_id in label_ids
            if int(label_id) in self.present_ids and int(label_id) not in self._world_points
        }
        if not requested:
            return

        selected = np.isin(self.labels, list(requested))
        ijk = np.argwhere(selected)
        values = self.labels[tuple(ijk.T)] if ijk.size else np.empty(0, np.int32)

        for label_id in sorted(requested):
            coords = ijk[values == label_id]
            if len(coords) > MAX_POINTS_PER_LABEL:
                stride = int(math.ceil(len(coords) / MAX_POINTS_PER_LABEL))
                coords = coords[::stride]
            homogeneous = np.ones((len(coords), 4), dtype=np.float32)
            homogeneous[:, :3] = coords.astype(np.float32)
            world = homogeneous @ self.affine.astype(np.float32).T
            self._world_points[label_id] = np.ascontiguousarray(world)

    def project_masks(
        self,
        projection_matrix: np.ndarray,
        label_ids: Optional[Iterable[int]],
        height: int,
        width: int,
        splat_radius: int = 1,
    ) -> dict[int, np.ndarray]:
        ids = self.present_ids if label_ids is None else {int(x) for x in label_ids}
        ids &= self.present_ids
        self.ensure(ids)
        matrix_t = np.asarray(projection_matrix, dtype=np.float32).T
        masks: dict[int, np.ndarray] = {}

        for label_id in sorted(ids):
            points = self._world_points.get(label_id)
            if points is None or not len(points):
                continue
            projected = points @ matrix_t
            depth = projected[:, 2]
            valid = np.isfinite(depth) & (depth > 1e-5)
            if not np.any(valid):
                continue
            u = np.rint(projected[valid, 0] / depth[valid]).astype(np.int32)
            v = np.rint(projected[valid, 1] / depth[valid]).astype(np.int32)
            inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            if not np.any(inside):
                continue
            mask = np.zeros((height, width), dtype=bool)
            mask[v[inside], u[inside]] = True
            if splat_radius > 0:
                mask = ndimage.binary_dilation(mask, iterations=int(splat_radius))
                mask = ndimage.binary_closing(mask, iterations=1)
            masks[label_id] = mask
        return masks


class DeepDRRRenderer:
    """Own the DeepDRR volume, device, projector, and optional label cache."""

    def __init__(self):
        self.ct = None
        self.device = None
        self.projector = None
        self.label_cache: Optional[LabelPointCloudCache] = None
        self.ct_path: Optional[Path] = None
        self.label_path: Optional[Path] = None
        self.pose_center_anatomical: Optional[np.ndarray] = None
        self._lock = threading.RLock()

    @property
    def ready(self) -> bool:
        return self.projector is not None

    def free(self) -> None:
        with self._lock:
            if self.projector is not None:
                try:
                    self.projector.free()
                except Exception:
                    traceback.print_exc()
            self.projector = None
            self.device = None
            self.ct = None
            self.label_cache = None
            self.pose_center_anatomical = None

    def load(self, ct_path: Path, label_path: Optional[Path], params: dict) -> None:
        if deepdrr is None:
            raise RuntimeError("DeepDRR is not installed in this Python environment.")
        if nib is None:
            raise RuntimeError("nibabel is not installed in this Python environment.")
        ct_path = Path(ct_path)
        if not ct_path.exists():
            raise FileNotFoundError(f"CT image not found: {ct_path}")
        if not (ct_path.name.endswith(".nii") or ct_path.name.endswith(".nii.gz")):
            raise ValueError("This simulator currently expects a NIfTI CT (.nii or .nii.gz).")

        with self._lock:
            self.free()
            self.ct_path = ct_path
            self.label_path = Path(label_path) if label_path is not None else None

            # Use the centre of the NIfTI voxel-centres, matching the centre
            # used by TorchIO/nanoDRR.  DeepDRR's Volume.get_center() uses
            # shape/2, which is half a voxel away from this position.
            ct_image = nib.load(str(ct_path))
            center_ijk = (np.asarray(ct_image.shape[:3], dtype=np.float64) - 1.0) / 2.0
            center_h = np.concatenate([center_ijk, [1.0]])
            self.pose_center_anatomical = (
                np.asarray(ct_image.affine, dtype=np.float64) @ center_h
            )[:3]

            # Threshold material decomposition is much faster than the learned
            # V-Net material segmentation and is suitable for an interactive app.
            self.ct = deepdrr.Volume.from_nifti(ct_path, use_thresholding=True)
            self.device = deepdrr.SimpleDevice(
                sensor_height=int(params["height"]),
                sensor_width=int(params["width"]),
                pixel_size=float(params["pixel_size"]),
                source_to_detector_distance=float(params["sdd"]),
            )
            self.projector = deepdrr.Projector(
                self.ct,
                device=self.device,
                step=float(params["ray_step"]),
                spectrum=str(params["spectrum"]),
                add_noise=False,
                neglog=True,
            )
            self.projector.initialize()

            if self.label_path is not None:
                if not self.label_path.exists():
                    raise FileNotFoundError(f"Label map not found: {self.label_path}")
                self.label_cache = LabelPointCloudCache(
                    self.label_path,
                    tuple(ct_image.shape[:3]),
                    np.asarray(ct_image.affine),
                )

    def _set_geometry(self, params: dict) -> None:
        self.device.sensor_height = int(params["height"])
        self.device.sensor_width = int(params["width"])
        self.device.pixel_size = float(params["pixel_size"])
        self.device.source_to_detector_distance = float(params["sdd"])
        self.projector.step = float(params["ray_step"])

        sid = float(params["sid"])
        sdd = float(params["sdd"])
        if not 0 < sid < sdd:
            raise ValueError(f"SID must be positive and smaller than SDD ({sdd:g} mm).")

        rotation = rotation_matrix_nanodrr_degrees(
            float(params["rot_x"]),
            float(params["rot_y"]),
            float(params["rot_z"]),
        )
        # nanoDRR's orientation="AP" places the source on +Y and sends the
        # central ray toward -Y at zero rotation.
        direction_anatomical = rotation @ np.array([0.0, -1.0, 0.0])
        up_anatomical = rotation @ np.array([0.0, 0.0, 1.0])
        translation_anatomical = np.array(
            [params["trans_x"], params["trans_y"], params["trans_z"]],
            dtype=np.float64,
        )

        world_from_anatomical = np.asarray(self.ct.world_from_anatomical, dtype=np.float64)
        world_rotation = world_from_anatomical[:3, :3]
        if self.pose_center_anatomical is None:
            raise RuntimeError("The CT pose centre has not been initialised.")
        point = (
            world_from_anatomical
            @ np.concatenate([self.pose_center_anatomical, [1.0]])
        )[:3]
        point = point + world_rotation @ translation_anatomical
        direction = world_rotation @ direction_anatomical
        up = world_rotation @ up_anatomical
        direction /= np.linalg.norm(direction)
        up /= np.linalg.norm(up)

        self.device.set_view(
            point=point,
            direction=direction,
            up=up,
            source_to_point_distance=sid,
        )

    def render(self, params: dict) -> np.ndarray:
        if not self.ready:
            raise RuntimeError("Load a CT before rendering.")
        with self._lock:
            self._set_geometry(params)
            image = self.projector()
            display = normalise_for_display(image, invert=bool(params["invert"]))
            rgb = greyscale_to_rgb_uint8(display)

            if params.get("show_segmentations") and self.label_cache is not None:
                projection = self.device.get_camera_projection()
                matrix = camera_projection_matrix(projection)
                masks = self.label_cache.project_masks(
                    matrix,
                    params.get("visible_label_ids"),
                    int(params["height"]),
                    int(params["width"]),
                    int(params.get("label_splat_radius", 1)),
                )
                rgb = overlay_masks_on_rgb(
                    rgb,
                    masks,
                    float(params["fill_alpha"]),
                    int(params["border_width"]),
                )
            # DeepDRR returns detector rows in the opposite vertical display
            # order to the radiological convention used by the companion
            # nanoDRR GUI. Flip the completed frame (DRR and overlay together)
            # so superior anatomy appears at the top in both the GUI and PNGs.
            return np.ascontiguousarray(np.flipud(rgb))


class FloatSlider(QWidget):
    value_changed = Signal(float)

    def __init__(self, name: str, minimum: float, maximum: float, default: float, step=1.0, decimals=1):
        super().__init__()
        self.name = name
        self.step = float(step)
        self.scale = int(round(1.0 / self.step)) if self.step < 1.0 else 1
        self.decimals = decimals
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(
            int(round(minimum * self.scale)), int(round(maximum * self.scale))
        )
        self.slider.setValue(int(round(default * self.scale)))
        self.slider.setTracking(True)
        self.label = QLabel()
        self.label.setMinimumWidth(170)
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        self.slider.valueChanged.connect(self._emit_value)
        self._update_label(default)

    def value(self) -> float:
        return self.slider.value() / self.scale

    def set_value(self, value: float) -> None:
        self.slider.setValue(int(round(value * self.scale)))

    def _update_label(self, value: float) -> None:
        self.label.setText(f"{self.name}: {value:.{self.decimals}f}")

    def _emit_value(self, raw_value: int) -> None:
        value = raw_value / self.scale
        self._update_label(value)
        self.value_changed.emit(float(value))


class RenderSignals(QObject):
    result = Signal(object, float, object)
    error = Signal(str, object)
    finished = Signal()


class RenderWorker(QRunnable):
    def __init__(self, renderer: DeepDRRRenderer, params: dict):
        super().__init__()
        self.renderer = renderer
        self.params = params
        self.signals = RenderSignals()

    def run(self) -> None:
        try:
            start = time.perf_counter()
            rgb = self.renderer.render(self.params)
            self.signals.result.emit(rgb, time.perf_counter() - start, self.params)
        except Exception:
            self.signals.error.emit(traceback.format_exc(), self.params)
        finally:
            self.signals.finished.emit()


class FluoroControlWindow(QMainWindow):
    def __init__(
        self,
        image_path: Optional[Path],
        label_path: Optional[Path],
        mapping_path: Optional[Path],
    ):
        super().__init__()
        self.setWindowTitle("DeepDRR Fluoroscopy Simulator: C-arm Control")
        self.resize(1280, 800)

        self.image_path = image_path
        self.label_path = label_path
        self.mapping_path = mapping_path
        self.renderer = DeepDRRRenderer()
        self.threadpool = QThreadPool(self)
        self.threadpool.setMaxThreadCount(1)
        self.render_running = False
        self.render_requested = False
        self.render_counter = 0
        self.current_rgb: Optional[np.ndarray] = None
        self.pending_capture: Optional[dict] = None
        self.label_checkboxes: dict[int, QCheckBox] = {}

        try:
            self.label_id_to_name, self.label_name_to_id = load_label_mapping(mapping_path)
        except Exception:
            traceback.print_exc()
            self.label_id_to_name, self.label_name_to_id = {}, {}

        self.params = {
            "rot_x": 0.0,
            "rot_y": 0.0,
            "rot_z": 0.0,
            "trans_x": 0.0,
            "trans_y": 0.0,
            "trans_z": 0.0,
            "sid": 850.0,
            "sdd": 1020.0,
            "pixel_size": 0.6,
            "height": 512,
            "width": 512,
            "ray_step": 0.5,
            "spectrum": "90KV_AL40",
            "invert": True,
            "show_segmentations": False,
            "fill_alpha": 0.30,
            "border_width": 2,
            "label_splat_radius": 1,
            "visible_label_ids": choose_starting_visible_ids(
                self.label_id_to_name, self.label_name_to_id
            ),
        }

        self._build_ui()
        self._rebuild_label_list()
        if self.image_path is not None and self.image_path.exists():
            self.reload_selected_files(initial=True)
        else:
            self.status_label.setText("Choose a NIfTI CT image, then click Load selected files.")

    def _build_ui(self) -> None:
        central = QWidget()
        main_layout = QHBoxLayout(central)
        self.setCentralWidget(central)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFixedWidth(450)
        controls_scroll.setWidget(self._make_controls_panel())
        main_layout.addWidget(controls_scroll)
        main_layout.addWidget(self._make_image_panel(), stretch=1)

    def _make_controls_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(410)
        layout = QVBoxLayout(panel)

        self.device_label = QLabel(self._device_text())
        self.device_label.setWordWrap(True)
        layout.addWidget(self.device_label)

        file_group = QGroupBox("Input files")
        file_layout = QVBoxLayout(file_group)
        self.image_path_edit = QLineEdit("" if self.image_path is None else str(self.image_path))
        self.label_path_edit = QLineEdit("" if self.label_path is None else str(self.label_path))
        self.mapping_path_edit = QLineEdit("" if self.mapping_path is None else str(self.mapping_path))
        for edit in (self.image_path_edit, self.label_path_edit, self.mapping_path_edit):
            edit.setReadOnly(True)

        file_layout.addWidget(QLabel("CT image"))
        file_layout.addWidget(self.image_path_edit)
        browse_ct = QPushButton("Browse CT image")
        browse_ct.clicked.connect(self.browse_image_file)
        file_layout.addWidget(browse_ct)
        file_layout.addWidget(QLabel("Segmentation label map (optional)"))
        file_layout.addWidget(self.label_path_edit)
        label_buttons = QHBoxLayout()
        browse_labels = QPushButton("Browse labels")
        clear_labels = QPushButton("Clear")
        browse_labels.clicked.connect(self.browse_label_file)
        clear_labels.clicked.connect(self.clear_label_file)
        label_buttons.addWidget(browse_labels)
        label_buttons.addWidget(clear_labels)
        file_layout.addLayout(label_buttons)
        file_layout.addWidget(QLabel("Vessel label mapping JSON/CSV (optional)"))
        file_layout.addWidget(self.mapping_path_edit)
        browse_mapping = QPushButton("Browse vessel mapping")
        browse_mapping.clicked.connect(self.browse_mapping_file)
        file_layout.addWidget(browse_mapping)
        load_button = QPushButton("Load selected files")
        load_button.clicked.connect(self.reload_selected_files)
        file_layout.addWidget(load_button)
        layout.addWidget(file_group)

        seg_group = QGroupBox("Segmentation overlay")
        seg_layout = QVBoxLayout(seg_group)
        self.seg_button = QPushButton("Impose segmentations")
        self.seg_button.setCheckable(True)
        self.seg_button.setEnabled(False)
        self.seg_button.toggled.connect(self.set_segmentation_overlay)
        seg_layout.addWidget(self.seg_button)

        self.fill_alpha_box = QDoubleSpinBox()
        self.fill_alpha_box.setRange(0.0, 1.0)
        self.fill_alpha_box.setSingleStep(0.05)
        self.fill_alpha_box.setDecimals(2)
        self.fill_alpha_box.setValue(self.params["fill_alpha"])
        self.fill_alpha_box.valueChanged.connect(
            lambda value: self.set_param("fill_alpha", float(value))
        )
        seg_layout.addWidget(QLabel("Fill opacity"))
        seg_layout.addWidget(self.fill_alpha_box)

        self.border_width_box = QSpinBox()
        self.border_width_box.setRange(1, 6)
        self.border_width_box.setValue(self.params["border_width"])
        self.border_width_box.valueChanged.connect(
            lambda value: self.set_param("border_width", int(value))
        )
        seg_layout.addWidget(QLabel("Border width (pixels)"))
        seg_layout.addWidget(self.border_width_box)

        select_buttons = QHBoxLayout()
        defaults_button = QPushButton("Vascular defaults")
        all_button = QPushButton("All")
        none_button = QPushButton("None")
        defaults_button.clicked.connect(self.select_default_labels)
        all_button.clicked.connect(self.select_all_labels)
        none_button.clicked.connect(self.clear_all_labels)
        select_buttons.addWidget(defaults_button)
        select_buttons.addWidget(all_button)
        select_buttons.addWidget(none_button)
        seg_layout.addLayout(select_buttons)

        label_scroll = QScrollArea()
        label_scroll.setWidgetResizable(True)
        label_scroll.setFixedHeight(220)
        self.label_list_widget = QWidget()
        self.label_list_layout = QVBoxLayout(self.label_list_widget)
        label_scroll.setWidget(self.label_list_widget)
        seg_layout.addWidget(QLabel("Visible labels"))
        seg_layout.addWidget(label_scroll)
        layout.addWidget(seg_group)

        pose_group = QGroupBox("C-arm pose")
        pose_layout = QVBoxLayout(pose_group)
        pose_note = QLabel(
            "AP at zero rotation. Rotation controls match nanoDRR's ZXY "
            "convention. Target shifts are relative to the CT centre; "
            "nanoDRR (0, 850, 0) corresponds here to shift (0, 0, 0) "
            "with SID 850 mm."
        )
        pose_note.setWordWrap(True)
        pose_layout.addWidget(pose_note)
        self.rot_x_slider = FloatSlider("Rotation X (deg)", -90, 90, 0)
        self.rot_y_slider = FloatSlider("Rotation Y (deg)", -90, 90, 0)
        self.rot_z_slider = FloatSlider("Rotation Z (deg)", -180, 180, 0)
        self.trans_x_slider = FloatSlider("Target shift X (mm)", -250, 250, 0, 5)
        self.trans_y_slider = FloatSlider("Target shift Y (mm)", -250, 250, 0, 5)
        self.trans_z_slider = FloatSlider("Target shift Z (mm)", -250, 250, 0, 5)
        sliders = [
            (self.rot_x_slider, "rot_x"),
            (self.rot_y_slider, "rot_y"),
            (self.rot_z_slider, "rot_z"),
            (self.trans_x_slider, "trans_x"),
            (self.trans_y_slider, "trans_y"),
            (self.trans_z_slider, "trans_z"),
        ]
        for slider, key in sliders:
            pose_layout.addWidget(slider)
            slider.value_changed.connect(lambda value, name=key: self.set_param(name, value))
        layout.addWidget(pose_group)

        detector_group = QGroupBox("Detector and image formation")
        detector_layout = QVBoxLayout(detector_group)
        self.sid_box = self._double_box(200, 1900, self.params["sid"], 10, 1)
        self.sdd_box = self._double_box(500, 2000, self.params["sdd"], 10, 1)
        self.pixel_size_box = self._double_box(0.1, 3.0, self.params["pixel_size"], 0.1, 2)
        self.height_box = QSpinBox()
        self.height_box.setRange(128, 1024)
        self.height_box.setSingleStep(64)
        self.height_box.setValue(self.params["height"])
        self.width_box = QSpinBox()
        self.width_box.setRange(128, 1024)
        self.width_box.setSingleStep(64)
        self.width_box.setValue(self.params["width"])
        self.ray_step_box = self._double_box(0.1, 2.0, self.params["ray_step"], 0.1, 1)
        self.spectrum_box = QComboBox()
        self.spectrum_box.addItems(["60KV_AL35", "90KV_AL40", "120KV_AL43"])
        self.spectrum_box.setCurrentText(self.params["spectrum"])
        self.invert_box = QCheckBox("Invert to fluoroscopy-style display")
        self.invert_box.setChecked(self.params["invert"])

        for label, widget in [
            ("Source-to-isocentre distance, SID (mm)", self.sid_box),
            ("Source-to-detector distance, SDD (mm)", self.sdd_box),
            ("Pixel size (mm)", self.pixel_size_box),
            ("Detector height (pixels)", self.height_box),
            ("Detector width (pixels)", self.width_box),
            ("Ray step (mm; smaller = slower/sharper)", self.ray_step_box),
            ("X-ray spectrum", self.spectrum_box),
        ]:
            detector_layout.addWidget(QLabel(label))
            detector_layout.addWidget(widget)
        detector_layout.addWidget(self.invert_box)
        layout.addWidget(detector_group)

        self.sid_box.valueChanged.connect(lambda v: self.set_param("sid", float(v)))
        self.sdd_box.valueChanged.connect(lambda v: self.set_param("sdd", float(v)))
        self.pixel_size_box.valueChanged.connect(lambda v: self.set_param("pixel_size", float(v)))
        self.height_box.valueChanged.connect(lambda v: self.set_param("height", int(v)))
        self.width_box.valueChanged.connect(lambda v: self.set_param("width", int(v)))
        self.ray_step_box.valueChanged.connect(lambda v: self.set_param("ray_step", float(v)))
        self.spectrum_box.currentTextChanged.connect(self.set_spectrum)
        self.invert_box.toggled.connect(lambda v: self.set_param("invert", bool(v)))

        reset_button = QPushButton("Reset pose")
        reset_button.clicked.connect(self.reset_pose)
        layout.addWidget(reset_button)
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addStretch()
        return panel

    @staticmethod
    def _double_box(minimum, maximum, value, step, decimals) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setSingleStep(step)
        box.setDecimals(decimals)
        box.setValue(value)
        return box

    def _make_image_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        title = QLabel("Live DeepDRR simulated fluoroscopy")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(500, 500)
        self.image_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.image_label, stretch=1)
        save_button = QPushButton("Save displayed image as PNG")
        save_button.clicked.connect(self.save_current_image)
        layout.addWidget(save_button)

        preset_buttons = QHBoxLayout()
        save_ap_button = QPushButton("Save AP screenshot")
        save_lateral_button = QPushButton("Save lateral screenshot")
        save_ap_button.setToolTip(
            "Render and save the canonical AP pose (all rotations and target shifts zero)."
        )
        save_lateral_button.setToolTip(
            "Render and save the canonical lateral pose (+90 degrees about anatomical Z)."
        )
        save_ap_button.clicked.connect(lambda: self.capture_preset("AP"))
        save_lateral_button.clicked.connect(lambda: self.capture_preset("LATERAL"))
        preset_buttons.addWidget(save_ap_button)
        preset_buttons.addWidget(save_lateral_button)
        layout.addLayout(preset_buttons)
        return panel

    def _device_text(self) -> str:
        if cp is None:
            return "DeepDRR CUDA device: unavailable until the DeepDRR environment is active"
        try:
            props = cp.cuda.runtime.getDeviceProperties(0)
            name = props.get("name", props.get(b"name", "CUDA GPU"))
            if isinstance(name, bytes):
                name = name.decode(errors="replace")
            return f"DeepDRR CUDA device: {name}"
        except Exception as exc:
            return f"DeepDRR CUDA device check failed: {exc}"

    def _rebuild_label_list(self) -> None:
        while self.label_list_layout.count():
            item = self.label_list_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.label_checkboxes.clear()
        visible = self.params.get("visible_label_ids")
        if not self.label_id_to_name:
            label = QLabel("Load a label map to discover its label IDs, or provide a mapping JSON/CSV.")
            label.setWordWrap(True)
            self.label_list_layout.addWidget(label)
        else:
            for label_id, label_name in sorted(
                self.label_id_to_name.items(), key=lambda item: item[1].lower()
            ):
                checkbox = QCheckBox(f"{label_name} [ID {label_id}]")
                checkbox.setChecked(visible is None or label_id in visible)
                checkbox.toggled.connect(
                    lambda checked, lid=label_id: self.on_label_checkbox_toggled(lid, checked)
                )
                self.label_checkboxes[label_id] = checkbox
                self.label_list_layout.addWidget(checkbox)
        self.label_list_layout.addStretch()

    def browse_image_file(self) -> None:
        start = self.image_path.parent if self.image_path else Path.home()
        path, _ = QFileDialog.getOpenFileName(self, "Select CT NIfTI", str(start), "NIfTI (*.nii *.nii.gz)")
        if path:
            self.image_path = Path(path)
            self.image_path_edit.setText(path)

    def browse_label_file(self) -> None:
        start = self.image_path.parent if self.image_path else Path.home()
        path, _ = QFileDialog.getOpenFileName(self, "Select label-map NIfTI", str(start), "NIfTI (*.nii *.nii.gz)")
        if path:
            self.label_path = Path(path)
            self.label_path_edit.setText(path)

    def clear_label_file(self) -> None:
        self.label_path = None
        self.label_path_edit.clear()

    def browse_mapping_file(self) -> None:
        start = self.mapping_path.parent if self.mapping_path else Path.home()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select vessel label mapping",
            str(start),
            "Label mappings (*.json *.csv);;JSON (*.json);;CSV (*.csv)",
        )
        if path:
            self.mapping_path = Path(path)
            self.mapping_path_edit.setText(path)

    def reload_selected_files(self, initial: bool = False) -> None:
        if self.render_running:
            QMessageBox.information(self, "Render in progress", "Wait for the current render to finish, then load the files.")
            return
        if self.image_path is None:
            QMessageBox.warning(self, "No CT selected", "Choose a CT image first.")
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.status_label.setText("Loading CT and compiling the DeepDRR projector…")
        QApplication.processEvents()
        try:
            if self.mapping_path is not None:
                self.label_id_to_name, self.label_name_to_id = load_label_mapping(self.mapping_path)
            self.renderer.load(self.image_path, self.label_path, self.params)

            if self.renderer.label_cache is not None:
                present_ids = set(self.renderer.label_cache.present_ids)
                self.label_id_to_name = {
                    label_id: self.label_id_to_name.get(label_id, f"label_{label_id}")
                    for label_id in sorted(present_ids)
                }
                self.label_name_to_id = {
                    name: label_id for label_id, name in self.label_id_to_name.items()
                }
                if self.params["visible_label_ids"] is None:
                    self.params["visible_label_ids"] = choose_starting_visible_ids(
                        self.label_id_to_name, self.label_name_to_id
                    )
                else:
                    self.params["visible_label_ids"] = (
                        set(self.params["visible_label_ids"]) & present_ids
                    )
                self.seg_button.setEnabled(True)
            else:
                self.params["show_segmentations"] = False
                self.seg_button.setChecked(False)
                self.seg_button.setEnabled(False)
            self._rebuild_label_list()
            self.device_label.setText(self._device_text())
            self.status_label.setText("Loaded. Rendering first view…")
            self.request_render()
        except Exception as exc:
            traceback.print_exc()
            self.status_label.setText("File load failed; see the error dialog.")
            if not initial or self.isVisible():
                QMessageBox.critical(self, "DeepDRR load error", f"Could not load the selected files:\n\n{exc}")
        finally:
            QApplication.restoreOverrideCursor()

    def set_param(self, key: str, value) -> None:
        self.params[key] = value
        self.request_render()

    def set_spectrum(self, value: str) -> None:
        if value == self.params["spectrum"]:
            return
        self.params["spectrum"] = value
        if self.renderer.ready:
            QMessageBox.information(
                self,
                "Reload required",
                "The X-ray spectrum is fixed when a DeepDRR Projector is created. "
                "Click 'Load selected files' to apply this spectrum.",
            )

    def set_segmentation_overlay(self, checked: bool) -> None:
        self.params["show_segmentations"] = bool(checked)
        self.seg_button.setText("Hide segmentations" if checked else "Impose segmentations")
        self.request_render()

    def on_label_checkbox_toggled(self, label_id: int, checked: bool) -> None:
        visible = self.params.get("visible_label_ids")
        if visible is None:
            visible = set(self.label_id_to_name)
        visible = set(visible)
        visible.add(label_id) if checked else visible.discard(label_id)
        self.params["visible_label_ids"] = visible
        if self.params["show_segmentations"]:
            self.request_render()

    def set_visible_labels(self, ids: Iterable[int]) -> None:
        visible = {int(x) for x in ids}
        self.params["visible_label_ids"] = visible
        for label_id, checkbox in self.label_checkboxes.items():
            checkbox.blockSignals(True)
            checkbox.setChecked(label_id in visible)
            checkbox.blockSignals(False)
        if self.params["show_segmentations"]:
            self.request_render()

    def select_default_labels(self) -> None:
        ids = {
            self.label_name_to_id[name]
            for name in DEFAULT_VISIBLE_LABEL_NAMES
            if name in self.label_name_to_id
        }
        self.set_visible_labels(ids if ids else set(self.label_id_to_name))

    def select_all_labels(self) -> None:
        self.set_visible_labels(set(self.label_id_to_name))

    def clear_all_labels(self) -> None:
        self.set_visible_labels(set())

    def reset_pose(self) -> None:
        defaults = {key: 0.0 for key in ("rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z")}
        widgets = [
            (self.rot_x_slider, "rot_x"),
            (self.rot_y_slider, "rot_y"),
            (self.rot_z_slider, "rot_z"),
            (self.trans_x_slider, "trans_x"),
            (self.trans_y_slider, "trans_y"),
            (self.trans_z_slider, "trans_z"),
        ]
        for widget, key in widgets:
            widget.blockSignals(True)
            widget.set_value(defaults[key])
            widget.blockSignals(False)
        self.params.update(defaults)
        self.request_render()

    def request_render(self) -> None:
        if not self.renderer.ready:
            return
        if self.render_running:
            self.render_requested = True
            return
        self.start_render()

    def start_render(self) -> None:
        self.render_running = True
        self.render_requested = False
        self.status_label.setText("Rendering with DeepDRR…")
        params_snapshot = dict(self.params)
        self.render_counter += 1
        params_snapshot["_render_id"] = self.render_counter
        visible = params_snapshot.get("visible_label_ids")
        if visible is not None:
            params_snapshot["visible_label_ids"] = set(visible)
        worker = RenderWorker(self.renderer, params_snapshot)
        worker.signals.result.connect(self.on_render_result)
        worker.signals.error.connect(self.on_render_error)
        worker.signals.finished.connect(self.on_render_finished)
        self.threadpool.start(worker)

    def on_render_result(
        self, rgb: np.ndarray, elapsed: float, rendered_params: dict
    ) -> None:
        self.current_rgb = rgb
        self.update_image_label(rgb)
        height, width = rgb.shape[:2]
        overlay = "overlay ON" if rendered_params["show_segmentations"] else "overlay OFF"
        visible = rendered_params.get("visible_label_ids")
        selected = "all labels" if visible is None else f"{len(visible)} selected"
        self.status_label.setText(
            f"DeepDRR | {width} × {height} | {elapsed:.3f} s | {overlay} | {selected}"
        )
        self._complete_pending_capture(rgb, rendered_params)

    def on_render_error(self, error_text: str, rendered_params: dict) -> None:
        print(error_text, file=sys.stderr)
        final_line = error_text.strip().splitlines()[-1] if error_text.strip() else "Unknown error"
        self.status_label.setText(f"Render failed: {final_line}")
        if self._is_pending_capture_render(rendered_params):
            self.pending_capture = None

    def on_render_finished(self) -> None:
        self.render_running = False
        if self.render_requested:
            self.start_render()

    def update_image_label(self, rgb: np.ndarray) -> None:
        height, width, channels = rgb.shape
        if channels != 3:
            raise ValueError(f"Expected RGB output, got {rgb.shape}.")
        image = QImage(rgb.data, width, height, 3 * width, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)

    @staticmethod
    def _preset_pose(view_name: str) -> dict[str, float]:
        pose = {
            "rot_x": 0.0,
            "rot_y": 0.0,
            "rot_z": 0.0,
            "trans_x": 0.0,
            "trans_y": 0.0,
            "trans_z": 0.0,
        }
        if view_name == "LATERAL":
            # The matched nanoDRR convention maps the first GUI rotation
            # control to rotation about anatomical Z.
            pose["rot_x"] = 90.0
        elif view_name != "AP":
            raise ValueError(f"Unknown preset view: {view_name}")
        return pose

    def _set_pose(self, pose: dict[str, float]) -> None:
        widgets = {
            "rot_x": self.rot_x_slider,
            "rot_y": self.rot_y_slider,
            "rot_z": self.rot_z_slider,
            "trans_x": self.trans_x_slider,
            "trans_y": self.trans_y_slider,
            "trans_z": self.trans_z_slider,
        }
        for key, widget in widgets.items():
            widget.blockSignals(True)
            widget.set_value(pose[key])
            widget.blockSignals(False)
        self.params.update(pose)

    def _default_capture_path(self, view_name: str) -> str:
        if self.image_path is None:
            folder = Path.home()
            stem = "fluoroscopy"
        else:
            folder = self.image_path.parent
            name = self.image_path.name
            stem = name[:-7] if name.lower().endswith(".nii.gz") else self.image_path.stem
        return str(folder / f"{stem}_deepdrr_{view_name}.png")

    @staticmethod
    def _save_rgb_png(rgb: np.ndarray, path: str) -> bool:
        height, width, _ = rgb.shape
        image = QImage(
            rgb.data, width, height, 3 * width, QImage.Format_RGB888
        ).copy()
        return image.save(path, "PNG")

    def capture_preset(self, view_name: str) -> None:
        if not self.renderer.ready:
            QMessageBox.information(self, "No CT loaded", "Load a CT image first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            f"Save {view_name.title()} DeepDRR screenshot",
            self._default_capture_path(view_name),
            "PNG (*.png)",
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"

        pose = self._preset_pose(view_name)
        self.pending_capture = {
            "view": view_name,
            "path": path,
            "pose": pose,
            "minimum_render_id": self.render_counter + 1,
        }
        self._set_pose(pose)
        self.status_label.setText(f"Rendering {view_name.title()} view for capture…")
        self.request_render()

    def _is_pending_capture_render(self, rendered_params: dict) -> bool:
        if self.pending_capture is None:
            return False
        expected = self.pending_capture["pose"]
        is_new_render = int(rendered_params.get("_render_id", 0)) >= int(
            self.pending_capture["minimum_render_id"]
        )
        return is_new_render and all(
            math.isclose(float(rendered_params[key]), float(value), abs_tol=1e-6)
            for key, value in expected.items()
        )

    def _complete_pending_capture(
        self, rgb: np.ndarray, rendered_params: dict
    ) -> None:
        if not self._is_pending_capture_render(rendered_params):
            return
        capture = self.pending_capture
        self.pending_capture = None
        if self._save_rgb_png(rgb, capture["path"]):
            self.status_label.setText(
                f"Saved {capture['view'].title()} screenshot: {capture['path']}"
            )
        else:
            QMessageBox.critical(
                self, "Save failed", f"Could not save {capture['path']}"
            )

    def save_current_image(self) -> None:
        if self.current_rgb is None:
            QMessageBox.information(self, "Nothing to save", "Render an image first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save simulated fluoroscopy", "deepdrr_fluoroscopy.png", "PNG (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        if not self._save_rgb_png(self.current_rgb, path):
            QMessageBox.critical(self, "Save failed", f"Could not save {path}")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.current_rgb is not None:
            self.update_image_label(self.current_rgb)

    def closeEvent(self, event) -> None:
        if self.render_running:
            self.threadpool.waitForDone()
        self.renderer.free()
        super().closeEvent(event)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive DeepDRR fluoroscopy simulator")
    parser.add_argument("--ct", type=Path, default=DEFAULT_IMAGE_PATH, help="CT NIfTI path")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABEL_PATH, help="Optional label-map NIfTI path")
    parser.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_MAPPING_CSV,
        help="Optional vessel_mapping.json or label_name/label_id CSV",
    )
    return parser.parse_args()


def main() -> None:
    numpy_major = int(np.__version__.split(".", maxsplit=1)[0])
    if numpy_major >= 2:
        raise SystemExit(
            f"DeepDRR compatibility error: NumPy {np.__version__} is installed. "
            "Install numpy==1.26.4 and cupy-cuda12x==13.6.0 in this environment."
        )
    if deepdrr is None:
        raise SystemExit(
            "DeepDRR import failed. The underlying error was:\n"
            f"{type(DEEPDRR_IMPORT_ERROR).__name__}: {DEEPDRR_IMPORT_ERROR}\n\n"
            "Use the pinned Blackwell environment described in README_vessel_drr_comparison.md."
        )
    if nib is None:
        raise SystemExit("nibabel is required: pip install nibabel")
    args = parse_args()
    app = QApplication(sys.argv)
    window = FluoroControlWindow(args.ct, args.labels, args.mapping)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
