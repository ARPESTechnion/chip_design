"""
First pipeline step: reconstruct a posterized image from the trained
channels_model's cluster selection, before marker detection ever runs.

Mirrors GDS_Code.m's own Prepare_Polygons() -- S.mp28 = flat gray(64) with
only the operator-selected clusters (S.LiSel) painted back in with their real
measured color -- except the "operator selection" is now the trained ONNX
model's prediction instead of a human clicking channels in the MATLAB GUI.

Reuses channels_model.matlab_clustering.create_channels_cached for the same
clustering every training image used, so an image already seen during
training costs zero MATLAB calls here.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    from channels_model.matlab_clustering import create_channels_cached
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from channels_model.matlab_clustering import create_channels_cached

from flake_detection.config import DetectionConfig


def _background_rgb_for(detection: DetectionConfig) -> tuple[int, int, int]:
    """
    Background fill for unselected-cluster pixels, at THIS dataset's own
    substrate_hsv_range midpoint -- guaranteed to fall inside that range
    regardless of what it is, unlike a single fixed RGB constant shared
    across every dataset (what this function replaced).

    GDS_Code.m's own Prepare_Polygons() uses a literal flat gray(64,64,64)
    here, but that turned out to break this pipeline's marker detection
    empirically: markers.find_top_markers isolates candidates via
    markers._non_substrate_mask, an HSV in-range check tuned for the
    REAL substrate color -- a zero-saturation gray fails that range, so
    the whole reconstructed background reads as "non-substrate" and
    swamps template matching with a uniform foreground (0 crosshair
    candidates found on every test image). A single fixed RGB tuned to
    sit inside ONE dataset's own substrate range (confirmed working for
    flakes/all, TS34001, orange_flakes_tuned's shared H 210-290deg/S
    15.7-100%/V 7.8-62.7% range) was the first fix -- but that's still
    just a fixed constant, and it broke again the same way the moment a
    new dataset (Ta2009) used a genuinely different substrate range
    (H 260-300deg/S 45-75%/V 45-75% -- the old constant's own H=248deg/
    S=31%/V=35% falls outside it on all three axes). Deriving the fill
    from THIS call's own detection.substrate_hsv_range instead makes it
    correct for any dataset by construction, not by coincidence.
    """
    s = detection.substrate_hsv_range
    h_mid = (s.h_min + s.h_max) / 2
    s_mid = (s.s_min + s.s_max) / 2
    v_mid = (s.v_min + s.v_max) / 2
    hue_cv = max(0, min(179, round(h_mid / 2)))
    sat_cv = max(0, min(255, round(s_mid / 100 * 255)))
    val_cv = max(0, min(255, round(v_mid / 100 * 255)))
    bgr = cv2.cvtColor(np.uint8([[[hue_cv, sat_cv, val_cv]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))  # RGB, matching canvas_rgb's own channel order

_session = None
_session_path = None


def _get_session(onnx_path: str | Path):
    """Lazily start (once per process, per path) and return the shared
    onnxruntime InferenceSession -- loading the model is not free, and every
    image in a run uses the same one."""
    global _session, _session_path
    onnx_path = str(onnx_path)
    if _session is None or _session_path != onnx_path:
        import onnxruntime
        import torch
        providers = (
            ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if torch.cuda.is_available() else ['CPUExecutionProvider']
        )
        _session = onnxruntime.InferenceSession(onnx_path, providers=providers)
        _session_path = onnx_path
    return _session


@dataclass
class ChannelSelection:
    reconstructed_bgr: np.ndarray     # (H, W, 3) uint8 BGR
    selected_channels: list[int]      # 1-indexed cluster numbers, prob > threshold
    probs: np.ndarray                 # (n_clusters,) float32 raw sigmoid outputs
    channels: np.ndarray              # (H, W, n_clusters) uint8 {0,1} -- source cluster masks
    palette: np.ndarray               # (n_clusters+1, 3) float64 RGB; row 0 unused, rows 1..n
                                       # are each cluster's own real measured color -- see
                                       # polygons.py, which colors each exported polygon by its
                                       # source cluster's entry here rather than a re-measured
                                       # pixel mean.


def select_and_reconstruct(
    image_bgr: np.ndarray,
    image_path: str | Path,
    config,
    detection: DetectionConfig,
    debug_dir: Path | None = None,
) -> ChannelSelection:
    """
    Run the trained channels_model on image_path's color clusters and
    reconstruct a posterized image from only the model-selected clusters.

    detection is THIS run's own DetectionConfig (see run.py: already
    loaded once via config.load_detection_config()) -- its
    substrate_hsv_range determines the background fill color for
    unselected-cluster pixels (_background_rgb_for), so the reconstructed
    background is always correctly recognized as substrate by
    markers._non_substrate_mask, whatever this dataset's own substrate
    color happens to be.
    """
    channels, palette = create_channels_cached(
        image_path, config.channels_cache_dir,
        config.channels_n_clusters, config.channels_cluster_median_win,
        expected_shape=image_bgr.shape[:2],
    )

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb_input = rgb.transpose(2, 0, 1)[None].astype(np.float32)
    channels_input = channels.transpose(2, 0, 1)[None].astype(np.float32)

    session = _get_session(config.channels_model_onnx_path)
    (probs_batch,) = session.run(None, {'rgb': rgb_input, 'channels': channels_input})
    probs = probs_batch[0]

    n_clusters = config.channels_n_clusters
    selected = [k + 1 for k in range(n_clusters) if probs[k] > config.channels_model_threshold]

    canvas_rgb = np.empty_like(rgb)
    canvas_rgb[:] = _background_rgb_for(detection)
    for k in selected:
        mask = channels[..., k - 1] > 0
        canvas_rgb[mask] = np.clip(palette[k], 0, 255).astype(np.uint8)

    reconstructed_bgr = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR)

    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / f"{Path(image_path).stem}.png"), reconstructed_bgr)

    return ChannelSelection(
        reconstructed_bgr=reconstructed_bgr, selected_channels=selected, probs=probs,
        channels=channels, palette=palette,
    )
