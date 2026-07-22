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

_BACKGROUND_RGB = (66, 62, 90)
"""
Background fill for unselected-cluster pixels.

GDS_Code.m's own Prepare_Polygons() uses a literal flat gray(64,64,64) here,
but that turned out to break this pipeline's marker detection empirically:
markers.find_top_markers isolates candidates via
markers._non_substrate_mask, an HSV in-range check tuned for the REAL
substrate color (DetectionConfig.substrate_hsv_range, h 105-145 / s 40-255 /
v 20-160) -- a zero-saturation gray fails that range, so the whole
reconstructed background reads as "non-substrate" and swamps template
matching with a uniform foreground (0 crosshair candidates found on every
test image). This RGB was chosen to sit inside that same HSV range (roughly
h=125, s=80, v=90) so the reconstructed background is correctly recognized
as substrate again, while still being a flat/textureless fill that Canny-
based polygon extraction (polygons.py) treats as empty background exactly
like the literal gray would.
"""

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
    debug_dir: Path | None = None,
) -> ChannelSelection:
    """
    Run the trained channels_model on image_path's color clusters and
    reconstruct a posterized image from only the model-selected clusters.
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
    canvas_rgb[:] = _BACKGROUND_RGB
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
