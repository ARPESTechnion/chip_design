"""
Color-cluster "channel" creation -- calls MATLAB's real rgb2ind-based
minimum-variance quantization (the actual algorithm GDS_Code.m's operator
used to pick channels, see channels_model/matlab/reduce_channels.m) rather
than approximating it in Python.

This module is pure numpy/opencv/matlab-engine -- NO torch dependency -- so
it can be imported by grid_preperation (or anything else) without pulling in
the ML stack. A from-scratch Python reimplementation of minimum-variance
quantization was tried and only reached 24.5% exact pixel-level agreement
with real MATLAB output on a labeled sample image -- nowhere near reliable
enough to preserve correspondence with hand-made labels, hence calling the
real MATLAB algorithm instead.

Starting a MATLAB engine session is slow (~4-10s) -- this module keeps ONE
process-wide session alive across calls (see _get_engine) rather than
starting one per image. An atexit hook quits it on interpreter shutdown so
MATLAB.exe processes don't linger.
"""

from __future__ import annotations

import atexit
from pathlib import Path

import cv2
import numpy as np

_MATLAB_DIR = Path(__file__).resolve().parent / "matlab"

_engine = None


def _get_engine():
    """Lazily start (once) and return the shared MATLAB engine session."""
    global _engine
    if _engine is None:
        import matlab.engine  # deferred import -- only needed if this path is actually used
        _engine = matlab.engine.start_matlab()
        _engine.addpath(str(_MATLAB_DIR), nargout=0)
        atexit.register(_close_engine)
    return _engine


def _close_engine() -> None:
    global _engine
    if _engine is not None:
        _engine.quit()
        _engine = None


def create_channels(
    image_path: str | Path,
    n_clusters: int = 8,
    median_win: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run MATLAB's real color-reduction algorithm (reduce_channels.m, a
    callable transcription of GDS_Code.m's UI_Act2_reduce) on an image file.

    Passing a path (not an in-memory array) is deliberate: MATLAB does its
    own imread, which is dramatically faster than marshalling a whole image
    array through the Python<->MATLAB engine bridge (~0.6s vs ~4-5s for a
    2048x1536 image, measured).

    Returns (channels, palette):
      channels : (H, W, n_clusters) uint8 {0,1} -- channels[...,k] is the
                 binary mask for 1-indexed cluster k+1, matching the
                 existing label files' 1-indexed convention.
      palette  : (n_clusters+1, 3) float64 -- row 0 is always [0,0,0]
                 (reserved/unused); rows 1..n_clusters are the cluster RGB
                 colors, sorted ascending by (R,G,B).
    """
    eng = _get_engine()
    indexed, palette = eng.reduce_channels(str(image_path), n_clusters, median_win, nargout=2)
    indexed_np = np.array(indexed, dtype=np.uint8)
    palette_np = np.array(palette, dtype=np.float64)

    channels = np.stack(
        [(indexed_np == (k + 1)).astype(np.uint8) for k in range(n_clusters)],
        axis=-1,
    )
    return channels, palette_np


def _cache_paths(image_path: str | Path, cache_dir: str | Path) -> tuple[Path, Path]:
    stem = Path(image_path).stem
    cache_dir = Path(cache_dir)
    return cache_dir / f"{stem}_indexed.png", cache_dir / f"{stem}_palette.csv"


def create_channels_cached(
    image_path: str | Path,
    cache_dir: str | Path,
    n_clusters: int = 8,
    median_win: int = 5,
    expected_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Same contract as create_channels, but checks a disk cache first --
    MATLAB round-trips are not free at hundreds of images, and the
    clustering is a pure function of the image (deterministic given
    n_clusters/median_win), so there's no reason to ever recompute it for
    the same image. Cache format mirrors the indexed-image + palette-CSV
    convention already used for MATLAB ground-truth verification this
    session (see channels_model/matlab/reduce_channels.m's docstring).

    expected_shape: optional (H, W) of the real image at image_path, if the
    caller already has it decoded -- used to validate a cache hit against
    the ACTUAL current image rather than trusting the cache by filename
    stem alone. If omitted, this function decodes image_path itself just to
    check. This guards against a real bug found empirically when
    grid_preperation started sharing this cache with channels_model's
    training cache_dir: two different images can share a stem (e.g.
    flakes/all/AU.png vs channels_model/data/images/AU.png -- unrelated
    images, different resolutions) and silently collide in the same
    cache_dir. A shape mismatch is treated as a stale/foreign cache entry
    and recomputed; if the shapes happened to match too, this check can't
    catch it -- keep cache_dir specific to one image source when possible.
    """
    indexed_path, palette_path = _cache_paths(image_path, cache_dir)

    if indexed_path.exists() and palette_path.exists():
        indexed_np = cv2.imread(str(indexed_path), cv2.IMREAD_UNCHANGED)
        if expected_shape is None:
            probe = cv2.imread(str(image_path))
            if probe is None:
                raise FileNotFoundError(f"could not read image {image_path}")
            expected_shape = probe.shape[:2]
        if tuple(indexed_np.shape[:2]) == tuple(expected_shape):
            palette_np = np.loadtxt(palette_path, delimiter=',')
            channels = np.stack(
                [(indexed_np == (k + 1)).astype(np.uint8) for k in range(n_clusters)],
                axis=-1,
            )
            return channels, palette_np
        # else: stale/foreign cache entry under this stem -- fall through and recompute

    channels, palette_np = create_channels(image_path, n_clusters, median_win)

    indexed_path.parent.mkdir(parents=True, exist_ok=True)
    indexed_np = np.zeros(channels.shape[:2], dtype=np.uint8)
    for k in range(n_clusters):
        indexed_np[channels[..., k] > 0] = k + 1
    cv2.imwrite(str(indexed_path), indexed_np)
    np.savetxt(palette_path, palette_np, delimiter=',')

    return channels, palette_np
