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
import hashlib
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
    """
    Cache filenames are keyed on BOTH the filename stem (for human-readable
    debugging -- see create_channels_cached's own reconstructed/*.png debug
    dumps, named by stem alone) AND a short hash of the image's full
    resolved path -- confirmed as a real, currently-occurring bug without
    the hash: two different folders' images sharing both a filename stem
    AND resolution (e.g. flakes/all/WB.png vs
    flakes/TS34001/Post Exfo/WB.png -- different real flakes, same
    standard camera resolution) collided in the shared cache_dir, and
    create_channels_cached's own shape-based staleness check can't catch a
    collision when the shapes also happen to match (its own docstring
    already flagged this as a known gap) -- one image's reconstruction
    silently reused the OTHER's cached clustering data, which then fed
    wrong marker/calibration input downstream. The hash makes every
    distinct source path its own cache entry regardless of filename,
    closing this class of collision entirely (not just the specific
    flake_all/flake_TS34001 pair already caught) -- at the cost of a
    one-time cache miss for any image path already cached under the old,
    hash-less naming.
    """
    stem = Path(image_path).stem
    path_hash = hashlib.sha1(str(Path(image_path).resolve()).encode()).hexdigest()[:10]
    cache_dir = Path(cache_dir)
    return cache_dir / f"{stem}_{path_hash}_indexed.png", cache_dir / f"{stem}_{path_hash}_palette.csv"


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
    the ACTUAL current image rather than trusting the cache blindly. If
    omitted, this function decodes image_path itself just to check. A
    shape mismatch is treated as a stale/foreign cache entry and
    recomputed -- belt-and-suspenders alongside _cache_paths' own per-path
    hash (see its docstring): the hash is what actually PREVENTS a
    collision between two different images that happen to share a
    filename stem (confirmed as a real, previously-occurring bug -- e.g.
    flakes/all/WB.png vs flakes/TS34001/Post Exfo/WB.png, different real
    flakes, same resolution, silently swapped reconstructions before the
    hash was added); this shape check catches the OTHER case, a genuinely
    corrupted/truncated cache file for the SAME path.
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
