"""
Real-MATLAB integration check for matlab_clustering.py -- NOT part of the
fast smoke test (see smoke_test.py, which deliberately avoids MATLAB via a
pre-populated cache). This exercises the actual MATLAB engine + the real
reduce_channels.m algorithm against real labeled images, and confirms the
disk cache actually avoids repeat MATLAB calls on a second run.

Requires MATLAB (R2026a+ Image Processing Toolbox) and the MATLAB Engine
API for Python to be installed and working.

Usage:
  python -m channels_model.tests.matlab_integration_check
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

try:
    from ..config import ChannelsModelConfig
    from ..matlab_clustering import create_channels, create_channels_cached
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from channels_model.config import ChannelsModelConfig
    from channels_model.matlab_clustering import create_channels, create_channels_cached

IMAGES_TO_CHECK = ["AA.png", "AX.png"]


def _check_well_formed(channels: np.ndarray, palette: np.ndarray, n_clusters: int, label: str) -> None:
    assert channels.ndim == 3 and channels.shape[2] == n_clusters, \
        f"{label}: expected (H,W,{n_clusters}), got {channels.shape}"
    assert set(np.unique(channels).tolist()) <= {0, 1}, \
        f"{label}: channels not binary, unique values {np.unique(channels)}"
    assert palette.shape == (n_clusters + 1, 3), \
        f"{label}: expected palette shape ({n_clusters + 1}, 3), got {palette.shape}"
    assert np.allclose(palette[0], [0, 0, 0]), f"{label}: palette row 0 should be [0,0,0]"
    # Every pixel should belong to exactly one channel (indexed 0 is the
    # reserved/unused black entry and isn't represented in `channels` at all).
    coverage = channels.sum(axis=2)
    assert set(np.unique(coverage).tolist()) <= {0, 1}, \
        f"{label}: some pixel assigned to >1 channel simultaneously"
    print(f"  {label}: well-formed OK (shape={channels.shape}, "
          f"per-channel pixel counts={channels.sum(axis=(0,1)).tolist()})")


def main() -> None:
    config = ChannelsModelConfig()
    images_dir = Path(config.images_dir)

    print("=== Direct create_channels (no cache) ===")
    for name in IMAGES_TO_CHECK:
        path = images_dir / name
        t0 = time.time()
        channels, palette = create_channels(path, n_clusters=config.n_clusters, median_win=config.cluster_median_win)
        elapsed = time.time() - t0
        _check_well_formed(channels, palette, config.n_clusters, f"{name} (direct, {elapsed:.2f}s)")

    print("\n=== create_channels_cached: miss then hit ===")
    temp_cache = Path(tempfile.mkdtemp(prefix='channels_model_matlab_check_'))
    try:
        for name in IMAGES_TO_CHECK:
            path = images_dir / name

            t0 = time.time()
            channels_miss, palette_miss = create_channels_cached(
                path, temp_cache, n_clusters=config.n_clusters, median_win=config.cluster_median_win,
            )
            miss_elapsed = time.time() - t0
            _check_well_formed(channels_miss, palette_miss, config.n_clusters, f"{name} (cache MISS, {miss_elapsed:.2f}s)")

            t0 = time.time()
            channels_hit, palette_hit = create_channels_cached(
                path, temp_cache, n_clusters=config.n_clusters, median_win=config.cluster_median_win,
            )
            hit_elapsed = time.time() - t0
            assert np.array_equal(channels_miss, channels_hit), f"{name}: cache hit channels differ from miss"
            assert np.allclose(palette_miss, palette_hit), f"{name}: cache hit palette differs from miss"
            assert hit_elapsed < 1.0, f"{name}: cache hit took {hit_elapsed:.2f}s -- expected near-instant, not a real MATLAB call"
            print(f"  {name}: cache HIT OK ({hit_elapsed:.3f}s, matches miss result exactly)")
    finally:
        shutil.rmtree(temp_cache, ignore_errors=True)

    print("\n=== MATLAB INTEGRATION CHECK PASSED ===")


if __name__ == '__main__':
    main()
