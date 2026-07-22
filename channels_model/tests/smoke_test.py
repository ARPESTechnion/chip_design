"""
End-to-end smoke test for the channels_model pipeline, using synthetic data.

Real training data (channels_model/data/images, channels_model/data/labels)
is available now, but running against it exercises a real MATLAB engine
call per uncached image (slow to start, ~4-10s, and requires MATLAB + the
Image Processing Toolbox installed) -- this smoke test instead PRE-POPULATES
the color-cluster channel cache with synthetic random data (matching
matlab_clustering.create_channels_cached's on-disk cache file format
exactly), so it never touches MATLAB and stays fast for quick iteration.
The real MATLAB path is exercised separately, see
tests/matlab_integration_check.py.

Usage:
  python -m channels_model.tests.smoke_test
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
import torch

try:
    from ..config import ChannelsModelConfig
    from ..preprocess import NUM_CHANNELS
    from ..train import run
except ImportError:
    # Running as `python channels_model/tests/smoke_test.py` rather than
    # `python -m channels_model.tests.smoke_test` -- add the project root to
    # sys.path so absolute imports resolve correctly.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from channels_model.config import ChannelsModelConfig
    from channels_model.preprocess import NUM_CHANNELS
    from channels_model.train import run

NUM_IMAGES = 40
NATIVE_HEIGHT, NATIVE_WIDTH = 1536, 2048  # matches real chip images' 4:3 native resolution
CHANNEL_ON_PROB = 0.4


def _generate_synthetic_dataset(root: Path) -> tuple[Path, Path, Path]:
    """Synthetic images/labels, PLUS a pre-populated color-cluster channel
    cache (random data, matching matlab_clustering's cache file naming
    exactly: <cache_dir>/<stem>_indexed.png + <stem>_palette.csv) so the
    training pipeline never has to call MATLAB for this synthetic data."""
    images_dir = root / 'images'
    labels_dir = root / 'labels'
    cache_dir = root / 'channels_cache'
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)

    rng = np.random.default_rng(seed=0)
    for i in range(NUM_IMAGES):
        stem = f"img_{i:03d}"
        image = rng.integers(0, 256, size=(NATIVE_HEIGHT, NATIVE_WIDTH, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / f"{stem}.png"), image)

        channels_on = [b for b in range(1, NUM_CHANNELS + 1) if rng.random() < CHANNEL_ON_PROB]
        with open(labels_dir / f"{stem}.txt", 'w') as f:
            f.write('\n'.join(str(b) for b in channels_on) + ('\n' if channels_on else ''))

        # Random indexed "cluster" image (values 0..NUM_CHANNELS) + a
        # matching random palette -- same file format create_channels_cached
        # writes for real, so it's read back exactly like a real cache hit.
        indexed = rng.integers(0, NUM_CHANNELS + 1, size=(NATIVE_HEIGHT, NATIVE_WIDTH), dtype=np.uint8)
        cv2.imwrite(str(cache_dir / f"{stem}_indexed.png"), indexed)
        palette = np.vstack([[0, 0, 0], rng.integers(0, 256, size=(NUM_CHANNELS, 3))])
        np.savetxt(cache_dir / f"{stem}_palette.csv", palette, delimiter=',')

    print(f"Generated {NUM_IMAGES} synthetic image/label pairs (+ pre-cached random channels) -> {root}")
    return images_dir, labels_dir, cache_dir


def _extra_onnx_check(onnx_path: Path, images_dir: Path) -> None:
    """Beyond export.py's own internal self-check (which already runs
    during train.run()), independently load the exported model and run it
    on a REAL sample image from the synthetic dataset, at its native
    resolution, exactly as an external downstream consumer would."""
    sample_path = sorted(images_dir.glob('*.png'))[0]
    image_bgr = cv2.imread(str(sample_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb_tensor = image_rgb.transpose(2, 0, 1)[None].astype(np.float32)  # (1,3,H,W), [0,255]

    rng = np.random.default_rng(seed=1)
    channels_tensor = (rng.random((1, NUM_CHANNELS, *image_rgb.shape[:2])) > 0.5).astype(np.float32)

    providers = (
        ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if torch.cuda.is_available() else ['CPUExecutionProvider']
    )
    sess = onnxruntime.InferenceSession(str(onnx_path), providers=providers)
    out = sess.run(None, {'rgb': rgb_tensor, 'channels': channels_tensor})[0]

    assert out.shape == (1, 8), f"unexpected output shape {out.shape}"
    assert not np.isnan(out).any(), "NaN in ONNX output"
    assert (out >= 0).all() and (out <= 1).all(), f"output outside [0,1]: {out}"
    print(f"Independent ONNX inference on a real synthetic sample OK: {out.round(3).tolist()}")


def main() -> None:
    temp_root = Path(tempfile.mkdtemp(prefix='channels_model_smoke_'))
    try:
        images_dir, labels_dir, cache_dir = _generate_synthetic_dataset(temp_root)
        output_dir = temp_root / 'outputs'

        config = ChannelsModelConfig(
            images_dir=str(images_dir),
            labels_dir=str(labels_dir),
            channels_cache_dir=str(cache_dir),
            output_dir=str(output_dir),
            resize_height=48,
            resize_width=64,
            num_folds=2,
            epochs=2,
            final_epochs=2,
            batch_size=4,
            num_workers=0,
            onnx_dummy_export_size=(96, 128),
        )

        print("\n=== Running full pipeline (train.run) against synthetic data ===\n")
        run(config)

        onnx_path = output_dir / 'channels_model.onnx'
        assert onnx_path.exists(), f"expected ONNX model at {onnx_path}"
        checkpoint_path = output_dir / 'final_model.pt'
        assert checkpoint_path.exists(), f"expected checkpoint at {checkpoint_path}"
        cv_results_path = output_dir / 'cv_results.json'
        assert cv_results_path.exists(), f"expected CV results at {cv_results_path}"

        _extra_onnx_check(onnx_path, images_dir)

        print("\n=== SMOKE TEST PASSED ===")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == '__main__':
    main()
