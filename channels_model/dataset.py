"""
Dataset for the color-cluster channel-selection model.

images_dir/<stem>.png <-> labels_dir/<stem>.txt (newline-separated channel
numbers 1-8 that are "on" for that image -- see matlab_clustering.py for
what a "channel" actually is: one of 8 color clusters from MATLAB's real
rgb2ind-based quantization, 1-indexed ascending by cluster RGB value).

Source images on disk are ordinary color PNGs. This Dataset decodes each to
RGB and, via matlab_clustering.create_channels_cached, computes its 8
binary cluster-channel maps (cached to disk -- MATLAB round-trips are not
free) BEFORE augmentation, so augmentation operates on the same RGB image
and channel maps the model actually consumes.

Images are resized to config.resize_height/resize_width here (see
ChannelsDataset._rgb_resize/_channels_resize) -- real source images are NOT
all the same native resolution, so torch's default batch collate needs a
common shape to stack. This is a separate, antialiased-for-rgb (nearest-for-
channels, to keep them binary) resize from preprocess.Preprocess's own
(non-antialiased, ONNX-exportable) internal resize -- training batches are
already at the target size by the time they reach the model, so Preprocess's
resize is a no-op for them; it only does real work for an external caller
(e.g. the flake pipeline) passing in an arbitrary-resolution image directly.

Augmentation keeps rgb and channels spatially aligned: geometric transforms
(flip/rotation) are applied to BOTH (concatenated, then split) with nearest-
neighbor interpolation throughout (so the channel maps never get blurred off
{0,1}); Gaussian noise applies to rgb only -- adding noise to a binary mask
is meaningless. Brightness/contrast jitter was deliberately left out: it
would have created (jittered RGB, unjittered channels) training pairs that
don't correspond to any real photo (the channels were computed by MATLAB
from the image's true, un-jittered colors) -- a standard and normally-fine
augmentation tradeoff elsewhere, but skipped here by explicit choice given
how tightly RGB and channels are coupled for this task.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from .matlab_clustering import create_channels_cached
from .preprocess import NUM_CHANNELS

SUPPORTED_EXTENSIONS = {'.png'}


def parse_label_file(path: str | Path) -> torch.Tensor:
    """
    Parse a label .txt (one channel number 1-8 per line) into an 8-d
    multi-hot float tensor. Raises ValueError naming the offending file on
    any malformed/out-of-range line -- bad label data should fail loudly,
    not silently produce a wrong vector.
    """
    vec = torch.zeros(NUM_CHANNELS, dtype=torch.float32)
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    for line in lines:
        try:
            channel = int(line)
        except ValueError as e:
            raise ValueError(f"{path}: non-integer label line {line!r}") from e
        if not (1 <= channel <= NUM_CHANNELS):
            raise ValueError(f"{path}: label {channel} out of range [1, {NUM_CHANNELS}]")
        vec[channel - 1] = 1.0
    return vec


def _find_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    """
    Match every images_dir/*.png with labels_dir/<stem>.txt. Images or label
    files without a counterpart are skipped (warned, not fatal) -- a
    hand-built dataset is prone to a stray unlabeled image or an orphaned
    label file, and that shouldn't crash the whole run.
    """
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    image_stems = {p.stem for p in image_paths}
    pairs = []
    skipped = 0
    for img_path in image_paths:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"  skipping {img_path.name}: no matching label file {label_path.name}")
            skipped += 1
            continue
        pairs.append((img_path, label_path))

    orphan_labels = {p.stem for p in labels_dir.glob('*.txt')} - image_stems
    for stem in sorted(orphan_labels):
        print(f"  skipping label {stem}.txt: no matching image")
        skipped += 1

    if skipped:
        print(f"Dataset: {len(pairs)} paired sample(s), {skipped} skipped (missing counterpart).")
    return pairs


def _filter_valid_labels(pairs: list[tuple[Path, Path]]) -> list[tuple[Path, Path]]:
    """
    Validate every label file up front (rather than letting a malformed one
    crash training later, deep inside some arbitrary fold/epoch) -- skip and
    warn on any file parse_label_file rejects, printing every offender in
    one pass so a hand-built dataset's bad files are all visible at once.
    """
    valid = []
    skipped = 0
    for img_path, label_path in pairs:
        try:
            parse_label_file(label_path)
        except ValueError as e:
            print(f"  skipping {img_path.name}: invalid label file ({e})")
            skipped += 1
            continue
        valid.append((img_path, label_path))
    if skipped:
        print(f"Dataset: {len(valid)} valid sample(s), {skipped} skipped (invalid label file).")
    return valid


def build_pairs(images_dir: str | Path, labels_dir: str | Path) -> list[tuple[Path, Path]]:
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images_dir does not exist: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"labels_dir does not exist: {labels_dir}")
    pairs = _find_pairs(images_dir, labels_dir)
    pairs = _filter_valid_labels(pairs)
    if not pairs:
        raise RuntimeError(f"No paired image/label samples found in {images_dir} / {labels_dir}")
    return pairs


class ChannelsDataset(Dataset):
    """
    Args:
        pairs: list of (image_path, label_path) -- shared, in the same
            order, across the augmented/non-augmented view of the same
            underlying data (see build_train_eval_datasets) so KFold Subset
            indices line up between the two views.
        augment: whether to apply train-time augmentation, applied BEFORE
            the model's own Preprocess step.
        config: ChannelsModelConfig (cache_images_in_memory, n_clusters,
            cluster_median_win, channels_cache_dir, the aug_* fields).
        cache: a pre-decoded (rgb, channels) cache to reuse (see
            build_train_eval_datasets) instead of decoding pairs again --
            decoding + clustering real, full-resolution images is not
            cheap, and the augmented/non-augmented views over the same
            pairs would otherwise each redo it independently for no reason.
    """

    def __init__(
        self, pairs: list[tuple[Path, Path]], augment: bool, config,
        cache: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        self.pairs = pairs
        self.augment = augment
        self.config = config
        self.cache = cache
        if self.cache is None and config.cache_images_in_memory:
            self.cache = [
                self._decode(p, config)
                for p, _ in tqdm(pairs, desc='Decoding images + computing channels', unit='image')
            ]

        # See module docstring: rgb gets a higher-quality antialiased
        # resize; channels must stay a clean {0,1} mask, so nearest-only.
        self._rgb_resize = v2.Resize([config.resize_height, config.resize_width], antialias=True)
        self._channels_resize = v2.Resize(
            [config.resize_height, config.resize_width],
            interpolation=v2.InterpolationMode.NEAREST,
        )

        self._geom_transform = None
        if augment:
            # fill: RGB channels get the configured mid-gray fill; the 8
            # cluster channels get 0 ("no cluster present") -- a single
            # scalar fill would incorrectly paint 127 into the binary masks.
            fill = [config.aug_rotate_fill] * 3 + [0] * NUM_CHANNELS
            self._geom_transform = v2.Compose([
                v2.RandomHorizontalFlip(p=config.aug_hflip_p),
                v2.RandomVerticalFlip(p=config.aug_vflip_p),
                v2.RandomRotation(
                    degrees=config.aug_rotate_degrees,
                    interpolation=v2.InterpolationMode.NEAREST,
                    fill=fill,
                ),
            ])

    @staticmethod
    def _decode(image_path: Path, config) -> tuple[np.ndarray, np.ndarray]:
        """Decode to RGB and compute the 8 color-cluster channels (cached to
        disk -- see matlab_clustering.py, this is a real MATLAB call on a
        cache miss). Returns (rgb (H,W,3) uint8, channels (H,W,8) uint8
        {0,1})."""
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"could not read image {image_path}")
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        channels, _palette = create_channels_cached(
            image_path, config.channels_cache_dir,
            n_clusters=config.n_clusters, median_win=config.cluster_median_win,
            expected_shape=image_bgr.shape[:2],
        )
        return rgb, channels

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path, label_path = self.pairs[idx]
        if self.cache is not None:
            rgb, channels = self.cache[idx]
        else:
            rgb, channels = self._decode(image_path, self.config)

        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()            # (3,H,W) uint8
        channels_t = torch.from_numpy(channels).permute(2, 0, 1).contiguous()  # (8,H,W) uint8 {0,1}

        rgb_t = self._rgb_resize(rgb_t)
        channels_t = self._channels_resize(channels_t)

        if self._geom_transform is not None:
            combined = torch.cat([rgb_t, channels_t], dim=0)  # (11,H,W)
            combined = self._geom_transform(combined)
            rgb_t, channels_t = combined[:3], combined[3:]

        if self.augment and self.config.aug_gaussian_noise_std > 0:
            noise = torch.randn(rgb_t.shape, dtype=torch.float32) * (self.config.aug_gaussian_noise_std * 255.0)
            rgb_t = (rgb_t.to(torch.float32) + noise).clamp(0, 255).round().to(torch.uint8)

        label = parse_label_file(label_path)
        return rgb_t.to(torch.float32), channels_t.to(torch.float32), label


def build_train_eval_datasets(pairs: list[tuple[Path, Path]], config) -> tuple[ChannelsDataset, ChannelsDataset]:
    """
    Two ChannelsDataset instances over the SAME pairs list (so indices line
    up 1:1 for KFold Subset splitting) -- one with augmentation for
    training, one without for eval, so eval never sees augmented data. The
    (expensive) decode+clustering is done exactly ONCE and shared between
    the two -- each independently doing it would double real MATLAB calls
    for no reason (the disk cache in matlab_clustering already prevents
    re-running MATLAB itself across separate process runs, but this avoids
    even the redundant cache-file reads within a single run).
    """
    cache = None
    if config.cache_images_in_memory:
        cache = [
            ChannelsDataset._decode(p, config)
            for p, _ in tqdm(pairs, desc='Decoding images + computing channels', unit='image')
        ]
    train_ds = ChannelsDataset(pairs, augment=config.augment, config=config, cache=cache)
    eval_ds = ChannelsDataset(pairs, augment=False, config=config, cache=cache)
    return train_ds, eval_ds
