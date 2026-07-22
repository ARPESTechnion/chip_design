"""
Configuration for the bit-plane channel-selection model.

Mirrors the dataclass + JSON save/load convention used by flake_detection.config
and grid_preperation.config (single source of truth, reproducible via
--save-config), even though this subpackage is otherwise standalone.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional


@dataclass
class ChannelsModelConfig:
    """
    Full configuration for training + exporting the channel-selection model.

    Pipeline overview:
      1. Data       — images_dir/*.png + labels_dir/<stem>.txt (channel numbers
                       1-8, one per line) -- see dataset.py. Each image's 8
                       binary "channels" are color-quantization clusters from
                       a REAL MATLAB rgb2ind call (matlab_clustering.py, the
                       actual algorithm GDS_Code.m's operator used -- a Python
                       reimplementation was tried and only reached 24.5% pixel
                       agreement, not reliable enough), computed here BEFORE
                       augmentation, not inside the model -- every caller
                       (this training pipeline, and the flake pipeline at
                       inference time) must compute channels identically.
      2. Preprocess — resize + normalize only -- baked into the model itself
                       (see preprocess.py) so training and ONNX inference can
                       never diverge on THESE steps.
      3. Model      — dual-branch (RGB + 8 cluster channels) residual CNN ->
                       8-d multi-label logits -- see model.py.
      4. Training   — num_folds-fold CV to pick an epoch budget by
                       super-accuracy, then a final refit on 100% of the data
                       for that budget, then ONNX export -- see train.py.
    """

    # ------------------------------------------------------------------ #
    # Paths                                                                #
    # ------------------------------------------------------------------ #

    images_dir: str = "channels_model/data/images"
    """Folder of <image>.png color images. Supplied later -- this default is
    just where the placeholder data/ folder lives; override via config or
    --images-dir once real data exists."""

    labels_dir: str = "channels_model/data/labels"
    """Folder of <image>.txt label files -- newline-separated channel numbers
    (1-8) that are "on" for that image. See dataset.parse_label_file."""

    output_dir: str = "channels_model/outputs"
    """Destination for checkpoints, CV results, effective config, and the
    exported ONNX model."""

    channels_cache_dir: str = "channels_model/data/channels_cache"
    """
    Disk cache for computed color-cluster channels (matlab_clustering.
    create_channels_cached) -- one real MATLAB call per image, ever;
    every subsequent run (this one or a later one) reads the cached
    result instead. Safe to delete to force recomputation (e.g. after
    changing n_clusters/cluster_median_win).
    """

    # ------------------------------------------------------------------ #
    # Preprocessing (see preprocess.py for the exact math)                 #
    # ------------------------------------------------------------------ #

    resize_height: int = 192
    resize_width: int = 256
    """
    Target size the model resizes every input image to internally (baked
    into the ONNX graph, not done by the caller). Kept at the source images'
    4:3 aspect ratio (2048x1536), just smaller -- ~8x downsample, small
    enough for a compact model on a small dataset while still resolving
    grid-scale structure.
    """

    normalize_mean: float = 0.5
    normalize_std: float = 0.5
    """Applied to the RGB branch's input only ((rgb/255 - mean) / std),
    uniformly across all 3 channels -- kept deliberately simple (no per-
    channel ImageNet-style stats) given the small-model/small-dataset spec.
    The cluster-channel branch is inherently already 0/1 and needs no
    normalizing."""

    n_clusters: int = 8
    """
    Number of color-quantization clusters (P.Ncol in GDS_Code.m). Tightly
    coupled to the label format (labels are 1..n_clusters) and the model's
    8-d output -- do not change without also changing what the labels mean.
    """

    cluster_median_win: int = 5
    """Median filter window (P.Median_win in GDS_Code.m) applied before AND
    after quantization -- see matlab_clustering.py / channels_model/matlab/
    reduce_channels.m."""

    # ------------------------------------------------------------------ #
    # Cross-validation / training schedule                                #
    # ------------------------------------------------------------------ #

    num_folds: int = 4
    """K for K-fold CV. Used to report per-class/super-accuracy metrics and
    to derive final_epochs -- NOT the model that gets exported (see
    final_epochs)."""

    epochs: int = 60
    """Epoch budget for each CV fold."""

    final_epochs: Optional[int] = None
    """
    Epoch budget for the final, all-data refit that actually gets exported.
    If None (default), train.py derives it automatically as the mean of each
    fold's best-super-accuracy epoch across the num_folds CV runs, and prints
    the derived value. Set explicitly to skip re-deriving on a rerun (e.g.
    once you already know a good value from a previous CV run).
    """

    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: Literal['adamw', 'adam', 'sgd'] = 'adamw'
    use_lr_scheduler: bool = True
    """Cosine-anneal the LR over the run's own epoch budget (per-fold epochs
    for CV runs, final_epochs for the final refit)."""

    grad_clip_norm: Optional[float] = 1.0
    """Max gradient norm (None disables clipping). Helps stability on a
    small dataset / small batch size."""

    seed: int = 42

    # ------------------------------------------------------------------ #
    # Device / precision                                                   #
    # ------------------------------------------------------------------ #

    device: Literal['auto', 'cuda', 'cpu'] = 'auto'
    """'auto' picks cuda if available, else cpu."""

    use_amp: Optional[bool] = None
    """None = auto (True iff the resolved device is cuda). Explicit True/
    False overrides. train.py always prints the resolved device + AMP state
    once at startup."""

    # ------------------------------------------------------------------ #
    # Data loading                                                         #
    # ------------------------------------------------------------------ #

    num_workers: int = 0
    """DataLoader worker count. Defaults to 0 (main-process loading) --
    safer on Windows (spawn-based multiprocessing overhead/pitfalls) and
    fine given cache_images_in_memory for a small dataset."""

    cache_images_in_memory: bool = True
    """Decode every PNG once at Dataset init and keep it as an in-memory
    uint8 array, rather than re-reading from disk every __getitem__. Assumes
    a small dataset (a few hundred images at ~9.4MB/image raw is trivial
    RAM); disable if the real dataset turns out to be much larger."""

    # ------------------------------------------------------------------ #
    # Train-time augmentation (applied to the RGB image + cluster channels,#
    # BEFORE the model's own Preprocess step -- see dataset.py)            #
    # ------------------------------------------------------------------ #

    augment: bool = True
    aug_hflip_p: float = 0.5
    aug_vflip_p: float = 0.0
    """Off by default -- the chip images likely have a canonical orientation
    (markers at the top), so vertical flips may not reflect real variation."""

    aug_rotate_degrees: float = 10.0
    """Small-angle-only random rotation, to avoid heavy border artifacts."""

    aug_rotate_fill: int = 127
    aug_gaussian_noise_std: float = 0.02
    """
    Fraction of the 0-255 range; 0 disables. Applied to rgb only (adding
    noise to a binary cluster-channel mask is meaningless).

    No brightness/contrast jitter here (deliberately removed) -- it would
    create (jittered RGB, unjittered channels) training pairs that don't
    correspond to any real photo, since the channels were computed by
    MATLAB from the image's true, un-jittered colors. A standard and
    normally-fine tradeoff for ordinary image augmentation, but skipped
    here given how tightly RGB and channels are coupled for this task.
    """

    aug_channel_shuffle: bool = True
    """
    Per-sample random permutation of the 8 cluster-channel inputs (see
    model.ChannelsModel.forward's channel_perm and train.py's channel-
    shuffle helpers), applied at train time only, never at eval or
    inference -- forces the channels branch to learn features that
    generalize across channel position rather than memorizing "channel
    slot 5 is always cluster 6", which is otherwise an easy shortcut for a
    small model on a small dataset. The label vector is permuted with the
    EXACT SAME per-sample permutation before computing the loss, so
    label[k] still means "is whatever cluster now sits at channel k
    selected".
    """

    # ------------------------------------------------------------------ #
    # ONNX export                                                          #
    # ------------------------------------------------------------------ #

    onnx_opset: int = 18
    """
    Preprocess originally used an antialiased resize, which would have
    needed opset 18+ for the Resize op's antialias attribute -- but
    aten::_upsample_bilinear2d_aa turned out to have no ONNX symbolic
    mapping at all in torch 2.6's exporter (confirmed empirically), so
    antialias was dropped instead (see preprocess.py). 18 is kept anyway
    as a reasonably current opset for everything else in the graph.
    """

    onnx_dummy_export_size: tuple = (300, 400)
    """(H, W) of the dummy input used to trace the ONNX graph. Deliberately
    different from (resize_height, resize_width) -- if export/self-check
    still passes, it proves the exported graph really does accept dynamic
    input resolutions rather than silently baking in the training size."""

    # ------------------------------------------------------------------ #
    # Serialization helpers                                               #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Persist config to JSON for reproducibility."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> ChannelsModelConfig:
        """Load config from a JSON file. Keys prefixed with '_' are comments."""
        with open(path) as f:
            data = json.load(f)
        data = {k: v for k, v in data.items() if not k.startswith('_')}
        if 'onnx_dummy_export_size' in data:
            data['onnx_dummy_export_size'] = tuple(data['onnx_dummy_export_size'])
        return cls(**data)

    @classmethod
    def from_args(cls, args) -> ChannelsModelConfig:
        """Build a config from a parsed argparse.Namespace: load from
        args.config if given, then apply non-None CLI overrides on top."""
        if getattr(args, 'config', None) is not None:
            config = cls.load(args.config)
            print(f"Loaded config from {args.config}")
        else:
            config = cls()

        overrides = {
            'images_dir': getattr(args, 'images_dir', None),
            'labels_dir': getattr(args, 'labels_dir', None),
            'output_dir': getattr(args, 'output_dir', None),
            'epochs': getattr(args, 'epochs', None),
            'final_epochs': getattr(args, 'final_epochs', None),
            'num_folds': getattr(args, 'folds', None),
            'batch_size': getattr(args, 'batch_size', None),
            'lr': getattr(args, 'lr', None),
        }
        for attr, value in overrides.items():
            if value is not None:
                setattr(config, attr, value)

        return config
