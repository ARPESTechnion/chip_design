"""
Shared, parameter-free preprocessing -- resize + normalize, used identically
for training AND baked into the exported ONNX graph, via one nn.Module with
no learnable parameters and no other code path.

Color-cluster channel creation is NOT part of this module -- it happens in
the surrounding pipeline instead (dataset.py for training; the flake
pipeline's own preprocessing for inference), because it's computed by a real
MATLAB call (see matlab_clustering.py) rather than anything expressible as
ONNX-traceable tensor ops. This means correctness depends on every caller of
this model computing the 8 cluster channels the SAME way (matlab_clustering.
create_channels / create_channels_cached, n_clusters=8, median_win=5) -- a
different clustering between training and inference would silently produce
different "channel" semantics.

NUM_CHANNELS = 8 matches config.n_clusters and the existing label files'
1-indexed convention (label k -> channel k-1 here, 0-indexed).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

NUM_CHANNELS = 8


class Preprocess(nn.Module):
    """
    Contract:
      rgb      : (N, 3, H, W) float32, RGB, values in [0, 255] (NOT
                 pre-normalized).
      channels : (N, 8, H, W) float32, binary cluster-channel maps (see
                 matlab_clustering.create_channels), values in {0., 1.}.
    H and W may be any size (need not match between calls) -- the resize
    target is fixed internally from config, so the exported ONNX graph
    accepts arbitrary input resolutions (see export.py's dynamic_axes on
    height/width).

    Returns (rgb_norm, channels_resized):
      rgb_norm         : (N, 3, resize_height, resize_width) float32,
                          normalized, for the RGB CNN branch.
      channels_resized : (N, 8, resize_height, resize_width) float32 in
                          {0., 1.}, for the cluster-channel CNN branch.
    """

    def __init__(self, resize_height: int, resize_width: int,
                 normalize_mean: float, normalize_std: float) -> None:
        super().__init__()
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

    def forward(self, rgb: torch.Tensor, channels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # RGB: NOT antialiased -- antialiased bilinear resize
        # (aten::_upsample_bilinear2d_aa) has no ONNX symbolic mapping in
        # torch 2.6's exporter (confirmed empirically -- torch.onnx.export
        # raises UnsupportedOperatorError at opset 18, the highest opset
        # this torch version supports, so there is no opset fallback that
        # fixes it). Plain bilinear resize is still correct, just slightly
        # blurrier on this ~8x downsample than an antialiased resize would
        # be -- acceptable since the model has to learn robust features
        # over the resized image regardless.
        rgb_r = F.interpolate(
            rgb, size=(self.resize_height, self.resize_width),
            mode='bilinear', align_corners=False,
        )
        rgb_norm = (rgb_r / 255.0 - self.normalize_mean) / self.normalize_std

        # Channels: nearest-neighbor -- bilinear (antialiased or not) would
        # blur a binary {0,1} mask into non-binary values at edges. Nearest
        # is trivially ONNX-exportable at any opset.
        channels_r = F.interpolate(
            channels, size=(self.resize_height, self.resize_width),
            mode='nearest',
        )

        return rgb_norm, channels_r
