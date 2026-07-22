"""
Dual-branch residual CNN: RGB image + 8 color-cluster channel maps in, an
8-d multi-label logit vector out (which color clusters are "the channels to
take" -- see matlab_clustering.py for how the clusters are computed).

Deliberately small (few hundred thousand params) -- the spec calls for a
small model to match a small dataset. See preprocess.py for how the two
branches' inputs are resized/normalized, and the module docstring below for
the logits-vs-probabilities split.
"""

from __future__ import annotations

import torch
from torch import nn

from .preprocess import NUM_CHANNELS, Preprocess


class ResidualBlock(nn.Module):
    """Conv3x3-BN-ReLU-Conv3x3-BN, + identity skip, ReLU. Stride 1,
    in_channels == out_channels throughout this model -- all spatial/width
    transitions happen in the stride-2 convs between residual blocks, so no
    projection shortcut is ever needed here."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


def _down_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Stride-2 Conv-BN-ReLU -- halves spatial resolution, changes channel
    width."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _branch(in_channels: int) -> nn.Sequential:
    """Stem shared by both the RGB and cluster-channel branches, differing
    only in in_channels (3 vs 8): downsample to 32 channels at 1/4
    resolution, 2 residual blocks after each downsample."""
    return nn.Sequential(
        _down_block(in_channels, 16),
        ResidualBlock(16),
        ResidualBlock(16),
        _down_block(16, 32),
        ResidualBlock(32),
        ResidualBlock(32),
    )


class ChannelsModel(nn.Module):
    """
    forward(rgb, channels) -> logits (N, 8).
      rgb      : RAW (N,3,H,W) float32 RGB [0,255] image.
      channels : (N,8,H,W) float32 {0,1} color-cluster channel maps (see
                 matlab_clustering.create_channels) -- computed in the
                 caller (dataset.py for training; the flake pipeline's own
                 preprocessing for inference), NOT in this model, since
                 clustering is a real MATLAB call, not an ONNX-traceable
                 tensor op. See preprocess.Preprocess for the resize/
                 normalize contract and matlab_clustering.py for why every
                 caller must compute channels identically.

    Returns raw logits (not probabilities) so training can use
    BCEWithLogitsLoss's numerically-stable formulation. For a deployable
    [0,1]-valued ONNX output, see ChannelsModelExport below (used only by
    export.py, never by training).

    channel_perm (optional): per-sample cluster-channel-shuffle
    augmentation -- see train.py's channel-shuffle helpers. Defaults to
    None (identity, i.e. no shuffle), which is what every inference/export
    path uses (ChannelsModelExport never passes this, so the traced ONNX
    graph contains no gather/shuffle ops at all) -- shuffling is a
    TRAINING-ONLY regularization, never something that should touch a real
    deployed prediction.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.preprocess = Preprocess(
            resize_height=config.resize_height,
            resize_width=config.resize_width,
            normalize_mean=config.normalize_mean,
            normalize_std=config.normalize_std,
        )
        self.rgb_branch = _branch(3)
        self.channels_branch = _branch(NUM_CHANNELS)

        # Fusion: combine the two branches into a single stream.
        self.fuse_proj = nn.Sequential(
            nn.Conv2d(32 + 32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.fuse_res = nn.Sequential(ResidualBlock(64), ResidualBlock(64))
        self.fuse_down = _down_block(64, 96)
        self.fuse_res2 = ResidualBlock(96)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.3)
        self.head = nn.Linear(96, NUM_CHANNELS)

    def forward(
        self, rgb: torch.Tensor, channels: torch.Tensor,
        channel_perm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rgb_norm, channels_r = self.preprocess(rgb, channels)

        if channel_perm is not None:
            # (N, NUM_CHANNELS) int64 -- a per-sample permutation of the 8
            # cluster channels, reordering WHICH cluster sits at which
            # channel index. rgb_norm (the true, unshuffled image) still
            # flows through the RGB branch unshuffled -- only the channels
            # branch's input is reordered.
            n, c, h, w = channels_r.shape
            idx = channel_perm.view(n, c, 1, 1).expand(n, c, h, w)
            channels_r = torch.gather(channels_r, dim=1, index=idx)

        rgb_feat = self.rgb_branch(rgb_norm)
        chan_feat = self.channels_branch(channels_r)

        fused = torch.cat([rgb_feat, chan_feat], dim=1)
        fused = self.fuse_proj(fused)
        fused = self.fuse_res(fused)
        fused = self.fuse_down(fused)
        fused = self.fuse_res2(fused)

        pooled = self.pool(fused).flatten(1)
        pooled = self.dropout(pooled)
        return self.head(pooled)


class ChannelsModelExport(nn.Module):
    """ChannelsModel + a final sigmoid -- ONNX-export-only wrapper so the
    deployed graph's single output is already [0,1]-valued and directly
    thresholdable ("Output: the channels to take"), without training code
    ever having to know about it."""

    def __init__(self, model: ChannelsModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, rgb: torch.Tensor, channels: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(rgb, channels))
