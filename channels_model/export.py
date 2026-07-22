"""
ONNX export + self-check for a trained ChannelsModel.

Usage:
  python -m channels_model.export --checkpoint channels_model/outputs/final_model.pt \\
      --config channels_model/outputs/effective_config.json \\
      --output channels_model/outputs/channels_model.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime
import torch

try:
    from .config import ChannelsModelConfig
    from .model import ChannelsModel, ChannelsModelExport
    from .preprocess import NUM_CHANNELS
except ImportError:
    # Running as `python channels_model/export.py` rather than
    # `python -m channels_model.export` -- add the project root to sys.path
    # so absolute imports resolve correctly.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from channels_model.config import ChannelsModelConfig
    from channels_model.model import ChannelsModel, ChannelsModelExport
    from channels_model.preprocess import NUM_CHANNELS


def export_onnx(model: ChannelsModel, config: ChannelsModelConfig, path: str | Path) -> None:
    """Export model (wrapped with a final sigmoid, see ChannelsModelExport)
    to ONNX with dynamic batch/height/width axes on BOTH inputs, then
    self-check the exported graph against the live PyTorch model before
    trusting it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    export_model = ChannelsModelExport(model).eval().to('cpu')
    h, w = config.onnx_dummy_export_size
    dummy_rgb = torch.rand(1, 3, h, w) * 255.0  # (N,3,H,W) RGB -- see preprocess.py's input contract
    dummy_channels = (torch.rand(1, NUM_CHANNELS, h, w) > 0.5).float()  # (N,8,H,W) {0,1}

    torch.onnx.export(
        export_model,
        (dummy_rgb, dummy_channels),
        str(path),
        input_names=['rgb', 'channels'],
        output_names=['channel_probs'],
        dynamic_axes={
            'rgb': {0: 'batch', 2: 'height', 3: 'width'},
            'channels': {0: 'batch', 2: 'height', 3: 'width'},
            'channel_probs': {0: 'batch'},
        },
        opset_version=config.onnx_opset,
        do_constant_folding=True,
    )
    print(f"Exported ONNX model -> {path}")

    _self_check(export_model, path, dummy_rgb, dummy_channels, config)


def _self_check(
    export_model: ChannelsModelExport,
    path: Path,
    dummy_rgb: torch.Tensor,
    dummy_channels: torch.Tensor,
    config: ChannelsModelConfig,
) -> None:
    """Load the exported graph back via onnxruntime and verify it: (a) runs
    on the exact dummy inputs used to trace it, (b) ALSO runs on
    differently-shaped inputs -- proving the dynamic H/W axes actually work,
    not just the traced shape -- and (c) numerically matches the live
    PyTorch model within a tight tolerance, catching any silent export
    divergence."""
    providers = (
        ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if torch.cuda.is_available() else ['CPUExecutionProvider']
    )
    sess = onnxruntime.InferenceSession(str(path), providers=providers)

    def run_and_compare(rgb: torch.Tensor, channels: torch.Tensor, label: str) -> None:
        onnx_out = sess.run(None, {
            'rgb': rgb.numpy().astype(np.float32),
            'channels': channels.numpy().astype(np.float32),
        })[0]
        with torch.no_grad():
            torch_out = export_model(rgb, channels).numpy()
        assert onnx_out.shape == torch_out.shape == (rgb.shape[0], 8), \
            f"{label}: unexpected output shape {onnx_out.shape}"
        assert (onnx_out >= 0).all() and (onnx_out <= 1).all(), \
            f"{label}: output outside [0,1]"
        assert np.allclose(onnx_out, torch_out, atol=1e-4), \
            f"{label}: ONNX/PyTorch output mismatch (max abs diff {np.abs(onnx_out - torch_out).max()})"
        print(f"  self-check OK ({label}): shape={onnx_out.shape}, "
              f"max|onnx-torch|={np.abs(onnx_out - torch_out).max():.2e}")

    run_and_compare(dummy_rgb, dummy_channels, f"dummy input {tuple(dummy_rgb.shape)}")

    # A DIFFERENT shape than the one traced with -- proves dynamic_axes on
    # height/width actually works, not just the traced shape.
    dh, dw = config.onnx_dummy_export_size
    other_h, other_w = max(64, dh // 2), max(64, dw // 2)
    other_rgb = torch.rand(1, 3, other_h, other_w) * 255.0
    other_channels = (torch.rand(1, NUM_CHANNELS, other_h, other_w) > 0.5).float()
    run_and_compare(other_rgb, other_channels, f"different shape {tuple(other_rgb.shape)}")

    print(f"ONNX export verified. Providers in use: {sess.get_providers()}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='channels_model.export',
        description='Export a trained ChannelsModel checkpoint to ONNX.',
    )
    p.add_argument('--checkpoint', type=Path, required=True, help='Path to a .pt state_dict.')
    p.add_argument('--config', '-c', type=Path, default=None,
                    help='Path to the ChannelsModelConfig JSON used for training. Defaults to built-ins.')
    p.add_argument('--output', '-o', type=Path, required=True, help='Destination .onnx path.')
    return p


def main() -> None:
    args = _build_parser().parse_args()
    config = ChannelsModelConfig.load(args.config) if args.config is not None else ChannelsModelConfig()
    model = ChannelsModel(config)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    model.eval()
    export_onnx(model, config, args.output)


if __name__ == '__main__':
    main()
