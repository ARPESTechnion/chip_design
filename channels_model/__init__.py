"""
channels_model -- color-cluster channel-selection model.

Imitates the manual step in the legacy MATLAB tool (GDS_Code.m) where a user
picks which of an image's 8 color-quantization clusters (from a real
rgb2ind minimum-variance quantization call, see matlab_clustering.py -- NOT
grayscale bit-planes, an earlier assumption this project started with and
corrected once GDS_Code.m was actually read) carry useful structure. Trains
a small dual-branch residual CNN (RGB image + the 8 binary cluster-channel
maps) to predict that same 8-d selection, and exports it to ONNX with
resize/normalize preprocessing baked into the graph (cluster-channel
creation itself is NOT part of the graph, since it's a real MATLAB call --
see matlab_clustering.py).

Quick start:
    python -m channels_model.train --images-dir path/to/images --labels-dir path/to/labels

See channels_model/prompt.docx for the original spec.

matlab_clustering.py (create_channels / create_channels_cached) is meant to
be importable by other packages (e.g. grid_preperation) WITHOUT pulling in
torch -- its own code has no torch dependency, but a plain `from .model
import ...` here would still drag torch in as a side effect of importing
this package at all (Python runs __init__.py before any submodule import).
ChannelsModel/ChannelsModelExport/Preprocess are therefore resolved lazily
(PEP 562 module __getattr__) so `import channels_model.matlab_clustering`
stays torch-free; ChannelsModelConfig has no torch dependency itself and is
imported eagerly.
"""

from .config import ChannelsModelConfig

__all__ = [
    'ChannelsModelConfig',
    'ChannelsModel',
    'ChannelsModelExport',
    'Preprocess',
]

_LAZY = {
    'ChannelsModel': ('.model', 'ChannelsModel'),
    'ChannelsModelExport': ('.model', 'ChannelsModelExport'),
    'Preprocess': ('.preprocess', 'Preprocess'),
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        module_name, attr_name = _LAZY[name]
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value  # cache so subsequent access skips __getattr__
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
