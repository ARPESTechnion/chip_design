"""
flake_detection — superconductor chip flake detection pipeline.

Quick start:
    from flake_detection import detect_flakes, DetectionConfig

    config = DetectionConfig()
    # Set config.hsv_range (or rgb_range) to your material's color,
    # then tune texture and isolation thresholds using results.json output.

    results = detect_flakes('/path/to/chip/images', config)
    good_images = [r.image_path for r in results]
"""

from .config import DetectionConfig, RGBRange, HSVRange
from .detector import detect_flakes, ImageResult, FlakeMetrics, DarkSpotInfo

__all__ = [
    'DetectionConfig',
    'RGBRange',
    'HSVRange',
    'detect_flakes',
    'ImageResult',
    'FlakeMetrics',
    'DarkSpotInfo',
]
