"""
Diagnostic visualization -- NOT part of the production pipeline.

For every image, draws:
  - red crosshair marks at the two pixel positions find_top_markers() (the
    real calibration code) uses for placement -- since the fix (see
    markers.py module docstring), this IS the anchor square's own centroid,
    not the raw crosshair template-match position.
  - green rings at the center of every solid pink grid-dot "square" found in
    the image (markers._find_squares).

The two should now coincide almost exactly at the two anchor markers (that
was the point of the fix) -- this tool is what surfaced the original
crosshair-vs-square bias and is kept around to re-verify it stays fixed.

Usage:
  python -m grid_preperation.debug_markers --input_folder flakes/all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from .config import CifExportConfig
    from . import markers
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from grid_preperation.config import CifExportConfig
    from grid_preperation import markers

from flake_detection.config import DetectionConfig

SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

CROSS_COLOR = (0, 0, 255)     # red (BGR) -- detected crosshair position
SQUARE_COLOR = (0, 255, 0)    # green (BGR) -- detected square center
ALERT_COLOR = (0, 0, 255)


def _draw_cross(img: np.ndarray, x: float, y: float, color, size: int = 25, gap: int = 9, thickness: int = 2) -> None:
    """
    Plus-sign with a gap around the center (rather than solid lines through
    it) and a small center dot -- leaves room for _draw_dot's ring to sit
    concentrically around the center without either mark hiding the other,
    since a crosshair and its square are often only ~1px apart.
    """
    xi, yi = int(round(x)), int(round(y))
    cv2.line(img, (xi - size, yi), (xi - gap, yi), color, thickness)
    cv2.line(img, (xi + gap, yi), (xi + size, yi), color, thickness)
    cv2.line(img, (xi, yi - size), (xi, yi - gap), color, thickness)
    cv2.line(img, (xi, yi + gap), (xi, yi + size), color, thickness)
    cv2.circle(img, (xi, yi), 2, color, -1)


def _draw_dot(img: np.ndarray, x: float, y: float, color, radius: int = 9) -> None:
    """Hollow ring, radius chosen to sit just outside _draw_cross's center
    gap so a concentric cross+ring stay both visible instead of overlapping."""
    cv2.circle(img, (int(round(x)), int(round(y))), radius, color, 2)


def _draw_legend(img: np.ndarray) -> None:
    h = img.shape[0]
    cv2.putText(img, "red = detected crosshair (used for placement)", (15, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, CROSS_COLOR, 2, cv2.LINE_AA)
    cv2.putText(img, "green = detected square center", (15, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, SQUARE_COLOR, 2, cv2.LINE_AA)


def _process_image(
    img_path: Path,
    config: CifExportConfig,
    detection: DetectionConfig,
) -> tuple[np.ndarray | None, dict]:
    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        return None, {'image': img_path.name, 'status': 'alert', 'reason': 'could not read image'}

    annotated = image_bgr.copy()
    _draw_legend(annotated)

    squares = markers._find_squares(image_bgr, config)
    result: dict = {'image': img_path.name, 'num_squares': len(squares)}

    try:
        pair = markers.find_top_markers(image_bgr, config, detection)
        _draw_cross(annotated, *pair.left_px, CROSS_COLOR)
        _draw_cross(annotated, *pair.right_px, CROSS_COLOR)
        result['status'] = 'ok'
        result['left_px'] = [round(v, 1) for v in pair.left_px]
        result['right_px'] = [round(v, 1) for v in pair.right_px]
    except markers.MarkerDetectionError as e:
        result['status'] = 'alert'
        result['reason'] = str(e)
        cv2.putText(annotated, f"ALERT: {e}", (15, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ALERT_COLOR, 2, cv2.LINE_AA)

    # Drawn after the crosshairs so the ring is never hidden by the cross's
    # center dot when the two nearly coincide (see _draw_cross/_draw_dot).
    for sx, sy, _area in squares:
        _draw_dot(annotated, sx, sy, SQUARE_COLOR)

    return annotated, result


def run(input_folder: Path, config: CifExportConfig, output_folder: Path | None = None) -> list[dict]:
    output_folder = Path(output_folder or (input_folder / 'marker_debug'))
    output_folder.mkdir(parents=True, exist_ok=True)
    detection = config.load_detection_config()

    image_paths = sorted(
        p for p in input_folder.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not image_paths:
        print(f"No supported images found in {input_folder}")
        return []

    print(f"Processing {len(image_paths)} images...")
    results = []
    for img_path in image_paths:
        annotated, result = _process_image(img_path, config, detection)
        if annotated is not None:
            cv2.imwrite(str(output_folder / img_path.name), annotated)
        results.append(result)
        if result['status'] == 'ok':
            print(f"  {img_path.name}: crosses at {result['left_px']} / {result['right_px']}, "
                  f"{result['num_squares']} square(s)")
        else:
            print(f"  {img_path.name}: ALERT - {result['reason']} ({result['num_squares']} square(s) found anyway)")

    with open(output_folder / 'marker_debug_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    num_ok = sum(1 for r in results if r['status'] == 'ok')
    print(f"\nDone. {num_ok}/{len(results)} images with both crosshairs found.")
    print(f"Annotated images -> {output_folder}")
    return results


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='grid_preperation.debug_markers',
        description='Overlay detected crosshair and square-marker positions on every image.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--input_folder', type=Path, required=True,
                    help='Folder of images (png/jpg/tif/bmp).')
    p.add_argument('--output', '-o', type=Path, default=None, dest='output_folder',
                    help='Output folder for annotated images. Defaults to <input_folder>/marker_debug/.')
    p.add_argument('--config', '-c', type=Path, default=None,
                    help='Path to a CifExportConfig JSON. Defaults to CifExportConfig() built-ins.')
    return p


def main() -> None:
    args = _build_parser().parse_args()
    config = CifExportConfig.load(args.config) if args.config is not None else CifExportConfig()
    run(args.input_folder, config, args.output_folder)


if __name__ == '__main__':
    main()
