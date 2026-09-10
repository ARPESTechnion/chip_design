"""
CLI entry point for the flake detection pipeline.

Usage examples:

  # Run with all defaults (accept-all thresholds — good for first inspection)
  python -m flake_detection.run /path/to/chip/images

  # Specify output folder
  python -m flake_detection.run /path/to/images --output /path/to/results

  # Load a saved config file
  python -m flake_detection.run /path/to/images --config my_config.json

  # Override color mode on the command line
  python -m flake_detection.run /path/to/images --color-mode rgb

  # Return only the top 20 images
  python -m flake_detection.run /path/to/images --top-n 20

  # Save the effective config used (useful for reproducibility)
  python -m flake_detection.run /path/to/images --save-config run_config.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .config import DetectionConfig
    from .detector import detect_flakes
    from .ui import maybe_review
except ImportError:
    # Running as `python flake_detection/run.py` rather than `python -m flake_detection.run`.
    # Add the project root to sys.path so absolute imports resolve correctly.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from flake_detection.config import DetectionConfig
    from flake_detection.detector import detect_flakes
    from flake_detection.ui import maybe_review


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='flake_detection',
        description='Detect candidate superconductor flakes in chip images.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        '--input_folder',
        type=Path,
        help='Folder containing chip images (png/jpg/tif/bmp).',
    )
    p.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        dest='output_folder',
        help='Output folder for annotated images and results.json. '
             'Defaults to <input_folder>/flake_detection_results/.',
    )
    p.add_argument(
        '--config', '-c',
        type=Path,
        default="flake_detection/config/orange_flakes_tuned.json",
        help='Path to a JSON config file (produced by DetectionConfig.save()). '
             'Command-line overrides below are applied on top.',
    )
    p.add_argument(
        '--save-config',
        type=Path,
        default=None,
        metavar='PATH',
        help='Save the effective config to a JSON file after parsing all arguments.',
    )
    p.add_argument(
        '--color-mode',
        choices=['rgb', 'hsv'],
        default=None,
        help='Color space for masking. Overrides config file value.',
    )
    p.add_argument(
        '--min-area',
        type=int,
        default=None,
        metavar='PX',
        help='Minimum flake area in pixels.',
    )
    p.add_argument(
        '--max-area',
        type=int,
        default=None,
        metavar='PX',
        help='Maximum flake area in pixels.',
    )
    p.add_argument(
        '--min-score',
        type=float,
        default=None,
        metavar='[0-1]',
        help='Minimum final score for a flake to be included in results.',
    )
    p.add_argument(
        '--top-n',
        type=int,
        default=None,
        metavar='N',
        help='Return only the top N images by best flake score.',
    )

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config = DetectionConfig.from_args(args)

    if args.save_config is not None:
        config.save(args.save_config)
        print(f"Config saved to {args.save_config}")

    output_folder = args.output_folder or (args.input_folder / 'flake_detection_results')
    results = detect_flakes(args.input_folder, config, output_folder)

    maybe_review(results, output_folder, config)

    # Print a summary of the top results
    if results:
        print("\nTop results:")
        # for i, r in enumerate(results[:10], 1):
        for i, r in enumerate(results, 1):
            print(f"  {i:2d}. {Path(r.image_path).name}  "
                  f"best_score={r.best_score:.3f}  "
                  f"flakes={len(r.flakes)}")


if __name__ == '__main__':
    main()
