"""
CLI entry point for the full pipeline: flake_detection selects/ranks candidate
images, then grid_preperation exports the selected images to CIF and splices
them into the master template.

Usage examples:

  # Run with defaults from pipeline_config.json
  python all_code.py --input_folder flakes/all

  # Use a different top-level config (which stage configs to use)
  python all_code.py --input_folder flakes/all --config pipeline_config.json

  # Export only, no placement into a master template
  python all_code.py --input_folder flakes/all --no-placement
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flake_detection.config import DetectionConfig
from flake_detection.detector import detect_flakes
from grid_preperation.config import CifExportConfig
from grid_preperation.run import run as run_grid_preperation
from pipeline_config import PipelineConfig


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='all_code',
        description='Run the full pipeline: flake_detection then grid_preperation.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--input_folder', type=Path, required=True,
                    help='Folder of raw chip images (png/jpg/tif/bmp).')
    p.add_argument('--output', '-o', type=Path, default=None, dest='output_folder',
                    help='Root output folder. Defaults to <input_folder>/pipeline_results/.')
    p.add_argument('--config', '-c', type=Path, default=Path('pipeline_config.json'),
                    help='Path to a PipelineConfig JSON pointing at the per-stage configs.')
    p.add_argument('--template', type=Path, default=None,
                    help='Override the master template CIF path for the grid_preperation stage.')
    p.add_argument('--no-placement', action='store_true',
                    help='Only write per-image CIF files; skip placement into a master template.')
    return p


def main() -> None:
    args = _build_parser().parse_args()
    pipeline_config = PipelineConfig.load(args.config)

    output_folder = args.output_folder or (args.input_folder / 'pipeline_results')
    detection_output = output_folder / 'flake_detection_results'
    cif_output = output_folder / 'cif_export_results'

    print(f"=== Stage 1/2: flake_detection ({pipeline_config.detection_config_path}) ===")
    detection_config = DetectionConfig.load(pipeline_config.detection_config_path)
    results = detect_flakes(args.input_folder, detection_config, detection_output)

    if not results:
        print("\nNo images passed flake_detection -- nothing to export.")
        return

    selected_paths = [Path(r.image_path) for r in results]
    print(f"\n{len(selected_paths)} image(s) selected for CIF export.")

    if pipeline_config.cif_export_config_path is not None:
        cif_config = CifExportConfig.load(pipeline_config.cif_export_config_path)
    else:
        cif_config = CifExportConfig()
    # Never let the two stages disagree on color/marker ranges.
    cif_config.detection_config_path = pipeline_config.detection_config_path
    if args.template is not None:
        cif_config.template_cif_path = str(args.template)
    if args.no_placement:
        cif_config.template_cif_path = None

    print("\n=== Stage 2/2: grid_preperation ===")
    run_grid_preperation(args.input_folder, cif_config, cif_output, image_paths=selected_paths)


if __name__ == '__main__':
    main()
