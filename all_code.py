"""
CLI entry point for the full pipeline: flake_detection -> grid_preperation.

Runs flake_detection on a folder of raw chip images, then feeds every image
with at least one qualifying flake into grid_preperation to export CIF
polygons (and optionally place them into a master template).

Usage examples:

  # Run the full pipeline with default configs (input_folder read from
  # config/pipeline_config.json if set there; otherwise pass --input_folder)
  python all_code.py

  # Override input_folder on the command line (takes priority over the config file)
  python all_code.py --input_folder flakes/all

  # Use a custom top-level pipeline config (paths to each stage's own config)
  python all_code.py --config config/pipeline_config.json

  # Export only, skip placement into a master template
  python all_code.py --input_folder flakes/all --no-placement

Each stage's own settings (color ranges, marker thresholds, channels_model
paths, etc.) live in that stage's own config file -- see
flake_detection/config.py and grid_preperation/config.py. This file's own
config (PipelineConfig, config/pipeline_config.json) only holds PATHS to
those two files, so each stage remains independently runnable/configurable
via its own `python -m flake_detection.run` / `python -m grid_preperation.run`.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from flake_detection.config import DetectionConfig
from flake_detection.detector import detect_flakes
from grid_preperation.config import CifExportConfig
from grid_preperation.run import run as run_grid_preperation

_DEFAULT_PIPELINE_CONFIG_PATH = "config/pipeline_config.json"


@dataclass
class PipelineConfig:
    """Paths to each pipeline stage's own config file -- not the stage
    settings themselves. See this module's docstring."""

    flake_detection_config_path: str = "flake_detection/config/orange_flakes_tuned.json"
    grid_preperation_config_path: str = "grid_preperation/config/example_config.json"

    input_folder: Optional[str] = None
    """Folder of raw chip images. Set here to avoid passing --input_folder
    on every run; the CLI flag, when given, overrides this."""

    def save(self, path: str | Path) -> None:
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> PipelineConfig:
        with open(path) as f:
            data = json.load(f)
        data = {k: v for k, v in data.items() if not k.startswith('_')}
        return cls(**data)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='all_code',
        description='Run the full pipeline: flake_detection -> grid_preperation.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--input_folder', type=Path, default=None,
                    help='Folder of raw chip images (png/jpg/tif/bmp). Overrides '
                         'PipelineConfig.input_folder from --config; one of the two is required.')
    p.add_argument('--output', '-o', type=Path, default=None, dest='output_folder',
                    help='Base output folder. Defaults to <input_folder>/pipeline_results/. '
                         'flake_detection writes to <output>/flake_detection_results/, '
                         'grid_preperation writes to <output>/cif_export_results/.')
    p.add_argument('--config', '-c', type=Path, default=Path(_DEFAULT_PIPELINE_CONFIG_PATH),
                    help=f'Path to a PipelineConfig JSON (paths to each stage\'s own config). '
                         f'Defaults to {_DEFAULT_PIPELINE_CONFIG_PATH}.')
    p.add_argument('--template', type=Path, default=None,
                    help='Override grid_preperation\'s master template CIF path.')
    p.add_argument('--no-placement', action='store_true',
                    help='grid_preperation: only write per-image CIF files; skip placement '
                         'into a master template.')
    return p


def run_pipeline(
    input_folder: Path,
    pipeline_config: PipelineConfig,
    output_folder: Path | None = None,
    template_override: Path | None = None,
    no_placement: bool = False,
) -> tuple[list, list[dict]]:
    """Run flake_detection then grid_preperation. Returns
    (flake_detection_results, grid_preperation_results)."""
    input_folder = Path(input_folder)
    output_folder = Path(output_folder) if output_folder is not None else input_folder / 'pipeline_results'

    detection_config = DetectionConfig.load(pipeline_config.flake_detection_config_path)
    print(f"=== Stage 1/2: flake_detection (config: {pipeline_config.flake_detection_config_path}) ===")
    detection_results = detect_flakes(
        input_folder, detection_config, output_folder / 'flake_detection_results',
    )

    selected = [r for r in detection_results if r.flakes]
    if not selected:
        print("\nNo images with qualifying flakes -- nothing to pass to grid_preperation.")
        return detection_results, []

    staging_dir = output_folder / 'selected_for_grid_preperation'
    staging_dir.mkdir(parents=True, exist_ok=True)
    for r in selected:
        src = Path(r.image_path)
        shutil.copy2(src, staging_dir / src.name)
    print(f"\n{len(selected)} image(s) with qualifying flakes -> staged in {staging_dir}")

    export_config = CifExportConfig.load(pipeline_config.grid_preperation_config_path)
    if template_override is not None:
        export_config.template_cif_path = str(template_override)
    if no_placement:
        export_config.template_cif_path = None
    # Always wired to THIS run's own flake_detection output, overriding
    # whatever static value the grid_preperation config file happens to
    # have -- contact_placement's contour matching (see
    # CifExportConfig.flake_detection_output_dir) needs the two stages'
    # outputs to come from the SAME run. A stale/mismatched path here fails
    # silently (grid_preperation just finds no contours.json to match and
    # writes empty flakes/*.json, no error) -- confirmed empirically: an
    # earlier hardcoded config value pointed at a directory from a manual,
    # one-off `flake_detection.run` invocation outside this orchestrator,
    # and once that stray directory was gone, every image's flakes/*.json
    # silently went empty.
    export_config.flake_detection_output_dir = str(output_folder / 'flake_detection_results')

    print(f"\n=== Stage 2/2: grid_preperation (config: {pipeline_config.grid_preperation_config_path}) ===")
    export_results = run_grid_preperation(
        staging_dir, export_config, output_folder / 'cif_export_results',
    )

    return detection_results, export_results


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    pipeline_config = (
        PipelineConfig.load(args.config) if args.config.exists() else PipelineConfig()
    )

    input_folder = args.input_folder or (
        Path(pipeline_config.input_folder) if pipeline_config.input_folder else None
    )
    if input_folder is None:
        parser.error(
            "no input folder given -- pass --input_folder or set input_folder in "
            f"{args.config}"
        )

    run_pipeline(
        input_folder, pipeline_config, args.output_folder,
        template_override=args.template, no_placement=args.no_placement,
    )


if __name__ == '__main__':
    main()
