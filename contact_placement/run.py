"""
CLI entry point for the automated contact/cutting-mask placement stage.

Usage examples:

  # Run with all defaults against a grid_preperation run's own --output folder
  python -m contact_placement.run --grid_preperation_output_dir /path/to/gp_output

  # Use a saved ContactPlacementConfig
  python -m contact_placement.run --grid_preperation_output_dir /path/to/gp_output --config my_config.json

  # Save the effective config used (useful for reproducibility)
  python -m contact_placement.run --grid_preperation_output_dir /path/to/gp_output --save-config run_config.json

Headless counterpart to `ui.py`'s interactive preview-then-export flow: runs
`layout.place_all` and commits STRAIGHT to CIF, no preview step -- use
`ui.py` instead when you want to inspect a placement before committing.

Safety: like grid_preperation.run, this never modifies the ORIGINAL master
template in place -- it splices into the working template COPY
grid_preperation.run() already made inside grid_preperation_output_dir
(found by locating the one .cif file sitting directly in that folder; see
cif_export.find_working_template).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from .cif_export import find_working_template, place_in_cif, write_placement_cif
    from .config import ContactPlacementConfig
    from .layout import place_all
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from contact_placement.cif_export import find_working_template, place_in_cif, write_placement_cif
    from contact_placement.config import ContactPlacementConfig
    from contact_placement.layout import place_all


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='contact_placement',
        description='Place electrical contacts + cutting masks onto already-exported flakes.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--grid_preperation_output_dir', type=Path, default=None,
                    help="A grid_preperation run's own --output folder (contains cif/, flakes/, "
                         "and the working template copy). Overrides the config file's own value "
                         "if both are given.")
    p.add_argument('--output', '-o', type=Path, default=None, dest='output_folder',
                    help='Output folder for the exported contacts CIF and results.json. '
                         'Defaults to <grid_preperation_output_dir>/contact_placement_results/.')
    p.add_argument('--config', '-c', type=Path, default=None,
                    help='Path to a ContactPlacementConfig JSON. Defaults to built-ins.')
    p.add_argument('--save-config', type=Path, default=None, metavar='PATH',
                    help='Save the effective config to a JSON file after parsing all arguments.')
    return p


def _load_config(args: argparse.Namespace) -> ContactPlacementConfig:
    config = ContactPlacementConfig.load(args.config) if args.config is not None else ContactPlacementConfig()
    if args.grid_preperation_output_dir is not None:
        config.grid_preperation_output_dir = str(args.grid_preperation_output_dir)
    if args.output_folder is not None:
        config.output_dir = str(args.output_folder)
    return config


def run(config: ContactPlacementConfig) -> list[dict]:
    if config.grid_preperation_output_dir is None:
        raise ValueError("grid_preperation_output_dir is required (--grid_preperation_output_dir or config file)")
    grid_dir = Path(config.grid_preperation_output_dir)

    output_dir = Path(config.output_dir) if config.output_dir else grid_dir / 'contact_placement_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Placing contacts...")
    results = place_all(config)
    placed = [r for r in results if r.mode is not None]
    print(f"{len(placed)}/{len(results)} flakes placed "
          f"({sum(1 for r in placed if r.mode == 'hall')} hall, "
          f"{sum(1 for r in placed if r.mode == 'ring')} ring)")

    summary = [
        {
            'image': r.candidate.image, 'flake_index': r.candidate.flake_index,
            'final_score': r.candidate.final_score, 'mode': r.mode,
        }
        for r in results
    ]
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary -> {output_dir / 'results.json'}")

    if not placed:
        print("No flakes placed -- nothing to export.")
        return summary

    export_cif = output_dir / 'contacts.cif'
    write_placement_cif(results, config, export_cif)
    print(f"Exported CIF -> {export_cif}")

    template = find_working_template(grid_dir)
    if template is None:
        print(f"ALERT: no single working template .cif found directly in {grid_dir} -- "
              f"skipping placement into master template (contacts.cif above is still valid standalone).")
        return summary

    place_in_cif(export_cif, template)
    print(f"Spliced into working template -> {template}")
    return summary


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = _load_config(args)

    if args.save_config is not None:
        config.save(args.save_config)
        print(f"Config saved to {args.save_config}")

    run(config)


if __name__ == '__main__':
    main()
