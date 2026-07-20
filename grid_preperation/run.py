"""
CLI entry point for the automated flake -> CIF export & placement pipeline.

Usage examples:

  # Run with all defaults against a folder of images already selected by
  # flake_detection
  python -m grid_preperation.run --input_folder flakes/all

  # Use a saved CifExportConfig and an explicit master template
  python -m grid_preperation.run --input_folder flakes/all --config my_config.json --template grid_preperation/BaseGrid.cif

  # Export only, no placement into a master template
  python -m grid_preperation.run --input_folder flakes/all --no-placement

Safety: this never modifies template_cif_path in place. On first run it
copies the template into the output folder once; every placement (across
this run and any future runs pointed at the same --output) is spliced into
that working copy, which you can inspect and promote to replace your real
master template when you're satisfied with it.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

try:
    from .config import CifExportConfig
    from . import calibration, cif_io, markers, polygons
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from grid_preperation.config import CifExportConfig
    from grid_preperation import calibration, cif_io, markers, polygons

import cv2

SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='grid_preperation',
        description='Convert flake images into CIF polygons and place them into a master chip template.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--input_folder', type=Path, required=True,
                    help='Folder of images already selected by flake_detection (png/jpg/tif/bmp).')
    p.add_argument('--output', '-o', type=Path, default=None, dest='output_folder',
                    help='Output folder for per-image CIF files, the working template copy, '
                         'and results.json. Defaults to <input_folder>/cif_export_results/.')
    p.add_argument('--config', '-c', type=Path, default=None,
                    help='Path to a CifExportConfig JSON. Defaults to CifExportConfig() built-ins.')
    p.add_argument('--detection-config', type=Path, default=None,
                    help='Override the DetectionConfig JSON path (color/marker ranges).')
    p.add_argument('--template', type=Path, default=None,
                    help='Override the master template CIF path.')
    p.add_argument('--no-placement', action='store_true',
                    help='Only write per-image CIF files; skip placement into a master template.')
    p.add_argument('--save-config', type=Path, default=None, metavar='PATH',
                    help='Save the effective config to a JSON file after parsing all arguments.')
    return p


def _load_config(args: argparse.Namespace) -> CifExportConfig:
    config = CifExportConfig.load(args.config) if args.config is not None else CifExportConfig()
    if args.detection_config is not None:
        config.detection_config_path = str(args.detection_config)
    if args.template is not None:
        config.template_cif_path = str(args.template)
    if args.no_placement:
        config.template_cif_path = None
    return config


def _process_image(
    img_path: Path,
    config: CifExportConfig,
    detection,
    cif_dir: Path,
    working_template_path: Path | None,
    palette: cif_io.LayerPalette,
) -> dict:
    """Run the full pipeline on one image. Returns a result dict for the
    summary JSON — never raises; alert conditions are reported, not thrown."""
    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        return {'image': img_path.name, 'status': 'alert', 'reason': 'could not read image'}

    try:
        marker_pair = markers.find_top_markers(image_bgr, config, detection)
    except markers.MarkerDetectionError as e:
        return {'image': img_path.name, 'status': 'alert', 'reason': f'marker detection: {e}'}

    try:
        cal = calibration.calibrate(img_path, marker_pair, config)
    except calibration.CalibrationError as e:
        return {'image': img_path.name, 'status': 'alert', 'reason': f'calibration: {e}'}

    polys = polygons.extract_polygons(image_bgr, cal, config, detection)
    cif_path = cif_dir / f"{img_path.stem}.cif"
    cif_io.write_cif(polys, cif_path, palette)

    result = {
        'image': img_path.name,
        'status': 'ok',
        'grid_id': cal.left_id,
        'num_polygons': len(polys),
        'polygon_scores': [round(p.final_score, 4) for p in polys],
        'cif_path': str(cif_path),
        'placed': False,
    }

    if not polys:
        # Nothing cleared gds_min_area / min_color_fraction / min_flake_score
        # -- skip placement entirely rather than splicing an empty no-op.
        result['reason'] = 'no qualifying flakes (below min_flake_score or other filters) - not placed'
        return result

    if working_template_path is not None:
        try:
            cif_io.place_in_cif(cif_path, working_template_path, cal)
            result['placed'] = True
        except Exception as e:  # noqa: BLE001 - placement failure is an alert, not a crash
            result['status'] = 'alert'
            result['reason'] = f'placement: {e}'

    return result


def run(
    input_folder: Path,
    config: CifExportConfig,
    output_folder: Path | None = None,
    image_paths: list[Path] | None = None,
) -> list[dict]:
    """
    Args:
        image_paths: explicit images to process, e.g. the subset already
            selected by flake_detection (see all_code.py). Defaults to every
            supported image directly under input_folder.
    """
    output_folder = output_folder or (input_folder / 'cif_export_results')
    output_folder = Path(output_folder)
    cif_dir = output_folder / 'cif'
    cif_dir.mkdir(parents=True, exist_ok=True)

    detection = config.load_detection_config()
    palette = cif_io.LayerPalette(
        config.flake_layer_palette, config.palette_max_distance, config.palette_quantize_step
    )

    working_template_path = None
    if config.template_cif_path is not None:
        template_src = Path(config.template_cif_path)
        working_template_path = output_folder / template_src.name
        if not working_template_path.exists():
            shutil.copy2(template_src, working_template_path)
            print(f"Copied template {template_src} -> {working_template_path} (working copy; "
                  f"original left untouched)")
        else:
            print(f"Reusing existing working template copy at {working_template_path}")

    if image_paths is None:
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
        result = _process_image(img_path, config, detection, cif_dir, working_template_path, palette)
        results.append(result)
        if result['status'] == 'ok':
            placed_note = f", placed at {result['grid_id']}" if result['placed'] else ""
            print(f"  {img_path.name}: {result['num_polygons']} polygon(s){placed_note}")
        else:
            print(f"  {img_path.name}: ALERT - {result['reason']}")

    import json
    with open(output_folder / 'cif_export_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    num_ok = sum(1 for r in results if r['status'] == 'ok')
    num_alert = len(results) - num_ok
    print(f"\nDone. {num_ok} succeeded, {num_alert} alerted/skipped.")
    print(f"Per-image CIFs -> {cif_dir}")
    if working_template_path is not None:
        print(f"Working template -> {working_template_path}")
    print(f"Summary         -> {output_folder / 'cif_export_results.json'}")

    return results


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = _load_config(args)

    if args.save_config is not None:
        config.save(args.save_config)
        print(f"Config saved to {args.save_config}")

    run(args.input_folder, config, args.output_folder)


if __name__ == '__main__':
    main()
