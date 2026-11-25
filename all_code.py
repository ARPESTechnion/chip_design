"""
CLI entry point for the full pipeline: flake_detection -> grid_preperation ->
contact_placement.

Runs flake_detection on a folder of raw chip images, feeds every image with
at least one qualifying flake into grid_preperation to export CIF polygons
(and place them into a master template), then runs contact_placement
headlessly on that same output to place electrical contacts + cutting masks
and splice them into the same working template.

Usage examples:

  # Run the full pipeline with default configs (input_folder read from
  # config/pipeline_config.json if set there; otherwise pass --input_folder)
  python all_code.py

  # Override input_folder on the command line (takes priority over the config file)
  python all_code.py --input_folder flakes/all

  # Use a custom top-level pipeline config (paths to each stage's own config)
  python all_code.py --config config/pipeline_config.json

  # Export only, skip placement into a master template (also skips contact_placement --
  # there'd be no template to splice contacts into)
  python all_code.py --input_folder flakes/all --no-placement

  # Keep the master-template placement but skip the contact_placement stage
  python all_code.py --input_folder flakes/all --no-contacts

Each stage's own settings (color ranges, marker thresholds, channels_model
paths, contact geometry, etc.) live in that stage's own config file -- see
flake_detection/config.py, grid_preperation/config.py, and
contact_placement/config.py. This file's own config (PipelineConfig,
config/pipeline_config.json) only holds PATHS to those three files, so each
stage remains independently runnable/configurable via its own
`python -m flake_detection.run` / `python -m grid_preperation.run` /
`python -m contact_placement.run` (or `contact_placement.ui` for the
interactive preview-then-export flow instead of this file's headless one).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from contact_placement.config import ContactPlacementConfig
from contact_placement.run import run as run_contact_placement
from flake_detection.config import DetectionConfig
from flake_detection.detector import SUPPORTED_EXTENSIONS, detect_flakes
from flake_detection.ui import maybe_review
from grid_preperation.config import CifExportConfig
from grid_preperation.run import run as run_grid_preperation

_DEFAULT_PIPELINE_CONFIG_PATH = "config/pipeline_config.json"


@dataclass
class PipelineConfig:
    """Paths to each pipeline stage's own config file -- not the stage
    settings themselves. See this module's docstring."""

    flake_detection_config_path: str = "flake_detection/config/orange_flakes_tuned.json"
    grid_preperation_config_path: str = "grid_preperation/config/example_config.json"
    contact_placement_config_path: str = "contact_placement/config/example_config.json"

    input_folder: Optional[str] = None
    """Folder of raw chip images. Set here to avoid passing --input_folder
    on every run; the CLI flag, when given, overrides this."""

    interactive: bool = False
    """
    Master switch for every stage's own interactive UI (flake_detection's
    review window today; more stages may grow one later, e.g.
    contact_placement's own popping UI could gain a headless-pipeline hook
    the same way). False forces every such UI off for this run regardless
    of what any individual stage's own config file says -- e.g.
    DetectionConfig.interactive_review is only honored when this is also
    True. True does NOT force a stage's UI on by itself; each stage's own
    config flag still decides whether THAT stage pops one. Use this to
    quickly batch-run the full pipeline headlessly without editing every
    stage's config file back and forth.
    """

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
        description='Run the full pipeline: flake_detection -> grid_preperation -> contact_placement.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--input_folder', type=Path, default=None,
                    help='Folder of raw chip images (png/jpg/tif/bmp). Overrides '
                         'PipelineConfig.input_folder from --config; one of the two is required.')
    p.add_argument('--output', '-o', type=Path, default=None, dest='output_folder',
                    help='Base output folder. Defaults to <input_folder>/pipeline_results/. '
                         'flake_detection writes to <output>/flake_detection_results/, '
                         'grid_preperation writes to <output>/cif_export_results/ (contact_placement '
                         'writes into a contact_placement_results/ subfolder of that same folder).')
    p.add_argument('--config', '-c', type=Path, default=Path(_DEFAULT_PIPELINE_CONFIG_PATH),
                    help=f'Path to a PipelineConfig JSON (paths to each stage\'s own config). '
                         f'Defaults to {_DEFAULT_PIPELINE_CONFIG_PATH}.')
    p.add_argument('--template', type=Path, default=None,
                    help='Override grid_preperation\'s master template CIF path.')
    p.add_argument('--no-placement', action='store_true',
                    help='grid_preperation: only write per-image CIF files; skip placement '
                         'into a master template. Also skips contact_placement -- with no working '
                         'template copy to splice contacts into, there is nothing for it to do.')
    p.add_argument('--no-contacts', action='store_true',
                    help='Skip the contact_placement stage (headless -- see contact_placement.run); '
                         'grid_preperation\'s own placement into the master template still runs. Use '
                         'this if you want to place contacts yourself later via contact_placement.ui\'s '
                         'interactive preview instead of this file\'s headless auto-accept.')
    p.add_argument('--keep-template', action='store_true',
                    help='Do NOT reset the working template copy before this run -- splice into '
                         'whatever is already there (grid_preperation\'s own default "reuse if present" '
                         'behavior). Off by default: this file always starts from a fresh copy of the '
                         'master template, since re-running it otherwise silently ACCUMULATES geometry '
                         'on top of a previous run\'s own output (confirmed as a real, repeatable bug -- '
                         'contact_placement\'s own splice step has no awareness of what it already wrote, '
                         'so a flake whose Hall/Ring assignment changes between runs ends up with BOTH '
                         'sets of contacts spliced in, not just the latest one).')
    p.add_argument('--interactive', action='store_true',
                    help='Force PipelineConfig.interactive on for this run (the master switch for '
                         'every stage\'s own interactive UI) even if config/pipeline_config.json has '
                         'it false. One-directional: does NOT turn it off if the config file already '
                         'has it true.')
    p.add_argument('--detection_interactive', action='store_true',
                    help='Force flake_detection\'s own DetectionConfig.interactive_review on for this '
                         'run even if its own config file has it false. One-directional, same as '
                         '--interactive. Still requires --interactive (or the config file\'s '
                         'PipelineConfig.interactive) to also be on for the review window to actually '
                         'appear -- the master switch always has the final say.')
    p.add_argument('--grid_interactive', action='store_true',
                    help='Force grid_preperation\'s own CifExportConfig.interactive_alignment on for '
                         'this run even if its own config file has it false. One-directional, same as '
                         '--interactive. Still requires --interactive (or the config file\'s '
                         'PipelineConfig.interactive) to also be on for the alignment review window to '
                         'actually appear -- the master switch always has the final say.')
    return p


def _finish_timing(output_folder: Path, timings: dict, pipeline_start: float, n_images_input: int) -> None:
    """Stamps `timings` with the total elapsed since `pipeline_start`,
    prints a one-line-per-stage summary, and writes timing.json to
    output_folder -- called right before EVERY return point in
    run_pipeline (there are 3: no qualifying flakes, no_placement/
    no_contacts, and normal completion), so a timing.json always exists
    reflecting exactly how far the run actually got, never only on full
    completion. Also stamps started_at/finished_at (wall-clock timestamps,
    not just elapsed duration) -- useful for correlating a run against
    other timestamped things afterward (e.g. cache file mtimes) without
    having to remember to note the time yourself before kicking it off.
    Seconds are wall-clock (time.perf_counter), same convention
    already used by flake_coco_eval.Timer/evaluator.py elsewhere in this
    repo for reporting inference runtime -- deliberately NOT a fixed
    warmup-discard average like those (this measures one real end-to-end
    run of the actual CLI entry point, not a benchmark loop)."""
    finished_at = datetime.now()
    timings['total_seconds'] = round(time.perf_counter() - pipeline_start, 2)
    timings['n_images_input'] = n_images_input
    if n_images_input:
        timings['seconds_per_input_image'] = round(timings['total_seconds'] / n_images_input, 3)
    timings['started_at'] = (finished_at - timedelta(seconds=timings['total_seconds'])).isoformat(timespec='seconds')
    timings['finished_at'] = finished_at.isoformat(timespec='seconds')

    print("\n=== Timing ===")
    for key in ('flake_detection_seconds', 'grid_preperation_seconds', 'contact_placement_seconds'):
        if key in timings:
            print(f"  {key.replace('_seconds', '')}: {timings[key]:.1f}s")
    print(f"  total: {timings['total_seconds']:.1f}s"
          + (f" ({timings['seconds_per_input_image']:.2f}s / input image, {n_images_input} images)" if n_images_input else ""))
    print(f"  finished at: {timings['finished_at']}")

    output_folder.mkdir(parents=True, exist_ok=True)
    with open(output_folder / 'timing.json', 'w') as f:
        json.dump(timings, f, indent=2)
    print(f"  -> {output_folder / 'timing.json'}")


def run_pipeline(
    input_folder: Path,
    pipeline_config: PipelineConfig,
    output_folder: Path | None = None,
    template_override: Path | None = None,
    no_placement: bool = False,
    no_contacts: bool = False,
    keep_template: bool = False,
    interactive: bool = False,
    detection_interactive: bool = False,
    grid_interactive: bool = False,
) -> tuple[list, list[dict], list[dict]]:
    """Run flake_detection, then grid_preperation, then contact_placement.
    Returns (flake_detection_results, grid_preperation_results,
    contact_placement_results) -- the third is [] if skipped (no_placement,
    no_contacts, or nothing was placed).

    interactive/detection_interactive/grid_interactive are one-directional
    CLI overrides (see --interactive/--detection_interactive/
    --grid_interactive): each forces the corresponding config flag ON for
    this run only, never OFF -- a config file that already has it on is
    left alone."""
    pipeline_start = time.perf_counter()
    timings: dict = {}
    input_folder = Path(input_folder)
    output_folder = Path(output_folder) if output_folder is not None else input_folder / 'pipeline_results'
    n_images_input = sum(1 for p in input_folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)

    if interactive:
        pipeline_config.interactive = True

    detection_config = DetectionConfig.load(pipeline_config.flake_detection_config_path)
    if detection_interactive:
        detection_config.interactive_review = True
    if not pipeline_config.interactive:
        # Master switch off -- never pop flake_detection's review window for
        # this run, regardless of what its own config file says.
        detection_config.interactive_review = False
    print(f"=== Stage 1/3: flake_detection (config: {pipeline_config.flake_detection_config_path}) ===")
    detection_folder = output_folder / 'flake_detection_results'
    stage_start = time.perf_counter()
    detection_results = detect_flakes(input_folder, detection_config, detection_folder)

    maybe_review(detection_results, detection_folder, detection_config)
    # Timed through the end of the (optional) review window, not just
    # detect_flakes itself -- interactive_review time is real wall-clock
    # this run spent in Stage 1, whether or not a human sat through it.
    timings['flake_detection_seconds'] = round(time.perf_counter() - stage_start, 2)

    # any(f.chosen ...), not just `r.flakes` truthy -- a review pass can
    # reject every flake in an image without emptying its flakes list (see
    # FlakeMetrics.chosen), so a plain non-empty check would still stage
    # that image for grid_preperation.
    selected = [r for r in detection_results if any(f.chosen for f in r.flakes)]
    if not selected:
        print("\nNo images with qualifying flakes -- nothing to pass to grid_preperation.")
        _finish_timing(output_folder, timings, pipeline_start, n_images_input)
        return detection_results, [], []

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
    if grid_interactive:
        export_config.interactive_alignment = True
    if not pipeline_config.interactive:
        # Master switch off -- never pop grid_preperation's alignment review
        # window for this run either, same gate as detection_config above.
        export_config.interactive_alignment = False
    # Always wired to the SAME DetectionConfig flake_detection itself just
    # ran with (pipeline_config.flake_detection_config_path), overriding
    # whatever static value grid_preperation's own config file happens to
    # have -- these used to be two independent fields that had to be kept
    # in sync by hand across every stage's config file whenever either
    # changed (confirmed as a real maintenance smell: nothing enforced
    # they actually matched, so the two stages could silently score/detect
    # flakes and markers against different color ranges). grid_preperation's
    # own field still exists for its standalone CLI (`python -m
    # grid_preperation.run --detection-config ...`), just no longer
    # trusted when running through this orchestrator.
    export_config.detection_config_path = pipeline_config.flake_detection_config_path
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

    # Reset the working template copy before this run, unless the caller
    # explicitly asked to keep it (--keep-template). Without this,
    # grid_preperation.run() reuses whatever working copy is already
    # sitting in cif_export_dir (its own default "reuse if present"
    # behavior, meant for incrementally growing a layout across separate
    # calls) -- but contact_placement's own splice step has NO awareness of
    # what it already wrote there, so re-running this file's own Stage 3 on
    # top of a previous run's output doesn't replace it, it ACCUMULATES:
    # confirmed as a real, repeatable bug -- a flake whose Hall/Ring
    # assignment changed between two runs ended up with BOTH sets of
    # contacts spliced in at once, not just the latest one. Deleting the
    # working copy here (not the original master template, which
    # grid_preperation never touches) makes every run of THIS file
    # self-contained by default, matching how someone actually expects
    # "run it again" to behave for a one-shot full-pipeline entry point.
    cif_export_dir = output_folder / 'cif_export_results'
    if not keep_template and export_config.template_cif_path is not None:
        working_template_path = cif_export_dir / Path(export_config.template_cif_path).name
        if working_template_path.exists():
            working_template_path.unlink()
            print(f"Reset working template copy (removed stale {working_template_path}; "
                  f"pass --keep-template to accumulate onto it instead).")

    print(f"\n=== Stage 2/3: grid_preperation (config: {pipeline_config.grid_preperation_config_path}) ===")
    stage_start = time.perf_counter()
    export_results = run_grid_preperation(staging_dir, export_config, cif_export_dir)
    timings['grid_preperation_seconds'] = round(time.perf_counter() - stage_start, 2)

    if no_placement or no_contacts:
        reason = "--no-placement (no working template to splice contacts into)" if no_placement else "--no-contacts"
        print(f"\nSkipping contact_placement ({reason}).")
        _finish_timing(output_folder, timings, pipeline_start, n_images_input)
        return detection_results, export_results, []

    contact_config = ContactPlacementConfig.load(pipeline_config.contact_placement_config_path)
    contact_config.grid_preperation_output_dir = str(cif_export_dir)
    contact_config.output_dir = None  # default: <cif_export_dir>/contact_placement_results/

    print(f"\n=== Stage 3/3: contact_placement (config: {pipeline_config.contact_placement_config_path}) ===")
    stage_start = time.perf_counter()
    contact_results = run_contact_placement(contact_config)
    timings['contact_placement_seconds'] = round(time.perf_counter() - stage_start, 2)

    _finish_timing(output_folder, timings, pipeline_start, n_images_input)
    return detection_results, export_results, contact_results


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
        template_override=args.template, no_placement=args.no_placement, no_contacts=args.no_contacts,
        keep_template=args.keep_template,
        interactive=args.interactive, detection_interactive=args.detection_interactive,
        grid_interactive=args.grid_interactive,
    )


if __name__ == '__main__':
    main()
