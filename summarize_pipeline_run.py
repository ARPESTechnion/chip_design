"""
Extracts the requested per-dataset metrics from one all_code.py run's output
folder:
  - flakes detected (chosen by flake_detection) + total raw candidates
  - images with >=1 chosen flake
  - images placed into the master CIF template (grid_preperation stage)
  - flakes carried into grid_preperation's own per-image flake exports
    (<output>/cif_export_results/flakes/*.json -- the real, calibrated flake
    candidates available for contact_placement to consider)
  - flakes placed with a contact scheme (contact_placement stage, mode != None)
  - flakes placed in 'hall' mode
  - flakes placed in 'ring' mode

Usage:
    python summarize_pipeline_run.py pipeline_runs/TaS2009 --name TaS2009
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def summarize(output_folder: Path, input_folder: Path | None = None) -> dict:
    fd_results_path = output_folder / "flake_detection_results" / "results.json"
    cif_results_path = output_folder / "cif_export_results" / "cif_export_results.json"
    flakes_dir = output_folder / "cif_export_results" / "flakes"
    contact_results_path = output_folder / "cif_export_results" / "contact_placement_results" / "results.json"

    summary = {"output_folder": str(output_folder)}

    if input_folder is not None and Path(input_folder).exists():
        summary["images_total_in_input_folder"] = sum(
            1 for p in Path(input_folder).iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    else:
        summary["images_total_in_input_folder"] = None

    # --- Stage 1: flake_detection ---
    # NOTE: results.json only contains images detect_flakes returned, i.e.
    # images with >=1 flake OR dark-spot candidate -- images with ZERO
    # candidates at all are silently absent from it (see detect_flakes's own
    # docstring). images_total_in_input_folder above is the true denominator.
    if fd_results_path.exists():
        fd = json.loads(fd_results_path.read_text())
        total_candidates = sum(len(r["flakes"]) for r in fd)
        total_chosen = sum(sum(1 for f in r["flakes"] if f.get("chosen")) for r in fd)
        images_with_chosen = sum(1 for r in fd if any(f.get("chosen") for f in r["flakes"]))
        summary["images_with_any_candidate"] = len(fd)
        summary["flakes_detected_chosen"] = total_chosen
        summary["flakes_candidates_total"] = total_candidates
        summary["images_with_flakes"] = images_with_chosen
    else:
        summary["images_with_any_candidate"] = None
        summary["flakes_detected_chosen"] = None
        summary["flakes_candidates_total"] = None
        summary["images_with_flakes"] = None

    # --- Stage 2: grid_preperation ---
    if cif_results_path.exists():
        cif_results = json.loads(cif_results_path.read_text())
        images_placed = sum(1 for r in cif_results if r.get("placed"))
        images_ok = sum(1 for r in cif_results if r.get("status") == "ok")
        summary["images_staged_for_grid_prep"] = len(cif_results)
        summary["images_ok_grid_prep"] = images_ok
        summary["images_placed"] = images_placed
    else:
        summary["images_staged_for_grid_prep"] = None
        summary["images_ok_grid_prep"] = None
        summary["images_placed"] = None

    if flakes_dir.exists():
        total_flakes_exported = 0
        for p in flakes_dir.glob("*.json"):
            data = json.loads(p.read_text())
            total_flakes_exported += len(data)
        summary["flakes_exported_to_grid_prep"] = total_flakes_exported
    else:
        summary["flakes_exported_to_grid_prep"] = None

    # --- Stage 3: contact_placement ---
    if contact_results_path.exists():
        contact_results = json.loads(contact_results_path.read_text())
        placed = [r for r in contact_results if r.get("mode") is not None]
        n_hall = sum(1 for r in placed if r["mode"] == "hall")
        n_ring = sum(1 for r in placed if r["mode"] == "ring")
        summary["flakes_considered_for_contacts"] = len(contact_results)
        summary["flakes_with_contacts"] = len(placed)
        summary["flakes_with_hall"] = n_hall
        summary["flakes_with_ring"] = n_ring
    else:
        summary["flakes_considered_for_contacts"] = None
        summary["flakes_with_contacts"] = None
        summary["flakes_with_hall"] = None
        summary["flakes_with_ring"] = None

    # --- Timing (all_code.py now writes this itself -- see _finish_timing --
    # no more need for the caller to time the run externally and pass
    # --runtime_seconds by hand). Falls back to that flag if timing.json
    # doesn't exist (e.g. a run from before this was added).
    timing_path = output_folder / "timing.json"
    if timing_path.exists():
        timing = json.loads(timing_path.read_text())
        summary["runtime_seconds_total"] = timing.get("total_seconds")
        summary["runtime_seconds_per_image"] = timing.get("seconds_per_input_image")
        summary["runtime_flake_detection_seconds"] = timing.get("flake_detection_seconds")
        summary["runtime_grid_preperation_seconds"] = timing.get("grid_preperation_seconds")
        summary["runtime_contact_placement_seconds"] = timing.get("contact_placement_seconds")

    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("output_folder", type=Path)
    ap.add_argument("--input_folder", type=Path, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--runtime_seconds", type=float, default=None,
                     help="Manual override for total wall-clock seconds. Usually unnecessary now -- "
                          "all_code.py writes its own timing.json (see _finish_timing there), which "
                          "this script reads automatically. Only needed for a run from before that "
                          "existed, or to override with an externally-measured time.")
    args = ap.parse_args()

    summary = summarize(args.output_folder, args.input_folder)
    summary["dataset"] = args.name or args.output_folder.name
    if args.runtime_seconds is not None:
        summary["runtime_seconds_total"] = round(args.runtime_seconds, 1)
        n_imgs = summary.get("images_total_in_input_folder")
        if n_imgs:
            summary["runtime_seconds_per_image"] = round(args.runtime_seconds / n_imgs, 2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
