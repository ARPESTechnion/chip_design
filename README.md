# ARPES Chip Flake Detection & CIF Export Pipeline

Automates finding candidate 2D-material flakes in chip microscope images and
exporting their geometry as CIF polygons, placed into a master chip template
for fabrication (electron-beam lithography).

**Paper**: O. Berman and I. Mangel, *FlakeForge: automated lithography-ready
device layout from optical images of exfoliated van der Waals flakes*,
arXiv:2611.29509 (2026) — https://arxiv.org/abs/2611.29509

**Materials** (trained model and a toy dataset of optical images for trying
the pipeline):
https://drive.google.com/drive/folders/1PFAKjvdyGy75H3g4cxbodw90LaiyOWCD?usp=drive_link

The pipeline has three parts, each independently runnable:

1. **`flake_detection`** — scans a folder of raw chip images and scores/flags
   the ones with candidate flakes worth reviewing.
2. **`grid_preperation`** — for each flagged image: reconstructs a cleaned-up
   image (via `channels_model`, see below), finds the two calibration
   crosshair markers, calibrates pixel↔micron, extracts polygons, and
   writes/places CIF files.
3. **`channels_model`** — a small trained model, used internally by
   `grid_preperation`, that imitates the manual "which color layers are real
   content" decision `GDS_Code.m`'s operator used to make by hand. You don't
   normally interact with it directly — see the bottom of this README.

`all_code.py` runs all of this end to end. Each stage can also be run on its
own (e.g. to re-export CIFs without re-running detection).

## Setup

```
.\setup.ps1
```

Creates `.\venv` (reused if it already exists) and installs everything from
the one `requirements.txt` — including the `torch`/`onnxruntime` ML stack
`channels_model` needs, which is required even if you only run
`grid_preperation`, not just `channels_model` training (see below).
`.\setup.ps1 -Force` deletes and recreates an existing venv.

Equivalent by hand:

```
python -m venv venv
venv\Scripts\activate                      # Windows
pip install -r requirements.txt
```

`requirements.txt` pins CUDA-specific `torch`/`torchvision` wheels via
`--extra-index-url` — edit that line first if you're not on CUDA 12.4, or
just accept CPU-only wheels.

**MATLAB**: `grid_preperation` calls MATLAB's real `rgb2ind` (via the
`matlabengine` package) to compute color clusters — a from-scratch Python
reimplementation wasn't reliable enough. You need a local MATLAB install
with the **Image Processing Toolbox**. Results are disk-cached per image, so
MATLAB is only actually invoked the first time a given image is processed.

`matlabengine` is version-locked 1:1 to a specific MATLAB release (e.g. a
local R2026a needs exactly `matlabengine==26.1`), so `requirements.txt`
deliberately does **not** pin it — `setup.ps1` detects your local MATLAB
release (scanning `Program Files\MATLAB\R*`) and installs the matching
`matlabengine` version itself. `setup.ps1` cannot install MATLAB itself (a
licensed product) or confirm the Image Processing Toolbox is actually
licensed (only that its files are present on disk); if it can't find a local
MATLAB, or the version it installs doesn't match, install it by hand:
`venv\Scripts\python.exe -m pip install matlabengine==<version>` — release
`R20YYa` → `YY.1`, `R20YYb` → `YY.2` (e.g. `R2026a` → `26.1`).

**Trained model**: download it from the materials folder
**https://drive.google.com/drive/folders/1PFAKjvdyGy75H3g4cxbodw90LaiyOWCD?usp=drive_link**
and place the files at `channels_model/outputs/` (you should end up with
`channels_model/outputs/channels_model.onnx`). It's not in git — see
`.gitignore`.

**Master template CIF**: `grid_preperation/BaseGrid.cif` is also not
tracked in git. Make sure you have your own copy at that path (or point
`template_cif_path` at wherever yours lives — see Configuration below)
before running placement.

## Configuration

Every stage has its own JSON config (a serialized dataclass — see the
`_comment*` keys in each file for inline docs) — this section covers the
parameters you'll actually need to touch, not every field.

### `config/pipeline_config.json` (top level)

Only holds **paths to each stage's config**, plus where to find your
images — it doesn't itself contain any detection/export tuning:

```json
{
  "flake_detection_config_path": "flake_detection/config/orange_flakes_tuned.json",
  "grid_preperation_config_path": "grid_preperation/config/example_config.json",
  "input_folder": "flakes/all",
  "interactive": false
}
```

**`interactive`** — master switch for every stage's own interactive UI
(`flake_detection`'s review window, `grid_preperation`'s alignment review
window). `False` forces them all off for this run regardless of what any
individual stage's own config file says. `True` doesn't force any of them
*on* by itself — each stage's own flag (`flake_detection`'s
`interactive_review`, `grid_preperation`'s `interactive_alignment`) still
decides whether that specific stage pops one. See **Interactive review**
and **Interactive alignment** below.

### `flake_detection`'s config (e.g. `flake_detection/config/orange_flakes_tuned.json`)

- **`substrate_hsv_range`** — the bare-chip background color (HSV). Everything
  *outside* this range is treated as potential flake material. Tune this
  first if detection is missing real flakes or flagging substrate as one.
- **`color_mode`** (`'hsv'` or `'rgb'`) + **`hsv_range`**/**`rgb_range`** — the
  target flake color used for scoring (not for the substrate mask above).
- **`min_color_fraction`** — hard cutoff: a candidate region is discarded
  outright if less than this fraction of its pixels match the target color.
- **`min_flake_area`** / **`max_flake_area`** — pixel-area bounds for a
  candidate to be considered at all (filters out dust and grid markers).
- **`reject_border_touching`** — if `True` (default), a candidate whose
  bounding box touches any image edge is rejected outright — it's clipped by
  the frame, so its true area/shape/color can't be measured, and it may well
  continue beyond the captured image.
- **`min_solidity`** / **`min_solidity_strict`** / **`solidity_relief_area_lo`**
  / **`solidity_relief_area_hi`** — a candidate is rejected if `contour area /
  convex-hull area` (`1.0` = perfectly convex; lower = a more concave/irregular
  outline — star-shaped debris, several touching fragments merged into one
  contour, jagged noise) falls below a **size-dependent** threshold: candidates
  at/below `solidity_relief_area_lo` px must clear `min_solidity_strict`
  (default `0.72`); candidates at/above `solidity_relief_area_hi` px only need
  to clear `min_solidity` (default `0.4`); anything in between is linearly
  interpolated. **Why size-dependent, not one flat number**: a real,
  single-piece flake can legitimately be *more* concave the bigger it is — a
  long, thin, gently curved strip produces one big concave region purely from
  being bent, even though it's completely legitimate, not junk; bigger flakes
  are also just more likely to have real bends or multiple domains. A flat
  `0.85`, then a flat `0.7`, were each tried first and both rejected real,
  visually-verified flakes (including one scoring `0.93` on every other
  metric) before landing on this size-relief scheme — the defaults above are
  verified safe (comfortable margin) against every known-good flake across
  this project's labeled datasets. Adjust only after inspecting `solidity`
  and `area_px` together in `results.json` for your own known good/bad flakes
  — a threshold safe for one dataset's typical flake shape/size isn't
  guaranteed safe for another.
- **`min_final_score`** — minimum combined color/texture/isolation score
  [0-1] for a flake to appear in results. Start at `0.0`, inspect
  `results.json`'s score distribution, then raise to cut false positives.
- **`top_n`** — if set, only the top N images by best flake score are kept.
- **`interactive_review`** — if `True`, pops an interactive review window
  right after detection finishes, letting you manually correct its
  automatic chosen/rejected calls before anything downstream reads them.
  See **Interactive review** below. Default `False`.

### `grid_preperation`'s config (e.g. `grid_preperation/config/example_config.json`)

**`template_cif_path`** — the master chip CIF that exported polygons get
placed into. **This is the one most people need to change first**: point it
at your own master template (e.g. a copy of `BaseGrid.cif`), or set it to
`null` to only ever write standalone per-image CIFs and skip placement
entirely (same effect as `--no-placement` on the CLI). The original file at
this path is never modified — a working copy is made once in the output
folder and every run's placements are spliced into that copy.

- **`detection_config_path`** — which `flake_detection` config to reuse for
  color ranges and marker detection, so the two stages never disagree about
  what counts as "flake color". Keep this in sync with whatever config you
  actually ran `flake_detection` with.
- **`grid_pitch_um`**, **`placement_x_offset_um`**, **`placement_y_offset_um`**
  — the real-world spacing between adjacent grid markers (e.g. AA to AB) and
  where grid column/row 'A' sits on your master template. These must match
  your actual chip layout, not a printed scale-bar overlay on the images —
  wrong values here silently misplace every flake.
- **`crosshair_match_threshold`** — how confidently a shape must match the
  crosshair template to count as a marker. Lower it if real markers are
  being missed; raise it if noise is being picked up as a marker.
- **`channels_model_onnx_path`**, **`channels_n_clusters`**,
  **`channels_cluster_median_win`** — which trained model to use and what it
  was trained with. The cluster/median-window values must match how that
  specific `.onnx` file was actually trained (see its `effective_config.json`)
  or you'll silently feed it the wrong kind of input.
- **`gds_min_area`** — minimum contour area (px) to export as a polygon at
  all; raise if tiny specks are showing up in the CIF.
- **`min_flake_score`** — like `flake_detection`'s `min_final_score`, but
  applied per-region within an already-selected image, not per-image.
  Defaults to `0.0` (export everything past `gds_min_area`).
- **`max_vertex_n`** — CIF polygon vertex cap; larger contours are split
  automatically. Don't raise this casually — it exists because very-high-
  vertex-count polygons can break downstream CAD tooling.
- **`output_dir`** — defaults to `<input_folder>/cif_export_results/` if unset.
- **`interactive_alignment`** — if `True`, pauses between marker detection
  and everything downstream to let you manually drag-correct each image's
  two calibration crosshairs. See **Interactive alignment** below. Default
  `False`.

After changing any of these, check `<output>/cif_export_results.json` — every
image is reported `ok` or `alert` with a concrete reason, so you can see
exactly what a parameter change affected.

## Running the full pipeline

```
python all_code.py                                   # input_folder from config/pipeline_config.json
python all_code.py --input_folder flakes/all -o out   # explicit input + output folder
python all_code.py --input_folder flakes/all --no-placement   # export CIFs, skip template placement
python all_code.py --input_folder flakes/all --template my_master.cif
python all_code.py --input_folder flakes/all --interactive --detection_interactive   # pop flake_detection's review window
python all_code.py --input_folder flakes/all --interactive --grid_interactive        # pop grid_preperation's alignment window
```

`--interactive`, `--detection_interactive`, and `--grid_interactive` are
one-directional CLI overrides for the config flags of the same purpose
above — each forces the corresponding flag **on** for this run only; they
never turn one *off* if a config file already has it `true`.
`--detection_interactive`/`--grid_interactive` alone does nothing without
`--interactive` (or `"interactive": true` in `config/pipeline_config.json`)
also being on — the master switch always has the final say.

This runs `flake_detection` on every image, stages the ones with at least
one qualifying flake into `<output>/selected_for_grid_preperation/`, then
runs `grid_preperation` on that staged set. Outputs land under:

```
<output>/flake_detection_results/     # annotated images + results.json
<output>/selected_for_grid_preperation/
<output>/cif_export_results/          # per-image CIFs + working template copy + cif_export_results.json
```

## Running `flake_detection` alone

```
python -m flake_detection.run --input_folder flakes/all
python -m flake_detection.run --input_folder flakes/all --config flake_detection/config/orange_flakes_tuned.json
python -m flake_detection.run --input_folder flakes/all --top-n 20
python -m flake_detection.run --input_folder flakes/all --save-config my_run_config.json
```

Writes `<output>/annotated/` (images with detection overlays) and
`<output>/results.json` (per-image, per-flake scores) — output defaults to
`<input_folder>/flake_detection_results/`.

Every candidate contour in an annotated image is drawn in the color of the
**first stage it failed**, so you can tell at a glance where the pipeline
stopped on it (see `flake_detection/detector.py`'s `_STAGE_COLORS`):

| Color | Stage failed | Label shown |
|---|---|---|
| 🔴 Red | Area filter (`min_flake_area`/`max_flake_area`) | `a:<area>` |
| 🔵 Blue | Bounding box touches the image border (`reject_border_touching`) | `border` |
| ⚪ Grey | Too irregular/spiky, size-relieved (`min_solidity`/`min_solidity_strict`) | `sol:<solidity><threshold>` |
| 🟠 Orange | Color fraction too low (`min_color_fraction`) | `cf:<value>` |
| 🟡 Yellow | Texture — weakest sub-score | `t:<texture_score> cf:<cf>` |
| 🟣 Purple | Isolation — weakest sub-score | `i:<isolation_score> cf:<cf>` |
| 🟢 Green (thicker outline) | Passed everything | `<final_score> cf:<cf>` |
| 🟣 Magenta rectangle | Dark-spot alert (unrelated to the above filter chain) | `dark <brightness>` |

A contour reaching texture/isolation scoring already cleared the area,
border, shape, and color filters, so yellow/purple mean "close, but not
good enough" rather than "wrong color entirely" (that's orange) or
"unmeasurable/malformed shape" (that's blue/grey). Tune whichever config
threshold matches the color you're seeing too much of.

### Interactive review

Set `"interactive_review": true` in the `flake_detection` config you run
with (and, if going through `all_code.py`, also `"interactive": true` in
`config/pipeline_config.json`, or pass `--interactive --detection_interactive`
— see above) to get a manual checkpoint right after detection finishes,
before anything downstream (`grid_preperation`, `contact_placement`) ever
reads the results:

```
python -m flake_detection.run --input_folder flakes/all --config flake_detection/config/orange_flakes_tuned.json
# (with interactive_review: true in that config file)
```

A maximized window opens, one image at a time. Every contour that cleared
the **size filter** is outlined — green if currently chosen, red if
rejected — regardless of which later stage (border, shape, color, or final
score) actually rejected it, so you can both reject a false positive the
pipeline accepted and rescue a near-miss it didn't. Only contours that
failed the size filter itself are never shown (bulk dust/noise/grid-marker
debris — the pipeline keeps no geometry for those at all).

| Action | How |
|---|---|
| Toggle a flake | Click it (chosen ⇄ rejected) |
| Previous / next image | `◀`/`▶` buttons, or `←`/`→` keys |
| Save Changes | Rewrites this image's `contours/<stem>.json` (what `grid_preperation` reads) and `scores/<stem>.txt` to the currently-chosen flakes only, and rewrites `results.json` (which keeps every flake either way, chosen or not, for later inspection). Button, or `Enter` |
| Reset Changes | Reverts the current image to whatever was last saved to disk |
| Skip All | Ends the review (always asks "are you sure?" first) — any image whose edits were never explicitly saved is left exactly as `detect_flakes()` originally wrote it |

Changes only take effect for `grid_preperation`/`contact_placement` runs
that happen *after* you Save — `all_code.py`'s orchestrator already gets
this ordering right (review happens before images are staged for
`grid_preperation`), but a standalone `grid_preperation`/`contact_placement`
run pointed at an old `flake_detection` output will only see whatever was
last saved there.

## Running `grid_preperation` alone

Expects a folder of already-selected images (e.g. `flake_detection`'s
output, or any folder you've curated by hand):

```
python -m grid_preperation.run --input_folder flakes/all
python -m grid_preperation.run --input_folder flakes/all --no-placement
python -m grid_preperation.run --input_folder flakes/all --config grid_preperation/config/example_config.json
python -m grid_preperation.run --input_folder flakes/all --template grid_preperation/BaseGrid.cif
```

Per image, this: runs `channels_model` to reconstruct a posterized image →
finds the 2 crosshair markers → calibrates pixel↔µm → extracts polygons
(each colored/layered by its own real source cluster color) → writes a
per-image CIF → splices it into the working template copy. Check
`<output>/cif_export_results.json` for per-image status — anything that
couldn't be confidently placed is reported as `alert` with a reason, never
silently guessed.

Debug output worth checking when tuning: `<output>/reconstructed/` (the
posterized image each source image got reduced to before marker/polygon
detection — shows what `channels_model` actually selected).

### Interactive alignment

Set `"interactive_alignment": true` in the `grid_preperation` config you run
with (and, if going through `all_code.py`, also `"interactive": true` in
`config/pipeline_config.json`, or pass `--interactive --grid_interactive` —
see above) to get a manual checkpoint between marker **detection** and
everything that consumes it (calibration, polygon extraction, CIF write,
template placement) — nothing gets written or spliced for ANY image until
you've had a chance to fix every image's alignment:

```
python -m grid_preperation.run --input_folder flakes/all --config grid_preperation/config/example_config.json
# (with interactive_alignment: true in that config file)
```

A maximized window opens, one image at a time, showing its reconstructed
image with the two calibration crosshairs drawn — red for the left/filename
marker, green for the right (one grid column over), same colors as the
`draw_marker_centers` debug output. An image where automatic detection
failed outright still opens for review, seeded with two placeholder crosses
to place from scratch instead of just alerting.

| Action | How |
|---|---|
| Select a cross | Click it (red or green) |
| Move the selected cross | Drag it, or use arrow keys (1px; Shift+arrow = 10px) |
| Deselect (arrow keys navigate images again) | Click empty canvas space |
| Previous / next image | `◀`/`▶` buttons, or `←`/`→` keys (when nothing's selected) |
| Drop this image from the run entirely | "Drop Image" / "Include Image" toggle |
| Save Changes | Writes `marker_overrides/<stem>.json` — a **permanent** override: every future run, interactive or not, uses this position/drop decision and skips automatic detection for that image. Button, or `Enter` |
| Reset Changes | Reverts the current image to whatever was last saved (or, if never saved, back to the original starting position) |
| Save All & Continue | Saves every image's current state, then ends the review (asks "are you sure?" first) |
| Skip All | Ends the review without saving (asks "are you sure?" first) — any image never explicitly saved is left exactly as it started |

Two aids for placing a cross precisely (mouse precision alone is coarser
than calibration wants — marker separation sets `scale_um_per_px` for the
**whole image**): while a cross is selected, a magnifier loupe appears in
the top-right corner showing a zoomed, full-resolution crop around it (sized
to fit a whole real crosshair, not just its core), with a full-span 1px
yellow guide cross through the exact position for judging fine alignment by
eye, plus a small color-matched cross marking the position itself. A live
readout under the title also shows both crosses' exact pixel coordinates and
the separation/angle/implied `scale_um_per_px` they'd produce.

Like flake_detection's review, changes only take effect for runs that
happen *after* you Save — a saved override or drop decision persists across
every later run of `grid_preperation.run`, headless or not, until its
`marker_overrides/<stem>.json` file is deleted by hand.

## `channels_model`

You normally don't run this directly — `grid_preperation` calls the
pretrained model automatically once it's downloaded (see Setup). Retraining
needs labeled data (`channels_model/data/images/<name>.png` +
`channels_model/data/labels/<name>.txt`, newline-separated channel numbers
1-8 that are "real content" for that image):

```
python -m channels_model.train --config channels_model/config/default_config.json
```

This runs k-fold cross-validation, refits on all the data, and exports
`channels_model.onnx` + `effective_config.json` to `channels_model/outputs/`
— the same layout the Drive download provides. Never regenerate
`channels_model/config/default_config.json` by re-running `Config().save(...)`
after changing the dataclass — merge new fields in by hand instead, or you'll
overwrite hand-tuned values.

## Repo layout

```
setup.ps1                      creates venv + installs requirements.txt
all_code.py                    full-pipeline CLI
config/pipeline_config.json    paths to each stage's own config + input_folder
flake_detection/                image -> candidate-flake scoring
  run.py  config.py  detector.py  ui.py  config/*.json
grid_preperation/                image -> CIF polygons -> placed in master template
  run.py  config.py  channel_selection.py  markers.py  calibration.py  polygons.py  cif_io.py  ui.py
  BaseGrid.cif  (gitignored -- supply your own, see Setup)
  crosshair_template.png
  data/channels_cache/           (gitignored -- disk cache, rebuilt on demand)
channels_model/                  color-cluster "which channels are real content" model
  train.py  model.py  dataset.py  matlab_clustering.py  export.py  config.py
  outputs/  (gitignored -- download from Drive, see Setup)
  data/     (gitignored -- your own training images/labels + disk cache)
```
