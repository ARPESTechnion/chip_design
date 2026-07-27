# ARPES Chip Flake Detection & CIF Export Pipeline

Automates finding candidate 2D-material flakes in chip microscope images and
exporting their geometry as CIF polygons, placed into a master chip template
for fabrication (electron-beam lithography). Built to replace the manual
steps of the legacy MATLAB tool `GDS_Code.m`.

The pipeline has three parts, each independently runnable:

1. **`flake_detection`** — scans a folder of raw chip images and scores/flags
   the ones with candidate flakes worth reviewing.
2. **`channels_model`** — a small trained model that imitates the manual
   "which color layers are real content" decision `GDS_Code.m`'s operator
   used to make by hand. Used internally by `grid_preperation`.
3. **`grid_preperation`** — for each flagged image: reconstructs a cleaned-up
   image using `channels_model`, finds the two calibration crosshair
   markers, calibrates pixel↔micron, extracts polygons, and writes/places
   CIF files.

`all_code.py` runs all of this end to end. Each stage can also be run on its
own (e.g. to re-export CIFs without re-running detection).

## Setup

```
python -m venv venv
venv\Scripts\activate                      # Windows
pip install -r requirements.txt -r channels_model/requirements.txt
```

`channels_model/requirements.txt` pins CUDA-specific `torch`/`torchvision`
wheels — edit its `--extra-index-url` line first if you're not on CUDA 12.4,
or just accept CPU-only wheels.

### MATLAB (required for `channels_model` / `grid_preperation`)

Color-cluster computation (`channels_model.matlab_clustering`) calls
MATLAB's real `rgb2ind` via the MATLAB Engine API (`matlabengine`, already
in `requirements.txt`) — a from-scratch Python reimplementation wasn't
reliable enough. You need:

- A local MATLAB install with the **Image Processing Toolbox**, matching a
  release the installed `matlabengine` version supports.
- Results are disk-cached per image (`grid_preperation/data/channels_cache/`,
  `channels_model/data/channels_cache/`) — MATLAB is only actually invoked
  on a cache miss (a new image never processed before).

### Download the trained `channels_model`

The trained model isn't in git (binary files under `channels_model/outputs/`
and `channels_model/data/` are gitignored). Download it here:

**https://drive.google.com/drive/folders/1UgtzAjijtjCXCGo_lJOhxZYN9KE9eSsn?usp=drive_link**

Place the downloaded files at `channels_model/outputs/` (i.e. you should end
up with `channels_model/outputs/channels_model.onnx`, matching the default
`channels_model_onnx_path` in `grid_preperation`'s config). `grid_preperation`
checks this path exists up front and fails clearly if it doesn't.

### Master template CIF

`grid_preperation/BaseGrid.cif` (the master chip template polygons get
placed into) is **also not tracked in git** (matches the `*.cif` gitignore
rule). Make sure you have your own copy of the master template at that path
— or point `--template` / `template_cif_path` in `grid_preperation`'s config
at wherever yours actually lives — before running placement.

## Running the full pipeline

```
python all_code.py --input_folder flakes/all
```

`--input_folder` overrides `input_folder` in `config/pipeline_config.json`
if set there — you can omit the flag entirely once that's configured.
`config/pipeline_config.json` only holds *paths* to each stage's own config
file (plus `input_folder`); tune actual detection/export behavior in
`flake_detection/config/orange_flakes_tuned.json` and
`grid_preperation/config/example_config.json` themselves.

```
python all_code.py                                   # input_folder from config/pipeline_config.json
python all_code.py --input_folder flakes/all -o out   # explicit input + output folder
python all_code.py --input_folder flakes/all --no-placement   # export CIFs, skip template placement
python all_code.py --input_folder flakes/all --template my_master.cif
```

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
`<input_folder>/flake_detection_results/`. Use `results.json` to tune
`min_final_score`, `min_area`/`max_area`, color ranges, etc. in the config.

## Running `grid_preperation` alone

Expects a folder of already-selected images (e.g. `flake_detection`'s
output, or any folder you've curated by hand):

```
python -m grid_preperation.run --input_folder flakes/all
python -m grid_preperation.run --input_folder flakes/all --no-placement
python -m grid_preperation.run --input_folder flakes/all --template grid_preperation/BaseGrid.cif
```

Per image, this: runs `channels_model` to reconstruct a posterized image →
finds the 2 crosshair markers → calibrates pixel↔µm → extracts polygons
(each colored/layered by its own real source cluster color) → writes a
per-image CIF → splices it into a working copy of the master template (never
modifies the original template in place). Check `<output>/cif_export_results.json`
for per-image status — anything that couldn't be confidently placed is
reported as `alert` with a reason, never silently guessed.

Debug output worth checking when tuning: `<output>/reconstructed/` (the
posterized image each image got reduced to before marker/polygon detection
— useful for seeing what `channels_model` actually selected).

## Using `channels_model`

### Just using the pretrained model (most common case)

Nothing to run directly — once you've downloaded it (see Setup above),
`grid_preperation` calls it automatically via
`grid_preperation/channel_selection.py`.

### Retraining

Needs labeled data: `channels_model/data/images/<name>.png` +
`channels_model/data/labels/<name>.txt` (newline-separated channel numbers
1-8 that are "real content" for that image — see
`channels_model.dataset.parse_label_file`).

```
python -m channels_model.train --config channels_model/config/default_config.json
python -m channels_model.train --images-dir path/to/images --labels-dir path/to/labels
```

Runs k-fold cross-validation (reporting per-class accuracy, confusion
matrices, and "super accuracy" — all 8 channels correct — each epoch),
derives an epoch budget from the CV run, refits on all the data, and
exports `channels_model.onnx` + `effective_config.json` + `cv_results.json`
to `channels_model/outputs/` — the same layout the Drive download provides.

Never edit `channels_model/config/default_config.json` (or any other saved
config JSON in this repo) by re-running `Config().save(...)` after changing
the dataclass — that blindly overwrites the whole file and destroys any
hand-tuned values (`epochs`, `lr`, etc.). Merge new/changed fields in by
hand instead.

## Repo layout

```
all_code.py                    full-pipeline CLI
config/pipeline_config.json    paths to each stage's own config + input_folder
flake_detection/                image -> candidate-flake scoring
  run.py  config.py  detector.py  config/*.json
channels_model/                  color-cluster "which channels are real content" model
  train.py  model.py  dataset.py  matlab_clustering.py  export.py  config.py
  outputs/  (gitignored -- download from Drive, see Setup)
  data/     (gitignored -- your own training images/labels + disk cache)
grid_preperation/                image -> CIF polygons -> placed in master template
  run.py  config.py  channel_selection.py  markers.py  calibration.py  polygons.py  cif_io.py
  BaseGrid.cif  (gitignored -- supply your own, see Setup)
  crosshair_template.png
  data/channels_cache/           (gitignored -- disk cache, rebuilt on demand)
```
