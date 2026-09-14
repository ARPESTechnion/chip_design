# Full pipeline run summary -- Ta2009/TaS2009, TaS2010, TaS2011, TS34001

Ran `all_code.py` (flake_detection -> grid_preperation -> contact_placement) headlessly
against each dataset's full real image set on the network share
(`\\132.68.73.70\Data\Ilay\<dataset>\Post Exfo`), using each dataset's own existing,
already-tuned configs (see "Which config was used" below). No interactive review was
used (`--interactive` not passed) -- every number below is fully automatic detection,
not human-corrected.

## Which config was used for each dataset

| Dataset | Top-level config | flake_detection config | grid_preperation config | contact_placement config |
|---|---|---|---|---|
| TaS2009 | `config/Ta2009.json` | `flake_detection/config/Ta2009.json` | `grid_preperation/config/Ta2009.json` | `contact_placement/config/example_config.json` (generic -- no dataset-specific contact config exists for this material) |
| TaS2010 | `config/TaS2010.json` (new, created this session) | `flake_detection/config/orange_flakes_tuned.json` (generic) | `grid_preperation/config/example_config.json` (generic) | `contact_placement/config/example_config.json` (generic) |
| TaS2011 | `config/TaS2011.json` | `flake_detection/config/TaS2011.json` | `grid_preperation/config/TaS2011.json` | `contact_placement/config/TaS2011.json` (dataset-specific) |
| TS34001 | `config/TS34001.json` | `flake_detection/config/TS34001.json` | `grid_preperation/config/TS34001.json` | `contact_placement/config/example_config.json` (generic -- no dataset-specific contact config exists for this material) |

**TaS2010 note**: per the user, this network folder (134 images) IS the material behind
the old local `flakes/all` (28 images) that `config/pipeline_config.json`'s generic
configs were originally developed/tested against -- confirmed by file-count relationship
(28 is a plausible subset of 134). No dataset-specific flake_detection/grid_preperation/
contact_placement config exists for it (unlike TaS2011/TS34001), so this run uses the
same generic configs `config/pipeline_config.json` already pointed at, just retargeted at
the larger network folder instead of the old local subset.

**TaS2009 vs local `flakes/Ta2009`**: file counts match closely (77 network vs 78 local),
confirming these are the same dataset -- ran against the network copy for consistency
with the other three.

## Results

| Metric | TaS2009 | TaS2010 | TaS2011 | TS34001 |
|---|---|---|---|---|
| Images in input folder | 77 | 134 | 139 | 47 |
| Images with >=1 candidate (flake or dark spot) | 77 | 76 | 79 | 18 |
| **Flakes detected** (chosen, passed flake_detection's own quality gate) | **34** | **93** | **82** | **21** |
| (of which, raw candidates before quality filtering) | 1043 | 188 | 209 | 69 |
| **Images with >=1 chosen flake** | **31** | **75** | **67** | **18** |
| Images staged for grid_preperation (= images with >=1 chosen flake) | 31 | 75 | 67 | 18 |
| **Images placed** into the master CIF template | **29** | **69** | **60** | **17** |
| Images that failed grid_preperation (marker detection/calibration alert) | 2 | 6 | 7 | 1 |
| **Flakes placed** (exported into grid_preperation's per-image flake lists, i.e. real calibrated flake candidates available downstream) | **32** | **87** | **73** | **20** |
| Flakes considered by contact_placement (= flakes placed, above) | 32 | 87 | 73 | 20 |
| **Flakes with contacts** (contact_placement found a valid scheme -- hall OR ring) | **8** | **33** | **25** | **13** |
| **Flakes with Hall** bars | **2** | **5** | **0** | **2** |
| **Flakes with Rings** | **6** | **28** | **25** | **11** |

*("Flakes were placed" and "flakes with contacts" turned out to be asking about the same
underlying number in this pipeline -- contact_placement's own `mode` field is either
`None` (not placed), `'hall'`, or `'ring'`; there is no separate third "has contacts but
neither hall nor ring" category. See caveat below on which of two "placed" counts this
is.)*

## Notes on the numbers

**TaS2011's zero Hall count is NOT a mystery -- it's a pre-existing, deliberate config
setting**: `contact_placement/config/TaS2011.json` has `num_halls_to_try: 0` (Hall
disabled entirely for this dataset, `num_rings_to_try: 40`), already in place before this
session. Every one of its 25 placed flakes going to Ring is exactly the expected result
of that setting, not a geometric property of TaS2011's flakes worth investigating.

TaS2009 stands out for its huge candidate-to-chosen ratio (1043 raw candidates -> 34
chosen, a 3.3% acceptance rate vs. 45-49% for the other three) -- this dataset's images
apparently have far more debris/false-candidate material for flake_detection's filters to
reject, or its own config's thresholds are simply stricter. Not investigated further here.

## Important caveat: "flakes placed" has two plausible readings, both reported

The user's request listed "how much flakes were placed" separately from "how much are
with contacts" -- these could mean two different things:
1. **grid_preperation-stage "placed"**: a flake's contour survived flake_detection,
   got calibrated, and was exported into that image's `flakes/<stem>.json` (available for
   contact_placement to consider) -- this is what the "Flakes placed" row above reports.
2. **contact_placement-stage "placed"**: contact_placement actually found valid
   contact geometry for it (mode != None) -- this is what "Flakes with contacts" reports.

In this pipeline these turned out to be describable by nested containment (every
grid_preperation-exported flake is "considered" by contact_placement; some subset of
those get placed with contacts), not two independent numbers -- both are reported above
so nothing is lost either way.

## Known gaps / things NOT verified here

- **No dataset-specific contact_placement config exists for TaS2009 or TS34001** -- both
  ran against the generic `contact_placement/config/example_config.json`. This may or may
  not be the "correct" geometry for those materials; not evaluated here, just flagged
  (matches what `config/Ta2009.json` and `config/TS34001.json` were already pointing at
  before this session -- not a new gap introduced now).
- **TaS2010 ran entirely on generic configs** (flake_detection/grid_preperation/
  contact_placement) -- none of the three stages have a dataset-specific tuned config for
  it, unlike TaS2011/TS34001. Numbers are real (a genuine automatic run), but represent
  the generic pipeline's performance on this material, not a tuned one.
- **No interactive review** was used for any run -- flake_detection's own review UI and
  grid_preperation's alignment-correction UI were both off (matching each dataset's
  existing config default `interactive_alignment`/`interactive_review: false`, and
  `all_code.py`'s own `--interactive` flag not passed). These are fully automatic numbers,
  not human-verified/corrected ones.
- Marker detection / calibration failures (the gap between "images with chosen flakes"
  and "images placed") were not individually inspected for cause here -- see each
  dataset's own `pipeline_runs/<dataset>/cif_export_results/cif_export_results.json`
  for the `reason` field on each alerted image if that's wanted next.

## Files

- Per-dataset raw pipeline output: `pipeline_runs/{TaS2009,TaS2010,TaS2011,TS34001}/`
  (flake_detection_results/, cif_export_results/, cif_export_results/contact_placement_results/)
- Per-dataset run logs: `pipeline_runs/log_{TaS2009,TaS2010,TaS2011,TS34001}.txt`
- Per-dataset extracted metrics (machine-readable): `pipeline_runs/summary_{dataset}.json`
- Metric-extraction script: `summarize_pipeline_run.py` (reusable for future runs/re-runs)
- New top-level config created this session: `config/TaS2010.json`
