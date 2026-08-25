"""
Configuration for the automated flake -> CIF export & placement pipeline.

Reuses flake_detection's DetectionConfig for color/marker ranges (single source
of truth with the detection pipeline) and adds fields specific to calibration,
polygon export, and master-template placement.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

try:
    from flake_detection.config import DetectionConfig
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from flake_detection.config import DetectionConfig


@dataclass
class CifExportConfig:
    """
    Full configuration for the flake -> CIF export & placement pipeline.

    Pipeline overview:
      1. Marker detection — find the top-2 crosshair markers (geometric only,
         no OCR: the filename gives the left marker's grid ID, the right
         marker's grid ID is inferred as the next column letter).
      2. Calibration     — 2-marker pixel<->um affine transform.
      3. Polygon export  — whole-image target-color contour extraction,
         vertex-limit splitting, transform to um.
      4. CIF write       — write a per-image CIF, optionally splice into a
         master template CIF at the filename-derived grid placement.
    """

    # ------------------------------------------------------------------ #
    # Detection config (color ranges, marker pink/bright ranges)          #
    # ------------------------------------------------------------------ #

    detection_config_path: str = "flake_detection/config/orange_flakes_tuned.json"
    """Path to a DetectionConfig JSON — reused as-is for flake color range and
    marker pink/bright-threshold ranges, so detection and export never disagree
    on what counts as "flake color" or "marker"."""

    flake_detection_output_dir: Optional[str] = None
    """A flake_detection run's own --output folder (contains contours/,
    written by flake_detection.detector.write_contours_json). If set, each
    image's own flake_detection contours + real scores are read from
    <this>/contours/<image_stem>.json and transformed through THIS image's
    own calibration into <output>/flakes/<image_stem>.json -- the input
    contact_placement uses to know which regions are real, flake_detection-
    selected flakes (with their real score) rather than re-deriving/matching
    a second, independent segmentation. If None, or no matching contours
    file exists for an image, that image's flakes/<stem>.json is just an
    empty list -- never a guess."""

    # ------------------------------------------------------------------ #
    # Channel selection (channels_model) -- reconstructs a posterized       #
    # image from the trained model's cluster selection, BEFORE marker      #
    # detection / calibration / polygon extraction all run against it      #
    # instead of the raw original.                                         #
    # ------------------------------------------------------------------ #

    channels_model_onnx_path: str = "channels_model/outputs/channels_model.onnx"
    """Trained channels_model export (see channels_model/export.py). Must
    exist before running the pipeline -- checked once up front in run(),
    rather than failing confusingly per-image."""

    channels_model_threshold: float = 0.5
    """Sigmoid decision threshold for "is this cluster real content" --
    matches channels_model.metrics.PRED_THRESHOLD, the same threshold used
    at training/eval time, for consistency."""

    channels_n_clusters: int = 8
    channels_cluster_median_win: int = 5
    """Must match whatever channels_model_onnx_path was actually trained
    with (see channels_model/outputs/effective_config.json) -- these values
    determine which color clusters get computed and fed to the model, so a
    mismatch here silently feeds the model channels it was never trained on.
    Defaults match the current trained model (n_clusters=8, median_win=5)."""

    channels_cache_dir: str = "grid_preperation/data/channels_cache"
    """Disk cache for computed color-cluster channels (see
    channels_model.matlab_clustering.create_channels_cached). Deliberately
    NOT channels_model's own training cache dir: confirmed empirically that
    grid_preperation's image folder(s) (e.g. flakes/all) and
    channels_model/data/images can contain UNRELATED images sharing the
    same stem at different resolutions (e.g. both have an AU.png) -- reusing
    the training cache_dir guarantees cache misses (and wasted probe reads)
    for every such collision. create_channels_cached validates cache hits
    against the real image's shape and recomputes on a mismatch, so this is
    a safety/performance choice, not a correctness requirement -- but a
    dedicated cache_dir avoids the collision entirely."""

    # ------------------------------------------------------------------ #
    # Calibration (marker geometry + pixel<->um transform)                #
    # ------------------------------------------------------------------ #

    grid_pitch_um: float = 200.0
    """
    Real-world spacing between adjacent grid letters (e.g. AA to AB) — one
    single physical grid, used both to derive scale_um_per_px + rotation from
    the two detected marker pixel positions AND to convert a filename-derived
    grid ID into an absolute placement position on the master template (see
    placement_x_offset_um / placement_y_offset_um below). Confirmed by the
    user: markers AA/AB etc. are 200um apart, and the master template
    (grid_preperation/BaseGrid.cif) uses that same grid.

    This — along with the offsets below — is the one thing in this file you
    should expect to change per dataset: the pipeline works for any grid
    pitch and any image scale, as long as this value matches what was
    actually used to lay out the markers on your chip. Do not assume a
    printed scale-bar overlay (if present) reflects this value; it may not.
    """

    placement_x_offset_um: float = -2400.0
    """X (um) of grid column 'A' (idx 0). X_um = placement_x_offset_um +
    col_idx * grid_pitch_um."""

    placement_y_offset_um: float = 2400.0
    """Y (um) of grid row 'A' (idx 0). Y_um = placement_y_offset_um -
    row_idx * grid_pitch_um."""

    marker_search_top_fraction: float = 0.35
    """Only search this top fraction of the image height for crosshair markers
    — they are always the top 2 boxes."""

    crosshair_core_min_area: int = 200
    crosshair_core_max_area: int = 5000
    crosshair_core_min_extent: float = 0.75
    """
    min_area/max_area are WIDE sanity bounds only, not the real filter --
    the actual per-candidate area window is auto-detected PER IMAGE from
    the image's own real content (see markers._typical_core_area), because
    a fixed absolute pixel-area range does not generalize across
    datasets/batches captured at different pixel scales: plain grid-dot
    squares measured ~800-980px area, ~30x30px bbox on a real TS34001
    image, but ~1300-1800px area, ~40-50px bbox on flakes/all -- both real
    core sizes, not noise, and a filter tight enough for one silently
    rejected almost every real core in the other (confirmed as a real
    regression: flakes/all's own success rate fell from 17/19 to 6/19
    before this was made scale-adaptive). min_area/max_area just need to
    comfortably bracket EVERY dataset's real core size with margin.

    crosshair_core_min_extent is the fill ratio (area / bounding-box area)
    a blob must still clear regardless of the auto-detected size window --
    i.e. a solid, squarish blob. A real CROSSHAIR's own core measured
    extent 0.83 on a real TS34001 image (small notches where its arms
    connect eat into its own fill ratio slightly), clearly lower than an
    isolated grid dot's own 0.92-0.98 -- this must stay comfortably below
    that (0.75, not 0.85) or real crosshair cores get filtered out before
    arm-evidence is even checked (confirmed as a real bug this threshold
    caused). Matches a crosshair's own core AND a plain grid-dot square
    equally (same shape) — see crosshair_arm_min_count/
    crosshair_arm_min_fraction for how the two get told apart. NOT
    template-matched against crosshair_template.png (an earlier version of
    this used a core-only sub-crop of the template for that, and it was a
    real dead end: the template's own core renders much larger/fuzzier
    within its own crop than a real core's crisp, small, flat-colored
    square in the reconstructed image — a genuine shape/scale mismatch,
    not a threshold problem).
    """

    crosshair_whole_match_threshold: float = 0.5
    """
    Minimum normalized cross-correlation score (cv2.TM_CCOEFF_NORMED,
    matched against the FULL crosshair_template.png -- core AND all 4
    arms together) for a candidate from markers._find_whole_template_candidates,
    the ORIGINAL whole-shape detection this module used before
    core+arm-evidence (crosshair_core_*/crosshair_arm_*) was added
    alongside it as a complementary source, not a replacement -- confirmed
    necessary: core+arm-evidence alone, even made scale-adaptive, still
    missed real markers on flakes/all whose full shape was actually
    intact (the common case); this whole-shape match catches those
    directly and cheaply, while core+arm-evidence covers the less common
    case of one broken arm. See markers._find_candidates for how the two
    are merged.
    """

    crosshair_arm_min_fraction: float = 0.3
    """A candidate's own arm-evidence strip (see markers._arm_evidence_count)
    in a given direction (left/right/up/down) counts as "arm present" once
    at least this fraction of it is non-substrate. Loose on purpose — real
    arms are thin (1-2px) within a much wider check strip, so even a
    partially-broken arm should still clear this."""

    crosshair_arm_min_count: int = 2
    """A core candidate is confirmed as a real crosshair once at least this
    many of its 4 arm directions show evidence (see crosshair_arm_min_fraction) —
    deliberately less than 4, so ANY single arm broken by color variance
    doesn't sink an otherwise-real marker (the whole reason this two-stage
    core+arm design replaced the old single whole-shape template match).
    A plain grid-dot square (same core shape, zero arms) should score 0 here
    regardless of this threshold; raise toward 3-4 only if real debris blobs
    are still being confirmed as markers."""

    crosshair_nms_radius_px: float = 100.0
    """Non-max-suppression radius for merging duplicate correlation peaks
    belonging to the same physical marker (observed duplicate peaks within
    ~60px of the true peak) — must stay well below the true marker-to-marker
    separation (observed ~1100-1200px) so the two real markers are never
    merged into one."""

    crosshair_score_gap_min: float = 0.03
    """If a 3rd candidate's score is within this margin of the 2nd, the
    top-2 ranking is ambiguous -> alert rather than silently picking a
    pair. Score is now an arm-evidence count (0-4, see
    markers._find_candidates), an integer -- this margin matters only when
    two candidates tie exactly on arm count; a lower-arm-count 3rd
    candidate is already excluded from top-2 by the sort itself regardless
    of this gap."""

    marker_min_separation_px: float = 200.0
    """Minimum pixel distance between the two top markers. Below this, the pair
    is almost certainly wrong (picked noise or duplicate peaks of the same
    marker) -> alert."""

    marker_max_separation_px: float = 1800.0
    """Maximum plausible pixel distance between the two top markers -> alert if
    exceeded (implies wrong blobs were picked)."""

    max_marker_row_skew_ratio: float = 0.15
    """The two markers must be roughly horizontally aligned (same grid row).
    If |dy_px| / |dx_px| exceeds this ratio, alert — implies the wrong blobs
    were selected or the image has unexpected rotation for the base case."""

    marker_refine_half_px: int = 25
    """
    After template matching + validation (which use the coarser, more
    robust template-matched position), each marker position is refined to
    the crosshair's actual solid core within a window of this half-size —
    per image, not a single fixed template-wide correction. Falls back to
    the unrefined position if no plausible core is found nearby (e.g. too
    much nearby clutter) rather than guessing.
    """

    marker_refine_color_hsv: list = field(default_factory=lambda: [327, 21])
    marker_refine_hue_tolerance_deg: float = 24.0
    marker_refine_sat_tolerance_pct: float = 12.0
    """
    Refinement now runs on the reconstructed image (see channel_selection.py
    / markers._refine_marker_center), where the crosshair's solid core
    renders as a flat, exact color -- it's just a channels_model color
    cluster like any other, painted with that cluster's own real measured
    color, not the noisy antialiased blend it was in the original photo
    (which is what motivated the OLD Canny-edge-based refinement, now
    replaced).

    Matched in HSV, not RGB -- the same real-world crosshair color renders
    at a different RGB per batch/image because of lighting (confirmed: an
    earlier fixed-RGB reference, (238,190,221), worked for one batch but
    not others), but lighting is mostly a brightness/exposure effect, which
    HSV isolates into V. H/S stay comparatively stable across lighting, so
    only they're matched here -- V is intentionally UNCONSTRAINED (see
    _refine_marker_center, no v_lo/v_hi tolerance field exists). Same
    principle DetectionConfig.substrate_hsv_range already relies on (wide v
    range, tighter h range).

    marker_refine_color_hsv is [H, S] in the COMMON color-picker scale --
    H: 0-360 degrees, S: 0-100 percent (e.g. Windows Paint's "Edit colors"
    dialog on a reconstructed/*.png debug image reads these out directly,
    no manual conversion needed) -- NOT OpenCV's own H:0-179/S:0-255 scale.
    _refine_marker_center converts to OpenCV's scale internally
    (hue_cv = round(H/2) % 180, sat_cv = round(S/100*255)) right before
    thresholding; the tolerances below follow the same common-scale
    convention (hue in degrees, saturation in percent), converted the same
    way. This batch's own measured value: H=327 deg, S=21%.
    marker_refine_hue_tolerance_deg is the looser of the two on purpose --
    the user's own observation, also confirmed by comparing this
    measurement against the older RGB reference converted to the same
    scale (~321 deg / 20% -- a ~6-degree hue drift, a much smaller relative
    shift in saturation). Re-measure and update marker_refine_color_hsv
    (not just widen tolerance further) if a new batch's lighting is
    different enough that these tolerances still miss it -- never guess a
    huge tolerance instead of a real per-batch reference.

    A marker position refines to the centroid of the best same-colored
    square-ish blob found within the search window (marker_refine_half_px);
    falls back to the unrefined position if none is found (e.g. that
    marker's cluster wasn't selected by channels_model) -- never guesses.
    """

    marker_refine_min_area: int = 100
    marker_refine_max_area: int = 1500
    marker_refine_min_extent: float = 0.55
    """Shape filter for the refined core contour: plausible area range (px)
    and minimum fill ratio (area / bounding-box area). Loose floor, even
    though the core should now render as a clean filled square (extent close
    to 1.0) on the reconstructed image."""

    # ------------------------------------------------------------------ #
    # Polygon export                                                       #
    # ------------------------------------------------------------------ #

    edge_blur_kernel: int = 5
    """
    Gaussian blur kernel (odd, px) applied before edge detection (see
    polygons._edge_foreground_mask) — denoise so fine sensor/substrate
    texture doesn't itself register as edges.
    """

    edge_canny_low: float = 30.0
    edge_canny_high: float = 90.0
    """
    Canny edge detector thresholds. Canny (not a plain Sobel/gradient-
    magnitude threshold, which was tried and picked up substrate texture as
    edges almost everywhere) is used specifically because its hysteresis
    thresholding suppresses weak/noisy edges while keeping strong,
    connected ones — real content boundaries survive, substrate texture
    mostly doesn't. Foreground/background here is decided by edges, not
    color, so it's immune to the substrate's lighting gradient (which has
    near-zero *local* gradient even though its absolute color varies a lot
    across the frame — this is what every earlier color-based approach,
    fixed HSV threshold and k-means clustering alike, got tripped up by).
    """

    edge_close_kernel: int = 5
    edge_close_iterations: int = 2
    """
    Morphological closing (kernel px, iterations) applied to the Canny edge
    map before contour-finding, to bridge small gaps so real content's edges
    form closed loops that can be filled into solid regions.
    """

    gds_min_area: int = 500
    """Minimum contour area (px) to export as a polygon. Defaults to match
    DetectionConfig.min_flake_area so export and detection agree on "real" vs.
    noise. Cheap pre-filter, applied before the more expensive min_flake_score
    scoring below."""

    min_flake_score: float = 0.0
    """
    Minimum quality score [0-1] a candidate region must reach to be exported
    as a polygon at all — this pipeline's own threshold, independent of
    flake_detection's min_final_score (which only controls which *images* are
    worth reviewing, not which regions within an already-selected image get
    exported). Uses the exact same scoring formula as flake_detection
    (color_fraction / texture / isolation, weighted by the referenced
    DetectionConfig's color_fraction_weight / texture_score_weight /
    isolation_score_weight) so the two thresholds mean the same thing, just
    apply at different stages. Start at 0.0 (export everything past
    gds_min_area), inspect cif_export_results.json's per-polygon scores,
    then raise to filter.
    """

    closing_kernel_size: int = 0
    """Morphological closing kernel (odd; 0 = disabled) applied to the target-
    color mask before contour extraction, to bridge small internal gaps in a
    single flake without merging separate flakes."""

    max_vertex_n: int = 100
    """CIF polygon vertex cap — matches P.Max_Vertex_N in GDS_Code.m. Any
    contour with more vertices is recursively split."""

    flake_layer_palette: list = field(default_factory=list)
    """
    CIF layer number -> representative RGB color, used to SEED
    cif_io.LayerPalette. A polygon's color is quantized (palette_quantize_step)
    and reuses an existing entry only on an EXACT match; otherwise it gets
    its own dynamically-allocated layer with its own color (see
    cif_io.LayerPalette, shared across the whole run so the same real-world
    color always lands on the same layer everywhere, and different classes
    never collide onto the same layer just for being nearby in RGB space --
    an earlier nearest-neighbor-radius version of this did exactly that and
    was removed).

    Empty by default -- deliberately NOT the 6 "Flake" layers (101-106,
    dark-to-light gradient) pre-declared in grid_preperation/BaseGrid.cif's
    own layer-names header. Those were a fixed legacy thickness-gradient
    scheme from before polygons were colored by their real channels_model
    cluster color (see polygons.py's module docstring); confirmed empirically
    that they actively cause wrong colors now -- e.g. a real cluster color of
    RGB(160,111,124) landed within the old max_distance=60 of that palette's
    layer 104 RGB(202,99,101) and got silently forced onto the wrong color.
    Since every polygon's color now comes from an actual measured cluster
    (not an approximate flake-thickness guess), no fixed seed palette is
    needed or wanted -- every distinct real cluster color earns its own
    layer, quantized (palette_quantize_step) only for genuine near-duplicates
    across images, never snapped onto an unrelated color. Only set this if
    you specifically want new colors to defer to some other pre-existing
    palette (e.g. a different --template with its own layer scheme).
    """

    palette_quantize_step: int = 24
    """Dynamically-allocated layer colors are rounded to the nearest
    multiple of this (per RGB channel) so near-duplicate colors reuse the
    same dynamic layer instead of each spawning a new one."""

    # ------------------------------------------------------------------ #
    # CIF output / placement                                              #
    # ------------------------------------------------------------------ #

    template_cif_path: Optional[str] = "grid_preperation/BaseGrid.cif"
    """Master CIF to splice exported polygons into. If None, only per-image
    CIF files are written and no placement is attempted."""

    output_dir: Optional[str] = None
    """Destination for per-image CIF files and the results summary. Defaults
    to <input_folder>/cif_export_results/."""

    # ------------------------------------------------------------------ #
    # Serialization helpers                                               #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Persist config to JSON for reproducibility."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> CifExportConfig:
        """Load config from a JSON file. Keys prefixed with '_' are comments."""
        with open(path) as f:
            data = json.load(f)
        data = {k: v for k, v in data.items() if not k.startswith('_')}
        return cls(**data)

    def load_detection_config(self) -> DetectionConfig:
        """Load the referenced DetectionConfig (color/marker ranges)."""
        return DetectionConfig.load(self.detection_config_path)
