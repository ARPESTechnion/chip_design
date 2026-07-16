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

    crosshair_match_threshold: float = 0.5
    """
    Minimum normalized cross-correlation score (cv2.TM_CCOEFF_NORMED, matched
    against crosshair_template.png over the substrate-exclusion mask — see
    markers.py) for a candidate to count as a crosshair marker. The reticle is
    a faint, partially alpha-blended overlay with no fixed solid color, so it's
    located by shape-correlation rather than a color-range threshold. Tuned
    empirically against the flakes/all/ sample set; expect occasional
    borderline images (e.g. SY.png) to sit right at this boundary — revisit
    with a larger real dataset.
    """

    crosshair_nms_radius_px: float = 100.0
    """Non-max-suppression radius for merging duplicate correlation peaks
    belonging to the same physical marker (observed duplicate peaks within
    ~60px of the true peak) — must stay well below the true marker-to-marker
    separation (observed ~1100-1200px) so the two real markers are never
    merged into one."""

    crosshair_score_gap_min: float = 0.03
    """If a 3rd candidate's score is within this margin of the 2nd, the
    top-2 ranking is ambiguous -> alert rather than silently picking a pair."""

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

    marker_refine_canny_low: float = 15.0
    marker_refine_canny_high: float = 50.0
    """Canny thresholds for finding the core's edge within the refinement
    window (see markers._refine_marker_center) — lower than the whole-image
    polygon-extraction Canny thresholds since this is a small, low-contrast
    region and needs to be more sensitive."""

    marker_refine_min_area: int = 100
    marker_refine_max_area: int = 1500
    marker_refine_min_extent: float = 0.55
    """Shape filter for the refined core contour: plausible area range (px)
    and minimum fill ratio (area / bounding-box area). The core is a
    rounded pentagon/house shape, not a perfect square, so this is
    deliberately looser than a strict squareness check."""

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

    flake_layer_palette: list = field(default_factory=lambda: [
        {"layer": 101, "rgb": [109, 59, 24]},
        {"layer": 102, "rgb": [138, 27, 34]},
        {"layer": 103, "rgb": [147, 68, 64]},
        {"layer": 104, "rgb": [202, 99, 101]},
        {"layer": 105, "rgb": [253, 132, 127]},
        {"layer": 106, "rgb": [235, 180, 131]},
    ])
    """
    CIF layer number -> representative RGB color. A polygon whose own
    measured color is close enough (palette_max_distance) to one of these
    reuses that layer; anything else (markers, debris — not flake-colored at
    all) gets its own dynamically-allocated layer with its own color instead
    of being forced onto a mismatched "Flake" layer (see cif_io.LayerPalette,
    shared across the whole run so the same real-world color always lands on
    the same layer everywhere). Defaults to the 6 "Flake" layers (101-106)
    already pre-declared with these exact colors in
    grid_preperation/BaseGrid.cif's own layer-names header (a dark-to-light
    gradient, evidently meant to represent flake thickness) — override this
    if you point --template at a different master CIF with a different
    palette.
    """

    palette_max_distance: float = 60.0
    """Max Euclidean RGB distance for a polygon to reuse an existing palette
    layer (base or already-dynamically-allocated) rather than getting a new
    dynamic layer of its own."""

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
