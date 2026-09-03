"""
Configuration for the flake detection pipeline.

All parameters are documented with their purpose and tuning guidance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Optional

import numpy as np


@dataclass
class RGBRange:
    """
    Color range in RGB space. All channel values in [0, 255].
    Used as the target color for scoring when color_mode='rgb'.
    """
    r_min: int = 0
    r_max: int = 255
    g_min: int = 0
    g_max: int = 255
    b_min: int = 0
    b_max: int = 255


@dataclass
class HSVRange:
    """
    Color range in HSV space, in the COMMON color-picker scale -- Hue in
    degrees [0, 360], Saturation/Value in percent [0, 100] -- e.g. what
    Windows Paint's "Edit colors" dialog (or any standard color picker)
    shows directly, NOT OpenCV's own native H:[0,179]/S:[0,255]/V:[0,255]
    scale. Every config file (and every human measuring a color from a
    screenshot) works in this scale; to_opencv_bounds() does the
    conversion right before a cv2 call, so no caller/config-editor ever
    does it by hand. Same convention already established elsewhere in this
    codebase (see grid_preperation.markers._refine_marker_center's own
    HSV handling) -- unified here so it's the ONE rule for every HSV field
    in any config, not something to re-derive per field.

      Hue        : [0, 360]  (0=red, 30=orange, 60=yellow, 120=green, 180=cyan, 240=blue, 300=magenta)
      Saturation : [0, 100]  (0=grey/white, 100=fully saturated)
      Value      : [0, 100]  (0=black, 100=bright)

    Used for both substrate detection and target color scoring.
    """
    h_min: float = 0.0
    h_max: float = 360.0
    s_min: float = 0.0
    s_max: float = 100.0
    v_min: float = 0.0
    v_max: float = 100.0

    def to_opencv_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """(lower, upper) uint8 arrays in OpenCV's own H:0-179/S:0-255/
        V:0-255 scale, ready for cv2.inRange. Does NOT handle a
        hue range that wraps past 0/360 (e.g. h_min=350, h_max=10) --
        none of this codebase's real ranges need that; a single
        cv2.inRange call assumes h_min <= h_max in OpenCV's own 0-179
        space too, same as before this class stored the common scale.
        Clamped, not modular, at the top end -- 360deg means "the same hue
        as 0deg" physically, but clamping to OpenCV's real max index (179)
        instead of wrapping to 0 is what makes h_min=0/h_max=360 (the
        natural "match every hue" spelling) actually cover the full range,
        rather than collapsing to a single point (confirmed as a real bug
        the first version of this method had: round(360/2) % 180 == 0)."""
        def _hue(deg: float) -> int:
            return max(0, min(179, round(deg / 2)))

        def _pct(p: float) -> int:
            return max(0, min(255, round(p / 100 * 255)))

        lower = np.array([_hue(self.h_min), _pct(self.s_min), _pct(self.v_min)], dtype=np.uint8)
        upper = np.array([_hue(self.h_max), _pct(self.s_max), _pct(self.v_max)], dtype=np.uint8)
        return lower, upper


@dataclass
class DetectionConfig:
    """
    Full configuration for the flake detection pipeline.

    Pipeline overview:
      1. Preprocess     — Gaussian blur
      2. Candidate mask — invert the substrate HSV mask to find all visible material;
                          opening removes noise, closing bridges cracks within flakes
      3. Contours       — find connected non-substrate regions; filter by size, then by
                          border-touching and solidity (irregular/spiky shapes)
      4. Color fraction — fraction of each candidate matching the target color;
                          hard-filtered by min_color_fraction before scoring
      5. Texture        — Laplacian variance + intensity std inside each contour
      6. Isolation      — non-substrate pixel density in the surrounding neighborhood;
                          captures ALL debris (orange and green alike)
      7. Score          — weighted combination of color, texture, isolation; filter and rank
    """

    # ------------------------------------------------------------------ #
    # Marker removal (Stage 1 — preprocessing)                            #
    # ------------------------------------------------------------------ #

    marker_removal_enabled: bool = True
    """
    If True, location marker pixels (pink dots + white crosshair arms) are
    excluded from all candidate detection.

    The full marker region is built as:
      1. Pink pixels    — pixels inside marker_rgb_range
      2. Bright pixels  — pixels where ALL channels >= marker_bright_threshold
                          (catches the white crosshair arm lines)
      3. Dilation       — the combined mask is dilated by marker_dilation_px
                          to cover the bright-substrate halo around each marker

    Removing the full marker region prevents large crosshair blobs from
    appearing as candidates and generating spurious red annotations.
    """

    marker_rgb_range: RGBRange = field(
        default_factory=lambda: RGBRange(
            r_min=220, r_max=255, g_min=160, g_max=220, b_min=220, b_max=255
        )
    )
    """RGB range of the pink marker dot pixels."""

    marker_bright_threshold: int = 200
    """
    All-channel brightness threshold for detecting white crosshair arm pixels.
    Any pixel with R, G, and B all >= this value is treated as a marker.
    Set to 255 to disable bright-pixel exclusion.
    """

    marker_dilation_px: int = 25
    """
    Dilation radius applied to the combined marker mask after pink + bright
    detection. Covers the bright-substrate halo that surrounds each marker and
    breaks connectivity with genuine flake candidates.
    """

    # ------------------------------------------------------------------ #
    # Substrate detection (Stage 2)                                        #
    # ------------------------------------------------------------------ #

    substrate_hsv_range: HSVRange = field(
        default_factory=lambda: HSVRange(
            h_min=210, h_max=290, s_min=15.7, s_max=100.0, v_min=7.8, v_max=62.7
        )
    )
    """
    HSV range of the substrate background color.
    Everything outside this range is treated as potential flake material.

    The default targets a blue/purple SiO2 substrate:
      H 105-145 covers blue through blue-purple
      S 40-255  excludes near-neutral grey pixels
      V 20-160  captures the medium-dark substrate brightness

    Tune by widening until the background mask covers the substrate cleanly
    with no holes, then check that flake regions are excluded.
    """

    # ------------------------------------------------------------------ #
    # Target color (Stage 4 — scoring only)                               #
    # ------------------------------------------------------------------ #

    color_mode: Literal['rgb', 'hsv'] = 'hsv'
    """
    Color space for the target flake color mask.
    'hsv' is recommended — hue is decoupled from brightness.
    'rgb' is simpler but sensitive to illumination gradients.
    """

    rgb_range: RGBRange = field(default_factory=RGBRange)
    """Active when color_mode='rgb'. Defines the desired flake color."""

    hsv_range: HSVRange = field(default_factory=HSVRange)
    """Active when color_mode='hsv'. Defines the desired flake color."""

    min_color_fraction: float = 0.3
    """
    Hard minimum: the fraction of a candidate flake's pixels that must match
    the target color for it to be scored at all.
    A candidate below this threshold is discarded regardless of other scores.

    0.3 = at least 30% of the flake must be the target color.
    Raise for higher precision (fewer false positives), lower for higher recall.
    """

    # ------------------------------------------------------------------ #
    # Preprocessing                                                        #
    # ------------------------------------------------------------------ #

    blur_kernel_size: int = 5
    """
    Gaussian blur kernel (must be odd). Applied before masking.
    Suppresses sensor noise. Typical range: 3-9.
    """

    closing_kernel_size: int = 0
    """
    Morphological closing kernel (must be odd; 0 = disabled).
    Applied to the candidate mask after opening.
    Closing = dilate then erode: bridges narrow cracks so a fractured flake
    is detected as one contour. Increase if flakes are being split.
    Trade-off: too large may merge genuinely separate nearby flakes.
    """

    # ------------------------------------------------------------------ #
    # Size filtering (Stage 3)                                            #
    # ------------------------------------------------------------------ #

    min_flake_area: int = 500
    """
    Minimum candidate area in pixels. Drops dust, noise, and grid markers.
    """

    max_flake_area: int = 100_000
    """
    Maximum candidate area in pixels. Drops large substrate regions that
    happen to fall outside the substrate HSV range.
    """

    large_flake_alert_area: int = 20_000
    """
    Candidates with area_px >= this are flagged is_large=True and drawn in
    orange. They still pass through full scoring. Set 0 to disable.
    """

    min_solidity: float = 0.4
    """
    Minimum solidity (contour area / convex-hull area) [0-1] required for a
    candidate AT OR ABOVE solidity_relief_area_hi (the size-relief "large
    flake" end) -- see min_solidity_strict for the small-flake end and
    solidity_relief_area_lo/hi for how the two are interpolated between.
    1.0 = perfectly convex; solidity measures concavity, i.e. how much of
    the shape's own convex hull is "missing" into dents/notches -- it
    rejects irregular/spiky/multi-lobed shapes (star-shaped debris, several
    touching fragments merged into one contour, jagged noise).

    Why there are TWO thresholds (size-relief), not one flat value: a real,
    single-piece flake can be legitimately MORE concave the bigger it is --
    larger flakes are more likely to have real physical bends, multiple
    domains, or genuine notches from exfoliation, all of which lower
    solidity without the shape being junk. A single long, thin, curved
    strip is the sharpest example: it produces one big concave region (a
    large area deficit against the convex hull) purely from being bent,
    even though it's a completely legitimate single piece. Empirically
    confirmed across all 3 real datasets (flakes/all, TS34001, Ta2009)
    before this was caught: a flat 0.85 rejected 3 real flakes (Ta2009
    BH.png 0.813/IT.png 0.827/PW.png 0.848, all ordinary jagged-edge
    shapes), a flat 0.7 additionally rejected flakes/all's IH.png (solidity
    0.577, a long gently-S-curved strip scoring 0.93 on every OTHER
    metric -- clearly one real flake by eye). Size-relief lets the SMALL
    end be stricter (0.72, catching more junk there) while the LARGE end
    stays permissive (0.4) for exactly the shapes that legitimately need
    it -- verified safe (0.117 minimum margin, TS34001's WI.png) against
    every known-good flake across all 3 datasets at these exact values;
    don't raise either endpoint without re-checking against real
    `solidity`/`area_px` values in results.json first.

    If you need a stronger filter for genuinely multi-lobed/spiky shapes
    specifically (as opposed to single-piece curved ones), a
    convexity-DEFECT-COUNT metric (cv2.convexityDefects, counting distinct
    dents rather than summing their area) would likely separate that case
    better than solidity alone -- untried here, no real data to calibrate
    it against yet.
    """

    min_solidity_strict: float = 0.72
    """Minimum solidity required for a candidate AT OR BELOW
    solidity_relief_area_lo (the size-relief "small flake" end) -- see
    min_solidity's own docstring for the full size-relief rationale."""

    solidity_relief_area_lo: float = 1500.0
    """Candidate area (px) at/below which min_solidity_strict applies in
    full. Between this and solidity_relief_area_hi, the required solidity
    is linearly interpolated between min_solidity_strict and min_solidity."""

    solidity_relief_area_hi: float = 6500.0
    """Candidate area (px) at/above which min_solidity (the lenient,
    large-flake end) applies in full. See solidity_relief_area_lo."""

    reject_border_touching: bool = True
    """
    If True, a candidate whose bounding box touches any image edge is
    rejected outright (regardless of its other scores) -- such a contour
    is necessarily clipped by the frame, so its true area/shape/color
    fraction cannot be measured accurately, and it may well continue
    beyond the captured image entirely. Set False to allow border-touching
    candidates through (e.g. if the capture region is known to be a strict
    crop of a larger flake you still want scored).
    """

    # ------------------------------------------------------------------ #
    # Texture scoring (Stage 5)                                           #
    # ------------------------------------------------------------------ #

    contour_erosion_px: int = 3
    """
    Pixels to erode the contour mask inward before texture measurement.
    Excludes the flake boundary (always high-contrast) from internal metrics.
    """

    max_laplacian_var: float = 500.0
    """
    Laplacian variance threshold. Values above this map to texture_score=0.
    Measures internal edge density — high = cracks or contamination.
    Inspect laplacian_var in results.json for known good/bad flakes to tune.
    """

    max_intensity_std: float = 30.0
    """
    Intensity std threshold. Values above this map to texture_score=0.
    Measures color uniformity — high = contamination or mixed layers.
    Inspect intensity_std in results.json to tune.
    """

    laplacian_weight: float = 0.4
    """Weight of Laplacian variance in texture_score. Must sum to 1 with std_weight."""

    std_weight: float = 0.6
    """Weight of intensity std in texture_score. Must sum to 1 with laplacian_weight."""

    # ------------------------------------------------------------------ #
    # Isolation scoring (Stage 6)                                         #
    # ------------------------------------------------------------------ #

    neighborhood_factor: float = 3.0
    """
    The neighborhood checked for nearby material is this multiple of the
    candidate's bounding box. 3.0 = a 3× larger bounding box is examined.
    """

    max_neighbor_coverage: float = 0.10
    """
    Maximum fraction of the neighborhood that may be covered by other
    non-substrate material before isolation_score drops to 0.
    Now captures ALL debris (orange and green), unlike the old color-only mask.
    """

    # ------------------------------------------------------------------ #
    # Dark spot detection                                                  #
    # ------------------------------------------------------------------ #

    dark_spot_enabled: bool = True
    """Enable independent dark-spot detection pass."""

    dark_spot_brightness_max: int = 50
    """
    Grayscale brightness threshold [0-255]. Pixels below this are dark spots.
    Indicates thick deposits, contamination, or voids.
    """

    dark_spot_min_area: int = 100
    """Minimum dark spot area in pixels."""

    dark_spot_max_area: int = 10_000
    """Maximum dark spot area in pixels."""

    # ------------------------------------------------------------------ #
    # Final scoring (Stage 7)                                             #
    # ------------------------------------------------------------------ #

    color_fraction_weight: float = 0.4
    """
    Weight of color_score (= color_fraction) in the final score.
    Must sum to 1 with texture_score_weight + isolation_score_weight.
    """

    texture_score_weight: float = 0.2
    """Weight of texture_score in the final score."""

    isolation_score_weight: float = 0.4
    """Weight of isolation_score in the final score."""

    min_final_score: float = 0.0
    """
    Minimum combined score [0-1] to include a flake in results.
    Start at 0.0, inspect the score distribution, then raise to filter.
    """

    top_n: Optional[int] = None
    """If set, return only the top N images by best flake score."""

    # ------------------------------------------------------------------ #
    # Interactive review (post-processing)                                #
    # ------------------------------------------------------------------ #

    interactive_review: bool = False
    """
    If True, flake_detection.run launches an interactive Tkinter review
    window immediately after detection finishes (see flake_detection.ui):
    one chip image at a time, EVERY contour that cleared the size filter
    drawn green (chosen) or red (rejected for touching the border, low
    solidity, wrong color, or a low final score); click a flake to toggle
    it. This lets a human both reject a false positive AND rescue a
    near-miss the pipeline rejected at any stage -- only contours that
    failed the size filter itself never reach this stage (bulk dust/noise/
    grid-marker debris, with no geometry kept for them at all).

    Per-image "Save Changes" rewrites that image's own contours/<stem>.json
    (what grid_preperation reads) and scores/<stem>.txt to the currently-
    chosen flakes only, and rewrites results.json (which keeps every flake
    regardless of chosen, for later inspection). "Reset Changes" reverts the
    current image to whatever was last saved to disk. "Skip All" ends the
    review outright -- any image never explicitly saved is left exactly as
    detect_flakes() originally wrote it (i.e. its original chosen flags).

    When running through all_code.py's full pipeline, this flag is also
    gated by PipelineConfig.interactive (a master switch covering every
    stage's own interactive UI) -- set both to actually see the window.
    """

    # ------------------------------------------------------------------ #
    # Serialization helpers                                               #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Persist config to JSON for reproducibility."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> DetectionConfig:
        """
        Load config from a JSON file.
        Keys prefixed with '_' are treated as comments and ignored.
        """
        with open(path) as f:
            data = json.load(f)
        data = {k: v for k, v in data.items() if not k.startswith('_')}

        def _clean(d: dict) -> dict:
            return {k: v for k, v in d.items() if not k.startswith('_')}

        data['rgb_range'] = RGBRange(**_clean(data.pop('rgb_range')))
        data['hsv_range'] = HSVRange(**_clean(data.pop('hsv_range')))
        data['substrate_hsv_range'] = HSVRange(**_clean(data.pop('substrate_hsv_range')))
        if 'marker_rgb_range' in data:
            data['marker_rgb_range'] = RGBRange(**_clean(data.pop('marker_rgb_range')))
        return cls(**data)

    @classmethod
    def from_args(cls, args) -> DetectionConfig:
        """
        Build a DetectionConfig from a parsed argparse.Namespace.
        Loads from args.config if provided, then applies CLI overrides on top.
        """
        if getattr(args, 'config', None) is not None:
            config = cls.load(args.config)
            print(f"Loaded config from {args.config}")
        else:
            config = cls()

        overrides = {
            'color_mode':      getattr(args, 'color_mode', None),
            'min_flake_area':  getattr(args, 'min_area', None),
            'max_flake_area':  getattr(args, 'max_area', None),
            'min_final_score': getattr(args, 'min_score', None),
            'top_n':           getattr(args, 'top_n', None),
        }
        for attr, value in overrides.items():
            if value is not None:
                setattr(config, attr, value)

        return config
