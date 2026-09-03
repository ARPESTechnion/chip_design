"""
Geometric-only detection of the top-2 crosshair calibration markers.

No OCR / letter-reading anywhere: a marker's grid ID is derived from the image
filename and grid adjacency (see calibration.py), never read off the pixels.
This module only locates the two markers' pixel positions and validates that
the "base case" holds (exactly two clean crosshair markers near the top of the
image, roughly horizontally aligned) — anything else raises so the caller can
alert and skip the image rather than guess.

Detection approach: the crosshair reticle turns out to be a faint, partially
alpha-blended overlay with no fixed solid color (unlike the grid-dot squares
and letter labels, which are solid and match flake_detection's marker color
ranges cleanly) — a plain color-threshold + connected-components approach
could not reliably isolate it. Instead we locate it by shape: normalized
cross-correlation template matching (cv2.matchTemplate) against a fixed
reference crop (crosshair_template.png, shipped alongside this module),
applied to a non-substrate mask so matching is robust to per-image
brightness/color variation.

Two matching strategies, not one whole-shape template match alone: the
reticle's 4 thin arms turn out to be the LEAST color-stable part of a real
image -- under enough lighting/color variance, an individual arm's own
pixels can drift into the mask's own excluded color range and vanish
entirely, even though the marker itself is perfectly real (confirmed
directly on a real batch: one marker's left arm disappeared from the mask
in one image while its right and vertical arms survived fine). A single
correlation score against the WHOLE cross (core + all 4 arms,
_find_whole_template_candidates) is fragile to exactly this, so it's
paired with a second strategy tolerant of a broken arm: find solid
core-shaped blobs alone (_find_core_blobs) and confirm each is really a
crosshair via _arm_evidence_count, which only needs a MAJORITY of the 4
arms to show evidence, not all 4.

Two masks, not one, feed both strategies: the non-substrate mask
(_non_substrate_mask, hue/saturation-range based) and a per-image Otsu
V-channel split (_otsu_v_mask, brightness-based, no color dependence at
all). Each mask fails on different images for different reasons, so a
marker missed by one can still be found by the other. find_top_markers
tries each mask ALONE first (not pooled from the start) and only pools
them together as a last resort -- pooling up front let one mask's own
false positives contaminate an image the other mask would have solved
cleanly (confirmed empirically: the Otsu mask's whole-template match
picked up unrelated bright-blob false positives on one image that tripped
the ambiguous-ranking guard on what was otherwise a clean substrate-mask
result). See _find_candidates and find_top_markers for the full pipeline.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from flake_detection.config import DetectionConfig

from .config import CifExportConfig

_TEMPLATE_PATH = Path(__file__).resolve().parent / "crosshair_template.png"
_template_cache: np.ndarray | None = None
_template_core_offset_cache: tuple[float, float] | None = None


@dataclass
class MarkerPair:
    left_px: tuple[float, float]   # (x, y), the filename's grid marker
    right_px: tuple[float, float]  # (x, y), one grid column to the right


class MarkerDetectionError(Exception):
    """Raised when the top-2 crosshair markers can't be reliably located —
    callers should alert and skip the image, never guess."""


@dataclass
class MarkerStageResult:
    """
    Phase-A (detection-only) outcome for one image, produced by
    grid_preperation.run._detect_markers_for_image. Consumed by Phase B
    (grid_preperation.run._finish_image) and, when
    CifExportConfig.interactive_alignment is True, by
    grid_preperation.ui's manual-correction review window in between the
    two -- see that module's own docstring for why marker_pair here is
    never trusted directly by Phase B (which always re-checks the override
    file on disk instead).
    """
    img_path: Path
    chan_sel: Optional[Any]          # channel_selection's own reconstruct result; None only if channel selection itself failed (nothing to show/calibrate)
    marker_pair: Optional[MarkerPair]  # None if no override existed yet AND (automatic detection failed OR the image is dropped)
    source: str                      # 'auto' | 'override' | 'failed' | 'dropped'
    error: Optional[str] = None      # set only when source == 'failed'


@dataclass
class MarkerOverride:
    """A persisted per-image correction from grid_preperation.ui (see
    write_marker_override/load_marker_override) -- a corrected MarkerPair to
    use instead of automatic detection, a decision to drop the image from
    this run entirely (dropped=True, regardless of any position), or both at
    once (a position was saved before the image was later dropped -- kept
    around so un-dropping it later doesn't lose that position)."""
    marker_pair: Optional[MarkerPair]  # None only if dropped and no position was ever saved
    dropped: bool = False


def load_marker_override(overrides_dir: Path, stem: str) -> Optional[MarkerOverride]:
    """A previously human-corrected MarkerOverride for this image (see
    write_marker_override), or None if it was never corrected. Checked by
    BOTH grid_preperation.run's detection stage (so a correction saved in
    an earlier interactive session skips automatic detection entirely on
    later runs) and its finish stage (so a correction saved DURING the
    current run's own interactive review, after detection already ran, is
    still picked up -- see that module's own docstring)."""
    path = Path(overrides_dir) / f"{stem}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    marker_pair = None
    if 'left_px' in data and 'right_px' in data:
        marker_pair = MarkerPair(left_px=tuple(data['left_px']), right_px=tuple(data['right_px']))
    return MarkerOverride(marker_pair=marker_pair, dropped=bool(data.get('dropped', False)))


def write_marker_override(
    overrides_dir: Path,
    stem: str,
    marker_pair: Optional[MarkerPair],
    dropped: bool = False,
) -> None:
    """Persist a human-corrected MarkerOverride for this image, permanently
    overriding automatic detection for it (and/or dropping it from every
    future run entirely) until this file is deleted by hand -- see
    load_marker_override. marker_pair may be None (a dropped image with no
    position ever saved)."""
    overrides_dir = Path(overrides_dir)
    overrides_dir.mkdir(parents=True, exist_ok=True)
    path = overrides_dir / f"{stem}.json"
    data: dict = {'dropped': dropped}
    if marker_pair is not None:
        data['left_px'] = list(marker_pair.left_px)
        data['right_px'] = list(marker_pair.right_px)
    path.write_text(json.dumps(data, indent=2))


def _load_template() -> np.ndarray:
    global _template_cache
    if _template_cache is None:
        template = cv2.imread(str(_TEMPLATE_PATH), cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise FileNotFoundError(f"crosshair template not found at {_TEMPLATE_PATH}")
        _template_cache = template.astype(np.float32)
    return _template_cache


def _template_core_offset() -> tuple[float, float]:
    """
    (x, y) offset, within the template image, of the crosshair's actual
    solid core -- close to but not exactly the template's geometric center
    (tw/2, th/2).

    The template was cropped generously around a real crosshair without
    precisely centering the crop on it, so there's a small real offset
    between the two. Measuring it needs care: eroding away the thin 1-2px
    arms and centroiding *everything* that survives is contaminated by a
    fragment of the adjacent letter that wasn't fully cropped out of the
    template (confirmed visually -- a first attempt at this measured a
    bogus ~(-30, -32) offset dominated by that fragment, which made real
    output visibly worse, not better). Fix: erode more aggressively (fully
    removes both the arms and the letter fragment's thin parts, but the
    core -- being a solid blob -- survives) and take connected components,
    keeping the one nearest the geometric center. Stable across erosion
    kernel size 9-13px (~96, ~111 throughout) -- converges on ~(+11, -4)
    from geometric center, matching the small "few micron" drift originally
    reported, not a large one.

    Deliberately NOT scaled per-image to the real image's own apparent
    crosshair size (a scale-adaptive version of this offset, using
    _typical_core_area as the per-image scale reference, was tried and
    reverted -- see [[project_marker_detection_redesign]] memory notes:
    the per-image scale estimate was too noisy on TS34001 to reliably beat
    this fixed offset, made some images measurably worse, and the small
    average improvement wasn't worth the added noise/complexity).
    """
    global _template_core_offset_cache
    if _template_core_offset_cache is None:
        template = _load_template().astype(np.uint8)
        th, tw = template.shape
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        eroded = cv2.erode(template, erode_kernel)
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=8)
        best = None
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] < 50:
                continue
            cx, cy = centroids[i]
            dist = (cx - tw / 2) ** 2 + (cy - th / 2) ** 2
            if best is None or dist < best[0]:
                best = (dist, cx, cy)
        if best is None:
            raise RuntimeError("crosshair_template.png: no plausible core component found")
        _template_core_offset_cache = (float(best[1]), float(best[2]))
    return _template_core_offset_cache


def _non_substrate_mask(image_bgr: np.ndarray, detection: DetectionConfig) -> np.ndarray:
    """Pixels outside the substrate HSV range — everything that isn't bare
    substrate (markers, labels, flakes, debris)."""
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower, upper = detection.substrate_hsv_range.to_opencv_bounds()
    substrate_mask = cv2.inRange(image_hsv, lower, upper)
    return cv2.bitwise_not(substrate_mask)


def _otsu_v_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Alternative to _non_substrate_mask: split the HSV V (brightness)
    channel alone via Otsu's method -- an automatic, PER-IMAGE split point,
    not a fixed cutoff (a fixed global brightness threshold was tried and
    confirmed to NOT generalize: flakes/all 1/19, TS34001 2/18, the exact
    same fixed-vs-scale-adaptive lesson as _find_core_blobs's own redesign,
    see its docstring). A discrete "top-N brightest colors present" variant
    (exploiting the reconstructed image's posterized, small-palette nature)
    was also tried and reverted -- see [[project_marker_detection_redesign]]
    memory notes: it structurally can't include the crosshair's own arms
    (rendered dimmer than the core, so excluded from any small top-N
    selection no matter how N is tuned), and a follow-up letter-adjacency
    strategy meant to route around that regressed badly on flakes/all
    (multiple genuinely letter-labeled markers per search region tied at
    an identical flat score). Otsu's CONTINUOUS per-image split naturally
    keeps core and arms together (both much brighter than substrate, even
    though arms are dimmer than the core) without any of that. Complementary
    to _non_substrate_mask, not a replacement: sidesteps hue/saturation
    matching entirely, so it still catches a real marker whose arm color
    happens to fall inside detection.substrate_hsv_range (the exact
    failure mode that motivated this module's two-stage core+arm design in
    the first place) as long as that arm is still visibly brighter than
    the substrate. Both masks are fed through both matching strategies in
    _find_candidates, since the two masks fail on different images for
    different reasons."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    _, mask = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


_ARM_CHECK_MARGIN_FRAC = 0.27
"""Extra clearance past a candidate blob's own measured half-size before
the arm-evidence check window starts (see _arm_evidence_count), as a
fraction of that candidate's own half_extent -- keeps the core's own edge
pixels from being counted as "arm" evidence. A FRACTION of each
candidate's own size, not a fixed pixel count -- confirmed necessary:
different datasets/batches are captured at genuinely different pixel
scales (flakes/all's own grid dots measured ~1300-1800px area vs
TS34001's ~800-980px, a real ~1.3x difference in linear size, not noise),
so fixed absolute pixel geometry silently breaks on whichever dataset it
wasn't tuned against."""

_ARM_CHECK_LENGTH_FRAC = 5.3
"""How far out (past the margin above) each direction's arm-evidence strip
reaches, as a multiple of the candidate's own half_extent. Same
scale-relative reasoning as _ARM_CHECK_MARGIN_FRAC."""

_ARM_STRIP_HALF_WIDTH_FRAC = 0.53
"""Half-width of the strip checked for arm evidence in each of the 4
directions, as a multiple of the candidate's own half_extent. Same
scale-relative reasoning as _ARM_CHECK_MARGIN_FRAC."""

_TYPICAL_AREA_TOLERANCE = (0.55, 1.8)
"""(low, high) multipliers applied to a search region's own auto-detected
typical core area (see _typical_core_area) to decide which blobs count as
core candidates at all -- wide enough to include a real crosshair core
(measured ~14% smaller area than a plain grid dot in the same image, due
to small notches where its arms connect eating into its own fill ratio)
without also admitting a text label or a merged multi-blob."""


def _typical_core_area(mask: np.ndarray, config: CifExportConfig) -> float | None:
    """This image's own typical grid-dot/marker-core blob area (the
    median area among squarish, solid blobs in the search region) --
    self-calibrated per image/dataset rather than assuming one fixed
    absolute pixel-area range works everywhere (see
    _ARM_CHECK_MARGIN_FRAC's own docstring for why that assumption broke).
    Most blobs of this rough size/shape in the search region are plain
    grid dots (there are far more of them than there are crosshairs), so
    their median area is a robust per-image reference point even before
    any actual crosshair has been identified. None if too few
    plausible-square blobs exist to establish one (caller should raise
    MarkerDetectionError rather than guessing a fallback)."""
    n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = []
    for i in range(1, n_labels):
        x, y, w, h, area = stats[i]
        if w == 0 or h == 0 or not (config.crosshair_core_min_area <= area <= config.crosshair_core_max_area):
            continue
        extent = area / (w * h)
        aspect = max(w, h) / min(w, h)
        if extent >= config.crosshair_core_min_extent and aspect <= 1.5:
            areas.append(area)
    if len(areas) < 3:
        return None
    return float(np.median(areas))


def _find_core_blobs(mask: np.ndarray, config: CifExportConfig) -> list[tuple[float, float, float]]:
    """Every connected component in the real non-substrate mask that's
    plausibly a solid marker-style square core -- within
    _TYPICAL_AREA_TOLERANCE of this image's own auto-detected typical core
    area (_typical_core_area), and fill ratio (area / bounding-box area)
    >= crosshair_core_min_extent. Matches BOTH a crosshair's own core AND
    a plain grid-dot square equally (they're the same shape) -- see
    _arm_evidence_count for how the two get told apart. Returns (cx, cy,
    half_extent) per blob, half_extent being half its own bounding box's
    average side length (used to size the arm-evidence check windows
    proportionally per candidate, see _ARM_CHECK_MARGIN_FRAC).

    Deliberately NOT template-matching (an earlier version of this
    function cropped just the core out of crosshair_template.png and
    matched that instead) -- confirmed as a real dead end: the template's
    own core renders much LARGER/fuzzier within its own crop than a real
    core's crisp, small, flat-colored square in the reconstructed image,
    a genuine shape/scale mismatch (not a threshold problem) that gave
    real cores a lower correlation score than the OLD whole-shape match
    did. Connected-component blob stats sidestep needing the template to
    agree with the real image's own rendering at all."""
    typical_area = _typical_core_area(mask, config)
    if typical_area is None:
        return []
    lo, hi = _TYPICAL_AREA_TOLERANCE
    area_lo, area_hi = typical_area * lo, typical_area * hi

    n_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs: list[tuple[float, float, float]] = []
    for i in range(1, n_labels):
        x, y, w, h, area = stats[i]
        if not (area_lo <= area <= area_hi):
            continue
        extent = area / (w * h) if w * h else 0.0
        if extent < config.crosshair_core_min_extent:
            continue
        cx, cy = centroids[i]
        blobs.append((float(cx), float(cy), (w + h) / 4.0))
    return blobs


def _arm_evidence_count(mask: np.ndarray, cx: float, cy: float, half_extent: float, config: CifExportConfig) -> int:
    """How many of the 4 directions (left/right/up/down) from a candidate
    core position show a plausible arm in the REAL image's own
    non-substrate mask -- checked as "at least
    config.crosshair_arm_min_fraction of a thin strip, positioned just
    outside the candidate's OWN measured half-size (not a fixed
    template-derived size -- every candidate can be a slightly different
    apparent size), is non-substrate". Tolerant of any ONE arm being
    broken by color variance (a real crosshair only needs a majority of
    its 4 arms to survive, not all of them, per
    config.crosshair_arm_min_count) -- unlike the old whole-shape template
    match this design replaces, which required the complete cross intact
    in one correlation score."""
    h, w = mask.shape
    margin = half_extent * _ARM_CHECK_MARGIN_FRAC
    length = half_extent * _ARM_CHECK_LENGTH_FRAC
    strip_half_width = half_extent * _ARM_STRIP_HALF_WIDTH_FRAC
    skip = half_extent + margin
    count = 0
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        x0 = cx + dx * skip
        y0 = cy + dy * skip
        x1 = cx + dx * (skip + length)
        y1 = cy + dy * (skip + length)
        sx0, sx1 = sorted((x0, x1))
        sy0, sy1 = sorted((y0, y1))
        if dx != 0:
            sy0, sy1 = sy0 - strip_half_width, sy1 + strip_half_width
        else:
            sx0, sx1 = sx0 - strip_half_width, sx1 + strip_half_width
        ix0, iy0 = max(0, int(round(sx0))), max(0, int(round(sy0)))
        ix1, iy1 = min(w, int(round(sx1))), min(h, int(round(sy1)))
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        strip = mask[iy0:iy1, ix0:ix1]
        if strip.size == 0:
            continue
        if float(np.count_nonzero(strip)) / strip.size >= config.crosshair_arm_min_fraction:
            count += 1
    return count


_ARM_SEGMENT_MIN_ASPECT = 4.0
"""A candidate lone arm segment (see _find_arm_segments) must be at least
this elongated (long-side / short-side) to plausibly be a thin crosshair
arm rather than a flake fragment or letter-glyph stroke."""

_ARM_SEGMENT_LENGTH_TOLERANCE = (0.8, 3.5)
"""(low, high) multipliers applied to sqrt(this mask's own
_typical_core_area) to decide which elongated blobs are plausibly
crosshair-arm-scaled -- measured directly from a real, confirmed Ta2009
arm segment (length 87px against sqrt(typical_core_area)=48.8px, ratio
1.78), given generous headroom on both sides since only ONE real
measurement was available to calibrate against. Scale-relative, not a
fixed pixel range, per this module's established convention (see
_ARM_CHECK_MARGIN_FRAC's own docstring)."""

_ARM_PAIR_Y_TOLERANCE_FRAC = 0.25
"""Two horizontal (or two vertical) arm segments are only considered the
SAME crosshair's opposing pair if their cross-axis centers differ by less
than this fraction of their own average length -- real opposing arms
measured exactly aligned (0px apart) on Ta2009; this leaves headroom for
minor rendering noise while staying far tighter than the gap needed to
reject two coincidentally-similar unrelated segments elsewhere in an
image."""

_ARM_PAIR_LENGTH_RATIO_MAX = 2.0
"""Two segments can only pair up if their own lengths are within this
ratio of each other -- a real crosshair's opposing arms are the same
length; this rejects pairing e.g. a real arm with an unrelated, merely
similarly-oriented flake edge of very different length."""

_ARM_PAIR_GAP_TOLERANCE = (0.3, 3.0)
"""(low, high) multipliers applied to sqrt(typical_core_area) bounding the
gap between two candidate opposing arms' own facing inner edges -- where
the core would sit. Without this, two segments can share similar
length/cross-axis alignment by sheer coincidence while sitting nowhere
near each other, producing a bogus derived center in empty space between
them (confirmed as a real false positive this session: an unrelated
distant segment paired against a letter-glyph stroke on Ta2009's
IW.png). Measured directly on a real Ta2009 marker: gap 113px against
sqrt(typical_core_area) 48.8px, ratio 2.31 -- comfortably inside this
range."""

_ARM_PERP_MATCH_TOLERANCE_FRAC = 1.5
"""How close a perpendicular segment's own NEAR edge must be to a pair's
derived center, as a multiple of sqrt(typical_core_area), to corroborate
(raise confidence of, but not required for) that pair -- see
_triangulate_from_arms. NOT the segment's centroid, which sits a whole
arm-length further away: the true gap from a real core to where its arm
begins is the same order of magnitude as the pair's own half-gap
(confirmed directly: 56.5px measured on a real Ta2009 marker against
sqrt(typical_core_area)=48.8px, ratio ~1.16), so this needs real headroom
past 1.0, unlike _ARM_PAIR_Y_TOLERANCE_FRAC's much tighter cross-axis
alignment check."""


def _find_arm_segments(
    mask: np.ndarray, typical_area: float,
) -> tuple[list[tuple[float, float, float, float, float]], list[tuple[float, float, float, float, float]]]:
    """Every thin, elongated non-substrate blob in `mask`, classified by
    orientation -- candidate INDIVIDUAL crosshair arms, found independently
    of any hypothesized center (unlike _arm_evidence_count, which checks
    for arm evidence in a strip AROUND an already-known candidate
    position). Feeds _triangulate_from_arms, which recovers a crosshair's
    true center directly from 2+ surviving arms' own geometry -- needed
    for a real, confirmed Ta2009 failure mode: several markers are missing
    their core square AND one arm in the SOURCE PHOTO itself (confirmed by
    inspecting the raw, pre-reconstruction image directly -- not a masking
    or reconstruction artifact), so neither _find_whole_template_candidates
    nor _find_core_blobs/_arm_evidence_count can ever find them (the
    former has no core to correlate against; the latter has no core blob
    to start from at all). See [[project_marker_detection_redesign]]
    memory notes for the EU/AW/HX/ON case history that motivated this.

    Returns (horizontal, vertical) -- each a list of (cx, cy, near_edge,
    far_edge, length) along that segment's own long axis (near/far_edge in
    x for horizontal, y for vertical)."""
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    lo, hi = _ARM_SEGMENT_LENGTH_TOLERANCE
    length_lo, length_hi = math.sqrt(typical_area) * lo, math.sqrt(typical_area) * hi

    horizontal: list[tuple[float, float, float, float, float]] = []
    vertical: list[tuple[float, float, float, float, float]] = []
    for i in range(1, n_labels):
        x, y, w, h, _area = stats[i]
        if w == 0 or h == 0:
            continue
        length = max(w, h)
        if not (length_lo <= length <= length_hi):
            continue
        if max(w, h) / min(w, h) < _ARM_SEGMENT_MIN_ASPECT:
            continue
        cx, cy = centroids[i]
        if w >= h:
            horizontal.append((float(cx), float(cy), float(x), float(x + w), float(length)))
        else:
            vertical.append((float(cx), float(cy), float(y), float(y + h), float(length)))
    return horizontal, vertical


def _triangulate_from_arms(mask: np.ndarray, config: CifExportConfig) -> list[tuple[float, float, float]]:
    """Recover a crosshair's true center directly from 2-3 of its own
    surviving arms' geometry, with no dependence on a core blob or a
    matchable whole-shape template correlation existing at all -- see
    _find_arm_segments for why this is needed.

    A single opposing pair (two horizontal arms, or two vertical arms)
    ALREADY fully determines the 2D center on its own: the cross-axis
    coordinate is their shared alignment (average of their own centers),
    the along-axis coordinate is the midpoint of the gap between their
    facing inner edges (where the core sits). A lone perpendicular segment
    near that derived point corroborates it (raises the score from 2-arm
    to 3-arm confidence) but isn't required.

    Deliberately conservative about which pairs are even considered (see
    _ARM_PAIR_Y_TOLERANCE_FRAC / _ARM_PAIR_LENGTH_RATIO_MAX) -- this
    strategy has no core/whole-shape corroboration of its own to fall back
    on, so false pairings (two unrelated similarly-oriented segments
    elsewhere in the image) would otherwise be a real risk."""
    typical_area = _typical_core_area(mask, config)
    if typical_area is None:
        return []
    horizontal, vertical = _find_arm_segments(mask, typical_area)
    align_tolerance = math.sqrt(typical_area) * _ARM_PAIR_Y_TOLERANCE_FRAC
    edge_tolerance = math.sqrt(typical_area) * _ARM_PERP_MATCH_TOLERANCE_FRAC
    gap_lo, gap_hi = math.sqrt(typical_area) * _ARM_PAIR_GAP_TOLERANCE[0], math.sqrt(typical_area) * _ARM_PAIR_GAP_TOLERANCE[1]

    candidates: list[tuple[float, float, float]] = []
    for same_axis, perp_axis, axis_is_horizontal in ((horizontal, vertical, True), (vertical, horizontal, False)):
        for i in range(len(same_axis)):
            for j in range(i + 1, len(same_axis)):
                cx1, cy1, near1, far1, len1 = same_axis[i]
                cx2, cy2, near2, far2, len2 = same_axis[j]
                length_ratio = max(len1, len2) / min(len1, len2)
                if length_ratio > _ARM_PAIR_LENGTH_RATIO_MAX:
                    continue
                cross1, cross2 = (cy1, cy2) if axis_is_horizontal else (cx1, cx2)
                if abs(cross1 - cross2) > _ARM_PAIR_Y_TOLERANCE_FRAC * ((len1 + len2) / 2):
                    continue
                # Order the pair so `far1 <= near2` (segment 1 is the
                # near/left-or-top one) -- the gap between them is where
                # the core sits.
                if near1 > near2:
                    near1, far1, near2, far2 = near2, far2, near1, far1
                # The gap between the two segments' own FACING inner edges
                # must itself be plausibly core-plus-margin-scaled -- two
                # segments can otherwise share similar length/alignment by
                # sheer coincidence while sitting nowhere near each other
                # (confirmed as a real false positive on Ta2009's IW.png:
                # an unrelated distant segment paired with a letter-glyph
                # stroke, landing the derived "center" on neither).
                gap = near2 - far1
                if not (gap_lo <= gap <= gap_hi):
                    continue
                gap_mid = (far1 + near2) / 2.0
                cross_mid = (cross1 + cross2) / 2.0
                cx, cy = (gap_mid, cross_mid) if axis_is_horizontal else (cross_mid, gap_mid)

                # A perpendicular segment corroborates this pair's derived
                # center if it's ALIGNED with it (its own cross-axis center
                # close to (cx, cy)) AND its NEAR edge -- not its centroid,
                # which sits a whole arm-length away -- is close to it (the
                # gap between a real core and where its arm begins is the
                # same order of magnitude as this pair's own gap_mid
                # margin, confirmed directly: 56.5px measured vs the pair's
                # own 56.5px half-gap on a real Ta2009 marker).
                arm_count = 2
                for pcx, pcy, pnear, pfar, _plen in perp_axis:
                    if axis_is_horizontal:
                        aligned = abs(pcx - cx) <= align_tolerance
                        near_edge_dist = min(abs(pnear - cy), abs(pfar - cy))
                    else:
                        aligned = abs(pcy - cy) <= align_tolerance
                        near_edge_dist = min(abs(pnear - cx), abs(pfar - cx))
                    if aligned and near_edge_dist <= edge_tolerance:
                        arm_count = 3
                        break

                if arm_count < config.crosshair_arm_min_count:
                    continue
                candidates.append((cx, cy, 4.0 + arm_count / 4.0))
    return candidates


def _find_whole_template_candidates(
    search_mask: np.ndarray,
    config: CifExportConfig,
) -> list[tuple[float, float, float]]:
    """The ORIGINAL whole-shape approach this module used before the
    core+arm-evidence design above: correlate the FULL crosshair template
    (core + all 4 arms together) against the mask in one shot, NMS the
    peaks above config.crosshair_whole_match_threshold. Kept as a
    complementary candidate source (see _find_candidates), not replaced
    outright -- confirmed as a real, necessary fallback: the core+arm
    design alone, even after being made scale-adaptive, still missed
    plenty of real markers on flakes/all that this whole-shape match finds
    cleanly (whenever a marker's shape happens to be fully intact, which
    is the common case -- the two-stage core+arm design exists
    specifically for the LESS common case where one arm has broken from
    color variance, see this module's own docstring)."""
    template = _load_template()
    result = cv2.matchTemplate(search_mask.astype(np.float32), template, cv2.TM_CCOEFF_NORMED)
    core_dx, core_dy = _template_core_offset()

    ys, xs = np.where(result >= config.crosshair_whole_match_threshold)
    points = sorted(zip(xs, ys, result[ys, xs]), key=lambda p: -p[2])

    nms_radius_sq = config.crosshair_nms_radius_px ** 2
    kept: list[tuple[float, float, float]] = []
    for x, y, score in points:
        cx, cy = x + core_dx, y + core_dy
        if all((cx - kx) ** 2 + (cy - ky) ** 2 > nms_radius_sq for kx, ky, _ in kept):
            kept.append((float(cx), float(cy), float(score)))
    return kept


def _find_candidates(
    masks: list[np.ndarray],
    config: CifExportConfig,
) -> list[tuple[float, float, float, str]]:
    """All three matching strategies, run over EVERY mask in `masks`, all
    merged together: (1) the original whole-shape template match
    (_find_whole_template_candidates) -- fast, reliable whenever a marker's
    full cross is intact, which is the common case; (2) core-blob +
    arm-evidence (_find_core_blobs + _arm_evidence_count) -- tolerant of any
    ONE arm being broken, the less common case (1) alone misses; (3) direct
    arm triangulation (_triangulate_from_arms, gated behind
    config.crosshair_use_arm_triangulation) -- tolerant of the core AND
    one arm being missing, needed for markers with no matchable whole shape
    and no core blob to start an arm-evidence check from at all (a real,
    confirmed case on Ta2009 -- see that function's own docstring). Callers
    pass multiple masks (see find_top_markers: the substrate-HSV-range
    exclusion mask AND the per-image Otsu V-channel split, _otsu_v_mask) so
    the same marker has multiple independent chances to be found -- the two
    masks fail on different images for different reasons (one is
    hue/saturation-based and can lose an arm whose color drifts into
    substrate range; the other is brightness-based and has no color
    dependence at all, but can fail on low-contrast images), per the "more
    the merrier, combine all of them" design.

    Both scored onto the SAME [4.0, 5.0] range (whole-template: 4.0 +
    correlation; arm-evidence: 4.0 + arm_count/4) so the two compete
    fairly rather than one structurally always beating the other --
    confirmed as a real, necessary fix: a bare 0-4 arm-evidence score
    meant even the WEAKEST admissible whole-template hit (barely over
    crosshair_whole_match_threshold) always outranked even a PERFECT 4/4
    arm-evidence match, which let a real whole-template false positive (an
    unrelated flake blob, no crosshair structure at all) beat out a real
    marker's own confident arm-evidence candidate on a real dataset
    (Ta2009). Every mask x strategy combination's results are pooled into one list
    and NMS'd together, so the same physical marker found by several of
    them collapses to one entry (keeping whichever version was scored
    highest) -- a marker found by more sources isn't scored any higher for
    it, since NMS only keeps the single best-scoring hit per location, but
    it does mean a marker only ONE source manages to find is still found.
    Returns (x, y, score, source) in mask coordinates, (x, y) already each
    candidate's own true core center -- this IS the position later used as
    the calibration anchor (after _refine_marker_center's own further
    precision pass -- SKIPPED for 'tri'-sourced candidates, see
    find_top_markers), per the requirement that alignment is always
    anchored to the middle of the square inside the cross, never any other
    point on the marker. `source` is one of 'whole' / 'arm' / 'tri',
    identifying which strategy produced that candidate (needed by
    find_top_markers to decide whether refinement is even meaningful for
    it -- a 'tri' candidate exists precisely because no real core is
    present to refine toward)."""
    scored: list[tuple[float, float, float, str]] = []
    for mask in masks:
        scored += [(x, y, 4.0 + score, 'whole') for x, y, score in _find_whole_template_candidates(mask, config)]

        for cx, cy, half_extent in _find_core_blobs(mask, config):
            arm_count = _arm_evidence_count(mask, cx, cy, half_extent, config)
            if arm_count < config.crosshair_arm_min_count:
                continue
            # Rescaled onto the SAME [4.0, 5.0] range whole-template hits
            # use (4/4 arms -> 5.0, matching a perfect correlation; the
            # minimum admissible arm_count -> 4.0), not a bare 0-4 float.
            # A bare 0-4 score meant EVERY whole-template hit -- even one
            # that barely cleared crosshair_whole_match_threshold -- always
            # outranked even a PERFECT 4/4 arm-evidence match, no matter
            # how confident either signal actually was. Confirmed as a
            # real, serious bug on Ta2009: a whole-template match on a
            # plain flake blob (no crosshair structure at all, arm_count=1
            # there) outscored the real marker's own 4/4 arm-evidence
            # candidate, which never even got a chance to compete.
            scored.append((cx, cy, 4.0 + arm_count / 4.0, 'arm'))

        # Third strategy: triangulate directly from 2-3 surviving arms,
        # with no core blob or matchable whole shape needed at all -- see
        # _triangulate_from_arms for the real, confirmed Ta2009 case (a
        # marker missing its core AND one arm in the source photo itself)
        # this exists for. Same [4.0, 5.0] scoring convention as the other
        # two strategies, so it competes fairly in the shared NMS below.
        if config.crosshair_use_arm_triangulation:
            scored += [(x, y, score, 'tri') for x, y, score in _triangulate_from_arms(mask, config)]

    scored.sort(key=lambda p: -p[2])

    nms_radius_sq = config.crosshair_nms_radius_px ** 2
    kept: list[tuple[float, float, float, str]] = []
    for cx, cy, score, source in scored:
        if any((cx - kx) ** 2 + (cy - ky) ** 2 <= nms_radius_sq for kx, ky, _, _ in kept):
            continue
        kept.append((cx, cy, score, source))
    return kept


def _refine_marker_center(
    image_bgr: np.ndarray,
    x: float,
    y: float,
    config: CifExportConfig,
) -> tuple[float, float]:
    """
    Refine a template-matched marker position to the crosshair's actual
    solid-color square core, precisely, per image (rather than relying on a
    single fixed template-wide correction, OR a single manually-measured
    reference color that needs re-measuring whenever a batch's lighting
    shifts -- both tried, see marker_refine_half_px's own docstring for
    why the latter was replaced): within a small window, find the
    config.marker_refine_top_v_colors_n colors with the highest V
    (brightness) actually present, and for EACH, isolate its own
    exact-color blob nearest the window's center. Candidates must clear
    BOTH a minimum fill ratio (marker_refine_min_extent) AND a maximum
    aspect ratio (marker_refine_max_aspect) -- fill ratio alone doesn't
    reject a thin bar (a solid rectangle can have extent even closer to
    1.0 than a real rounded-corner core, since it doesn't penalize
    elongation at all; confirmed as a real bug: a thin fragment of an
    adjacent letter, sharing the core's own exact color, once won
    refinement over the real core this way). Whichever qualifying
    candidate has the highest fill ratio is treated as the core.
    Self-calibrates per marker, no reference color/tolerance config needed
    at all.

    Exact color match, not a tolerance range: this runs on the
    reconstructed image (see channel_selection.py), where the crosshair
    core (and every other real feature) renders as a flat, exact color --
    a channels_model cluster color, not the noisy antialiased blend it was
    in the original photo, so there's no antialiasing gradient to need a
    tolerance for in the first place (confirmed directly: a real window
    around a real core showed only a handful of exact, discrete colors).

    Falls back to the unrefined (x, y) if no plausible square blob is
    found nearby (e.g. that marker's cluster wasn't selected by
    channels_model) -- never guesses.
    """
    cx, cy = int(x), int(y)
    half = config.marker_refine_half_px
    h, w = image_bgr.shape[:2]
    y0, y1 = max(0, cy - half), min(h, cy + half)
    x0, x1 = max(0, cx - half), min(w, cx + half)
    window_bgr = image_bgr[y0:y1, x0:x1]
    win_h, win_w = window_bgr.shape[:2]
    win_cx, win_cy = win_w / 2.0, win_h / 2.0

    pixels = window_bgr.reshape(-1, 3)
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    unique_hsv = cv2.cvtColor(unique_colors.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    top_v_order = np.argsort(-unique_hsv[:, 2])[:config.marker_refine_top_v_colors_n]

    best: tuple[float, float, float] | None = None  # (extent, core_x, core_y)
    for idx in top_v_order:
        color_mask = np.all(window_bgr == unique_colors[idx], axis=-1).astype(np.uint8) * 255
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(color_mask, connectivity=8)
        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if not (config.marker_refine_min_area < area < config.marker_refine_max_area):
                continue
            cw, ch = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            extent = area / (cw * ch) if cw * ch else 0.0
            if extent < config.marker_refine_min_extent:
                continue
            aspect = max(cw, ch) / min(cw, ch) if min(cw, ch) else 999.0
            if aspect > config.marker_refine_max_aspect:
                continue
            bcx, bcy = centroids[i]
            # Must land within the window itself, not just anywhere a
            # same-colored blob happens to appear at the window's edge --
            # a real core candidate should already be roughly centered
            # here, since this window is built around a validated coarse
            # position.
            if math.hypot(bcx - win_cx, bcy - win_cy) >= half:
                continue
            if best is None or extent > best[0]:
                best = (extent, x0 + bcx, y0 + bcy)

    if best is None:
        return (x, y)
    return (best[1], best[2])


def _select_pair(
    candidates: list[tuple[float, float, float, str]],
    config: CifExportConfig,
) -> tuple[tuple[float, float, str], tuple[float, float, str]]:
    """Given a scored candidate list (see _find_candidates), pick the top-2
    and validate the base case holds -- raises MarkerDetectionError if not
    (fewer than 2 confident candidates, an ambiguous ranking, or the top-2
    pair fails a geometric sanity check). Split out of find_top_markers so
    it can be tried independently against each candidate source in turn
    (see find_top_markers) -- pooling every source's candidates into one
    list BEFORE this check runs would let a noisy source's spurious
    near-tied candidates trip the ambiguous-ranking guard on an image
    where a single cleaner source would have passed outright (confirmed
    empirically: the Otsu-V mask's own whole-template match found 2 extra,
    unrelated bright-blob false positives on one TS34001 image that the
    substrate-HSV mask alone never saw, and pooling them in broke that
    image's otherwise-clean, unambiguous HSV-only result)."""
    if len(candidates) < 2:
        raise MarkerDetectionError(
            f"found only {len(candidates)} crosshair marker candidate(s) "
            f"(core-shaped blob with >= {config.crosshair_arm_min_count} arms "
            f"showing evidence)"
        )

    top_two = candidates[:2]
    if len(candidates) > 2:
        gap = top_two[1][2] - candidates[2][2]
        if gap < config.crosshair_score_gap_min:
            raise MarkerDetectionError(
                f"found {len(candidates)} candidates with ambiguous ranking "
                f"(score gap between 2nd and 3rd = {gap:.3f} < "
                f"{config.crosshair_score_gap_min})"
            )

    pts = sorted(top_two, key=lambda p: p[0])
    left_px = (pts[0][0], pts[0][1], pts[0][3])
    right_px = (pts[1][0], pts[1][1], pts[1][3])

    dx = right_px[0] - left_px[0]
    dy = right_px[1] - left_px[1]
    separation = (dx ** 2 + dy ** 2) ** 0.5

    if not (config.marker_min_separation_px <= separation <= config.marker_max_separation_px):
        raise MarkerDetectionError(
            f"marker separation {separation:.1f}px outside plausible range "
            f"[{config.marker_min_separation_px}, {config.marker_max_separation_px}]"
        )

    if dx == 0 or abs(dy) / abs(dx) > config.max_marker_row_skew_ratio:
        raise MarkerDetectionError(
            f"markers not horizontally aligned (dx={dx:.1f}, dy={dy:.1f})"
        )

    return left_px, right_px


def find_top_markers(
    image_bgr: np.ndarray,
    config: CifExportConfig,
    detection: DetectionConfig,
    otsu_mask_dir: Path | None = None,
    image_stem: str | None = None,
) -> MarkerPair:
    """
    Locate the top-2 crosshair calibration markers.

    Tries each ENABLED candidate-generation source in turn -- the
    substrate-HSV exclusion mask (if config.crosshair_use_substrate_mask),
    then the per-image Otsu V-channel mask (if
    config.crosshair_use_otsu_mask), then both pooled together as a last
    resort (only if more than one source is enabled) -- returning the
    first that yields a clean, unambiguous top-2 pair (see _select_pair).
    Tried in this order (not pooled from the start) so a noisy source
    can't contaminate a mask that would have succeeded on its own; the
    pooled attempt exists for the remaining case where a real marker is
    only partially visible through each single mask, needing evidence
    from both to clear the >= 2 threshold at all. Raises
    MarkerDetectionError immediately if BOTH flags are False -- there is
    nothing to try.

    Raises MarkerDetectionError (the LAST attempt's reason) if every
    attempt fails: fewer than 2 confident candidates, an ambiguous ranking,
    or the top-2 pair failing a geometric sanity check, in all attempts.

    otsu_mask_dir/image_stem (same pattern as
    channel_selection.select_and_reconstruct's own debug_dir): a routine
    output of every run, the same category as the reconstructed image
    itself, not an opt-in debug extra -- always saves the Otsu-V mask
    (_otsu_v_mask) to otsu_mask_dir/{image_stem}.png regardless of
    crosshair_use_otsu_mask, a plain black/white image of exactly what
    that candidate source saw (or would see, if disabled), so it's always
    available for inspection without re-running anything. Skipped only
    when otsu_mask_dir is None (e.g. standalone/test callers that don't
    want any file output)."""
    img_h = image_bgr.shape[0]
    search_h = max(_load_template().shape[0], int(img_h * config.marker_search_top_fraction))

    otsu_mask = _otsu_v_mask(image_bgr[:search_h])
    if otsu_mask_dir is not None:
        otsu_mask_dir = Path(otsu_mask_dir)
        otsu_mask_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(otsu_mask_dir / f"{image_stem}.png"), otsu_mask)

    mask_sources: list[np.ndarray] = []
    if config.crosshair_use_substrate_mask:
        mask_sources.append(_non_substrate_mask(image_bgr[:search_h], detection))
    if config.crosshair_use_otsu_mask:
        mask_sources.append(otsu_mask)
    if not mask_sources:
        raise MarkerDetectionError(
            "no candidate mask sources enabled (crosshair_use_substrate_mask "
            "and crosshair_use_otsu_mask both False)"
        )

    attempts = [[m] for m in mask_sources]
    if len(mask_sources) > 1:
        attempts.append(mask_sources)

    left_px = right_px = None
    last_error: MarkerDetectionError | None = None
    for masks in attempts:
        candidates = _find_candidates(masks, config)
        try:
            left_px, right_px = _select_pair(candidates, config)
            break
        except MarkerDetectionError as e:
            last_error = e
    else:
        assert last_error is not None
        raise last_error

    # Validation above uses the robust (if slightly less precise) template-
    # matched positions. Refine to the crosshair's actual core for the final
    # positions actually used for calibration -- falls back to the
    # unrefined position per-marker if no clean core is found nearby.
    # SKIPPED for a 'tri'-sourced position (see _triangulate_from_arms):
    # that candidate exists precisely because the real core is missing
    # from the source image, not merely hard to isolate -- attempting to
    # refine it anyway searches the same window for SOME OTHER
    # plausible-looking blob (a letter-glyph fragment, most likely) and can
    # walk the correct triangulated position onto a wrong one (confirmed
    # as a real bug this session on Ta2009's IW.png).
    left_x, left_y, left_source = left_px
    right_x, right_y, right_source = right_px
    left_px = (left_x, left_y) if left_source == 'tri' else _refine_marker_center(image_bgr, left_x, left_y, config)
    right_px = (right_x, right_y) if right_source == 'tri' else _refine_marker_center(image_bgr, right_x, right_y, config)

    return MarkerPair(left_px=left_px, right_px=right_px)


def draw_marker_centers(image_bgr: np.ndarray, markers: MarkerPair) -> np.ndarray:
    """A copy of image_bgr with a cross drawn at each of markers' own final
    (refined) positions -- red for left_px (the filename/grid marker),
    green for right_px -- for visually auditing detection/calibration
    accuracy (see CifExportConfig.draw_marker_centers). Purely a debug
    rendering, never fed back into detection or calibration."""
    annotated = image_bgr.copy()
    for pos, color in ((markers.left_px, (0, 0, 255)), (markers.right_px, (0, 255, 0))):
        x, y = int(round(pos[0])), int(round(pos[1]))
        cv2.drawMarker(annotated, (x, y), color, cv2.MARKER_CROSS, 50, 3)
    return annotated
