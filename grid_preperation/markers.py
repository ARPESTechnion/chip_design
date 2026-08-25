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

from dataclasses import dataclass
from pathlib import Path

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
    s = detection.substrate_hsv_range
    substrate_mask = cv2.inRange(
        image_hsv,
        np.array([s.h_min, s.s_min, s.v_min], dtype=np.uint8),
        np.array([s.h_max, s.s_max, s.v_max], dtype=np.uint8),
    )
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
) -> list[tuple[float, float, float]]:
    """Both matching strategies, run over EVERY mask in `masks`, all merged
    together: (1) the original whole-shape template match
    (_find_whole_template_candidates) -- fast, reliable whenever a marker's
    full cross is intact, which is the common case; (2) core-blob +
    arm-evidence (_find_core_blobs + _arm_evidence_count) -- tolerant of any
    ONE arm being broken, the less common case (1) alone misses. Callers
    pass multiple masks (see find_top_markers: the substrate-HSV-range
    exclusion mask AND the per-image Otsu V-channel split, _otsu_v_mask) so
    the same marker has multiple independent chances to be found -- the two
    masks fail on different images for different reasons (one is
    hue/saturation-based and can lose an arm whose color drifts into
    substrate range; the other is brightness-based and has no color
    dependence at all, but can fail on low-contrast images), per the "more
    the merrier, combine all of them" design.

    A whole-template hit always outranks an arm-evidence-only hit (assigned
    score 4 + correlation, comfortably above arm-evidence's own 0-4 range)
    since a complete shape match is inherently the more confident signal;
    among arm-evidence-only candidates, more arms present ranks higher.
    Every mask x strategy combination's results are pooled into one list
    and NMS'd together, so the same physical marker found by several of
    them collapses to one entry (keeping whichever version was scored
    highest) -- a marker found by more sources isn't scored any higher for
    it, since NMS only keeps the single best-scoring hit per location, but
    it does mean a marker only ONE source manages to find is still found.
    Returns (x, y, score) in mask coordinates, (x, y) already each
    candidate's own true core center -- this IS the position later used as
    the calibration anchor (after _refine_marker_center's own further
    precision pass), per the requirement that alignment is always anchored
    to the middle of the square inside the cross, never any other point on
    the marker."""
    scored: list[tuple[float, float, float]] = []
    for mask in masks:
        scored += [(x, y, 4.0 + score) for x, y, score in _find_whole_template_candidates(mask, config)]

        for cx, cy, half_extent in _find_core_blobs(mask, config):
            arm_count = _arm_evidence_count(mask, cx, cy, half_extent, config)
            if arm_count < config.crosshair_arm_min_count:
                continue
            scored.append((cx, cy, float(arm_count)))

    scored.sort(key=lambda p: -p[2])

    nms_radius_sq = config.crosshair_nms_radius_px ** 2
    kept: list[tuple[float, float, float]] = []
    for cx, cy, score in scored:
        if any((cx - kx) ** 2 + (cy - ky) ** 2 <= nms_radius_sq for kx, ky, _ in kept):
            continue
        kept.append((cx, cy, score))
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
    single fixed template-wide correction): threshold directly for
    config.marker_refine_color_hsv (+-hue/sat tolerance, V unconstrained)
    within a small window and take the centroid of the best matching blob.

    Direct color thresholding, not edge detection: this runs on the
    reconstructed image (see channel_selection.py), where the crosshair core
    is just a channels_model color cluster painted with its own flat, real
    measured color -- no antialiasing/blending to fight, unlike the original
    photo this step was designed against before the pipeline switched to
    detecting markers on the reconstructed image (that used Canny edge
    detection instead; replaced since it's unnecessary now).

    HSV, not RGB: the same real-world crosshair color renders at a
    different RGB per batch/image because of lighting -- confirmed as a
    real problem (a fixed RGB reference worked for one batch, not others).
    Lighting is mostly a brightness/exposure effect, isolated into V by
    HSV, so only H/S are matched -- V is deliberately left unconstrained.
    See marker_refine_color_hsv's own docstring for the measurement this
    batch's reference came from and why hue gets the looser tolerance.

    Falls back to the unrefined (x, y) if no plausible blob is found nearby
    (e.g. that marker's cluster wasn't selected by channels_model) -- never
    guesses.
    """
    cx, cy = int(x), int(y)
    half = config.marker_refine_half_px
    h, w = image_bgr.shape[:2]
    y0, y1 = max(0, cy - half), min(h, cy + half)
    x0, x1 = max(0, cx - half), min(w, cx + half)
    window_bgr = image_bgr[y0:y1, x0:x1]
    window_hsv = cv2.cvtColor(window_bgr, cv2.COLOR_BGR2HSV)

    # config values are all in the COMMON picker scale (H degrees 0-360, S
    # percent 0-100), not OpenCV's own H:0-179/S:0-255 -- convert here so
    # every caller/config-editor can work in the scale a color picker (e.g.
    # Windows Paint) actually shows, never doing this conversion by hand.
    hue_deg, sat_pct = config.marker_refine_color_hsv
    hue = round(hue_deg / 2) % 180
    sat = round(sat_pct / 100 * 255)
    h_tol = round(config.marker_refine_hue_tolerance_deg / 2)
    s_tol = round(config.marker_refine_sat_tolerance_pct / 100 * 255)
    s_lo, s_hi = max(0, sat - s_tol), min(255, sat + s_tol)

    # OpenCV hue is circular, 0-179 -- a tolerance window can wrap past
    # either end (e.g. hue=2, tolerance=12 needs both [170,179] AND
    # [0,14]). Two in-range masks unioned handles both the wrapping and
    # non-wrapping case uniformly. Python's % always returns non-negative
    # for a positive divisor, so h_lo % 180 correctly wraps a negative
    # h_lo (e.g. -10 % 180 == 170); h_hi % 180 correctly wraps h_hi > 179
    # back down (e.g. 187 % 180 == 7).
    h_lo, h_hi = hue - h_tol, hue + h_tol
    if h_lo < 0 or h_hi > 179:
        mask_a = cv2.inRange(window_hsv, (h_lo % 180, s_lo, 0), (179, s_hi, 255))
        mask_b = cv2.inRange(window_hsv, (0, s_lo, 0), (h_hi % 180, s_hi, 255))
        color_mask = cv2.bitwise_or(mask_a, mask_b)
    else:
        color_mask = cv2.inRange(window_hsv, (h_lo, s_lo, 0), (h_hi, s_hi, 255))

    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        if not (config.marker_refine_min_area < area < config.marker_refine_max_area):
            continue
        cw, ch = cv2.boundingRect(c)[2:]
        extent = area / (cw * ch) if cw * ch else 0
        if extent < config.marker_refine_min_extent:
            continue
        m = cv2.moments(c)
        if m['m00'] == 0:
            continue
        core_x, core_y = x0 + m['m10'] / m['m00'], y0 + m['m01'] / m['m00']
        if best is None or area > best[0]:
            best = (area, core_x, core_y)

    if best is None:
        return (x, y)
    return (best[1], best[2])


def _select_pair(
    candidates: list[tuple[float, float, float]],
    config: CifExportConfig,
) -> tuple[tuple[float, float], tuple[float, float]]:
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
    left_px = (pts[0][0], pts[0][1])
    right_px = (pts[1][0], pts[1][1])

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
    left_px = _refine_marker_center(image_bgr, *left_px, config)
    right_px = _refine_marker_center(image_bgr, *right_px, config)

    return MarkerPair(left_px=left_px, right_px=right_px)
