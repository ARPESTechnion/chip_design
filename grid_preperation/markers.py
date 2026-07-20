"""
Geometric-only detection of the top-2 crosshair calibration markers.

No OCR / letter-reading anywhere: a marker's grid ID is derived from the image
filename and grid adjacency (see calibration.py), never read off the pixels.
This module only locates the two markers' pixel positions and validates that
the "base case" holds (exactly two clean crosshair markers near the top of the
image, roughly horizontally aligned) — anything else raises so the caller can
alert and skip the image rather than guess.

Two-stage detection:
  1. Coarse localization + validation: normalized cross-correlation template
     matching (cv2.matchTemplate) against a fixed reference crop
     (crosshair_template.png, shipped alongside this module), applied to a
     substrate-exclusion mask so matching is robust to per-image brightness/
     color variation. This identifies WHICH two of the many solid grid-dot
     squares in the image are the actual labeled anchor markers (only those
     two have a crosshair drawn on them) and validates the base-case geometry
     (exactly two, plausible separation, horizontally aligned).
  2. Precise position: each coarse candidate is snapped to the centroid of
     its own solid grid-dot square (see _find_squares), not used directly.

     A visual debug tool (debug_markers.py) that overlays both detections
     revealed the crosshair template match alone has a small but SYSTEMATIC
     bias that differs between the left and right marker (measured: left
     offset from its own square ~(+1, +1)px, right offset ~(-4, +3)px, across
     26 sample images) — because calibration.py derives both scale AND
     rotation from the left->right marker vector, this asymmetry doesn't just
     mislocate a marker, it injects a small but consistent false rotation/
     scale into every placement, growing with distance from the left marker.
     The square centroid -- simple color threshold + connected components,
     no template contamination to be biased by -- has no such asymmetry and
     is used as the actual calibration position instead. The crosshair match
     position is now only ever used for coarse localization/validation, never
     for the final returned coordinates.

     If a marker's own square can't be found near its coarse crosshair match
     (measured: 4/28 sample images — AU, AV, AX, QK — each missing a square
     for one or both markers), the image is alerted and skipped rather than
     silently placed using the less precise, potentially biased coarse
     position — consistent with this module's "never guess" rule everywhere
     else.
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


def _find_candidates(
    search_mask: np.ndarray,
    template: np.ndarray,
    config: CifExportConfig,
) -> list[tuple[float, float, float]]:
    """Template-match, return NMS'd (x, y, score) peaks (marker centers, in
    search_mask coordinates) above config.crosshair_match_threshold, sorted by
    score descending."""
    result = cv2.matchTemplate(search_mask.astype(np.float32), template, cv2.TM_CCOEFF_NORMED)
    core_dx, core_dy = _template_core_offset()

    ys, xs = np.where(result >= config.crosshair_match_threshold)
    points = sorted(zip(xs, ys, result[ys, xs]), key=lambda p: -p[2])

    nms_radius_sq = config.crosshair_nms_radius_px ** 2
    kept: list[tuple[float, float, float]] = []
    for x, y, score in points:
        cx, cy = x + core_dx, y + core_dy
        if all((cx - kx) ** 2 + (cy - ky) ** 2 > nms_radius_sq for kx, ky, _ in kept):
            kept.append((float(cx), float(cy), float(score)))
    return kept


def _rgb_range(rgb: tuple[int, int, int], margin: int) -> tuple[np.ndarray, np.ndarray]:
    """+/- margin box around an (R, G, B) target, returned as (lo, hi) in
    BGR order (cv2.inRange's pixel order)."""
    r, g, b = rgb
    lo = np.array([max(0, b - margin), max(0, g - margin), max(0, r - margin)], dtype=np.uint8)
    hi = np.array([min(255, b + margin), min(255, g + margin), min(255, r + margin)], dtype=np.uint8)
    return lo, hi


def _find_squares(
    image_bgr: np.ndarray,
    config: CifExportConfig,
) -> list[tuple[float, float, float]]:
    """
    Centers of every solid pink grid-dot square in the image (marker
    anchors AND plain unlabeled grid dots alike -- caller matches to the
    nearest one).

    Two-pass color match against config.square_marker_rgb, NOT flake_
    detection's much broader marker_rgb_range -- that range is wide enough
    to also catch part of the pale cyan/blue halo where each marker's
    bright color blends into the substrate, unevenly across a square's
    edges, which skews the centroid (see config.py):

      1. Coarse pass (square_color_margin): isolates each candidate blob
         (square or letter) from its neighbors via connected components,
         after eroding away thin fragments (crosshair arms, same trick as
         _template_core_offset). This margin has to be loose enough to
         admit each marker's real per-image core-color variation (the B
         channel especially) without being so loose that neighboring
         glyphs merge into one oversized blob -- precision isn't the goal
         here, isolating the right blob is. Letters happen to share the
         same ink color and mostly survive too, but are much larger/less
         square, so an area + aspect-ratio filter excludes them.
      2. Fine pass (square_core_margin, tighter): once a blob is isolated
         and passes the area/aspect filter, its centroid is recomputed
         using only the pixels within its own bounding box that tightly
         match square_marker_rgb -- excluding the halo. This can safely be
         much tighter than the coarse pass with no merge risk, since it
         only ever looks inside one already-isolated blob's small box.
         Falls back to the coarse centroid if nothing in the box matches
         that tightly.

    Returns (x, y, area_px) for each surviving blob, area from the COARSE
    pass (small, post-erosion; only useful for relative filtering, not
    real square size).
    """
    img_h, img_w = image_bgr.shape[:2]
    coarse_lo, coarse_hi = _rgb_range(config.square_marker_rgb, config.square_color_margin)
    mask = cv2.inRange(image_bgr, coarse_lo, coarse_hi)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.square_erode_kernel_px,) * 2)
    eroded = cv2.erode(mask, kernel)

    n_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=8)
    fine_lo, fine_hi = _rgb_range(config.square_marker_rgb, config.square_core_margin)
    squares = []
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (config.square_min_area <= area <= config.square_max_area):
            continue
        x0, y0, w, h = (
            stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
            stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT],
        )
        aspect = w / h if h else 0
        if not (config.square_min_aspect <= aspect <= config.square_max_aspect):
            continue

        pad = 4
        y0p, y1p = max(0, y0 - pad), min(img_h, y0 + h + pad)
        x0p, x1p = max(0, x0 - pad), min(img_w, x0 + w + pad)
        fine_mask = cv2.inRange(image_bgr[y0p:y1p, x0p:x1p], fine_lo, fine_hi)
        m = cv2.moments(fine_mask, binaryImage=True)
        if m['m00'] > 0:
            cx, cy = x0p + m['m10'] / m['m00'], y0p + m['m01'] / m['m00']
        else:
            cx, cy = centroids[i]

        squares.append((float(cx), float(cy), float(area)))
    return squares


def _snap_to_square(
    squares: list[tuple[float, float, float]],
    x: float,
    y: float,
    max_distance_px: float,
) -> tuple[float, float] | None:
    """
    Replace a coarse template-matched position with the centroid of its own
    grid-dot square -- see the module docstring for why (the crosshair match
    alone carries a small systematic left/right bias; the square doesn't).

    Returns None if no square is found within max_distance_px -- the caller
    treats this as a failed detection (alert + skip), not a silent fallback
    to the less precise, potentially-biased crosshair-match position. A
    marker we can't verify against its own square is one we shouldn't place
    with at all.
    """
    if not squares:
        return None
    sx, sy, _area = min(squares, key=lambda s: (s[0] - x) ** 2 + (s[1] - y) ** 2)
    if (sx - x) ** 2 + (sy - y) ** 2 > max_distance_px ** 2:
        return None
    return (sx, sy)


def find_top_markers(
    image_bgr: np.ndarray,
    config: CifExportConfig,
    detection: DetectionConfig,
) -> MarkerPair:
    """
    Locate the top-2 crosshair calibration markers.

    Raises MarkerDetectionError (human-readable reason) if the base case
    doesn't hold: fewer than 2 confident candidates, an ambiguous ranking, or
    the top-2 pair fails a geometric sanity check.
    """
    template = _load_template()
    img_h = image_bgr.shape[0]
    search_h = max(template.shape[0], int(img_h * config.marker_search_top_fraction))

    mask = _non_substrate_mask(image_bgr[:search_h], detection)
    candidates = _find_candidates(mask, template, config)

    if len(candidates) < 2:
        raise MarkerDetectionError(
            f"found only {len(candidates)} crosshair marker candidate(s) above "
            f"match threshold {config.crosshair_match_threshold} in top "
            f"{config.marker_search_top_fraction:.0%} of image"
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

    # Validation above uses the robust (if slightly less precise) template-
    # matched positions to confirm which two squares are the labeled anchor
    # markers. The actual calibration position is each one's own square
    # centroid instead (see module docstring for why) -- if either marker's
    # square can't be verified nearby, alert rather than placing with an
    # unverified (and potentially biased) coarse position.
    squares = _find_squares(image_bgr[:search_h], config)
    snapped_left = _snap_to_square(squares, *left_px, config.marker_square_snap_max_distance_px)
    if snapped_left is None:
        raise MarkerDetectionError(
            f"left marker crosshair matched at ({left_px[0]:.1f}, {left_px[1]:.1f}) but no "
            f"grid-dot square found within {config.marker_square_snap_max_distance_px}px"
        )
    snapped_right = _snap_to_square(squares, *right_px, config.marker_square_snap_max_distance_px)
    if snapped_right is None:
        raise MarkerDetectionError(
            f"right marker crosshair matched at ({right_px[0]:.1f}, {right_px[1]:.1f}) but no "
            f"grid-dot square found within {config.marker_square_snap_max_distance_px}px"
        )

    return MarkerPair(left_px=snapped_left, right_px=snapped_right)
