"""
Whole-image polygon extraction for CIF export.

Exports EVERY non-background region in the image — flakes, debris, and the
calibration markers/scaling marks themselves — not just orange-flake-colored
regions above a quality bar. Background/foreground is separated by edge
detection (see _edge_foreground_mask), not color: the substrate has a real
lighting vignette/gradient across the frame, which confused every color-based
approach tried before this (a fixed HSV threshold, k-means color clustering
with various background-class rules) into treating part of the substrate as
foreground. Edge detection sidesteps this entirely — a slow spatial gradient
has near-zero *local* gradient, so it's invisible to an edge detector, while
real content (flakes, markers, debris) has sharp edges against the substrate
regardless of the substrate's absolute brightness at that point in the frame.
Canny (not a plain Sobel/gradient-magnitude threshold, which was tried and
picked up substrate texture noise as "edges" almost everywhere) finds edges,
closing bridges small gaps into closed loops, and contours of those closed
loops are filled to get solid regions — those pixels are what gets exported.

Each exported polygon's color is its own measured mean color (cv2.mean over
its contour mask) — there's no cluster centroid to be loyal to anymore now
that backgrounds/foreground isn't decided by color at all.

color_fraction/texture/isolation scoring (same formula as flake_detection,
for comparability) is still computed and attached to each polygon, but only
used as a hard filter via config.min_flake_score, which defaults to 0.0
(nothing excluded).

Deliberately simpler and more precise than GDS_Code.m's approach (posterize
the image -> rotate the raster -> contour it): contours are found directly in
original pixel space and each vertex is transformed to absolute um via the
closed-form calibration transform — no raster rotation/resampling, so no
interpolation artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from flake_detection.config import DetectionConfig
from flake_detection.detector import (
    _build_candidate_mask,
    _build_target_color_mask,
    _compute_color_fraction,
    _compute_isolation_score,
    _compute_texture_score,
)

from .calibration import Calibration
from .config import CifExportConfig

_MAX_SPLIT_DEPTH = 24  # safety guard against runaway recursion on pathological shapes


def _edge_foreground_mask(
    image_bgr: np.ndarray,
    blur_kernel: int,
    canny_low: float,
    canny_high: float,
    close_kernel: int,
    close_iterations: int,
) -> np.ndarray:
    """
    Edge-detect the image (Canny, immune to the substrate's slow lighting
    gradient since that has near-zero local gradient), close small gaps so
    real-content edges form closed loops, then fill the enclosed contours —
    those filled pixels are the foreground mask.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(closed)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    return mask


@dataclass
class Polygon:
    vertices_um: list[tuple[float, float]]  # closed ring, ready for CIF output
    final_score: float  # the source region's quality score (see module docstring)
    mean_rgb: tuple[int, int, int]  # the source region's own measured mean color (R, G, B)


def _split_contour(
    contour: np.ndarray,
    max_vertex_n: int,
    vertical: bool = True,
    _depth: int = 0,
) -> list[np.ndarray]:
    """
    Recursively split a contour so every piece has <= max_vertex_n vertices.

    Mask-based equivalent of GDS_Code.m's Divide_Polygons (which used
    polyshape intersection): rasterize the contour into a local mask, cut it
    with a bisecting rectangle (alternating vertical/horizontal, same
    alternation rule as the legacy tool), re-extract contours from each half,
    and recurse. Avoids a shapely dependency.
    """
    if len(contour) <= max_vertex_n or _depth >= _MAX_SPLIT_DEPTH:
        return [contour]

    x, y, w, h = cv2.boundingRect(contour)
    if w < 2 or h < 2:
        return [contour]

    margin = 1
    mask = np.zeros((h + 2 * margin, w + 2 * margin), dtype=np.uint8)
    shifted = contour.reshape(-1, 2) - [x - margin, y - margin]
    cv2.drawContours(mask, [shifted.astype(np.int32)], -1, 255, thickness=cv2.FILLED)

    half_a = mask.copy()
    half_b = mask.copy()
    if vertical:
        mid = mask.shape[1] // 2
        half_a[:, mid:] = 0
        half_b[:, :mid] = 0
    else:
        mid = mask.shape[0] // 2
        half_a[mid:, :] = 0
        half_b[:mid, :] = 0

    pieces: list[np.ndarray] = []
    for half_mask in (half_a, half_b):
        sub_contours, _ = cv2.findContours(half_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for sub in sub_contours:
            if cv2.contourArea(sub) < 1:
                continue
            sub_shifted = sub.reshape(-1, 2) + [x - margin, y - margin]
            sub_contour = sub_shifted.reshape(-1, 1, 2).astype(np.int32)
            pieces.extend(_split_contour(sub_contour, max_vertex_n, vertical=not vertical, _depth=_depth + 1))

    return pieces


def extract_polygons(
    image_bgr: np.ndarray,
    calibration: Calibration,
    config: CifExportConfig,
    detection: DetectionConfig,
) -> list[Polygon]:
    """
    Extract every non-background region in the whole image scoring at least
    config.min_flake_score (default 0.0 -- everything) as one or more
    CIF-ready polygons (vertex-limited, transformed to absolute um).
    """
    foreground_mask = _edge_foreground_mask(
        image_bgr, config.edge_blur_kernel, config.edge_canny_low, config.edge_canny_high,
        config.edge_close_kernel, config.edge_close_iterations,
    )
    # Still built for isolation scoring (matches flake_detection's own
    # neighbor-coverage semantics) and for target_color_mask below.
    candidate_mask = _build_candidate_mask(image_bgr, detection)
    target_color_mask = _build_target_color_mask(image_bgr, detection)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons: list[Polygon] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < config.gds_min_area:
            continue

        contour_mask = np.zeros(foreground_mask.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)

        color_fraction, color_score = _compute_color_fraction(contour_mask, target_color_mask)
        _, _, texture_score = _compute_texture_score(image_gray, contour_mask, detection)
        bbox = cv2.boundingRect(contour)
        _, isolation_score = _compute_isolation_score(candidate_mask, contour_mask, bbox, detection)

        final_score = (
            detection.color_fraction_weight * color_score
            + detection.texture_score_weight * texture_score
            + detection.isolation_score_weight * isolation_score
        )
        if final_score < config.min_flake_score:
            continue

        mean_bgr = cv2.mean(image_bgr, mask=contour_mask)[:3]
        mean_rgb = (round(mean_bgr[2]), round(mean_bgr[1]), round(mean_bgr[0]))

        for piece in _split_contour(contour, config.max_vertex_n):
            pts_px = piece.reshape(-1, 2)
            if len(pts_px) < 3:
                continue
            vertices_um = [calibration.pixel_to_um(float(px), float(py)) for px, py in pts_px]
            polygons.append(Polygon(vertices_um=vertices_um, final_score=final_score, mean_rgb=mean_rgb))

    return polygons
