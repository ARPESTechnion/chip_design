"""
Flake detection pipeline for superconductor chip characterization.

Pipeline overview (one image at a time):
  1. Preprocess      — Gaussian blur to suppress sensor noise
  2. Candidate mask  — substrate-exclusion mask (opening only, NO closing):
                       every pixel outside the substrate HSV range is a flake candidate.
                       Opening removes noise; closing is intentionally omitted so nearby
                       differently-colored objects (orange flake + green debris) are never
                       merged. This mask is used for both contour detection and isolation.
  3. Contours        — find connected non-substrate regions; filter by area.
                       Detecting by substrate-border (not by color) means a flake with
                       internal color gradients is still found as one contour.
  4. Color fraction  — fraction of each contour's pixels that match the target orange color.
                       Hard-filtered by min_color_fraction: green blobs (≈0 orange) are
                       dropped here; gradient flakes with ≥35% orange pass through.
  5. Texture         — Laplacian variance + intensity std inside the eroded contour interior
  6. Isolation       — non-substrate pixel density in neighborhood (same candidate mask);
                       counts ALL debris — orange, green, and markers alike
  7. Score           — weighted combination; filter by min_final_score and rank

Output per run:
  output_folder/annotated/   — original images with detection overlays
  output_folder/results.json — per-image, per-flake metrics for threshold tuning
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import DetectionConfig


SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


# --------------------------------------------------------------------------- #
# Result types                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class FlakeMetrics:
    """All measurements for a single detected flake candidate."""
    contour_index: int
    area_px: float
    bbox: tuple                  # (x, y, w, h)
    is_large: bool               # True when area_px >= config.large_flake_alert_area

    # Color relevance (Stage 4)
    color_fraction: float        # Fraction of flake pixels matching target color [0-1]
    color_score: float           # = color_fraction, used directly in final score

    # Texture (Stage 5) — raw values useful for threshold tuning
    laplacian_var: float         # Internal edge density; lower = fewer cracks
    intensity_std: float         # Internal color spread; lower = cleaner surface
    texture_score: float         # Normalized [0-1]; higher = better quality

    # Isolation (Stage 6) — counts all non-substrate material in neighborhood
    neighbor_coverage: float     # Fraction of neighborhood covered by any material
    isolation_score: float       # Normalized [0-1]; higher = more isolated

    # Combined (Stage 7)
    final_score: float


@dataclass
class DarkSpotInfo:
    """A contiguous region significantly darker than dark_spot_brightness_max."""
    bbox: tuple
    area_px: float
    mean_brightness: float


@dataclass
class ImageResult:
    """Detection output for a single image."""
    image_path: str
    num_candidates: int          # Contours passing the size filter
    flakes: list                 # FlakeMetrics instances above min_final_score
    best_score: float
    dark_spots: list             # DarkSpotInfo instances


# --------------------------------------------------------------------------- #
# Helpers — color and substrate masks                                          #
# --------------------------------------------------------------------------- #

def _build_target_color_mask(image_bgr: np.ndarray, config: DetectionConfig) -> np.ndarray:
    """
    Build a mask of pixels matching the desired flake color (e.g. orange).
    Used for color_fraction scoring (Stage 4). Not used for contour detection.
    """
    if config.color_mode == 'rgb':
        r = config.rgb_range
        lower = np.array([r.b_min, r.g_min, r.r_min], dtype=np.uint8)
        upper = np.array([r.b_max, r.g_max, r.r_max], dtype=np.uint8)
        return cv2.inRange(image_bgr, lower, upper)

    h = config.hsv_range
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([h.h_min, h.s_min, h.v_min], dtype=np.uint8)
    upper = np.array([h.h_max, h.s_max, h.v_max], dtype=np.uint8)
    return cv2.inRange(image_hsv, lower, upper)


# --------------------------------------------------------------------------- #
# Stage 2 — Candidate mask (substrate exclusion, opening only)                #
# --------------------------------------------------------------------------- #

def _build_candidate_mask(image_bgr: np.ndarray, config: DetectionConfig) -> np.ndarray:
    """
    Build a mask of all non-substrate, non-marker pixels (opening only — no closing).

    Pixels matching the substrate HSV range OR the marker RGB range are treated
    as background. The remainder are flake candidates.
    """
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    s = config.substrate_hsv_range
    exclude = cv2.inRange(
        image_hsv,
        np.array([s.h_min, s.s_min, s.v_min], dtype=np.uint8),
        np.array([s.h_max, s.s_max, s.v_max], dtype=np.uint8),
    )
    if config.marker_removal_enabled:
        # Pink dots
        m = config.marker_rgb_range
        pink_mask = cv2.inRange(
            image_bgr,
            np.array([m.b_min, m.g_min, m.r_min], dtype=np.uint8),
            np.array([m.b_max, m.g_max, m.r_max], dtype=np.uint8),
        )
        # White crosshair arms — all channels bright
        t = config.marker_bright_threshold
        bright_mask = cv2.inRange(
            image_bgr,
            np.array([t, t, t], dtype=np.uint8),
            np.array([255, 255, 255], dtype=np.uint8),
        )
        marker_mask = cv2.bitwise_or(pink_mask, bright_mask)
        # Dilate to fill the bright-substrate halo around each marker
        if config.marker_dilation_px > 0:
            k = config.marker_dilation_px * 2 + 1
            dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            marker_mask = cv2.dilate(marker_mask, dil_kernel)
        exclude = cv2.bitwise_or(exclude, marker_mask)
    candidate_mask = cv2.bitwise_not(exclude)
    noise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, noise_kernel)
    return candidate_mask


# --------------------------------------------------------------------------- #
# Stage 4 — Color fraction scoring                                             #
# --------------------------------------------------------------------------- #

def _compute_color_fraction(
    contour_mask: np.ndarray,
    target_color_mask: np.ndarray,
) -> tuple[float, float]:
    """
    Compute the fraction of a candidate flake's pixels that match the target color.

    Returns:
        color_fraction — raw fraction in [0, 1]
        color_score    — equals color_fraction (used directly as a score component)
    """
    total = int(np.sum(contour_mask > 0))
    if total == 0:
        return 0.0, 0.0
    matching = int(np.sum((contour_mask > 0) & (target_color_mask > 0)))
    color_fraction = matching / total
    return color_fraction, color_fraction


# --------------------------------------------------------------------------- #
# Stage 5 — Texture scoring                                                    #
# --------------------------------------------------------------------------- #

def _compute_texture_score(
    image_gray: np.ndarray,
    contour_mask: np.ndarray,
    config: DetectionConfig,
) -> tuple[float, float, float]:
    """
    Measure internal texture quality of a flake.

    The mask is eroded by contour_erosion_px first to exclude the boundary
    (which always has high contrast against the substrate regardless of quality).

    Returns:
        laplacian_var  — internal edge density; lower = fewer cracks
        intensity_std  — pixel intensity spread; lower = cleaner, more uniform
        texture_score  — combined score in [0, 1]; 1 = perfectly clean
    """
    if config.contour_erosion_px > 0:
        kernel_size = config.contour_erosion_px * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        interior_mask = cv2.erode(contour_mask, kernel, iterations=1)
    else:
        interior_mask = contour_mask

    pixel_indices = interior_mask > 0
    if int(np.sum(pixel_indices)) < 10:
        return 0.0, 0.0, 0.0

    laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
    laplacian_var = float(np.var(laplacian[pixel_indices]))
    intensity_std = float(np.std(image_gray[pixel_indices].astype(np.float64)))

    lap_score = max(0.0, 1.0 - laplacian_var / config.max_laplacian_var)
    std_score = max(0.0, 1.0 - intensity_std / config.max_intensity_std)

    texture_score = config.laplacian_weight * lap_score + config.std_weight * std_score
    return laplacian_var, intensity_std, texture_score


# --------------------------------------------------------------------------- #
# Stage 6 — Isolation scoring                                                  #
# --------------------------------------------------------------------------- #

def _compute_isolation_score(
    candidate_mask: np.ndarray,
    contour_mask: np.ndarray,
    bbox: tuple,
    config: DetectionConfig,
) -> tuple[float, float]:
    """
    Measure how isolated a flake is from surrounding material.

    Uses the full candidate mask (all non-substrate pixels), so green/teal
    debris and other flakes all count — not just same-color neighbors.

    Returns:
        neighbor_coverage — fraction of neighborhood covered by other material
        isolation_score   — normalized [0, 1]; 1 = completely isolated
    """
    x, y, w, h = bbox
    img_h, img_w = candidate_mask.shape

    margin_x = int(w * (config.neighborhood_factor - 1) / 2)
    margin_y = int(h * (config.neighborhood_factor - 1) / 2)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(img_w, x + w + margin_x)
    y2 = min(img_h, y + h + margin_y)

    neighborhood_area = (x2 - x1) * (y2 - y1)
    if neighborhood_area == 0:
        return 1.0, 0.0

    neighborhood = candidate_mask[y1:y2, x1:x2].copy()
    neighborhood[contour_mask[y1:y2, x1:x2] > 0] = 0   # exclude the flake itself
    neighbor_pixels = int(np.sum(neighborhood > 0))

    neighbor_coverage = neighbor_pixels / neighborhood_area
    isolation_score = max(0.0, 1.0 - neighbor_coverage / config.max_neighbor_coverage)

    return neighbor_coverage, isolation_score


# --------------------------------------------------------------------------- #
# Dark spot detection                                                          #
# --------------------------------------------------------------------------- #

def _detect_dark_spots(
    image_gray: np.ndarray,
    config: DetectionConfig,
) -> list[DarkSpotInfo]:
    """Detect contiguous regions below dark_spot_brightness_max."""
    if not config.dark_spot_enabled:
        return []

    _, dark_mask = cv2.threshold(
        image_gray, config.dark_spot_brightness_max, 255, cv2.THRESH_BINARY_INV
    )
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    spots: list[DarkSpotInfo] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not (config.dark_spot_min_area <= area <= config.dark_spot_max_area):
            continue
        spot_mask = np.zeros(image_gray.shape, dtype=np.uint8)
        cv2.drawContours(spot_mask, [contour], -1, 255, thickness=cv2.FILLED)
        spots.append(DarkSpotInfo(
            bbox=cv2.boundingRect(contour),
            area_px=area,
            mean_brightness=float(np.mean(image_gray[spot_mask > 0])),
        ))

    return spots


# --------------------------------------------------------------------------- #
# Annotation                                                                   #
# --------------------------------------------------------------------------- #

# Hierarchical stage colors (BGR).  Each contour is drawn in the color of the
# first stage it failed, so you can tell at a glance where the pipeline stops.
_STAGE_COLORS = {
    'size':      (50,  50,  200),   # red      — failed area filter
    'color':     (0,   140, 255),   # orange   — wrong color (low color_fraction)
    'texture':   (0,   230, 230),   # yellow   — texture is the weakest sub-score
    'isolation': (200, 0,   200),   # purple   — isolation is the weakest sub-score
    'passed':    (0,   210, 0),     # green    — passed all stages
}
_STAGE_THICKNESS = {'size': 1, 'color': 1, 'texture': 1, 'isolation': 1, 'passed': 2}


def _annotate_image(
    image_bgr: np.ndarray,
    all_contours: list,
    contour_stages: dict[int, tuple[str, str]],
    dark_spots: list[DarkSpotInfo],
) -> np.ndarray:
    """
    Draw hierarchical stage colors on every contour that was evaluated.

    Color  Stage failed          Label shows
    -----  --------------------  ----------------------------
    Red    area filter           a:<area>
    Orange color fraction        cf:<value>
    Yellow texture (weakest)     t:<texture_score> cf:<cf>
    Purple isolation (weakest)   i:<isolation_score> cf:<cf>
    Green  passed everything     <final_score> cf:<cf>

    Magenta rectangle = dark spot alert.
    """
    annotated = image_bgr.copy()

    for i, contour in enumerate(all_contours):
        if i not in contour_stages:
            continue
        stage, label = contour_stages[i]
        color = _STAGE_COLORS[stage]
        thickness = _STAGE_THICKNESS[stage]
        cv2.drawContours(annotated, [contour], -1, color, thickness)
        if label:
            x, y, _, _ = cv2.boundingRect(contour)
            font_scale = 0.45 if stage == 'passed' else 0.35
            cv2.putText(
                annotated, label, (x, max(y - 4, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA,
            )

    for spot in dark_spots:
        x, y, w, h = spot.bbox
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(
            annotated, f"dark {spot.mean_brightness:.0f}",
            (x, max(y - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA,
        )

    return annotated


# --------------------------------------------------------------------------- #
# Per-image processing                                                         #
# --------------------------------------------------------------------------- #

def _process_image(
    img_path: Path,
    config: DetectionConfig,
    annotated_dir: Path,
) -> Optional[ImageResult]:
    """Run the full pipeline on a single image. Returns None if unreadable."""
    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        print(f"Warning: could not read {img_path.name} — skipping.")
        return None

    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Stage 1: Preprocess
    blurred = cv2.GaussianBlur(
        image_bgr, (config.blur_kernel_size, config.blur_kernel_size), 0
    )

    # Stage 2: Candidate mask — all non-substrate material (opening only, no closing).
    # Used for both contour detection and isolation scoring.
    # No closing here: closing would merge nearby differently-colored objects
    # (e.g. an orange flake adjacent to a green blob), producing mixed-color
    # candidates that corrupt both the color_fraction and isolation scores.
    # The color_fraction filter handles non-orange blobs naturally: green/blue
    # regions score near 0 and are dropped before any scoring takes place.
    candidate_mask = _build_candidate_mask(blurred, config)

    # Stage 3: Raw orange color mask — used for color_fraction scoring only.
    target_color_mask = _build_target_color_mask(blurred, config)

    # Find contours in the substrate-exclusion mask.
    # This detects the physical extent of each flake (including color-gradient
    # interiors) rather than just the orange pixels, so a flake that transitions
    # from orange to gold to green is still found as one connected region.
    contours, _ = cv2.findContours(
        candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # contour_stages maps each contour index to (stage, label).
    # Only contours above _MIN_DISPLAY_AREA are evaluated and annotated.
    _MIN_DISPLAY_AREA = 200
    contour_stages: dict[int, tuple[str, str]] = {}
    qualifying: list[FlakeMetrics] = []
    num_size_candidates = 0

    for i, contour in enumerate(contours):
        area = float(cv2.contourArea(contour))
        if area < _MIN_DISPLAY_AREA:
            continue

        bbox = cv2.boundingRect(contour)

        # Stage 3: size filter
        if not (config.min_flake_area <= area <= config.max_flake_area):
            contour_stages[i] = ('size', f"a:{area:.0f}")
            continue

        num_size_candidates += 1
        contour_mask = np.zeros(candidate_mask.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Stage 4: Color fraction
        color_fraction, color_score = _compute_color_fraction(contour_mask, target_color_mask)
        if color_fraction < config.min_color_fraction:
            contour_stages[i] = ('color', f"cf:{color_fraction:.2f}")
            continue

        # Stage 5: Texture
        laplacian_var, intensity_std, texture_score = _compute_texture_score(
            image_gray, contour_mask, config
        )

        # Stage 6: Isolation
        neighbor_coverage, isolation_score = _compute_isolation_score(
            candidate_mask, contour_mask, bbox, config
        )

        # Stage 7: Combined score
        final_score = (
            config.color_fraction_weight * color_score
            + config.texture_score_weight * texture_score
            + config.isolation_score_weight * isolation_score
        )

        is_large = config.large_flake_alert_area > 0 and area >= config.large_flake_alert_area

        if final_score >= config.min_final_score:
            contour_stages[i] = (
                'passed',
                f"{'L ' if is_large else ''}{final_score:.2f} cf:{color_fraction:.2f}",
            )
            qualifying.append(FlakeMetrics(
                contour_index=i,
                area_px=area,
                bbox=bbox,
                is_large=is_large,
                color_fraction=color_fraction,
                color_score=color_score,
                laplacian_var=laplacian_var,
                intensity_std=intensity_std,
                texture_score=texture_score,
                neighbor_coverage=neighbor_coverage,
                isolation_score=isolation_score,
                final_score=final_score,
            ))
        else:
            # Attribute failure to the weakest weighted sub-score
            if texture_score <= isolation_score:
                contour_stages[i] = ('texture', f"t:{texture_score:.2f} cf:{color_fraction:.2f}")
            else:
                contour_stages[i] = ('isolation', f"i:{isolation_score:.2f} cf:{color_fraction:.2f}")

    qualifying.sort(key=lambda f: f.final_score, reverse=True)
    best_score = qualifying[0].final_score if qualifying else 0.0

    dark_spots = _detect_dark_spots(image_gray, config)

    if contour_stages or dark_spots:
        annotated = _annotate_image(image_bgr, contours, contour_stages, dark_spots)
        cv2.imwrite(str(annotated_dir / img_path.name), annotated)

    return ImageResult(
        image_path=str(img_path),
        num_candidates=num_size_candidates,
        flakes=qualifying,
        best_score=best_score,
        dark_spots=dark_spots,
    )


# --------------------------------------------------------------------------- #
# JSON output                                                                  #
# --------------------------------------------------------------------------- #

def _write_results_json(results: list[ImageResult], path: Path) -> None:
    """
    Serialize results to JSON. Raw metric values are included so you can
    inspect real score distributions and tune thresholds.
    """
    data = [
        {
            'image_path': r.image_path,
            'num_candidates': r.num_candidates,
            'best_score': r.best_score,
            'flakes': [asdict(f) for f in r.flakes],
            'dark_spots': [asdict(s) for s in r.dark_spots],
        }
        for r in results
    ]
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


# --------------------------------------------------------------------------- #
# Public entry point                                                           #
# --------------------------------------------------------------------------- #

def detect_flakes(
    input_folder: str | Path,
    config: DetectionConfig,
    output_folder: Optional[str | Path] = None,
) -> list[ImageResult]:
    """
    Run the flake detection pipeline on all images in a folder.

    Args:
        input_folder:  Path to folder containing chip images.
        config:        Detection parameters (see DetectionConfig).
        output_folder: Destination for annotated images and results.json.
                       Defaults to <input_folder>/flake_detection_results/.

    Returns:
        List of ImageResult for images with at least one qualifying flake or
        dark spot, sorted by best flake score descending.
    """
    input_folder = Path(input_folder)
    if output_folder is None:
        output_folder = input_folder / 'flake_detection_results'
    output_folder = Path(output_folder)

    annotated_dir = output_folder / 'annotated'
    annotated_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in input_folder.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not image_paths:
        print(f"No supported images found in {input_folder}")
        return []

    print(f"Processing {len(image_paths)} images...")

    results: list[ImageResult] = []
    for img_path in image_paths:
        result = _process_image(img_path, config, annotated_dir)
        if result is not None and (result.flakes or result.dark_spots):
            results.append(result)
            parts = []
            if result.flakes:
                large = sum(1 for f in result.flakes if f.is_large)
                parts.append(
                    f"{len(result.flakes)} flake(s) (best {result.best_score:.3f})"
                    + (f", {large} large" if large else "")
                )
            if result.dark_spots:
                parts.append(f"{len(result.dark_spots)} dark spot(s)")
            print(f"  {img_path.name}: {', '.join(parts)}")

    results.sort(key=lambda r: r.best_score, reverse=True)

    if config.top_n is not None:
        results = results[:config.top_n]

    _write_results_json(results, output_folder / 'results.json')

    print(f"\nDone. {len(results)} image(s) with alerts.")
    print(f"Annotated images -> {annotated_dir}")
    print(f"Metrics          -> {output_folder / 'results.json'}")

    return results
