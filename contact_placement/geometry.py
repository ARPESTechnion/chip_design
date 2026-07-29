"""
Shared, independently-verifiable geometric primitives for contact placement.

Everything here works in real micron ("um") coordinates -- the same absolute
chip coordinate space grid_preperation's exported CIF vertices already use
(see grid_preperation.calibration.Calibration.pixel_to_um) -- never pixels.

RotatedRect is the shared building block: both largest_inscribed_rectangle
(Hall's "new flake") and blocking_rectangle (Ring's "blocking rectangle")
return one, giving callers a local coordinate frame (aligned to the
rectangle's own orientation, not necessarily the image's) to build further
geometry in via to_local/to_absolute -- mirrors how
grid_preperation.calibration.Calibration.pixel_to_um is the one place pixel
geometry becomes absolute um geometry, so hall.py/ring.py never have to
reason about rotation math themselves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

Point = tuple[float, float]


@dataclass
class RotatedRect:
    """
    A rectangle at an arbitrary rotation, plus the coordinate frame it
    defines: local +x is the rectangle's own "width" axis, local +y is its
    "height" axis, both through `center`, rotated by `angle_rad` from
    absolute +x/+y. width/height are the local x/y span (NOT necessarily
    width <= height -- see with_long_axis_as to normalize when a caller
    needs a specific axis to be the long one, e.g. Hall's "length must be at
    least 2x width").
    """
    center: Point
    angle_rad: float
    width: float
    height: float

    def to_local(self, x: float, y: float) -> Point:
        """Absolute um -> local (rectangle-aligned) um."""
        dx, dy = x - self.center[0], y - self.center[1]
        c, s = math.cos(-self.angle_rad), math.sin(-self.angle_rad)
        return dx * c - dy * s, dx * s + dy * c

    def to_absolute(self, x: float, y: float) -> Point:
        """Local (rectangle-aligned) um -> absolute um."""
        c, s = math.cos(self.angle_rad), math.sin(self.angle_rad)
        return self.center[0] + x * c - y * s, self.center[1] + x * s + y * c

    def corners(self) -> list[Point]:
        """The 4 corners, in absolute um, closed-ring order."""
        hw, hh = self.width / 2, self.height / 2
        return [
            self.to_absolute(-hw, -hh), self.to_absolute(hw, -hh),
            self.to_absolute(hw, hh), self.to_absolute(-hw, hh),
        ]

    def with_long_axis_as(self, axis: str) -> RotatedRect:
        """
        Return an equivalent RotatedRect (same physical rectangle) with the
        LONGER of width/height guaranteed to be the given local axis
        ('x' or 'y') -- rotates the frame 90 degrees if the current
        width/height are the wrong way round. Use this once, right after
        largest_inscribed_rectangle/blocking_rectangle, so all downstream
        code can assume a fixed "which axis is the long one" convention
        instead of branching on it everywhere.
        """
        long_is_y = self.height >= self.width
        wants_y = (axis == 'y')
        if long_is_y == wants_y:
            return self
        return RotatedRect(
            center=self.center, angle_rad=self.angle_rad + math.pi / 2,
            width=self.height, height=self.width,
        )


def _rasterize_local_polygon(
    local_pts: np.ndarray, resolution_um: float, margin_cells: int = 2,
) -> tuple[np.ndarray, float, float]:
    """
    Rasterize a polygon (already in some local coordinate frame, um) onto a
    boolean grid at resolution_um per cell. Returns (mask, x0, y0) where
    mask[row, col] covers local x in [x0+col*res, x0+(col+1)*res), y
    likewise -- i.e. (x0, y0) is the grid origin in the same local frame.
    """
    x0, y0 = float(local_pts[:, 0].min()), float(local_pts[:, 1].min())
    x1, y1 = float(local_pts[:, 0].max()), float(local_pts[:, 1].max())
    cols = int(np.ceil((x1 - x0) / resolution_um)) + 2 * margin_cells
    rows = int(np.ceil((y1 - y0) / resolution_um)) + 2 * margin_cells
    x0 -= margin_cells * resolution_um
    y0 -= margin_cells * resolution_um

    shifted = (local_pts - [x0, y0]) / resolution_um
    mask = np.zeros((rows, cols), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(shifted).astype(np.int32)], 1)
    return mask.astype(bool), x0, y0


def _max_rectangle_in_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    """
    Largest all-True axis-aligned rectangle in a 2D boolean array. Standard
    "maximal rectangle in a binary matrix" algorithm: per row, build a
    histogram of consecutive True-run heights ending at that row, then find
    the largest rectangle in that histogram via a monotonic stack (O(cols)
    amortized per row, O(rows*cols) total).

    Returns (row0, col0, row1, col1) -- half-open bounds, i.e. the rectangle
    covers rows [row0, row1) and cols [col0, col1).
    """
    rows, cols = mask.shape
    heights = np.zeros(cols, dtype=np.int64)
    best_area, best = 0, (0, 0, 0, 0)

    for r in range(rows):
        heights = np.where(mask[r], heights + 1, 0)
        stack: list[tuple[int, int]] = []  # (start_col, height)
        for c in range(cols + 1):
            h = int(heights[c]) if c < cols else 0
            start = c
            while stack and stack[-1][1] >= h:
                idx, height = stack.pop()
                start = idx
                area = height * (c - idx)
                if area > best_area:
                    best_area = area
                    best = (r - height + 1, idx, r + 1, c)
            stack.append((start, h))

    return best


def largest_inscribed_rectangle(vertices_um: list[Point], resolution_um: float = 0.1) -> RotatedRect:
    """
    The largest axis-aligned (in its own rotated frame) rectangle that fits
    entirely inside a polygon -- Hall's "new flake": "Find maximum rectangle
    that fits into the flake's contour."

    Rotation is taken from cv2.minAreaRect's own angle (the polygon's
    natural bounding-box orientation) -- everything else happens entirely in
    that LOCAL frame via a synthetic raster grid at resolution_um, never
    through calibration.pixel_to_um or any other rotation, so this can't
    compound rotations. A 1-cell erosion is applied before the search so the
    result can't poke past the true (rasterization-fuzzy) boundary.
    """
    pts = np.asarray(vertices_um, dtype=np.float64)
    (cx, cy), _, angle_deg = cv2.minAreaRect(pts.astype(np.float32))
    frame = RotatedRect(center=(float(cx), float(cy)), angle_rad=math.radians(float(angle_deg)), width=0, height=0)

    local_pts = np.array([frame.to_local(x, y) for x, y in pts])
    mask, x0, y0 = _rasterize_local_polygon(local_pts, resolution_um)
    mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)

    row0, col0, row1, col1 = _max_rectangle_in_mask(mask)
    if row1 <= row0 or col1 <= col0:
        raise ValueError("largest_inscribed_rectangle: no rectangle found (degenerate/too-small mask)")

    lx0, lx1 = x0 + col0 * resolution_um, x0 + col1 * resolution_um
    ly0, ly1 = y0 + row0 * resolution_um, y0 + row1 * resolution_um
    local_center = ((lx0 + lx1) / 2, (ly0 + ly1) / 2)
    abs_center = frame.to_absolute(*local_center)

    return RotatedRect(
        center=abs_center, angle_rad=frame.angle_rad,
        width=lx1 - lx0, height=ly1 - ly0,
    )


def blocking_rectangle(vertices_um: list[Point]) -> RotatedRect:
    """The smallest rectangle containing the polygon -- Ring's "blocking
    rectangle": "The smallest possible rectangle that fits the flake's
    contour." Just cv2.minAreaRect, wrapped into the shared RotatedRect
    frame convention."""
    pts = np.asarray(vertices_um, dtype=np.float32)
    (cx, cy), (w, h), angle_deg = cv2.minAreaRect(pts)
    return RotatedRect(center=(float(cx), float(cy)), angle_rad=math.radians(float(angle_deg)), width=w, height=h)


def tapered_quad(p0: Point, p1: Point, width0: float, width1: float) -> list[Point]:
    """
    A single straight tapered segment: 4-point polygon running from p0 to
    p1, width0 wide (perpendicular to the p0->p1 direction) at p0, tapering
    linearly to width1 at p1. The one primitive every current/voltage
    contact and every arm segment in this stage is built from.
    """
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy)
    if length == 0:
        raise ValueError("tapered_quad: zero-length segment")
    nx, ny = -dy / length, dx / length  # unit perpendicular

    return [
        (p0[0] + nx * width0 / 2, p0[1] + ny * width0 / 2),
        (p1[0] + nx * width1 / 2, p1[1] + ny * width1 / 2),
        (p1[0] - nx * width1 / 2, p1[1] - ny * width1 / 2),
        (p0[0] - nx * width0 / 2, p0[1] - ny * width0 / 2),
    ]


def tapered_path(waypoints: list[Point], width_start: float, width_end: float) -> ShapelyPolygon:
    """
    A tapered polygon along a multi-segment path (>= 2 waypoints), width
    interpolated linearly by cumulative arc-length from width_start (at
    waypoints[0]) to width_end (at waypoints[-1]) -- used for every arm
    connecting an inner contact to an external pad. Multiple straight
    tapered_quad segments are unioned via shapely so a bent path (mandatory
    per the spec: "Arms should not contain straight angle") comes out as one
    clean polygon rather than several overlapping/misaligned pieces.
    """
    if len(waypoints) < 2:
        raise ValueError("tapered_path: need at least 2 waypoints")

    seg_lengths = [
        math.hypot(waypoints[i + 1][0] - waypoints[i][0], waypoints[i + 1][1] - waypoints[i][1])
        for i in range(len(waypoints) - 1)
    ]
    total = sum(seg_lengths)
    if total == 0:
        raise ValueError("tapered_path: zero-length path")

    widths = [width_start]
    cumulative = 0.0
    for seg_len in seg_lengths:
        cumulative += seg_len
        widths.append(width_start + (width_end - width_start) * (cumulative / total))

    quads = [
        ShapelyPolygon(tapered_quad(waypoints[i], waypoints[i + 1], widths[i], widths[i + 1]))
        for i in range(len(waypoints) - 1)
    ]
    return unary_union(quads)
