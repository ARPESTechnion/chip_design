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
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

Point = tuple[float, float]

_COLLISION_AREA_EPSILON = 1e-9
"""Shapes in this stage are deliberately built to TOUCH at shared boundaries
(an arm meets its own inner contact exactly at one point/edge, a contact
meets the inner rectangle's edge, etc.) -- that's correct, not a collision.
shapely's .intersects() treats any shared boundary point as True, which
would flag every one of those legitimate joins as a failure. collides()
checks the intersection has real AREA instead, so touching is always fine
and only genuine overlap counts."""


def collides(a: BaseGeometry, b: BaseGeometry) -> bool:
    """True if a and b overlap by more than a floating-point-noise sliver --
    use this everywhere in contact_placement instead of shapely's own
    .intersects()/.overlaps(), which don't distinguish "touching at a shared
    edge" (expected, fine) from "actually overlapping" (a real collision)."""
    if not a.intersects(b):
        return False
    return a.intersection(b).area > _COLLISION_AREA_EPSILON


ChipBounds = tuple[float, float, float, float]
"""(minx, miny, maxx, maxy), absolute um -- the real chip/die extent, used
to reject placements that would land geometry off the physical chip (e.g. a
flake near the edge of its own imaged region getting a work-area-sized
external pad layout that overshoots the die boundary). See
cif_export.chip_bounds_from_cif for how this gets derived in practice."""


def within_bounds(geom: BaseGeometry, bounds: ChipBounds | None) -> bool:
    """True if `geom` is entirely within `bounds` (or bounds is None,
    meaning no constraint is known -- never silently reject in that case)."""
    if bounds is None:
        return True
    minx, miny, maxx, maxy = geom.bounds
    bx0, by0, bx1, by1 = bounds
    return minx >= bx0 and miny >= by0 and maxx <= bx1 and maxy <= by1


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

    def direction_to_absolute(self, dx: float, dy: float) -> Point:
        """Rotate-only (no translation) transform of a LOCAL direction
        vector into absolute space -- use for contact/arm exit directions,
        not positions (use to_absolute for those)."""
        c, s = math.cos(self.angle_rad), math.sin(self.angle_rad)
        return dx * c - dy * s, dx * s + dy * c

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


def edge_to_edge_taper(
    p0: Point, direction0: Point, width0: float,
    p1: Point, direction1: Point, width1: float,
) -> list[Point]:
    """
    A 4-point polygon connecting two EDGES exactly -- one centered at p0,
    perpendicular to direction0, width0 wide; the other centered at p1,
    perpendicular to direction1, width1 wide -- rather than tapered_quad's
    single shared perpendicular (which only matches an edge's true
    orientation when that edge happens to be perpendicular to the p0->p1
    travel direction itself).

    Needed for a direct straight connection between an inner contact and
    its pad: the contact's own true edge is perpendicular to ITS OWN
    direction (e.g. Ring current's length axis), the pad's own true edge is
    perpendicular to the (possibly different) pad direction (e.g. Ring's
    shared width axis) -- and the straight line between them is generally
    at neither angle. Using tapered_quad directly for this would draw both
    end caps perpendicular to that connecting line instead, matching
    NEITHER real edge -- confirmed as the exact defect flagged: "they are
    not fully connected to the internal contacts and the external
    rectangle. they should match exactly." This uses each edge's own real
    normal, so the resulting polygon's start/end edges coincide exactly
    with the contact's and pad's own edges (reduces to tapered_quad's own
    result whenever direction0/direction1 both already equal the p0->p1
    direction, e.g. Hall's straight-out arms).
    """
    n0 = _edge_normal(direction0)
    n1 = _edge_normal(direction1)
    return [
        (p0[0] + n0[0] * width0 / 2, p0[1] + n0[1] * width0 / 2),
        (p1[0] + n1[0] * width1 / 2, p1[1] + n1[1] * width1 / 2),
        (p1[0] - n1[0] * width1 / 2, p1[1] - n1[1] * width1 / 2),
        (p0[0] - n0[0] * width0 / 2, p0[1] - n0[1] * width0 / 2),
    ]


def _edge_normal(direction: Point) -> Point:
    length = math.hypot(direction[0], direction[1])
    if length == 0:
        raise ValueError("_edge_normal: zero-length direction")
    dx, dy = direction[0] / length, direction[1] / length
    return -dy, dx


def chamfer_right_angles(
    waypoints: list[Point], widths: list[float], chamfer_fraction: float = 0.3,
) -> tuple[list[Point], list[float]]:
    """
    Replace every ~90-degree interior corner in `waypoints` with a short
    diagonal cut (two points straddling the original corner instead of one
    sharp point), so the resulting tapered_path has no literal right angle
    anywhere -- Ring's own updated spec: "Arms should not contain right
    angles" (an earlier reading of the doc's previous wording, "straight
    angle", was that the arm must bend at LEAST once -- already true of
    place_external_contacts's escape/shift/approach routing, which is BUILT
    from axis-aligned segments and so turns in exact 90-degree corners at
    every bend; the updated wording flags those exact corners as the actual
    problem). Endpoints (waypoints[0]/[-1]) are never touched -- those are
    fixed connection points (an inner contact's own edge, a pad's own
    edge), not routing choices.

    Only touches corners near-perpendicular (within ~3 degrees); a corner
    that's already some other angle (e.g. a straight-through waypoint from
    upstream dedup) is left alone. `chamfer_fraction` of the SHORTER
    adjacent segment is consumed on each side of the cut, so it can never
    eat into (or reverse) either neighboring segment.
    """
    if len(waypoints) < 3:
        return waypoints, widths
    new_path = [waypoints[0]]
    new_widths = [widths[0]]
    for i in range(1, len(waypoints) - 1):
        prev, cur, nxt = waypoints[i - 1], waypoints[i], waypoints[i + 1]
        v1 = (cur[0] - prev[0], cur[1] - prev[1])
        v2 = (nxt[0] - cur[0], nxt[1] - cur[1])
        len1, len2 = math.hypot(*v1), math.hypot(*v2)
        if len1 < 1e-9 or len2 < 1e-9:
            new_path.append(cur)
            new_widths.append(widths[i])
            continue
        u1, u2 = (v1[0] / len1, v1[1] / len1), (v2[0] / len2, v2[1] / len2)
        cos_angle = u1[0] * u2[0] + u1[1] * u2[1]
        if abs(cos_angle) > math.cos(math.radians(87.0)):  # not (close to) perpendicular
            new_path.append(cur)
            new_widths.append(widths[i])
            continue
        c = chamfer_fraction * min(len1, len2)
        new_path.append((cur[0] - u1[0] * c, cur[1] - u1[1] * c))
        new_widths.append(widths[i])
        new_path.append((cur[0] + u2[0] * c, cur[1] + u2[1] * c))
        new_widths.append(widths[i])
    new_path.append(waypoints[-1])
    new_widths.append(widths[-1])
    return new_path, new_widths


def tapered_path(
    waypoints: list[Point], width_start: float, width_end: float,
    taper_last_segment_only: bool = False, widths: list[float] | None = None,
) -> ShapelyPolygon:
    """
    A tapered polygon along a multi-segment path (>= 2 waypoints).

    `widths`, if given, is an explicit width per waypoint (same length as
    waypoints, before duplicate-dropping) -- each segment tapers linearly
    between its two endpoints' given widths. This is the general form; use
    it when a path needs MORE than one taper zone (e.g. narrow down right
    after leaving a wide inner contact, stay narrow through the risky
    escape/shift/lane run, then widen again right before the pad -- see
    contacts.py, which needs exactly this whenever a contact's own width
    exceeds the pad-end arm_start_width).

    Without `widths`: by default width is interpolated linearly by
    cumulative arc-length from width_start (at waypoints[0]) to width_end
    (at waypoints[-1]). taper_last_segment_only=True instead holds
    width_start constant across every segment except the last, which alone
    tapers to width_end -- keeps the arm thin through the risky early
    segments (escaping the flake, shifting tangentially to line up with a
    pad) and only grows to full pad-connecting width in the final, confined
    approach segment (confirmed empirically: linear tapering let an arm's
    early, still-growing width bulge into a neighboring arm's own lane even
    though their centerlines never crossed).

    Multiple straight tapered_quad segments are unioned via shapely so a
    bent path (mandatory per the spec: "Arms should not contain straight
    angle") comes out as one clean polygon rather than several overlapping/
    misaligned pieces.
    """
    if len(waypoints) < 2:
        raise ValueError("tapered_path: need at least 2 waypoints")
    if widths is not None and len(widths) != len(waypoints):
        raise ValueError("tapered_path: widths must match waypoints length")

    # Drop consecutive near-duplicate waypoints -- callers legitimately pass
    # paths where a bend collapses to zero length in some cases (e.g. a
    # tangential-shift leg with offset=0 for a centered contact), and a
    # zero-length leg would otherwise blow up tapered_quad. Keep `widths` in
    # sync with whatever waypoints survive.
    deduped = [waypoints[0]]
    deduped_widths = [widths[0]] if widths is not None else None
    for i, p in enumerate(waypoints[1:], start=1):
        if math.hypot(p[0] - deduped[-1][0], p[1] - deduped[-1][1]) > 1e-9:
            deduped.append(p)
            if widths is not None:
                deduped_widths.append(widths[i])
    waypoints = deduped
    if len(waypoints) < 2:
        raise ValueError("tapered_path: all waypoints coincide")

    if widths is not None:
        point_widths = deduped_widths
    elif taper_last_segment_only:
        point_widths = [width_start] * len(waypoints)
        point_widths[-1] = width_end
    else:
        seg_lengths = [
            math.hypot(waypoints[i + 1][0] - waypoints[i][0], waypoints[i + 1][1] - waypoints[i][1])
            for i in range(len(waypoints) - 1)
        ]
        total = sum(seg_lengths)
        if total == 0:
            raise ValueError("tapered_path: zero-length path")
        point_widths = [width_start]
        cumulative = 0.0
        for seg_len in seg_lengths:
            cumulative += seg_len
            point_widths.append(width_start + (width_end - width_start) * (cumulative / total))

    quads = [
        ShapelyPolygon(tapered_quad(waypoints[i], waypoints[i + 1], point_widths[i], point_widths[i + 1]))
        for i in range(len(waypoints) - 1)
    ]
    return unary_union(quads)
