"""
Shared external-contact placement, used by both hall.py and ring.py: given
each mode's own inner current/voltage contacts (already built, in absolute
um), place their external bond pads within a work-area square and route a
tapered, bent arm from each inner contact out to its pad.

Deliberately mode-agnostic: hall.py/ring.py each know their own inner-contact
geometry (Hall's 2 current + 6 voltage vs Ring's 2 + 2, and exactly where/
which direction each one faces), but the "fit N pads of a given size into a
work-area square without collisions, then route an arm to each" problem is
identical either way -- see contact_placement/materials/geometric_placement_approach.docx
for the overall approach this implements.

Side-effect-free per the approved plan: takes an `occupied` shapely geometry
and returns new geometry or None on failure -- never mutates `occupied`
itself. layout.py owns committing accepted geometry into the shared
occupied state across flakes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .geometry import ChipBounds, Point, collides, tapered_path, within_bounds


@dataclass
class ContactConnection:
    """One inner contact's own connection point for an external-pad arm --
    built by hall.py/ring.py (they know their own inner-contact geometry),
    consumed here without any knowledge of which mode produced it."""
    position: Point       # absolute um -- the inner contact's own outer edge, centered
    direction: Point      # unit vector, absolute um -- straight-out direction from the flake
    width: float           # the inner contact's own width at `position` (the arm's end width)
    kind: Literal['current', 'voltage']


@dataclass
class PlacedContact:
    kind: Literal['current', 'voltage']
    pad: ShapelyPolygon
    arm: ShapelyPolygon


def _unit(v: Point) -> Point:
    n = math.hypot(v[0], v[1])
    if n == 0:
        raise ValueError("_unit: zero-length direction")
    return (v[0] / n, v[1] / n)


def _perp(v: Point) -> Point:
    return (-v[1], v[0])


def _group_by_direction(connections: list[ContactConnection]) -> list[list[ContactConnection]]:
    """Group connections whose direction points the same way (within ~1
    degree) -- these share one radial "side" of the work area and must be
    spread out tangentially so their pads don't overlap (matches the
    reference layout: e.g. Hall's 3 voltage contacts per side)."""
    groups: list[list[ContactConnection]] = []
    for conn in connections:
        angle = math.atan2(conn.direction[1], conn.direction[0])
        for group in groups:
            g_angle = math.atan2(group[0].direction[1], group[0].direction[0])
            if abs(math.remainder(angle - g_angle, 2 * math.pi)) < math.radians(1.0):
                group.append(conn)
                break
        else:
            groups.append([conn])
    return groups


def _pad_polygon(center: Point, direction: Point, pad_length: float, pad_width: float) -> ShapelyPolygon:
    """A pad_length (radial, along `direction`) x pad_width (tangential)
    rectangle centered at `center`, long axis along `direction` -- i.e. the
    pad "points" toward the flake. See module docstring's linked approach
    doc for why long-axis-radial is the only orientation that fits 3
    same-side voltage pads within a 900-1000um work area given
    min_parallel_voltage_pad_spacing=150."""
    d = _unit(direction)
    t = _perp(d)
    hl, hw = pad_length / 2, pad_width / 2
    corners_local = [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)]
    return ShapelyPolygon([
        (center[0] + lx * d[0] + tx * t[0], center[1] + lx * d[1] + tx * t[1])
        for lx, tx in corners_local
    ])


def place_external_contacts(
    connections: list[ContactConnection],
    center: Point,
    work_area: float,
    pad_length: float,
    pad_width: float,
    current_arm_start_width: float,
    voltage_arm_start_width: float,
    min_pad_spacing: float,
    min_parallel_voltage_pad_spacing: float,
    occupied: BaseGeometry,
    chip_bounds: ChipBounds | None = None,
) -> list[PlacedContact] | None:
    """
    Place one external pad + one routed arm per connection, within a
    work_area x work_area square centered on `center`. Returns None (never
    partial geometry) if any pad or arm can't be placed without violating
    spacing rules, intersecting `occupied`, or landing outside `chip_bounds`
    (if given -- e.g. a flake near the edge of the chip whose work-area
    square would otherwise let a pad land off the physical die) -- callers
    drop the flake for that mode on None, per the spec's "if unsuccessful
    drop flake".

    Does not mutate `occupied`; each pad/arm is checked against the UNION of
    the caller's `occupied` and whatever this call has placed so far (so
    pads/arms within this same contact system don't overlap each other
    either), but that placed-so-far geometry is only ever local to this call.
    """
    placed: list[PlacedContact] = []
    blocked = occupied

    # Shared "escape radius" every arm's first segment travels straight out
    # to (no tangential drift) before doing anything else -- at least as far
    # as the farthest-reaching connection, in ANY direction, so by
    # construction no arm's first segment can dip back into another
    # connection's own inner-contact territory. Only past this radius does
    # an arm shift tangentially to line up with its pad's lane.
    escape_radius = max(
        math.hypot(c.position[0] - center[0], c.position[1] - center[1]) for c in connections
    ) * 1.05  # strictly past every connection's own radial reach, else the connection
              # that sets this max would get a zero-length first segment (its own
              # position already sits exactly at the un-buffered escape radius)

    for group in _group_by_direction(connections):
        d = _unit(group[0].direction)
        t = _perp(d)
        n = len(group)

        # Assign offsets in the same tangential order the connections already sit
        # in -- each arm then only needs a SMALL tangential shift to reach its own
        # pad. Without this, offset assignment is arbitrary relative to `t` (whose
        # sign flips between opposite-facing groups), so a connection can end up
        # assigned a pad on the far side of the group, forcing its arm to sweep
        # across every other connection's own straight-out corridor.
        group = sorted(group, key=lambda c: (c.position[0] - center[0]) * t[0] + (c.position[1] - center[1]) * t[1])

        # `gap` here is CENTER-to-center spacing, but the spec's 50/150
        # values are EDGE-to-edge distances between rectangles -- since both
        # pads in a pair span pad_width around their own center, center
        # distance must be pad_width + the desired edge gap, not just
        # max(edge_gap, pad_width) (that older formula only guaranteed no
        # overlap, e.g. edge_gap=50 with pad_width=100 gave center
        # distance=100 -- pads exactly touching, 0 apart, not >=50 apart).
        #
        # Per-adjacent-pair spacing, not one spacing for the whole group -- a
        # group can be MIXED kind (e.g. Ring's 4 connections all exit the same
        # side: 2 voltage + 2 current together), so whether a given gap needs
        # the larger min_parallel_voltage_pad_spacing depends on THAT specific
        # adjacent pair, not the group as a whole. Reduces to the original
        # uniform spacing for a same-kind group (e.g. Hall's 3-per-side).
        cumulative = [0.0]
        for prev, cur in zip(group, group[1:]):
            same_voltage = prev.kind == 'voltage' and cur.kind == 'voltage'
            edge_gap = min_parallel_voltage_pad_spacing if same_voltage else min_pad_spacing
            cumulative.append(cumulative[-1] + pad_width + edge_gap)
        center_shift = (cumulative[0] + cumulative[-1]) / 2
        offsets = [c - center_shift for c in cumulative]

        radial_dist = work_area / 2 - pad_length / 2
        if radial_dist <= 0:
            return None

        # A single shared escape radius is NOT enough: pad offsets are almost
        # always spread much wider than the connections' own native
        # positions (pads sit up to work_area/2 out; native contacts sit
        # within a few tens of um of center), so a MORE CENTRAL connection's
        # tangential-shift sweep (from its own small |own_tangential| out to
        # its own, comparably distant, offset) almost always passes straight
        # through a MORE PERIPHERAL connection's own native position, even
        # though both own_tangential and offset are individually sorted
        # ascending across the group.
        #
        # Fix: peripheral connections (larger |own_tangential|) must shift
        # OUT OF THE WAY at a SMALLER radius, before more central connections
        # (smaller |own_tangential|) sweep past that same tangential
        # position at a LARGER radius. Ranking by |own_tangential| descending
        # guarantees every connection has cleared its own native tangential
        # position before any more-central connection's shift sweeps through
        # it.
        #
        # Staggered radii must stay below every PAD's own near edge
        # (radial_dist - pad_length/2), not just below radial_dist (the pad
        # CENTER) -- a pad has real radial extent (pad_length, centered on
        # radial_dist), so a tangential-shift leg at some radius inside
        # [radial_dist - pad_length/2, radial_dist + pad_length/2] can clip a
        # neighboring, already-placed pad even though it's nowhere near that
        # pad's own tangential lane.
        own_tangentials = [
            (c.position[0] - center[0]) * t[0] + (c.position[1] - center[1]) * t[1] for c in group
        ]
        shift_order = sorted(range(n), key=lambda i: -abs(own_tangentials[i]))
        pad_near_radius = radial_dist - pad_length / 2
        radius_step = (pad_near_radius - escape_radius) / (n + 1)
        if radius_step <= 0:
            return None
        conn_escape_radius = [0.0] * n
        for rank, i in enumerate(shift_order):
            conn_escape_radius[i] = escape_radius + (rank + 1) * radius_step

        for idx, (conn, offset) in enumerate(zip(group, offsets)):
            pad_center = (
                center[0] + d[0] * radial_dist + t[0] * offset,
                center[1] + d[1] * radial_dist + t[1] * offset,
            )
            if max(abs(pad_center[0] - center[0]), abs(pad_center[1] - center[1])) > work_area / 2:
                return None  # this connection's pad doesn't fit in the work area at all

            pad = _pad_polygon(pad_center, d, pad_length, pad_width)
            if collides(pad, blocked) or not within_bounds(pad, chip_bounds):
                return None

            # Escape straight out along `d`, preserving THIS connection's own
            # tangential position (not converging every connection in the
            # group onto one shared point on the group's center axis, which
            # would send their escape segments crossing each other/each
            # other's own inner contacts near the flake).
            own_tangential = own_tangentials[idx]
            r = conn_escape_radius[idx]
            escape_point = (
                center[0] + d[0] * r + t[0] * own_tangential,
                center[1] + d[1] * r + t[1] * own_tangential,
            )
            lane_point = (center[0] + d[0] * r + t[0] * offset, center[1] + d[1] * r + t[1] * offset)
            pad_near_edge = (pad_center[0] - d[0] * pad_length / 2, pad_center[1] - d[1] * pad_length / 2)
            arm_start_width = current_arm_start_width if conn.kind == 'current' else voltage_arm_start_width

            # Taper over a SHORT zone right next to the pad, not the whole
            # (possibly long) lane-approach segment -- confirmed empirically
            # that tapering over the full lane segment still let the
            # growing width bulge back into a neighboring arm's escape zone.
            # A short, pad-adjacent taper zone stays clear of that region.
            taper_length = min(pad_length * 0.1, (radial_dist - r) / 2)
            pre_taper_point = (pad_near_edge[0] - d[0] * taper_length, pad_near_edge[1] - d[1] * taper_length)

            # Symmetric taper at the SOURCE end too: if this connection's own
            # inner-contact width is wider than the pad-end arm_start_width,
            # carrying that full width all the way through the escape/shift/
            # lane run can bulge into a neighboring connection's lane even
            # though the two escape radii and centerlines never cross.
            # Narrow to min(conn.width, arm_start_width) right after the
            # escape point, matching whichever end is already the smaller of
            # the two -- a no-op when conn.width <= arm_start_width (the
            # common case).
            transit_width = min(conn.width, arm_start_width)
            arm = tapered_path(
                [conn.position, escape_point, lane_point, pre_taper_point, pad_near_edge],
                width_start=conn.width, width_end=arm_start_width,
                widths=[conn.width, transit_width, transit_width, transit_width, arm_start_width],
            )
            if collides(arm, blocked) or not within_bounds(arm, chip_bounds):
                return None

            placed.append(PlacedContact(kind=conn.kind, pad=pad, arm=arm))
            blocked = unary_union([blocked, pad, arm]) if not blocked.is_empty else unary_union([pad, arm])

    return placed
