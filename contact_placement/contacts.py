"""
Shared external-contact placement, used by both hall.py and ring.py: given
each mode's own inner current/voltage contacts (already built, in absolute
um), place their external bond pads within a work-area square and connect
each inner contact to its pad with ONE straight tapered arm -- never a bent
route. If no pad layout gives every arm a clear straight line, placement
fails and the caller drops the flake for that mode.

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

from shapely.geometry import Polygon as ShapelyPolygon, box
from shapely.geometry.base import BaseGeometry

from .geometry import ChipBounds, Point, collides, edge_to_edge_taper, within_bounds


@dataclass
class ContactConnection:
    """One inner contact's own connection point for an external-pad arm --
    built by hall.py/ring.py (they know their own inner-contact geometry),
    consumed here without any knowledge of which mode produced it."""
    position: Point       # absolute um -- the inner contact's own outer edge, centered
    direction: Point      # unit vector, absolute um -- straight-out direction from the flake
    width: float           # the inner contact's own width at `position` (the arm's end width)
    kind: Literal['current', 'voltage']
    group_direction: Point | None = None
    """Direction used to group this connection with its siblings and to
    orient its pad, when it differs from `direction` -- e.g. Hall's bent
    outer voltage taps (see hall.py) exit at an angle but still belong to
    their side's 3-pad group, with pads parallel to the straight centre
    tap's. None = same as `direction`."""

    @property
    def lane_direction(self) -> Point:
        return self.group_direction if self.group_direction is not None else self.direction


@dataclass
class PlacedContact:
    kind: Literal['current', 'voltage']
    pad: ShapelyPolygon
    arm: ShapelyPolygon
    radial_dist: float | None = None                    # pad-centre distance used for its group
    radial_bounds: tuple[float, float] | None = None    # (closest, farthest) distance allowed for that group


class _Obstacles:
    """Everything a pad/arm must not overlap: the caller's fixed `occupied`
    region plus the pads/arms this call has placed so far, kept as a list
    rather than unioned in. The pad-distance search tries hundreds of
    candidate layouts; re-unioning a chip-wide `occupied` with every trial
    pad/arm dominated runtime, while checking a handful of local shapes
    separately is cheap."""

    def __init__(
        self, base: BaseGeometry, local: tuple[BaseGeometry, ...] = (),
        other_pads: BaseGeometry | None = None, pad_spacing: float = 0.0,
    ):
        self.base = base
        self.local = local
        self.other_pads = other_pads
        self.pad_spacing = pad_spacing

    def hit(self, geom: BaseGeometry) -> bool:
        return collides(geom, self.base) or any(collides(geom, g) for g in self.local)

    def hit_pad(self, pad: BaseGeometry) -> bool:
        """Like hit(), plus: a pad must stay at least pad_spacing (edge to
        edge, exact distance) from every other flake's pad."""
        if self.hit(pad):
            return True
        return self.other_pads is not None and pad.distance(self.other_pads) < self.pad_spacing - 1e-9

    def plus(self, *geoms: BaseGeometry) -> _Obstacles:
        return _Obstacles(self.base, self.local + geoms, self.other_pads, self.pad_spacing)


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
        angle = math.atan2(conn.lane_direction[1], conn.lane_direction[0])
        for group in groups:
            g_angle = math.atan2(group[0].lane_direction[1], group[0].lane_direction[0])
            if abs(math.remainder(angle - g_angle, 2 * math.pi)) < math.radians(1.0):
                group.append(conn)
                break
        else:
            groups.append([conn])
    return groups


def _group_pads_at(
    group: list[ContactConnection], offsets: list[float], d: Point, t: Point, center: Point,
    radial_dist: float, work_area: float, pad_length: float, pad_width: float,
    current_pad_direction: Point | None, blocked: _Obstacles, chip_bounds: ChipBounds | None,
    min_gap_to_voltage: float | None, voltage_pads: list[ShapelyPolygon],
    current_arm_start_width: float, voltage_arm_start_width: float,
) -> tuple[list[PlacedContact], _Obstacles] | None:
    """(placed pads + arms, `blocked` plus every pad+arm this group placed)
    if ALL of `group`'s connections are placeable at this one shared
    `radial_dist` AND connectable via a direct (straight, unbent) arm alone
    -- in bounds, pad and direct-arm collision-free against `blocked` AND
    against each other (checked cumulatively, one connection at a time,
    exactly like the real per-connection loop later will), and -- for
    current-kind connections, when min_gap_to_voltage is given -- at least
    that far from every already-placed voltage pad. None the moment any one
    of them fails. The returned occupied region lets a caller (see
    _remaining_groups_placeable) chain a lookahead across multiple groups
    without re-deriving what this group would leave behind.

    The arm is always the single straight connection (edge_to_edge_taper)
    -- there is no bent fallback route, so a layout only counts if every
    arm is clear."""
    placed: list[PlacedContact] = []
    trial_blocked = blocked
    for conn, offset in zip(group, offsets):
        pad_center = (center[0] + d[0] * radial_dist + t[0] * offset, center[1] + d[1] * radial_dist + t[1] * offset)
        if max(abs(pad_center[0] - center[0]), abs(pad_center[1] - center[1])) > work_area / 2:
            return None
        pad_direction = (
            _unit(current_pad_direction) if (current_pad_direction is not None and conn.kind == 'current') else d
        )
        pad_aligned = abs(pad_direction[0] * d[0] + pad_direction[1] * d[1]) > 0.999
        extent_along_d = pad_length if pad_aligned else pad_width
        pad = _pad_polygon(pad_center, pad_direction, pad_length, pad_width)
        if trial_blocked.hit_pad(pad) or not within_bounds(pad, chip_bounds):
            return None
        if conn.kind == 'current' and min_gap_to_voltage is not None and voltage_pads:
            if min(pad.distance(vp) for vp in voltage_pads) < min_gap_to_voltage:
                return None

        pad_near_edge = (
            pad_center[0] - d[0] * extent_along_d / 2, pad_center[1] - d[1] * extent_along_d / 2,
        )
        arm_start_width = current_arm_start_width if conn.kind == 'current' else voltage_arm_start_width
        direct_arm = ShapelyPolygon(edge_to_edge_taper(
            conn.position, conn.direction, conn.width, pad_near_edge, d, arm_start_width,
        ))
        if trial_blocked.hit(direct_arm) or not within_bounds(direct_arm, chip_bounds):
            return None

        placed.append(PlacedContact(kind=conn.kind, pad=pad, arm=direct_arm))
        trial_blocked = trial_blocked.plus(pad, direct_arm)
    return placed, trial_blocked


def _remaining_groups_placeable(
    groups: list[list[ContactConnection]], blocked: _Obstacles, voltage_pads: list[ShapelyPolygon],
    r: float, center: Point, work_area: float, pad_length: float, pad_width: float,
    current_pad_direction: Point | None, chip_bounds: ChipBounds | None,
    current_voltage_min_gap: float | None,
    current_arm_start_width: float, voltage_arm_start_width: float,
    min_pad_spacing: float, min_parallel_voltage_pad_spacing: float,
) -> bool:
    """True if every group in `groups` -- the ones the real per-group loop
    in place_external_contacts still has left to process after whichever
    group is currently being sized -- is ALSO directly placeable at this
    SAME radial_dist `r` (not each independently re-minimized), placed in
    sequence starting from `blocked`. Mirrors place_external_contacts's own
    per-group sort-by-tangential-position + cumulative-spacing setup
    exactly, so the simulated placement matches what the real loop would
    actually do later.

    Deliberately reuses the SAME `r` for every remaining group rather than
    letting each one search for its own independent minimum: an earlier
    version did the latter and was WRONG -- two symmetric groups (e.g.
    Hall's own two voltage sides) each searching independently, mid-lookahead,
    landed at DIFFERENT radii (whichever was cheapest for each one alone),
    producing a lopsided trial placement that doesn't match what the real
    per-group loop would ever actually produce (every group there gets its
    own real, consistent search once its turn comes) -- confirmed as a real
    false negative: it rejected radii that a proper joint check confirms
    are genuinely fine. Checking "does everyone ALSO fit at this exact r"
    is a sufficient (if slightly conservative -- the real loop may later
    find an even smaller r for a given remaining group) and cheap
    (one _group_pads_at call per remaining group, not a nested search) test.

    Used as a lookahead so an earlier group's own greedy smallest-radius
    choice can't strand a later group with nowhere left to go -- confirmed
    as a real failure: Hall's voltage groups (searched first, per
    current_voltage_min_gap's own docstring) could converge on a much
    smaller radius once the current contacts' own internal footprint
    shrank (see hall.py), and that tighter voltage placement then stranded
    current's own group entirely -- even though a slightly larger (still
    perfectly valid, still minimal for ITS OWN purposes) voltage radius
    would have left current plenty of room."""
    trial_blocked = blocked
    trial_voltage_pads = list(voltage_pads)
    for group in groups:
        d = _unit(group[0].lane_direction)
        t = _perp(d)
        group = sorted(group, key=lambda c: (c.position[0] - center[0]) * t[0] + (c.position[1] - center[1]) * t[1])
        cumulative = [0.0]
        for prev, cur in zip(group, group[1:]):
            same_voltage = prev.kind == 'voltage' and cur.kind == 'voltage'
            edge_gap = min_parallel_voltage_pad_spacing if same_voltage else min_pad_spacing
            cumulative.append(cumulative[-1] + pad_width + edge_gap)
        center_shift = (cumulative[0] + cumulative[-1]) / 2
        offsets = [c - center_shift for c in cumulative]

        group_gap = current_voltage_min_gap if group[0].kind == 'current' else None
        result = _group_pads_at(
            group, offsets, d, t, center, r, work_area, pad_length, pad_width,
            current_pad_direction, trial_blocked, chip_bounds, group_gap, trial_voltage_pads,
            current_arm_start_width, voltage_arm_start_width,
        )
        if result is None:
            return False
        group_placed, trial_blocked = result
        if group[0].kind == 'voltage':
            trial_voltage_pads = trial_voltage_pads + [p.pad for p in group_placed]
    return True


def _find_radial_dist(
    group: list[ContactConnection], offsets: list[float], d: Point, t: Point, center: Point,
    work_area: float, pad_length: float, pad_width: float, current_pad_direction: Point | None,
    blocked: _Obstacles, chip_bounds: ChipBounds | None, min_gap_to_voltage: float | None,
    voltage_pads: list[ShapelyPolygon], escape_radius: float,
    current_arm_start_width: float, voltage_arm_start_width: float,
    remaining_groups: list[list[ContactConnection]] | None = None,
    current_voltage_min_gap: float | None = None,
    min_pad_spacing: float = 0.0, min_parallel_voltage_pad_spacing: float = 0.0,
    descending: bool = False,
    radial_level: tuple[int, int] | None = None,
) -> float | None:
    """First radial_dist (pad-center distance from `center`, along `d`)
    at which this whole group's pads AND straight arms are all placeable at
    once -- searched in fixed steps between a conservative minimum and the
    work-area-edge bound. Ascending by default (smallest distance: pads
    pulled in close); `descending=True` starts at the work-area edge and
    moves inward instead (keeps pads as far out as possible, only pulling
    them in when that's what it takes for every arm to be straight). None
    if no valid radial_dist exists within that range (caller fails
    closed).

    Replaces always pinning pads to the work-area edge (`work_area/2 -
    pad_length/2`, a FIXED value regardless of whether the pads actually
    need to be that far out) -- user: "I want the rectangles to be more
    closer together... like the image" (materials/debugging/hallExample.png
    vs now.png: the reference has the whole 8-pad assembly sitting close
    around the flake, not pushed out to the work area's own edge).

    The search never goes below the point where the pad's own near edge
    (`radial_dist - pad_length/2`) sits inside `escape_radius` (just past
    the farthest inner-contact connection point), so pads always stay
    outside the inner-contact region.

    `remaining_groups`, if given, is every group place_external_contacts's
    own per-group loop still has left to process after this one -- a
    candidate r is only accepted once _remaining_groups_placeable also
    confirms every one of them still has SOME valid radial_dist against the
    occupied region this group's own placement at r would leave behind (see
    that function's own docstring for the real failure mode this closes).

    `radial_level=(k, n)`, if given, replaces the search with ONE fixed
    distance: level k of n evenly spaced levels from the farthest (k=0,
    max_radial) to the closest (k=n-1, min_radial) -- used by the
    contact-placement search (search.py), where each distance level is its
    own option. None if the layout doesn't fit at exactly that distance."""
    step = 1.0
    min_radial, max_radial = _radial_bounds(work_area, pad_length, pad_width, escape_radius)
    n_steps = int((max_radial - min_radial) // step)
    radii = [min_radial + k * step for k in range(n_steps + 1)]
    if descending:
        radii.reverse()
    if radial_level is not None:
        if max_radial < min_radial:
            return None
        k, n = radial_level
        hi = _level_radius(k, n, min_radial, max_radial)
        if k >= n - 1:
            radii = [hi]
        else:  # band k: from its own level inward, up to (not including) the next level
            lo = _level_radius(k + 1, n, min_radial, max_radial)
            radii = [hi - j * step for j in range(int((hi - lo) / step) + 1) if hi - j * step > lo + 1e-9]
    for r in radii:
        result = _group_pads_at(
            group, offsets, d, t, center, r, work_area, pad_length, pad_width,
            current_pad_direction, blocked, chip_bounds, min_gap_to_voltage, voltage_pads,
            current_arm_start_width, voltage_arm_start_width,
        )
        if result is not None:
            group_placed, trial_blocked = result
            pads = [p.pad for p in group_placed]
            if remaining_groups is None or _remaining_groups_placeable(
                remaining_groups, trial_blocked,
                voltage_pads + pads if group[0].kind == 'voltage' else voltage_pads,
                r, center, work_area, pad_length, pad_width, current_pad_direction, chip_bounds,
                current_voltage_min_gap, current_arm_start_width, voltage_arm_start_width,
                min_pad_spacing, min_parallel_voltage_pad_spacing,
            ):
                return r
    return None


def _radial_bounds(work_area: float, pad_length: float, pad_width: float, escape_radius: float) -> tuple[float, float]:
    """(closest, farthest) pad-centre distance a group's pads may use: the
    farthest keeps the pad inside the work area; the closest keeps its near
    edge past escape_radius (outside the inner contacts)."""
    max_radial = work_area / 2 - pad_length / 2
    min_radial = max(max(pad_length, pad_width) / 2, escape_radius + pad_length / 2 + 1.0)
    return min_radial, max_radial


def _level_radius(k: int, n: int, min_radial: float, max_radial: float) -> float:
    """Distance of level k of n, evenly spaced from farthest (0) to closest (n-1)."""
    return max_radial - (max_radial - min_radial) * k / (n - 1) if n > 1 else max_radial


def distance_level(radial_dist: float, radial_bounds: tuple[float, float], n: int) -> int:
    """Which of n distance bands a pad distance falls in (band k = from
    level k inward to level k+1) -- maps a freely chosen distance (greedy)
    onto the search's distance levels."""
    lo, hi = radial_bounds
    if n <= 1 or hi <= lo:
        return 0
    t = (hi - radial_dist) / (hi - lo) * (n - 1)
    return min(n - 1, max(0, int(t + 1e-9)))


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
    shared_direction: Point | None = None,
    current_pad_direction: Point | None = None,
    minimize_radial_distance: bool = False,
    current_voltage_min_gap: float | None = None,
    other_pads: BaseGeometry | None = None,
    radial_level: tuple[int, int] | None = None,
) -> list[PlacedContact] | None:
    """
    Place one external pad + one straight arm per connection, within a
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

    `shared_direction`, if given, overrides the default by-direction
    grouping (`_group_by_direction`) with ONE group covering every
    connection, all pads sharing this single orientation -- for Ring, whose
    current connections now exit along a different local axis than voltage
    (see ring.py) but whose pads should still come out "all parallel to
    each other like there were before" (user's own words). A connection
    whose own direction differs from `shared_direction` still gets a single
    straight arm (edge_to_edge_taper matches both its own edge and the
    pad's edge exactly).

    `current_pad_direction`, if given, overrides ONLY current-kind pads'
    own ROTATION (not their position -- still placed along their own
    group's `d`/offset as usual) to this direction instead of their
    group's own `d`. For Hall, current's own native direction (length
    axis, top/bottom) is perpendicular to voltage's (width axis, left/
    right), so without this override current pads come out rotated 90
    degrees relative to voltage's -- flagged directly: "the current
    rectangle should be parallel to the voltage. that is the contact path
    should connect to its long side" (see materials/debugging/hallExample.png
    -- ALL 8 pads, current included, share one fixed landscape orientation;
    only voltage's own natural direction happens to already match it).
    hall.py passes voltage's own direction here; ring.py leaves this unset
    (Ring's current already has its own, separately-specified reasons for
    its geometry -- not in scope for this override).

    `minimize_radial_distance`, if True, searches for the SMALLEST radial
    distance each group's pads can sit at (see _find_radial_dist) -- user
    wanted the whole assembly pulled in close around the flake, matching
    materials/debugging/hallExample.png. If False, pads start at the
    work-area edge and are only pulled inward as far as needed for every
    arm to be straight and clear.

    `current_voltage_min_gap`, if given (only meaningful alongside
    minimize_radial_distance), additionally requires every current pad to
    sit at least this far (edge-to-edge) from the nearest already-placed
    voltage pad -- user: "make the current rectangles distance from the
    closest voltage rectangle 50." Voltage groups are placed FIRST (groups
    reordered up front) whenever this is set, so their final pads exist to
    measure against by the time a current group's own search runs.

    `other_pads`, if given, is every pad already placed for OTHER flakes;
    each new pad must keep at least min_pad_spacing (edge-to-edge) from
    all of them, not just avoid overlapping them.

    `radial_level=(k, n)`, if given, puts every group's pads at that one
    fixed distance level instead of searching (see _find_radial_dist).
    """
    placed: list[PlacedContact] = []
    # Only the part of `occupied` a pad or arm can reach can ever collide --
    # clip once up front. A pad's CENTRE stays inside the work-area square,
    # but the pad itself reaches up to its half-diagonal beyond it, and the
    # 50um spacing to other pads one step further.
    half = work_area / 2 + math.hypot(pad_length, pad_width) / 2 + min_pad_spacing + 1.0
    work_box = box(center[0] - half, center[1] - half, center[0] + half, center[1] + half)
    near_pads = None
    if other_pads is not None and not other_pads.is_empty:
        near_pads = other_pads.intersection(work_box)
        near_pads = None if near_pads.is_empty else near_pads
    blocked = _Obstacles(occupied.intersection(work_box), other_pads=near_pads, pad_spacing=min_pad_spacing)

    # Pads never sit inside this radius (just past the farthest-reaching
    # connection point) -- see _find_radial_dist.
    escape_radius = max(
        math.hypot(c.position[0] - center[0], c.position[1] - center[1]) for c in connections
    ) * 1.05

    groups = [connections] if shared_direction is not None else _group_by_direction(connections)
    if current_voltage_min_gap is not None:
        # Voltage groups must be FULLY placed (final pad polygons known)
        # before any current group's own search can measure distance to
        # them -- see current_voltage_min_gap's own docstring.
        groups = sorted(groups, key=lambda g: 0 if g[0].kind == 'voltage' else 1)
    for group_idx, group in enumerate(groups):
        remaining_groups = groups[group_idx + 1:]
        d = _unit(shared_direction) if shared_direction is not None else _unit(group[0].lane_direction)
        t = _perp(d)

        # Assign offsets in the same tangential order the connections already sit
        # in, so no arm has to cross a sibling's to reach its own pad.
        group = sorted(group, key=lambda c: (c.position[0] - center[0]) * t[0] + (c.position[1] - center[1]) * t[1])

        # `gap` here is CENTER-to-center spacing, but the spec's 50/150
        # values are EDGE-to-edge distances between rectangles -- so center
        # distance must be pad_width + the desired edge gap. Per-adjacent-pair
        # spacing: a group can be MIXED kind (e.g. Ring's 4 connections all
        # exit the same side: 2 voltage + 2 current together), so whether a
        # given gap needs min_parallel_voltage_pad_spacing depends on THAT
        # specific adjacent pair.
        cumulative = [0.0]
        for prev, cur in zip(group, group[1:]):
            same_voltage = prev.kind == 'voltage' and cur.kind == 'voltage'
            edge_gap = min_parallel_voltage_pad_spacing if same_voltage else min_pad_spacing
            cumulative.append(cumulative[-1] + pad_width + edge_gap)
        center_shift = (cumulative[0] + cumulative[-1]) / 2
        offsets = [c - center_shift for c in cumulative]

        group_gap = current_voltage_min_gap if group[0].kind == 'current' else None
        voltage_pads_so_far = [p.pad for p in placed if p.kind == 'voltage']
        radial_dist = _find_radial_dist(
            group, offsets, d, t, center, work_area, pad_length, pad_width,
            current_pad_direction, blocked, chip_bounds, group_gap, voltage_pads_so_far, escape_radius,
            current_arm_start_width, voltage_arm_start_width,
            remaining_groups=remaining_groups, current_voltage_min_gap=current_voltage_min_gap,
            min_pad_spacing=min_pad_spacing, min_parallel_voltage_pad_spacing=min_parallel_voltage_pad_spacing,
            descending=not minimize_radial_distance, radial_level=radial_level,
        )
        if radial_dist is None:
            return None  # no pad distance gives every arm in this group a clear straight line
        result = _group_pads_at(
            group, offsets, d, t, center, radial_dist, work_area, pad_length, pad_width,
            current_pad_direction, blocked, chip_bounds, group_gap, voltage_pads_so_far,
            current_arm_start_width, voltage_arm_start_width,
        )
        if result is None:
            return None
        group_placed, blocked = result
        bounds = _radial_bounds(work_area, pad_length, pad_width, escape_radius)
        for p in group_placed:
            p.radial_dist, p.radial_bounds = radial_dist, bounds
        placed += group_placed

    return placed
