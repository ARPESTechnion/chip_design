"""
Ring placement: derive the Ring inner geometry (blocking rectangle, cut
mask, 2 voltage + 2 current contacts) from a flake's real contour, then
call contacts.py for the 4 external pads/arms + the triangle "arrow"
decoration. See contact_placement/materials/prompt.docx's "Ring" section
and materials/ring_example_with_measurments.cif (a real, measured
reference instance) for the layout this implements.

Orientation: the cut mask sits at the blocking rectangle's own center,
"vertically to its length" -- the 2 voltage + 2 current contacts stack
symmetrically along the LENGTH axis on either side of it (closest:
voltage, then current -- matches "The voltage are the closest and then
the current").

The cut mask (L041) shape -- inner ring, outer ring, arms, and fingers --
was reverse-engineered by scanline analysis of
materials/ring_example_with_measurments.cif's real polygon vertices (a
small, single-instance file with C5Meas dimension annotations confirming
each of the 5 named quantities below) and verified byte-exact (shapely
symmetric-difference area = 0.0) against that file before being wired in
here -- see _ring_cut_mask for the exact formula. Per side (mirrored top/
bottom along the length axis), from the center outward:
  1. Inner ring: a solid cut square, `inner_ring` on a side.
  2. Outer ring: a BLANK (uncut) gap, `outer_ring` wide, surrounding the
     inner ring on all 4 sides -- this gap is what leaves a real, visible
     ring of uncut flake material; without it (an earlier version fused
     the arm directly onto the inner square) the shape doesn't read as a
     ring at all (confirmed with the user: "this is a single square with
     arms. the inner ring is missing").
  3. Arm: a cut strip, `arm_width` wide (tangential), extending from the
     outer ring's own edge out to wherever the voltage contact begins.
  4. Finger: a second cut block, WIDER than the arm (extends `finger_width`
     further out tangentially, past the outer ring's own edge) but no
     longer than the arm along the length axis -- positioned directly
     beyond the arm's own far edge. NOT a 4-sided frame around the inner
     ring (a separate, earlier version built one of those and was ALSO
     flagged as wrong: "it is not a square, it should be 2 long arms
     connected to a square but there is a space on 2 of the flanks that
     the arms are not touching" -- the left/right flanks of the whole
     assembly stay bare, no cut mask there at all).
`distance_between_arms` sets how close the arm/finger assembly's own
innermost edge sits to the center (independent of inner_ring/outer_ring).

Voltage exits along the WIDTH axis (sideways, toward exit_sign -- +X tried
first, then -X); current exits along the LENGTH axis instead (straight out
past the blocking rectangle's own length-axis end, in whichever sign
direction -- north/south -- it's already stacked in), confirmed directly by
the user after an earlier version had current exit sideways like voltage
too: "the current external connection should connect to the internal along
the flake's length axis, and should stick out more from this direction."
So current's 2 connections end up in 2 OPPOSITE directions (not grouped
together by contacts.py's own by-direction grouping), while voltage's 2
stay grouped together on whichever exit_sign side succeeds.

Pure function, side-effect-free: takes `occupied`, returns new geometry or
None on failure -- never mutates `occupied` itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shapely.affinity import affine_transform
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .config import RingConfig
from .contacts import ContactConnection, PlacedContact, place_external_contacts
from .geometry import ChipBounds, Point, RotatedRect, blocking_rectangle, collides, within_bounds


@dataclass
class RingResult:
    cut_mask: BaseGeometry            # L041 -- inner ring + 2 arm/finger assemblies (see _ring_cut_mask)
    internal_contacts: list[ShapelyPolygon]  # L042 -- 2 voltage + 2 current
    external: list[PlacedContact]      # L040 -- 4 pads + arms
    arrows: list[ShapelyPolygon]        # L040 -- 2 triangle decorations


def _rect_polygon(rect: RotatedRect) -> ShapelyPolygon:
    return ShapelyPolygon(rect.corners())


def _square(frame: RotatedRect, dim: float) -> RotatedRect:
    return RotatedRect(center=frame.center, angle_rad=frame.angle_rad, width=dim, height=dim)


def _ring_cut_mask(
    blocking: RotatedRect,
    inner_ring: float,
    outer_ring: float,
    distance_between_arms: float,
    arm_width: float,
    finger_width: float,
    far_reach: float,
) -> BaseGeometry:
    """
    The cut mask (L041) shape -- see module docstring for the named pieces
    this assembles (inner ring, outer ring gap, arm, finger). Formula,
    reverse-engineered from materials/ring_example_with_measurments.cif and
    verified byte-exact against its real polygon vertices (that file's own
    arms happen to run along ITS OWN local x -- irrelevant, since this
    function re-derives which absolute direction that maps to from the
    blocking rectangle's own orientation, not from the reference file's).

    Let inner_half = inner_ring/2, outer_half = inner_half + outer_ring,
    d_half = distance_between_arms/2, reach = d_half + arm_width. For each
    side (sign = +1/-1, along the blocking rectangle's WIDTH axis --
    "vertically to its length", i.e. perpendicular to length, confirmed
    with the user after an earlier version ran the arms ALONG the length
    axis instead and was flagged as wrong):
      - big_rect: local y in [-(outer_half+finger_width), outer_half+finger_width],
        local x in [sign*d_half, sign*reach]
      - notch: local y in [-outer_half, outer_half], local x in [sign*d_half, sign*outer_half]
      - arm: local y in [-arm_width/2, arm_width/2], local x in [sign*outer_half, sign*far_reach]
      - this side's contribution = (big_rect - notch) | arm
    Full mask = inner_square(inner_ring) | side(+1) | side(-1).

    All boolean ops (difference/union) are done in the blocking rectangle's
    own LOCAL frame (small-magnitude coordinates, axis-aligned pieces) and
    the fully-assembled result is rotated/translated to absolute um with a
    SINGLE affine_transform at the very end. This matters, not just style:
    doing each piece's own to_abs first and then running difference/union on
    already-rotated (non-axis-aligned, large-absolute-magnitude) polygons was
    empirically found to make shapely's GEOS boolean ops merge pieces that
    should stay separate (spurious "thin line" slivers bridging the inner
    square to the arm/finger assemblies) -- confirmed via a controlled
    comparison: local-frame pieces stayed cleanly separate (3-piece
    MultiPolygon) both before AND after an independent
    shapely.affinity.affine_transform rigid transform, but performing the
    same difference/union directly on already-absolute (rotated, translated
    far from the origin) pieces produced 1 merged Polygon instead. A true
    rigid transform can't create new topology, so the bug was specifically
    in running booleans on non-axis-aligned/large-magnitude input, not in
    the rotation math itself -- doing all booleans locally and transforming
    once sidesteps it entirely.
    """
    inner_half = inner_ring / 2
    outer_half = inner_half + outer_ring
    d_half = distance_between_arms / 2
    reach = d_half + arm_width
    finger_outer = outer_half + finger_width
    aw = arm_width / 2

    inner_square = ShapelyPolygon([(-inner_half, -inner_half), (inner_half, -inner_half),
                                    (inner_half, inner_half), (-inner_half, inner_half)])

    pieces = [inner_square]
    for sign in (1.0, -1.0):
        x_lo, x_hi = sorted((sign * d_half, sign * reach))
        big_rect = ShapelyPolygon([(x_lo, -finger_outer), (x_hi, -finger_outer),
                                    (x_hi, finger_outer), (x_lo, finger_outer)])
        n_lo, n_hi = sorted((sign * d_half, sign * outer_half))
        notch = ShapelyPolygon([(n_lo, -outer_half), (n_hi, -outer_half),
                                 (n_hi, outer_half), (n_lo, outer_half)])
        a_lo, a_hi = sorted((sign * outer_half, sign * far_reach))
        arm = ShapelyPolygon([(a_lo, -aw), (a_hi, -aw), (a_hi, aw), (a_lo, aw)])
        pieces.append(big_rect.difference(notch))
        pieces.append(arm)

    local_result = unary_union(pieces)
    c, s = math.cos(blocking.angle_rad), math.sin(blocking.angle_rad)
    cx, cy = blocking.center
    return affine_transform(local_result, [c, -s, s, c, cx, cy])


def _grow_contacts(config: RingConfig, available: float) -> tuple[float, float, float, float] | None:
    """
    (voltage_dist, voltage_len, current_dist, current_len) for one side,
    applying the spec's grow-if-space logic: minimums first, then grow
    current_len up to current_grow_cap, then scale the OTHER three
    proportionally to exactly fill `available`. None if even the minimum
    doesn't fit.

    Spec text: "first increase the current up to 3, then increase all
    contacts sizes (including distances) by the same proportions." Read
    current_len as PERMANENTLY capped at current_grow_cap once reached (not
    re-scaled in the second step too) -- the alternative reading (scale ALL
    FOUR, including the just-capped current_len, by the same factor) makes
    the "up to 3" cap nearly meaningless whenever there's a lot of leftover
    room (current_len would grow far past 3 again in the second step) --
    confirmed this matters empirically: a moderately large synthetic test
    blocking rectangle scaled current_len to ~40 under that reading.

    v_len is ALSO capped, at voltage_grow_cap, AFTER the proportional
    scaling step (not in prompt.docx, a later user addition: "make the
    maximum size of internal voltage width to be 3") -- any growth v_len
    can't absorb is redistributed to v_dist/c_dist instead, so the total
    still exactly fills `available`.
    """
    v_dist, v_len = config.voltage_min_distance_from_mask, config.voltage_min_width_in_blocking_rect
    c_dist, c_len = config.current_min_distance_from_voltage, config.current_min_length_in_blocking_rect

    required_min = v_dist + v_len + c_dist + c_len
    if required_min > available:
        return None

    leftover = available - required_min
    current_growth = min(leftover, config.current_grow_cap - c_len)
    c_len += current_growth
    leftover -= current_growth

    if leftover > 0:
        total = v_dist + v_len + c_dist
        factor = (total + leftover) / total
        v_dist *= factor
        v_len *= factor
        c_dist *= factor

        # v_len (voltage_min_width_in_blocking_rect, grown) is ALSO capped,
        # same "grow up to a cap, then redistribute the rest" pattern as
        # current_len above -- user: "make the maximum size of internal
        # voltage width to be 3." Not in prompt.docx (which only caps
        # current); redistribute whatever v_len couldn't absorb to v_dist/
        # c_dist instead, proportionally between just those two, so the
        # total still exactly fills `available`.
        if v_len > config.voltage_grow_cap:
            excess = v_len - config.voltage_grow_cap
            v_len = config.voltage_grow_cap
            remaining_total = v_dist + c_dist
            redistribute_factor = (remaining_total + excess) / remaining_total
            v_dist *= redistribute_factor
            c_dist *= redistribute_factor

    return v_dist, v_len, c_dist, c_len


def _try_ring(
    flake_polygon: ShapelyPolygon,
    blocking: RotatedRect,
    config: RingConfig,
    inner_ring: float,
    outer_ring: float,
    exit_sign: float,
    occupied: BaseGeometry,
    chip_bounds: ChipBounds | None,
    other_pads: BaseGeometry | None = None,
) -> RingResult | None:
    inner_square = _square(blocking, inner_ring)
    if not flake_polygon.contains(_rect_polygon(inner_square)):
        return None  # "If the inner ring is not inside the flakes contour drop"

    mask_half = inner_ring / 2
    half_l = blocking.height / 2
    available = half_l - mask_half
    grown = _grow_contacts(config, available)
    if grown is None:
        return None
    v_dist, v_len, c_dist, c_len = grown

    # Cut mask's arm reach -- arms run along the WIDTH axis (see
    # _ring_cut_mask) and must stick out config.arm_protrusion_beyond_blocking
    # past the blocking rectangle's own width edge (user: "the arms should
    # stick out 3 um beyond the blocking rectangle"). The internal contacts
    # below also exit along the WIDTH axis (v_x_far/v_x_near, c_x_far/c_x_near)
    # -- same axis as the cut mask's own arms, confirmed by direct inspection
    # of the constructed polygons' own edge lengths/directions (the long,
    # ~width-sized edge of each contact runs along blocking.direction_to_absolute(1,0),
    # the short, v_len/c_len-sized edge along direction_to_absolute(0,1)) --
    # so despite looking mismatched at a glance in a rotated render, both
    # pieces are already on the same local axis.
    far_reach = blocking.width / 2 + config.arm_protrusion_beyond_blocking
    cut_mask = _ring_cut_mask(
        blocking, inner_ring, outer_ring, config.distance_between_arms,
        config.arm_width, config.finger_width, far_reach,
    )

    internal_local: list[list[Point]] = []
    connections: list[ContactConnection] = []
    for sign in (1.0, -1.0):
        v_y0 = sign * (mask_half + v_dist)
        v_y1 = v_y0 + sign * v_len
        c_y0 = v_y1 + sign * c_dist

        v_x_far = exit_sign * (blocking.width / 2 + config.voltage_stackout_connecting_side)
        v_x_near = -exit_sign * (blocking.width / 2 + config.voltage_stackout)

        # Trapezoid on the connecting side: v_len wide until it clears the
        # current contact's own width edge (current_width/2, just past the
        # blocking rectangle's edge), then widening linearly to v_len *
        # voltage_taper_factor at v_x_far. Widens only on the side AWAY from
        # the cut mask (outward, toward the current's side), so the
        # voltage_min_distance_from_mask gap to the mask's protruding arms
        # is untouched; past current_width/2 there's no current contact
        # alongside, so current_min_distance_from_voltage is untouched too.
        current_width = blocking.width + 1.0  # see the current contact below
        x_taper = exit_sign * current_width / 2
        v_y1_far = v_y1 + sign * v_len * (config.voltage_taper_factor - 1.0)
        internal_local.append([(v_x_near, v_y0), (v_x_far, v_y0), (v_x_far, v_y1_far), (x_taper, v_y1), (v_x_near, v_y1)])
        connections.append(ContactConnection(
            position=blocking.to_absolute(v_x_far, (v_y0 + v_y1_far) / 2),
            direction=blocking.direction_to_absolute(exit_sign, 0.0),
            width=abs(v_y1_far - v_y0),
            kind='voltage',
        ))

        # Current, unlike voltage, exits along the LENGTH axis (user: "the
        # current external connection should connect to the internal along
        # the flake's length axis") -- it continues straight out past the
        # blocking rectangle's own length-axis end (half_l) by
        # current_min_stackout, in the SAME sign direction it's already
        # stacked in, rather than bending sideways toward exit_sign like
        # voltage does. Its cross-sectional width (perpendicular to this
        # travel direction) is the blocking rectangle's own width plus 1 --
        # user: "the width of the current contacts should be the same as
        # the blocking rectangle's width" then "lets change the width to be
        # 1 larger then the blocking rectangle" (was c_len, then plain
        # blocking.width, both flagged as too small).
        c_y1 = sign * (half_l + config.current_min_stackout)
        c_half_width = current_width / 2

        c_ylo, c_yhi = sorted((c_y0, c_y1))
        internal_local.append([(-c_half_width, c_ylo), (c_half_width, c_ylo), (c_half_width, c_yhi), (-c_half_width, c_yhi)])
        connections.append(ContactConnection(
            position=blocking.to_absolute(0.0, c_y1),
            direction=blocking.direction_to_absolute(0.0, sign),
            width=current_width,
            kind='current',
        ))

    internal_abs = [ShapelyPolygon([blocking.to_absolute(x, y) for x, y in pts]) for pts in internal_local]
    if collides(cut_mask, occupied) or any(collides(s, occupied) for s in internal_abs):
        return None
    if not within_bounds(cut_mask, chip_bounds) or any(not within_bounds(s, chip_bounds) for s in internal_abs):
        return None  # flake too close to the chip edge for even the inner ring geometry to fit

    external = place_external_contacts(
        connections, center=blocking.center, work_area=config.work_area,
        pad_length=config.external_pad_width, pad_width=config.external_pad_height,
        current_arm_start_width=config.current_arm_start_width,
        voltage_arm_start_width=config.voltage_arm_start_width,
        min_pad_spacing=config.min_pad_spacing,
        min_parallel_voltage_pad_spacing=config.min_parallel_voltage_pad_spacing,
        # Deliberately NOT including this flake's own internal_abs here
        # (unlike Hall, which does) -- with all 4 of Ring's own contacts
        # clustered tightly around the same small cut mask, a genuinely
        # straight connection from any one of them to its own pad almost
        # always grazes a SIBLING contact's own body right at the start
        # (confirmed: every direct attempt collided, ~0.4-11 sq-um overlap,
        # entirely against sibling internal contacts, not anything else).
        # Matches the same precedent already used for the cut mask itself
        # (never included here either): different CIF layers, fine to
        # overlap in a real fabrication stack. User: "try to connect the
        # rectangles to the inner contacts in the straightest way possible"
        # + a reference image showing exactly one straight trapezoid, no
        # bend -- this is what actually unblocks that.
        occupied=occupied,
        chip_bounds=chip_bounds,
        # All 4 pads share ONE orientation (voltage's own, matching the
        # pre-length-axis-change layout) even though current's own inner
        # contact now exits along a different axis -- user: "the external
        # contacts blocks should be all parallel to each other like there
        # were before."
        shared_direction=blocking.direction_to_absolute(exit_sign, 0.0),
        other_pads=other_pads,
    )
    if external is None:
        return None

    arrows = _build_arrows(external, config)

    return RingResult(cut_mask=cut_mask, internal_contacts=internal_abs, external=external, arrows=arrows)


def _build_arrows(external: list[PlacedContact], config: RingConfig) -> list[ShapelyPolygon]:
    """One isosceles-triangle arrow between each voltage/current pad pair on
    the same side (matched by nearest pad-centroid distance -- with exactly
    2 voltage + 2 current pads, each voltage pairs with its nearer current)."""
    voltages = [c for c in external if c.kind == 'voltage']
    currents = [c for c in external if c.kind == 'current']
    arrows = []
    remaining_currents = list(currents)
    for v in voltages:
        vc = v.pad.centroid
        nearest = min(remaining_currents, key=lambda c: vc.distance(c.pad.centroid))
        remaining_currents.remove(nearest)
        cc = nearest.pad.centroid
        dx, dy = cc.x - vc.x, cc.y - vc.y
        length = math.hypot(dx, dy)
        if length == 0:
            continue
        ux, uy = dx / length, dy / length
        nx, ny = -uy, ux
        mid = ((vc.x + cc.x) / 2, (vc.y + cc.y) / 2)
        apex = (mid[0] + ux * config.arrow_height / 2, mid[1] + uy * config.arrow_height / 2)
        base_center = (mid[0] - ux * config.arrow_height / 2, mid[1] - uy * config.arrow_height / 2)
        base_a = (base_center[0] + nx * config.arrow_base / 2, base_center[1] + ny * config.arrow_base / 2)
        base_b = (base_center[0] - nx * config.arrow_base / 2, base_center[1] - ny * config.arrow_base / 2)
        arrows.append(ShapelyPolygon([apex, base_a, base_b]))
    return arrows


def place_ring(
    flake_vertices_um: list[Point],
    config: RingConfig,
    occupied: BaseGeometry,
    chip_bounds: ChipBounds | None = None,
    other_pads: BaseGeometry | None = None,
) -> RingResult | None:
    """Attempt a Ring placement for one flake, searching inner_ring x
    outer_ring x exit_side candidates (smallest first, per config.frange())
    until one succeeds. Returns None (drop the flake for Ring) if no
    candidate works."""
    from .config import frange

    flake_polygon = ShapelyPolygon(flake_vertices_um)
    if not flake_polygon.is_valid:
        flake_polygon = flake_polygon.buffer(0)

    blocking = blocking_rectangle(flake_vertices_um).with_long_axis_as('y')

    for inner_ring in frange(config.inner_ring_min, config.inner_ring_max, config.inner_ring_step):
        for outer_ring in frange(config.outer_ring_min, config.outer_ring_max, config.outer_ring_step):
            for exit_sign in (1.0, -1.0):
                result = _try_ring(
                    flake_polygon, blocking, config, inner_ring, outer_ring, exit_sign, occupied, chip_bounds, other_pads,
                )
                if result is not None:
                    return result
    return None
