"""
Hall-bar placement: derive the Hall inner geometry (new-flake rectangle,
inner rectangle, contacts area, 2 current + 6 voltage contacts, cut mask)
from a flake's real contour, then call contacts.py for the 8 external
pads/arms. See contact_placement/materials/prompt.docx's "Hall" section and
hall_example(.png/_far.png) for the reference layout this implements, and
contact_placement/materials/geometric_placement_approach.docx for the
overall approach.

Orientation (derived from the reference images + the spec's own physics --
this is a real Hall-bar/Hall-cross device): current flows lengthwise through
the bar, so the 2 current contacts sit at the two ENDS along the flake's long
(length) axis, each spanning the full new-flake WIDTH. Voltage taps measure
the transverse (Hall) voltage, so the 6 voltage contacts sit 3-per-side along
the two LONG (width-direction) sides, evenly spaced along the length.

Pure function, side-effect-free (per the approved plan): takes `occupied`,
returns new geometry or None on failure -- never mutates `occupied` itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .config import HallConfig
from .contacts import ContactConnection, PlacedContact, place_external_contacts
from .geometry import (
    ChipBounds, Point, RotatedRect, collides, largest_inscribed_rectangle, tapered_quad, within_bounds,
)


@dataclass
class HallResult:
    cut_mask: BaseGeometry           # L041 -- may be a MultiPolygon
    internal_contacts: list[ShapelyPolygon]  # L042 -- 2 current + 6 voltage
    external: list[PlacedContact]     # L040 -- 8 pads + arms (see contacts.py)


def _rect_polygon(rect: RotatedRect) -> ShapelyPolygon:
    return ShapelyPolygon(rect.corners())


def place_hall(
    flake_vertices_um: list[Point],
    config: HallConfig,
    occupied: BaseGeometry,
    chip_bounds: ChipBounds | None = None,
) -> HallResult | None:
    """Attempt a Hall placement for one flake. Returns None (drop the flake
    for Hall -- never partial geometry) if any spec constraint or collision
    check fails."""
    new_flake = largest_inscribed_rectangle(flake_vertices_um).with_long_axis_as('y')
    if new_flake.height < config.min_length_to_width_ratio * new_flake.width:
        return None  # "length must be at least 2x width, else give up contour for hall"

    inner = RotatedRect(
        center=new_flake.center, angle_rad=new_flake.angle_rad,
        width=new_flake.width - config.inner_rect_width_margin,
        height=new_flake.height - config.inner_rect_length_margin,
    )
    contacts_area = RotatedRect(
        center=new_flake.center, angle_rad=new_flake.angle_rad,
        width=new_flake.width + config.contacts_area_margin,
        height=new_flake.height + config.contacts_area_margin,
    )
    if inner.width <= 0 or inner.height <= 0:
        return None

    internal_local: list[list[Point]] = []
    connections: list[ContactConnection] = []
    arm_stubs_local: list[list[Point]] = []

    # --- Current contacts: 2, at the two ends along the length axis ---
    for sign in (1.0, -1.0):
        y0 = sign * inner.height / 2
        y1 = sign * (contacts_area.height / 2 - config.contact_end_setback)
        quad = tapered_quad((0.0, y0), (0.0, y1), new_flake.width, config.current_taper_width)
        internal_local.append(quad)
        connections.append(ContactConnection(
            position=new_flake.to_absolute(0.0, y1),
            direction=new_flake.direction_to_absolute(0.0, sign),
            width=config.current_taper_width,
            kind='current',
        ))

    # --- Voltage contacts: 6, 3 per long side, evenly spaced along length ---
    # 3 points spread across the inner rectangle's length at the 1/4, 2/4,
    # 3/4 marks (i.e. -L/4, 0, +L/4) -- evenly spaced without sitting right
    # on the inner rectangle's own corners.
    half_l = inner.height / 2
    taps_y = [-half_l / 2, 0.0, half_l / 2]
    tap_widths = [config.voltage_side_taper_width, config.voltage_center_taper_width, config.voltage_side_taper_width]

    # "If impossible, drop flake from hall" -- adjacent taps must not overlap:
    # each tap's cross-section (its taper width, since it runs along x) needs
    # at least half its own width plus half its neighbor's within the gap
    # between them.
    gap = taps_y[1] - taps_y[0]
    for w_a, w_b in zip(tap_widths, tap_widths[1:]):
        if gap < (w_a + w_b) / 2:
            return None

    for sign in (1.0, -1.0):
        x0 = sign * (inner.width / 2 + config.voltage_offset_from_inner_rect)
        x1 = sign * (contacts_area.width / 2 - config.contact_end_setback)
        x_inner_edge = sign * inner.width / 2
        for y, taper_w in zip(taps_y, tap_widths):
            quad = tapered_quad((x0, y), (x1, y), config.voltage_width, taper_w)
            internal_local.append(quad)
            connections.append(ContactConnection(
                position=new_flake.to_absolute(x1, y),
                direction=new_flake.direction_to_absolute(sign, 0.0),
                width=taper_w,
                kind='voltage',
            ))
            # Un-masked arm bridging the inner-rectangle edge to this
            # contact, centered on it -- "The arm length should be 0.8, and
            # it should be centered relative to the contact."
            arm_half = config.voltage_arm_length / 2
            arm_center_x = x_inner_edge  # centered ON the inner-rectangle edge, per "centered relative to the contact"
            arm_stubs_local.append([
                (arm_center_x - sign * arm_half, y - config.voltage_width / 2),
                (arm_center_x + sign * arm_half, y - config.voltage_width / 2),
                (arm_center_x + sign * arm_half, y + config.voltage_width / 2),
                (arm_center_x - sign * arm_half, y + config.voltage_width / 2),
            ])

    # --- Cut mask: contacts_area minus inner_rect minus all contact
    # footprints, plus the voltage arm stubs re-added ---
    internal_abs = [ShapelyPolygon([new_flake.to_absolute(x, y) for x, y in pts]) for pts in internal_local]
    arm_stubs_abs = [ShapelyPolygon([new_flake.to_absolute(x, y) for x, y in pts]) for pts in arm_stubs_local]

    cut_region = _rect_polygon(contacts_area).difference(_rect_polygon(inner))
    for shape in internal_abs:
        cut_region = cut_region.difference(shape)
    for stub in arm_stubs_abs:
        cut_region = cut_region.difference(stub)

    if collides(cut_region, occupied) or any(collides(s, occupied) for s in internal_abs):
        return None
    if not within_bounds(cut_region, chip_bounds) or any(not within_bounds(s, chip_bounds) for s in internal_abs):
        return None  # flake too close to the chip edge for even the inner geometry to fit

    # Deliberately NOT including cut_region here: an arm legitimately passes
    # over/through the cut-masked strip right next to its own contact's tip
    # (different CIF layers -- L041 cut vs L040 metal -- commonly overlap in
    # a real fabrication stack, e.g. a via cut with metal filling it). Still
    # includes internal_abs (the actual contact shapes) + the caller's
    # `occupied` so an arm can't cross through a DIFFERENT internal contact
    # or another flake's territory.
    external = place_external_contacts(
        connections, center=new_flake.center, work_area=config.work_area,
        pad_length=config.external_pad_width, pad_width=config.external_pad_height,
        current_arm_start_width=config.current_arm_start_width,
        voltage_arm_start_width=config.voltage_arm_start_width,
        min_pad_spacing=config.min_pad_spacing,
        min_parallel_voltage_pad_spacing=config.min_parallel_voltage_pad_spacing,
        occupied=unary_union([occupied] + internal_abs),
        chip_bounds=chip_bounds,
        # Same as Ring: no right angles in the routed arms -- user: "the
        # contact paths should like the rings be straight and with no right
        # angles." See contacts.py's own avoid_right_angles docstring.
        avoid_right_angles=True,
        # Current's own native direction (length axis, top/bottom) is
        # perpendicular to voltage's (width axis, left/right) -- rotate
        # current's own pads to voltage's direction instead so all 8 pads
        # share one fixed orientation. User: "the current rectangle should
        # be parallel to the voltage. that is the contact path should
        # connect to its long side" (see materials/debugging/hallExample.png).
        current_pad_direction=new_flake.direction_to_absolute(1.0, 0.0),
        # Pull the whole assembly in close around the flake instead of
        # always pinning it to the work-area edge -- user: "I want the
        # rectangles to be more closer together... like the image", plus a
        # new explicit cross-kind gap: "make the current rectangles
        # distance from the closest voltage rectangle 50" (reusing
        # min_pad_spacing's own value -- same "adjacent pads, any kind
        # pairing, >=50 apart" concept this field already represents
        # within a single group).
        minimize_radial_distance=True,
        current_voltage_min_gap=config.min_pad_spacing,
    )
    if external is None:
        return None

    return HallResult(cut_mask=cut_region, internal_contacts=internal_abs, external=external)
