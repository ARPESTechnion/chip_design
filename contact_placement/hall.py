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

from shapely.geometry import GeometryCollection, Polygon as ShapelyPolygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .config import HallConfig
from .contacts import ContactConnection, PlacedContact, place_external_contacts
from .geometry import (
    ChipBounds, Point, RotatedRect, collides, largest_inscribed_rectangle, within_bounds,
)


def _local_rect(x0: float, x1: float, y0: float, y1: float) -> list[Point]:
    """A plain axis-aligned rectangle in local (flake-frame) coordinates,
    same 4-point convention as tapered_quad's output."""
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


@dataclass
class HallResult:
    cut_mask: BaseGeometry           # L041 -- may be a MultiPolygon
    internal_contacts: list[ShapelyPolygon]  # L042 -- 2 current + 6 voltage
    external: list[PlacedContact]     # L040 -- 8 pads + arms (see contacts.py)


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
    # Same rectangles as internal_local, grouped by which contact they came
    # from -- used below as the cut mask's own hole footprint. No dilation:
    # the mask sits flush against every contact edge, zero gap (user,
    # after reviewing a render with the earlier 0.1/1um keepout dilation:
    # "the internal contacts and the mask should have no gap between them").
    current_rects_local: dict[float, list[Point]] = {}
    voltage_rects_local: dict[float, list[list[Point]]] = {1.0: [], -1.0: []}

    # --- Current contacts: 2, at the two ends along the length axis, flush
    # against the inner rectangle -- plain rectangles, constant width =
    # new_flake.width (no taper; see materials/debugging/single_hall.cif).
    # The far end extends PAST the contacts area by contact_end_setback
    # (not short of it -- user: "current length is also the gap between the
    # inner rectangle and the contact area + 10"), so the external arm's own
    # connection point sits well beyond the cut mask's own domain (which
    # stays bounded by contacts_area, see below), naturally keeping the
    # internal-to-external connection far from the mask without needing any
    # special routing exclusion. ---
    for sign in (1.0, -1.0):
        y0 = sign * inner.height / 2
        y1 = sign * (contacts_area.height / 2 + config.contact_end_setback)
        half_w = new_flake.width / 2
        rect = _local_rect(-half_w, half_w, min(y0, y1), max(y0, y1))
        internal_local.append(rect)
        current_rects_local[sign] = rect
        connections.append(ContactConnection(
            position=new_flake.to_absolute(0.0, y1),
            direction=new_flake.direction_to_absolute(0.0, sign),
            width=new_flake.width,
            kind='current',
        ))

    # --- Voltage contacts: 6, 3 per long side, evenly spaced along length,
    # flush against the inner rectangle -- plain rectangles, constant width
    # = config.voltage_width (no more separate side/center taper widths). ---
    # 3 points spread across the inner rectangle's length at the 1/4, 2/4,
    # 3/4 marks (i.e. -L/4, 0, +L/4) -- evenly spaced without sitting right
    # on the inner rectangle's own corners; equidistant from each other AND
    # from the current contacts at the ends (both gaps are inner.height/4).
    half_l = inner.height / 2
    taps_y = [-half_l / 2, 0.0, half_l / 2]

    # "If impossible, drop flake from hall" -- adjacent taps must not overlap.
    gap = taps_y[1] - taps_y[0]
    if gap < config.voltage_width:
        return None

    for sign in (1.0, -1.0):
        # Contact's own near edge retreats mask_voltage_gap from the inner
        # rectangle (a real, visible gap to the mask on the flake-facing
        # side only -- user: "the internal contacts need to be 0.1 shorter
        # then the mask on the flake side (this is the mask-voltage
        # distances discussed)"). The HOLE used to cut the mask, in
        # contrast, extends mask_voltage_gap PAST the strip's own inner
        # edge (into inner-rectangle territory the strip never covers
        # anyway, so harmless) rather than landing exactly ON it -- an
        # exact boundary match there left a zero-width "pinch" connecting
        # what should be separate mask pieces (confirmed: a real exported
        # polygon whose ring walked out along the strip's own edge and
        # back, rendering as a thin connecting line the user flagged
        # directly). Both use the same constant since they're the same
        # physical gap, just measured from opposite sides of it.
        x0_contact = sign * (inner.width / 2 + config.mask_voltage_gap)
        x0_hole = sign * (inner.width / 2 - config.mask_voltage_gap)
        # Same "extend past contacts_area by contact_end_setback" rule as
        # current -- user: "the voltage contact length is the difference
        # between the contact area and the inner rectangle... + 10 (outside
        # of the contact area)".
        x1 = sign * (contacts_area.width / 2 + config.contact_end_setback)
        half_v = config.voltage_width / 2
        for y in taps_y:
            rect = _local_rect(min(x0_contact, x1), max(x0_contact, x1), y - half_v, y + half_v)
            internal_local.append(rect)
            voltage_rects_local[sign].append(_local_rect(min(x0_hole, x1), max(x0_hole, x1), y - half_v, y + half_v))
            connections.append(ContactConnection(
                position=new_flake.to_absolute(x1, y),
                direction=new_flake.direction_to_absolute(sign, 0.0),
                width=config.voltage_width,
                kind='voltage',
            ))

    def _abs(pts: list[Point]) -> ShapelyPolygon:
        return ShapelyPolygon([new_flake.to_absolute(x, y) for x, y in pts])

    internal_abs = [_abs(pts) for pts in internal_local]

    # --- Cut mask: NOT one frame-shaped region -- a collection of separate,
    # non-touching rectangles (user: "the cut mask is not a single polygon,
    # but a collection of rectangles that dont touch each other, as in the
    # examples"). Only exists in the LEFT/RIGHT margins (contacts_area's own
    # edge in to the inner rectangle's edge, full contacts_area HEIGHT) --
    # there's no separate top/bottom strip at all, since current's own
    # contact already spans that whole territory itself (its width covers
    # essentially the full inner region there, see materials/debugging/
    # single_hall.cif's own L041 layer: all 8 real mask pieces sit in the
    # left/right margins, none above/below current). Differencing current's
    # 2 contacts AND voltage's own 3-per-side contacts out of each
    # full-height margin strip naturally splits it into several disjoint
    # simple pieces (confirmed: KK -> 4 pieces per side, 0 interior rings
    # each, matching single_hall.cif's own 4-per-side split into 2 small
    # tap-gap pieces + 2 large corner pieces) -- no manual per-gap
    # construction needed. Mask sits flush against every contact edge (see
    # internal_local above) -- zero gap, per the user's own review.
    #
    # This ALSO fixes the single-merged-frame bug an earlier version of this
    # function had: subtracting every hole from one big "contacts_area minus
    # inner" frame let them all merge with the frame's own inner-rectangle
    # hole into one deeply concave ~37-vertex ring, which cif_export's
    # keyhole bridge-line technique couldn't flatten correctly -- confirmed
    # as a real bug: the mask rendered as a solid block completely covering
    # the contacts in the actual exported CIF (materials/debugging/
    # external_contacts_covered.png). Building only left/right (never a
    # frame that fully encloses the inner rectangle) sidesteps that
    # entirely: every piece here is simple from the start.
    half_ca_h = contacts_area.height / 2
    half_in_w = inner.width / 2
    current_rects_abs = [_abs(current_rects_local[1.0]), _abs(current_rects_local[-1.0])]
    mask_pieces: list[ShapelyPolygon] = []
    for sign in (1.0, -1.0):
        x0 = sign * half_in_w
        x1 = sign * (contacts_area.width / 2)
        strip = _abs(_local_rect(min(x0, x1), max(x0, x1), -half_ca_h, half_ca_h))
        for hole in [*current_rects_abs, *[_abs(pts) for pts in voltage_rects_local[sign]]]:
            strip = strip.difference(hole)
        # Zero-gap differencing can leave degenerate slivers (a notch
        # boundary landing exactly on the strip's own edge or another
        # notch) as zero-area LineStrings/Points rather than Polygons --
        # filter those out rather than pass them to the CIF writer, which
        # expects every piece to have a real .exterior.
        mask_pieces += [
            g for g in (strip.geoms if hasattr(strip, 'geoms') else [strip])
            if g.geom_type == 'Polygon' and g.area > 1e-6
        ]

    cut_region = GeometryCollection(mask_pieces)

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
