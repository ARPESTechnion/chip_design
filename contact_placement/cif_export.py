"""
Writes a layout.place_all() run's L040/L041/L042 geometry to a standalone
CIF file via grid_preperation.cif_io's shared low-level writer
(`_write_cif_text`), then splices it into the same working master template
grid_preperation already produced -- reusing `place_in_cif` as-is (it's
already generic to any file matching that writer's convention, needs no
changes for this).

Colors are the EXACT rgb BaseGrid.cif's own pre-existing L040/L041/L042
declarations already use (decoded from its CleWin hex comments -- green
(34,211,22)/blue (0,0,255)/red (249,28,14)), not just "a" blue/red/green,
so a standalone contact_placement CIF opened on its own still matches the
template's own convention. Since the target already declares these 3
layers, place_in_cif's own "insert missing header colors" step is a no-op
for them either way (see grid_preperation.cif_io._HEADER_COLOR_RE's fix --
without it, BaseGrid.cif's L040/L042 colors, which use a '05' CleWin
fill-style prefix instead of '0f', would have gone undetected as
"already declared" and gotten a duplicate, conflicting header line).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from shapely.geometry.base import BaseGeometry

try:
    from grid_preperation.cif_io import _ds_um_per_unit, _parse_ds, _write_cif_text, place_in_cif
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from grid_preperation.cif_io import _ds_um_per_unit, _parse_ds, _write_cif_text, place_in_cif

from .config import ContactPlacementConfig
from .geometry import ChipBounds, Point

if TYPE_CHECKING:
    # Deferred to avoid a real import cycle: layout.py needs
    # find_working_template/chip_bounds_from_cif from THIS module to derive
    # chip_bounds, while this module's write_placement_cif needs layout's
    # PlacementResult only as a type hint -- never at runtime.
    from .layout import PlacementResult

__all__ = ['write_placement_cif', 'place_in_cif', 'find_working_template', 'chip_bounds_from_cif']


def find_working_template(grid_preperation_output_dir: Path) -> Path | None:
    """The working master-template copy grid_preperation.run() makes sits
    directly in its own --output folder (named after the original template
    file), alongside -- not inside -- the cif/, flakes/, reconstructed/
    subfolders that same run also writes (see grid_preperation.run.run()).
    None if that's ambiguous (0 or >1 .cif files directly in the folder)."""
    candidates = [p for p in grid_preperation_output_dir.glob('*.cif') if p.is_file()]
    return candidates[0] if len(candidates) == 1 else None


_B_RE = re.compile(r'\bB\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*;')
_P_RE = re.compile(r'\bP\s+([\d\s\-]+?)\s*;', re.DOTALL)


def chip_bounds_from_cif(path: str | Path) -> ChipBounds | None:
    """
    The real chip/die extent, derived from the FULL geometric extent of an
    existing CIF file's own content (both `B` box commands -- e.g. the
    marker-dot grid -- and `P` polygon commands), converted through that
    file's own `DS` declaration. Used as a placement constraint so contacts
    can't land past the physical chip edge (confirmed as a real, previously
    unconstrained gap: a flake near the edge of the chip could get a
    work-area-sized external pad layout that overshoots the die boundary,
    since nothing previously checked against the chip's own real extent).

    Read this from the PRISTINE original template (or a working copy
    BEFORE any contact_placement geometry has been spliced into it) --
    calling this after contacts have already been written would just fold
    already-placed (possibly already-out-of-bounds) geometry into the
    "boundary", which defeats the purpose.

    None if the file has no B/P content at all (can't derive a boundary --
    never silently invent one; callers should treat None as "no known
    boundary, don't constrain placement").
    """
    text = Path(path).read_text()
    try:
        ds_scale, ds_length = _parse_ds(text)
    except ValueError:
        return None
    um_per_unit = _ds_um_per_unit(ds_scale, ds_length)

    xs: list[float] = []
    ys: list[float] = []
    for m in _B_RE.finditer(text):
        w, h, cx, cy = (int(v) for v in m.groups())
        xs += [(cx - w / 2) * um_per_unit, (cx + w / 2) * um_per_unit]
        ys += [(cy - h / 2) * um_per_unit, (cy + h / 2) * um_per_unit]
    for m in _P_RE.finditer(text):
        nums = [int(v) for v in m.group(1).split()]
        xs += [n * um_per_unit for n in nums[0::2]]
        ys += [n * um_per_unit for n in nums[1::2]]

    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _drop_close_duplicate(coords: list[Point]) -> list[Point]:
    return coords[:-1] if coords and coords[0] == coords[-1] else coords


def _keyhole_path(polygon) -> list[Point]:
    """
    Flatten one shapely Polygon (exterior + 0 or more interior rings/holes)
    into a SINGLE simple path via the standard "keyhole"/bridge technique --
    CIF's P command can only express one simple ring per polygon, no holes,
    so a donut shape (e.g. Ring's ring-arms cut-mask frame: an outer square
    minus a smaller concentric inner square, exactly a square annulus) needs
    this to render as an actual ring instead of silently becoming a solid
    filled box (confirmed empirically: an earlier version just took
    `.exterior.coords` and dropped every `.interiors` hole entirely --
    exactly what a user reported seeing: "the ring itself is missing, there
    is only a box").

    For each hole, cut a zero-width slit from the nearest exterior vertex to
    the nearest hole vertex, walk the whole hole ring, then cut back --
    traversed twice (out and back), that slit contributes zero net area and
    is safe under both nonzero and even-odd fill rules. shapely's own
    exterior/interior ring winding convention (exterior CCW, holes CW, i.e.
    already opposite) means no manual reversal is needed for the fill rule
    to correctly treat the hole as a subtraction.
    """
    path = _drop_close_duplicate(list(polygon.exterior.coords))
    for interior in polygon.interiors:
        hole = _drop_close_duplicate(list(interior.coords))
        if not hole:
            continue
        best = min(
            ((pi, hj) for pi in range(len(path)) for hj in range(len(hole))),
            key=lambda ij: (path[ij[0]][0] - hole[ij[1]][0]) ** 2 + (path[ij[0]][1] - hole[ij[1]][1]) ** 2,
        )
        pi, hj = best
        hole_loop = hole[hj:] + hole[:hj + 1]  # starts and ends at the bridge point -> full loop
        path = path[:pi + 1] + hole_loop + [path[pi]] + path[pi + 1:]
    return path


def _polygon_vertex_lists(geom: BaseGeometry) -> list[list[Point]]:
    """Flatten a shapely geometry (a single Polygon, or a MultiPolygon/
    GeometryCollection of them -- e.g. Ring's cut_mask, a union of the inner
    ring square + the ring-arm frame, can come back as either depending on
    whether they touch) into one simple vertex path per polygon (holes
    merged in via _keyhole_path), with the closing duplicate vertex dropped
    (CIF's own P command implicitly closes the ring, matching
    grid_preperation.polygons.Polygon.vertices_um's own convention --
    _write_cif_text expects exactly this shape)."""
    geoms = geom.geoms if hasattr(geom, 'geoms') else [geom]
    result: list[list[Point]] = []
    for g in geoms:
        if g.is_empty:
            continue
        result.append(_keyhole_path(g))
    return result


def write_placement_cif(results: list[PlacementResult], config: ContactPlacementConfig, path: str | Path) -> None:
    """One standalone CIF covering every successfully placed flake in
    `results` (mode=None entries -- dropped flakes -- are skipped), on the
    3 fixed layers from config.layer_cut/layer_internal/layer_external."""
    layers: dict[int, list[list[Point]]] = {
        config.layer_cut: [], config.layer_internal: [], config.layer_external: [],
    }
    for r in results:
        if r.mode is None:
            continue
        layers[config.layer_cut] += _polygon_vertex_lists(r.result.cut_mask)
        for contact in r.result.internal_contacts:
            layers[config.layer_internal] += _polygon_vertex_lists(contact)
        for placed in r.result.external:
            layers[config.layer_external] += _polygon_vertex_lists(placed.pad)
            layers[config.layer_external] += _polygon_vertex_lists(placed.arm)
        if r.mode == 'ring':
            for arrow in r.result.arrows:
                layers[config.layer_external] += _polygon_vertex_lists(arrow)

    layer_colors = {
        config.layer_cut: (0, 0, 255),
        config.layer_internal: (249, 28, 14),
        config.layer_external: (34, 211, 22),
    }
    text = _write_cif_text(layers, layer_colors, structure_name='Contacts', class_label='Contact')
    Path(path).write_text(text)
