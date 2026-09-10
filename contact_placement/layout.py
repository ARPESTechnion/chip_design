"""
Per-run orchestration: load every chosen flake across ALL images processed
by a grid_preperation run (`<grid_preperation_output_dir>/flakes/*.json` --
already real flake_detection contours+scores, transformed to absolute chip
um by grid_preperation's own calibration), try Hall then Ring placement for
each in descending score order, sharing ONE globally-growing `occupied`
state.

Global, not per-image, per the user's explicit direction: every image's
flake vertices already live in the SAME absolute chip coordinate system
(grid_preperation places each image at its own grid-derived position), so
`num_halls_to_try`/`num_rings_to_try` are whole-run budgets and a flake's
contacts must avoid every earlier flake's placed geometry AND every chosen
flake's own body -- not just flakes from the same source image.

Pure orchestration: hall.py/ring.py/contacts.py stay side-effect-free; this
module is the ONLY place `occupied` is grown, and only for flakes that
actually got a successful placement -- a flake dropped for both Hall and
Ring contributes nothing beyond its own seeded body extent (see `occupied`
seeding below) and is simply absent from the successful-placement output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from shapely.geometry import GeometryCollection, Polygon as ShapelyPolygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .cif_export import chip_bounds_from_cif, find_working_template
from .config import ContactPlacementConfig
from .geometry import ChipBounds, Point, largest_inscribed_rectangle
from .hall import HallResult, place_hall
from .ring import RingResult, place_ring


@dataclass
class FlakeCandidate:
    image: str
    flake_index: int
    vertices_um: list[Point]
    final_score: float


@dataclass
class PlacementResult:
    candidate: FlakeCandidate
    mode: Literal['hall', 'ring'] | None
    result: HallResult | RingResult | None


def _load_candidates(grid_preperation_output_dir: Path, min_flake_score: float) -> list[FlakeCandidate]:
    """All flakes from every `<output>/flakes/<image_stem>.json` (see
    grid_preperation.run's own writer) at or above min_flake_score, sorted
    by final_score descending -- the order Hall/Ring attempts are made in."""
    flakes_dir = grid_preperation_output_dir / 'flakes'
    candidates: list[FlakeCandidate] = []
    for flakes_path in sorted(flakes_dir.glob('*.json')):
        records = json.loads(flakes_path.read_text())
        for i, rec in enumerate(records):
            if rec['final_score'] < min_flake_score:
                continue
            candidates.append(FlakeCandidate(
                image=flakes_path.stem, flake_index=i,
                vertices_um=[tuple(p) for p in rec['vertices_um']],
                final_score=rec['final_score'],
            ))
    candidates.sort(key=lambda c: c.final_score, reverse=True)
    return candidates


def _committed_geometry(mode: Literal['hall', 'ring'], result: HallResult | RingResult) -> BaseGeometry:
    """Every shape a successful placement adds to `occupied` -- cut mask,
    internal contacts, and external pads/arms, plus Ring's arrow
    decorations (also real L040 geometry, must avoid later flakes too)."""
    pieces = [result.cut_mask, *result.internal_contacts]
    pieces += [p.pad for p in result.external] + [p.arm for p in result.external]
    if mode == 'ring':
        pieces += result.arrows
    return unary_union(pieces)


def place_all(config: ContactPlacementConfig) -> list[PlacementResult]:
    """
    Run Hall (up to config.num_halls_to_try attempts, in descending-score
    order, skipping flakes below config.min_flake_length_for_hall /
    min_flake_width_for_hall) then Ring
    (up to config.num_rings_to_try attempts, over whatever's left) across
    every qualifying flake from config.grid_preperation_output_dir. Returns
    one PlacementResult per candidate flake -- mode=None/result=None means
    dropped (never attempted, or attempted and failed both modes).
    """
    if config.grid_preperation_output_dir is None:
        raise ValueError("place_all: config.grid_preperation_output_dir is required")
    grid_dir = Path(config.grid_preperation_output_dir)

    # The chip's real extent, so a flake near the edge can't get an
    # external-pad layout that overshoots the physical die (confirmed as a
    # real, previously unconstrained gap). Derived from the working
    # template's CURRENT content -- at this point in a run that's just
    # grid_preperation's own markers + placed flakes, since contact_placement
    # itself hasn't written anything yet (see cif_export.chip_bounds_from_cif).
    # None (no constraint) if no single working template can be found --
    # never silently invent a boundary.
    chip_bounds: ChipBounds | None = None
    template = find_working_template(grid_dir)
    if template is not None:
        chip_bounds = chip_bounds_from_cif(template)
        if chip_bounds is not None:
            # Shrink inward by chip_border_margin_um on all four sides --
            # within_bounds (used by both place_hall and place_ring) already
            # rejects anything not entirely inside whatever bounds it's
            # given, so this is the only change needed to keep contacts off
            # a border strip near the real chip edge.
            minx, miny, maxx, maxy = chip_bounds
            m = config.chip_border_margin_um
            chip_bounds = (minx + m, miny + m, maxx - m, maxy - m)

    candidates = _load_candidates(grid_dir, config.min_flake_score)
    results = {id(c): PlacementResult(candidate=c, mode=None, result=None) for c in candidates}

    # Every chosen flake's own body, kept separate from `occupied_contacts`
    # (which only ever holds successfully PLACED geometry). A candidate's
    # own body must never be checked against ITSELF -- its cut mask/internal
    # contacts legitimately sit inside/at the edge of its own contour, which
    # is correct, not a collision (confirmed empirically: seeding a flake's
    # own body into its own `occupied` made every placement fail). Each
    # attempt instead gets occupied_contacts + every OTHER chosen flake's
    # body, so an early flake's contacts still can't intrude into a later,
    # not-yet-processed flake's body, without self-colliding.
    bodies: dict[int, BaseGeometry] = {id(c): ShapelyPolygon(c.vertices_um) for c in candidates}
    occupied_contacts: BaseGeometry = GeometryCollection()

    def occupied_for(candidate: FlakeCandidate) -> BaseGeometry:
        other_bodies = [b for cid, b in bodies.items() if cid != id(candidate)]
        return unary_union([occupied_contacts, *other_bodies]) if other_bodies else occupied_contacts

    hall_attempts = 0
    for candidate in candidates:
        if hall_attempts >= config.num_halls_to_try:
            break
        try:
            new_flake = largest_inscribed_rectangle(candidate.vertices_um)
        except ValueError:
            continue  # too small/degenerate to even measure -- not a real Hall attempt
        flake_length, flake_width = max(new_flake.width, new_flake.height), min(new_flake.width, new_flake.height)
        if flake_length < config.min_flake_length_for_hall or flake_width < config.min_flake_width_for_hall:
            continue  # below size gate -- not a real Hall attempt, doesn't consume the budget

        hall_attempts += 1
        hall_result = place_hall(candidate.vertices_um, config.hall, occupied_for(candidate), chip_bounds)
        if hall_result is not None:
            results[id(candidate)] = PlacementResult(candidate=candidate, mode='hall', result=hall_result)
            occupied_contacts = unary_union([occupied_contacts, _committed_geometry('hall', hall_result)])

    ring_attempts = 0
    for candidate in candidates:
        if results[id(candidate)].mode is not None:
            continue  # already placed via Hall
        if ring_attempts >= config.num_rings_to_try:
            break
        ring_attempts += 1
        ring_result = place_ring(candidate.vertices_um, config.ring, occupied_for(candidate), chip_bounds)
        if ring_result is not None:
            results[id(candidate)] = PlacementResult(candidate=candidate, mode='ring', result=ring_result)
            occupied_contacts = unary_union([occupied_contacts, _committed_geometry('ring', ring_result)])

    return [results[id(c)] for c in candidates]
