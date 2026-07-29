"""
Configuration for the electrical-contact / cutting-mask placement stage.

All geometry constants below are transcribed directly from
contact_placement/materials/prompt.docx (the spec document, already cross-
checked against the real example CIFs/images also in that folder) -- see
each field's docstring for the literal spec sentence it corresponds to.
Units are the same real micron ("um") coordinates every other stage in this
pipeline already works in (grid_preperation's exported CIF vertices,
grid_pitch_um, etc.) -- these are physical contact-pad/arm dimensions for
e-beam lithography, not pixels.

Mirrors the dataclass + JSON save/load convention used by flake_detection.config
and grid_preperation.config.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


def parse_range(value: str) -> list[float]:
    """
    Parse a "single number or a range" user-input field -- per the spec's
    own UI section: "Ranges are separated with comma". `"5"` -> `[5.0]`,
    `"4,5,6"` -> `[4.0, 5.0, 6.0]`. Values are tried in the given order
    (e.g. by hall.py/ring.py when searching for a size that satisfies all
    placement constraints) -- this function does no sorting/deduplication of
    its own so the caller's tried-order matches what the user typed.
    """
    return [float(v.strip()) for v in value.split(',') if v.strip()]


@dataclass
class HallConfig:
    """Hall-bar placement geometry -- see prompt.docx's "Hall" section."""

    inner_rect_width_margin: float = 3.0
    inner_rect_length_margin: float = 4.0
    """Inner rectangle = new-flake rectangle shrunk by (width_margin,
    length_margin), centered. "Inner rectangle: new flake width minus 3, new
    flake length minus 4, placed in center." """

    min_length_to_width_ratio: float = 2.0
    """"The length most be at least 2 times the width. Else give up contour
    for hall." Applied to the new-flake rectangle (the largest rectangle
    inscribed in the flake's own contour)."""

    contacts_area_margin: float = 20.0
    """Contacts area = new-flake rectangle grown by this on each side,
    centered on the new flake. "Contacts area: a rectangle new flake size
    plus 20." """

    current_taper_width: float = 15.0
    """Each of the 2 current contacts (one per long side of the new flake)
    starts at the new-flake width and tapers OUT to this width as it moves
    away from the flake. "They should get wider to 15 as it distant from the
    flake." """

    contact_end_setback: float = 10.0
    """Both current and voltage contacts run from the inner rectangle to
    this far short of the contacts area boundary. "There length should be
    from the inner rectangle until 10 from the contacts area." (stated
    separately for current and voltage in the doc, same value both times)."""

    voltage_width: float = 1.0
    """Voltage contact width at its narrow (inner) end. "Their width should
    be 1." """

    voltage_offset_from_inner_rect: float = 0.5
    """Each voltage contact sits this far outside the inner rectangle's own
    edge. "There distance from the inner rectangle is 0.5 outside." """

    voltage_center_taper_width: float = 3.0
    voltage_side_taper_width: float = 5.0
    """3 voltage contacts per short side, evenly spaced; the center one
    tapers to voltage_center_taper_width, the two side ones taper to
    voltage_side_taper_width. "The center contacts should get wider to 3,
    and the side ones should get wider to 5." """

    voltage_arm_length: float = 0.8
    """Within the cut-masked gap between contacts area and inner rectangle,
    a small un-masked arm is left behind each voltage contact, centered on
    it. "The arm length should be 0.8, and it should be centered relative to
    the contact." """

    external_pad_width: float = 300.0
    external_pad_height: float = 100.0
    num_current_pads: int = 2
    num_voltage_pads: int = 6
    """8 total external pads (2 current + 6 voltage), each 300x100um. "8
    300X100 rectangles. 2 for currant and 6 for voltage." """

    min_pad_spacing: float = 50.0
    min_parallel_voltage_pad_spacing: float = 150.0
    """"The distance between 2 adjacent rectangles should be at least 50.
    Distance between 2 parallal voltage rectangles should be at least 150." """

    work_area: float = 1000.0
    """Square region the 8 external pads must fit within, centered on the
    flake. The doc states "900X900" once in the restrictions section but
    "1000X1000" in the algorithm section for Hall specifically -- confirmed
    with the user that 1000 is correct for Hall (the 900 mention there was a
    typo); Ring genuinely uses 900 (see RingConfig.work_area)."""

    current_arm_start_width: float = 35.0
    voltage_arm_start_width: float = 20.0
    """Width of each external-pad-connecting arm at its pad end; the other
    end matches whichever inner contact it connects to. "Current arms start
    width 35, voltage arms starts width 20." """


@dataclass
class RingConfig:
    """Ring placement geometry -- see prompt.docx's "Ring" section."""

    voltage_min_distance_from_mask: float = 0.7
    """Minimum gap between each voltage contact and the inner-ring cut mask.
    "There is a minimal 0.7 distance between the voltage and the mask." """

    current_min_distance_from_voltage: float = 1.4
    """Minimum gap between each current contact and its adjacent voltage
    contact. "And a minimal 1.4 between the currant and its adjacent
    voltage." """

    current_min_length_in_blocking_rect: float = 1.5
    voltage_min_width_in_blocking_rect: float = 0.7
    """Hard minimums for the portion of each contact inside the blocking
    rectangle; drop the flake for Ring if these can't be met. "The current
    length inside the blocking rectangle should be 1.5 at least. The voltage
    width inside the blocking rectangle should be minimum 0.7. If impossible
    drop." """

    current_grow_cap: float = 3.0
    """When there's room to grow the inner contacts beyond their minimums:
    grow current length first, up to this cap, before scaling everything
    else. "To increase them inside the blocking rectangle (if there is
    space) follow this logic: first increase the current up to 3, then
    increase all contacts sizes (including distances) by the same
    proportions." """

    voltage_stackout: float = 2.0
    voltage_stackout_connecting_side: float = 5.0
    """How far each voltage contact sticks out past the blocking rectangle's
    own edge -- 2 normally, 5 on the side it connects outward to the
    external contacts from. "The voltage contacts should stack out 2 from
    the blocking rectangle and at 5 on the side where they are connected to
    the external contacts." """

    current_min_stackout: float = 6.0
    """"The current should stack out at least 6 from the blocking
    rectangle." """

    external_pad_width: float = 300.0
    external_pad_height: float = 100.0
    num_current_pads: int = 2
    num_voltage_pads: int = 2
    """4 total external pads (2 current + 2 voltage). "4 rectangles 300X100.
    2 for current 2 for voltage." """

    min_pad_spacing: float = 50.0
    min_parallel_voltage_pad_spacing: float = 150.0
    work_area: float = 900.0
    """Same spacing rules as Hall; Ring's work area is genuinely 900x900
    (only Hall's is the confirmed-1000 exception -- see HallConfig.work_area)."""

    current_arm_start_width: float = 16.0
    voltage_arm_start_width: float = 12.0

    arrow_base: float = 40.0
    arrow_height: float = 20.0
    """Isosceles-triangle "arrow" decoration between each voltage/current
    external pad pair. "There should be an arrow between each voltage to
    current: isosceles triangle with base 40 and height 20." """

    inner_ring_dim: str = "5"
    arm_distance: str = "3"
    """User input, single value or comma-separated range (spec: "Inner ring
    dim (can be single number or a range), distance from arm rings (can be
    single number or a range)"). Parse with parse_range(). Defaults here are
    placeholders -- there's no spec-given default, set these per your actual
    chip design before running."""


@dataclass
class ContactPlacementConfig:
    """
    Full configuration for placing electrical contacts + cutting masks onto
    already-exported flakes.

    Pipeline position: runs AFTER grid_preperation, reading its
    <output>/flakes/<image>.json -- flake_detection's own real per-flake
    contour + score (see flake_detection.detector.write_contours_json),
    transformed into absolute um by grid_preperation via that image's own
    calibration (see CifExportConfig.flake_detection_output_dir) -- and
    splices its own geometry into the SAME working master template
    grid_preperation already produced.
    """

    # ------------------------------------------------------------------ #
    # CIF layers -- FIXED numbers, not dynamically allocated. Confirmed    #
    # these are already declared (with matching colors) in                #
    # grid_preperation/BaseGrid.cif's own header -- an established        #
    # convention for this exact purpose, not a coincidence.               #
    # ------------------------------------------------------------------ #

    layer_cut: int = 41
    """L041, blue -- cutting masks ("gentle cut")."""

    layer_internal: int = 42
    """L042, red -- internal current/voltage contacts ("gentle writing")."""

    layer_external: int = 40
    """L040, green -- external bond pads + connecting arms ("rough
    writing")."""

    # ------------------------------------------------------------------ #
    # "Chosen flake" filtering, applied to grid_preperation's             #
    # <output>/flakes/<image>.json (see module docstring)                 #
    # ------------------------------------------------------------------ #

    min_flake_score: float = 0.3
    """A flake's real flake_detection score (see module docstring -- this is
    flake_detection's own final_score, transformed through calibration by
    grid_preperation, not re-derived) must be at least this to be considered
    a real, contact-worthy flake at all. No separate marker-exclusion filter
    is needed here -- flake_detection already excludes marker/grid-dot
    regions before scoring (marker_removal_enabled), so anything in
    <output>/flakes/<image>.json is already a real flake candidate."""

    # ------------------------------------------------------------------ #
    # User-tunable run parameters (the "popping UI" sets these)           #
    # ------------------------------------------------------------------ #

    num_halls_to_try: int = 10
    """"Number of halls to try to rich" -- cap on how many flakes get a Hall
    placement attempt (in score order) before the rest fall through to Ring."""

    min_flake_size_for_hall: float = 10.0
    """"min flake size for hall" -- new-flake rectangle area (um^2) below
    which Hall isn't attempted for that flake."""

    num_rings_to_try: int = 10
    """"Number of rings to try to rich" -- cap on how many remaining
    (non-Hall) flakes get a Ring placement attempt."""

    # ------------------------------------------------------------------ #
    # Mode-specific geometry                                              #
    # ------------------------------------------------------------------ #

    hall: HallConfig = field(default_factory=HallConfig)
    ring: RingConfig = field(default_factory=RingConfig)

    # ------------------------------------------------------------------ #
    # I/O                                                                  #
    # ------------------------------------------------------------------ #

    grid_preperation_output_dir: Optional[str] = None
    """A grid_preperation run's own --output folder (contains cif/, flakes/,
    and the working template copy) -- this stage's input. Required at run
    time; left None here since it's genuinely per-run, like
    grid_preperation's own --input_folder."""

    output_dir: Optional[str] = None
    """Destination for this stage's own results. Defaults to
    <grid_preperation_output_dir>/contact_placement_results/."""

    # ------------------------------------------------------------------ #
    # Serialization helpers                                               #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Persist config to JSON for reproducibility."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> ContactPlacementConfig:
        """Load config from a JSON file. Keys prefixed with '_' are comments."""
        with open(path) as f:
            data = json.load(f)
        data = {k: v for k, v in data.items() if not k.startswith('_')}
        if 'hall' in data:
            data['hall'] = HallConfig(**{k: v for k, v in data['hall'].items() if not k.startswith('_')})
        if 'ring' in data:
            data['ring'] = RingConfig(**{k: v for k, v in data['ring'].items() if not k.startswith('_')})
        return cls(**data)
