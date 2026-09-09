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


def frange(min_val: float, max_val: float, step: float) -> list[float]:
    """
    Inclusive stepped range [min_val, min_val+step, ..., max_val], per the
    updated spec's explicit "X - Y with Z jumps" phrasing for Ring's
    inner_ring/outer_ring fields (e.g. "Inner ring: 0.7 - 1 with 0.05
    jumps" -> frange(0.7, 1.0, 0.05)). Ascending order (smallest first),
    matching how place_ring searches candidates. Index-based (not repeated
    float addition) to avoid floating-point drift over many steps; rounded
    to 10 decimal places to avoid spurious values like 0.7999999999999999.
    """
    if step <= 0:
        raise ValueError(f"frange: step must be positive, got {step}")
    n = round((max_val - min_val) / step)
    return [round(min_val + i * step, 10) for i in range(n + 1)]


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
    """Contacts area = new-flake rectangle with each dimension grown by this
    (width+20, length+20; NOT doubled/per-side -- same single-addition
    convention as inner_rect_width_margin/inner_rect_length_margin's
    "minus" phrasing), centered on the new flake. "Contacts area: a
    rectangle new flake size plus 20." """

    contact_end_setback: float = 10.0
    """Both current and voltage contacts extend from the inner rectangle to
    this far PAST the contacts area boundary (not short of it) -- user:
    "current length is also the gap between the inner rectangle and the
    contact area + 10" / "voltage contact length is the difference between
    the contact area and the inner rectangle... + 10 (outside of the
    contact area)". Puts the external arm's own connection point well
    beyond the cut mask's own domain (which stays bounded by contacts_area),
    keeping that connection naturally far from the mask."""

    voltage_width: float = 1.0
    """Constant cross-sectional width of every voltage contact (all 3 per
    side, side and center alike -- no taper). Reverse-engineered from
    materials/debugging/single_hall.cif's own measured geometry (a plain
    rectangle, not a taper -- supersedes the earlier
    voltage_center_taper_width/voltage_side_taper_width/current_taper_width
    design, which tapered each contact wider away from the flake). Current
    contacts have no equivalent field -- their constant width is always
    new_flake.width directly, see hall.py."""

    mask_voltage_gap: float = 0.1
    """Gap, on the flake-facing (near/inner-rectangle) side only, between a
    voltage contact's own edge and the cut mask's own edge there -- user:
    "the internal contacts need to be 0.1 shorter then the mask on the
    flake side (this is the mask-voltage distances discussed)." Current has
    no equivalent field -- its own near edge sits flush with the mask,
    no gap, since it doesn't have the pinch problem this field also fixes
    (see hall.py)."""

    min_voltage_distance: float = 1.0
    """Minimum real gap between two adjacent voltage contacts' own facing
    edges (the 3 taps along one long side) -- user: "the distance between
    voltages should be at least 1." hall.py's own tap spacing check used
    to only require the two taps' CENTERS be voltage_width apart (i.e.
    just barely non-overlapping, a real gap of 0 was allowed); now
    requires centers be voltage_width + min_voltage_distance apart, so the
    actual edge-to-edge gap is always >= this value. Same "drop the flake
    from Hall if impossible" behavior as the old overlap check -- see
    hall.py's own comment there."""

    external_pad_width: float = 300.0
    external_pad_height: float = 100.0
    num_current_pads: int = 2
    num_voltage_pads: int = 6
    """8 total external pads (2 current + 6 voltage), each 300x100um. "8
    300X100 rectangles. 2 for currant and 6 for voltage." """

    min_pad_spacing: float = 50.0
    min_parallel_voltage_pad_spacing: float = 50.0
    """"The distance between 2 adjacent rectangles should be at least 50.
    Distance between 2 parallal voltage rectangles should be at least 50."
    (Updated spec value -- was "at least 150", matching Ring's own field
    before ITS update -- see RingConfig.min_parallel_voltage_pad_spacing.)"""

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

    inner_ring_min: float = 0.7
    inner_ring_max: float = 1.0
    inner_ring_step: float = 0.05
    """Search range for the inner ring (the solid cut-mask square)'s own
    dimension. Per the spec's updated "Ring:" measurements section: "Inner
    ring: 0.7 - 1 with 0.05 jumps" -- searched via config.frange(), smallest
    first. The reference file (materials/ring_example_with_measurments.cif)
    used 0.8, one valid step within this range -- used to derive/verify the
    exact cut-mask formula in _ring_cut_mask (confirmed byte-exact,
    symmetric-difference-zero, against that file's own polygon vertices)."""

    outer_ring_min: float = 0.25
    outer_ring_max: float = 0.3
    outer_ring_step: float = 0.025
    """Search range for the outer ring: the width of the BLANK (uncut) gap
    between the inner ring's own edge and where the arm/finger assembly
    begins -- this gap is what leaves a real, visible RING of uncut flake
    material around the inner square, the whole point of the shape
    (confirmed with the user after an earlier version fused the arms
    directly onto the square, no gap, and was flagged as wrong: "this is a
    single square with arms. the inner ring is missing"). Per the spec's
    updated measurements: "Outer ring: 0.25 - 0.3 with 0.025 jumps" --
    searched via config.frange(). The reference file used 0.2 (predates
    this exact range being pinned down; the shape formula itself was
    verified against that file regardless, see inner_ring_min above)."""

    distance_between_arms: float = 0.5
    """Fixed (not a search range) gap between the two arm/finger
    assemblies' own innermost edges, along the blocking rectangle's length
    axis. "Distance between arms: 0.5 constant." Confirmed by direct
    measurement of a C5Meas annotation in
    materials/ring_example_with_measurments.cif spanning exactly this
    gap."""

    arm_width: float = 0.7
    """Fixed cross-sectional (tangential) width of each arm -- NOT a search
    range. "Arm width: 0.7 constant." Also sets how far, along the length
    axis, the combined arm+finger assembly reaches inward toward the
    center: `distance_between_arms/2 + arm_width` (see _ring_cut_mask's
    `reach`) -- confirmed as one value driving both roles by reverse-
    engineering materials/ring_example_with_measurments.cif's real polygon
    vertices (verified byte-exact, symmetric-difference area = 0)."""

    finger_width: float = 0.35
    """Fixed tangential width the "finger" adds BEYOND the outer ring's own
    edge (i.e. the finger reaches `outer_ring + finger_width` from the
    inner ring's center, tangentially -- wider than the arm/outer-ring
    assembly, but confined to a narrow band along the length axis, see
    _ring_cut_mask's `big_rect`/`notch` construction for the exact
    formula). "Finger width: 0.35 constant." Reverse-engineered by
    scanline analysis of materials/ring_example_with_measurments.cif's
    real polygon vertices and verified byte-exact (symmetric-difference
    area = 0) before being wired in."""

    arm_protrusion_beyond_blocking: float = 3.0
    """How far each cut-mask arm's far tip extends past the blocking
    rectangle's own edge (along the width axis the arms run on -- see
    _ring_cut_mask). User: "the arms should stick out 3 um beyond the
    blocking rectangle." Sets _try_ring's far_reach directly:
    `blocking.width/2 + arm_protrusion_beyond_blocking` (replaces the
    earlier `mask_half + v_dist` placeholder, which had no real spec basis
    -- the updated 5-field Ring measurements section has no arm-length
    parameter of its own, so this was reusing voltage_min_distance_from_mask
    as a stand-in until the user specified the real rule)."""

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

    voltage_grow_cap: float = 3.0
    """Hard cap on the voltage contact's own grown width
    (voltage_min_width_in_blocking_rect, after the proportional growth step)
    -- user: "make the maximum size of internal voltage width to be 3."
    Not itself in prompt.docx (which only caps current); when the
    proportional-growth step would push v_len past this, the excess is
    redistributed to v_dist/c_dist instead (same "grow up to a cap, then
    redistribute the rest" pattern as current_grow_cap) -- see
    _grow_contacts."""

    voltage_stackout: float = 2.0
    voltage_stackout_connecting_side: float = 5.0
    """How far each voltage contact sticks out past the blocking rectangle's
    own WIDTH edge (the long edge, running along the length axis -- "vertically
    to its length") -- 2 on the non-connecting side, 5 on the side it connects
    outward to the external contacts from. "The voltage contacts should stack
    out 2 from the blocking rectangle vertically to its length and at 5 on the
    side where they are connected to the external contacts." """

    current_min_stackout: float = 6.0
    """Unlike voltage (which exits sideways along the WIDTH axis, see
    voltage_stackout), each current contact exits along the LENGTH axis --
    straight out past the blocking rectangle's own length-axis end, in the
    same sign direction (north/south) it's already stacked in, rather than
    bending toward exit_sign. This is how far past that end it extends on
    its own connecting side. "The current should stack out at least 6 from
    the blocking rectangle, on the side where they are connected to the
    external contacts. This side is along the rectangle's length." Confirmed
    directly by the user after an earlier version exited current sideways
    like voltage: "the current external connection should connect to the
    internal along the flake's length axis, and should stick out more from
    this direction." The current contact's own cross-sectional width
    (perpendicular to this travel direction) is the blocking rectangle's own
    width plus 1 -- "the width of the current contacts should be the same
    as the blocking rectangle's width" then "lets change the width to be 1
    larger then the blocking rectangle" (an earlier version reused
    current_min_length_in_blocking_rect's own value here instead, flagged as
    too small)."""

    external_pad_width: float = 300.0
    external_pad_height: float = 100.0
    num_current_pads: int = 2
    num_voltage_pads: int = 2
    """4 total external pads (2 current + 2 voltage). "4 rectangles 300X100.
    2 for current 2 for voltage." """

    min_pad_spacing: float = 50.0
    min_parallel_voltage_pad_spacing: float = 50.0
    """Distance between 2 adjacent pads, and between 2 parallel voltage
    pads specifically, are both 50 for Ring -- user: "Distance between 2
    parallel voltage external rectangles should be also 50" (overrides
    prompt.docx's own "at least 150" for Ring; Hall's own equivalent field,
    HallConfig.min_parallel_voltage_pad_spacing, is untouched, still 150)."""

    work_area: float = 900.0
    """Same spacing rules as Hall; Ring's work area is genuinely 900x900
    (only Hall's is the confirmed-1000 exception -- see HallConfig.work_area)."""

    current_arm_start_width: float = 20.0
    voltage_arm_start_width: float = 16.0
    """Width of each external-pad-connecting arm at its pad end; the other
    end matches whichever inner contact it connects to. "Current arms start
    width 20, voltage arms start width 16." (Updated spec value -- was
    16/12.)"""

    arrow_base: float = 40.0
    arrow_height: float = 20.0
    """Isosceles-triangle "arrow" decoration between each voltage/current
    external pad pair. "There should be an arrow between each voltage to
    current: isosceles triangle with base 40 and height 20." """



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
    # Chip geometry constraints                                           #
    # ------------------------------------------------------------------ #

    chip_border_margin_um: float = 400.0
    """
    No contact/cutting-mask geometry may be placed within this distance (um)
    of the chip/die's own real edge. Applied by layout.place_all by shrinking
    ChipBounds (see geometry.ChipBounds / cif_export.chip_bounds_from_cif) by
    this margin on all four sides before passing it to place_hall/place_ring
    -- both already reject any placement that doesn't fit entirely within
    whatever bounds they're given (geometry.within_bounds), so this is the
    only change needed to keep contacts away from the border, not a
    separate check. No effect if chip_bounds_from_cif itself returns None
    (no working template found, or it has no B/P content to derive an
    extent from) -- never silently invents a boundary to shrink.
    """

    # ------------------------------------------------------------------ #
    # User-tunable run parameters (the "popping UI" sets these)           #
    # ------------------------------------------------------------------ #

    num_halls_to_try: int = 10
    """"Number of halls to try to rich" -- cap on how many flakes get a Hall
    placement attempt (in score order) before the rest fall through to Ring."""

    min_flake_length_for_hall: float = 9.0
    """Minimum LONG-side length (um) of the new-flake inscribed rectangle
    below which Hall isn't attempted for that flake. Replaces the old
    single area-based min_flake_size_for_hall -- length and width are
    checked separately (not their product) since a Hall bar needs enough
    room along BOTH axes independently: a long, very thin sliver could
    clear an area threshold while still being too narrow for a real Hall
    channel (see min_flake_width_for_hall)."""

    min_flake_width_for_hall: float = 4.0
    """Minimum SHORT-side width (um) of the new-flake inscribed rectangle
    below which Hall isn't attempted for that flake. See
    min_flake_length_for_hall."""

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
