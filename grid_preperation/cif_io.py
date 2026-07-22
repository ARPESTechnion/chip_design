"""
CIF writer + master-template splicer.

Direct-um design: every polygon vertex here is already an absolute (X_um,
Y_um) chip position (see calibration.pixel_to_um), unlike GDS_Code.m (which
worked in raster-pixel-derived "exported units" tied to a legacy pixel-
doubling convention). Both the per-image CIF writer and the template splicer
use one explicit, self-documented unit convention and DS-ratio arithmetic,
ported from the same regex-based approach GDS_Code.m uses
(Parse_CIF_Polygons / UI_Act6_place_in_cif) — no CIF parser dependency.

Because exported vertices already encode an absolute chip position, placing
them into the master template needs only a unit conversion between the two
files' DS declarations, not a separate shift-to-placement step.

IMPORTANT: CIF's default base resolution is 1/100 micron (a "centimicron"),
per the format spec — `DS 1 A B;` scales *that* base unit by A/B, it does not
make A/B directly micron-per-unit. An earlier version of this module got this
wrong (treated `DS 1 A B` as `A/B` um/unit) and every placement ended up
~100x smaller than it should be, confirmed both against real rendered output
and by cross-checking against BaseGrid.cif's own pre-existing geometry (its
chip-edge coordinates only make physical sense — a few mm, not 250mm — under
the centimicron-corrected formula). Always go through _ds_um_per_unit, never
`ds_scale / ds_length` directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .calibration import Calibration
from .polygons import Polygon

# CIF's default base resolution: 1 unit = 1/100 micron unless a DS statement
# scales it. See module docstring.
_CIF_BASE_UM_PER_UNIT = 0.01

# This module's own exported CIF convention: DS 1 1 10 -> 1 CIF unit =
# 0.01 * (1/10) = 1/1000 um (1 nm). Declared via a standard
# `DS 1 <scale> <length>;` line, so it's read back generically (ratio-based,
# via _ds_um_per_unit) by _parse_ds — never hardcoded elsewhere.
_EXPORT_DS_SCALE = 1
_EXPORT_DS_LENGTH = 10


def _ds_um_per_unit(ds_scale: float, ds_length: float) -> float:
    """um-per-CIF-unit for a `DS 1 <ds_scale> <ds_length>;` declaration."""
    return _CIF_BASE_UM_PER_UNIT * ds_scale / ds_length


def _um_to_export_units(value_um: float) -> int:
    return round(value_um / _ds_um_per_unit(_EXPORT_DS_SCALE, _EXPORT_DS_LENGTH))


def _rgb_to_cif_hex(rgb: tuple[int, int, int]) -> str:
    """Same byte order GDS_Code.m used for its CleWin color comments: '0f' +
    hex(B) + hex(G) + hex(R)."""
    r, g, b = rgb
    return f"0f{b:02x}{g:02x}{r:02x}"


_HEADER_COLOR_RE = re.compile(r"^L\s+L(\d+);\s*\(CleWin:[^/\n]*/(0[fF][0-9a-fA-F]{6})", re.MULTILINE)


def _parse_header_colors(text: str) -> dict[int, str]:
    """layer_num -> hex color string (e.g. '0f7878a8'), parsed from
    '(CleWin: ...)'-commented layer-name header lines only -- not bare
    'L Lxxx;' body layer-selector lines, which carry no color info.
    Tolerant of either this module's own comment format ('.../Flake/hex
    hex') or BaseGrid.cif's pre-existing one ('.../L001/hex hex') -- both
    just need a hex code right after the first '/'."""
    return {int(m.group(1)): m.group(2) for m in _HEADER_COLOR_RE.finditer(text)}


def existing_layer_numbers(text: str) -> set[int]:
    """All layer numbers referenced anywhere in a CIF file (header color
    declarations AND body 'L Lxxx;' layer-selectors alike) -- pass as
    LayerPalette's reserved_layers to keep newly dynamically-allocated
    layers from colliding with layer numbers a master template already uses
    for its own, unrelated content."""
    return {int(m.group(1)) for m in re.finditer(r"L\s+L(\d+);", text)}


class LayerPalette:
    """
    Stateful layer registry shared across a whole run (all images), so the
    same real-world color always ends up on the same CIF layer number
    everywhere it's spliced into the target — not just within one image.

    Starts from config.flake_layer_palette (empty by default -- see
    CifExportConfig.flake_layer_palette). A polygon's color is quantized
    (palette_quantize_step) and looked up by EXACT match against already-
    registered (quantized) colors; no distinct quantized color is ever
    registered gets a fresh dynamically-allocated layer.

    Deliberately NOT nearest-neighbor-within-a-radius matching (an earlier
    version did this via a now-removed max_distance parameter): that let two
    genuinely different classes end up on the same layer/color whenever they
    happened to be within the radius of each other -- confirmed empirically,
    e.g. a crosshair's real cluster color (RGB ~238,190,221) could get
    silently forced onto an unrelated, already-registered dynamic layer's
    color instead of getting its own. Exact quantized-value matching is
    deterministic (doesn't depend on registration order) and only ever
    merges colors that round to the literal same bucket -- different classes
    with different real colors always get different layers.

    reserved_layers: layer numbers to never dynamically allocate, because a
    placement target (e.g. BaseGrid.cif) already uses them for its own,
    unrelated content (see cif_io.existing_layer_numbers). Confirmed
    empirically as a second, independent color bug: place_in_cif only
    splices GEOMETRY into a target, not this module's own color-declaration
    header, so if a dynamically-allocated layer number collides with a
    number the target already declares, the spliced-in geometry silently
    inherits the TARGET's unrelated pre-existing color (e.g. BaseGrid.cif's
    own layer 108 is a bright green (96,224,96) used for unrelated existing
    content -- any flake polygon that happened to land on layer 108 rendered
    with that color instead of its own real cluster color).
    """

    def __init__(
        self, base_palette: list[dict], quantize_step: int = 24,
        reserved_layers: set[int] = frozenset(),
    ):
        self._entries: dict[int, tuple[int, int, int]] = {
            entry["layer"]: tuple(entry["rgb"]) for entry in base_palette
        }
        self._quantize_step = quantize_step
        self._next_layer = max([*self._entries.keys(), *reserved_layers, 100]) + 1

    def layer_for(self, rgb: tuple[int, int, int]) -> int:
        step = self._quantize_step
        quantized = tuple(min(255, round(c / step) * step) for c in rgb)
        for layer, entry_rgb in self._entries.items():
            if entry_rgb == quantized:
                return layer

        new_layer = self._next_layer
        self._next_layer += 1
        self._entries[new_layer] = quantized
        return new_layer

    def rgb_for(self, layer_num: int) -> tuple[int, int, int]:
        return self._entries[layer_num]


_BARE_PREDECLARE_MAX_LAYER = 100
"""Matches GDS_Code.m's UI_Act5_export_cif (for i4 = 0:100 -> S.Txt.L0):
bare, colorless 'L Lxxx;' declarations for layers 0-100, written before the
real colored ones."""


def write_cif(polygons: list[Polygon], path: str | Path, palette: LayerPalette) -> None:
    """
    Write exported polygons (flakes, markers, debris -- everything past
    extract_polygons' filters) to a standalone per-image CIF file.

    Layer numbering/coloring matches GDS_Code.m's UI_Act5_export_cif (lines
    623-632) exactly: a polygon whose color came from a real channels_model
    cluster (poly.source_cluster is not None) goes on FIXED layer
    100+source_cluster, colored with that cluster's own real measured color
    (t.lay = 100+(S.LiSel(i4)); t.rgb = S.mp24(S.LiSel(i4),:) in the
    original) -- not a dynamically-allocated/deduped layer. Only polygons
    with no cluster overlap at all (source_cluster is None, a rare fallback
    -- see polygons.py) go through palette.layer_for's dynamic allocation,
    which is why LayerPalette still exists.
    """
    layer_colors: dict[int, tuple[int, int, int]] = {}
    by_layer: dict[int, list[Polygon]] = {}
    for poly in polygons:
        if poly.source_cluster is not None:
            layer_num = 100 + poly.source_cluster
            layer_colors[layer_num] = poly.mean_rgb
        else:
            layer_num = palette.layer_for(poly.mean_rgb)
            layer_colors[layer_num] = palette.rgb_for(layer_num)
        by_layer.setdefault(layer_num, []).append(poly)

    lines = [
        "(CIF written by grid_preperation - automated flake export);\n",
        "(Nanoelectronics Research Center);\n",
        "(Technion Israel Institute of Technology);\n",
        "\n",
        "(Layer names);\n",
    ]
    for i in range(_BARE_PREDECLARE_MAX_LAYER + 1):
        lines.append(f"L L{i:03d};\n")
    for layer_num in sorted(layer_colors):
        hexcode = _rgb_to_cif_hex(layer_colors[layer_num])
        lines.append(f"L L{layer_num:03d}; (CleWin: {layer_num:03d} Flake/{hexcode} {hexcode});\n")

    lines += [
        "\n(Top level);\n\n",
        f"DS 1 {_EXPORT_DS_SCALE} {_EXPORT_DS_LENGTH};\n\n",
        "9 Flakes;\n\n",
    ]

    for layer_num in sorted(by_layer):
        lines.append(f"L L{layer_num:03d};\n")
        for poly in by_layer[layer_num]:
            coords = " ".join(
                f"{_um_to_export_units(x)} {_um_to_export_units(y)}" for x, y in poly.vertices_um
            )
            lines.append(f"P {coords};\n")

    lines += ["\nDF;\n", "C 1;\n", "E\n"]
    Path(path).write_text("".join(lines))


@dataclass
class _ParsedLayer:
    layer_cmd: str
    polygons: list[list[tuple[float, float]]]


def _parse_ds(text: str) -> tuple[float, float]:
    match = re.search(r"DS\s*1\s+([\d.\-]+)\s+([\d.\-]+)", text)
    if not match:
        raise ValueError("DS command not found in CIF file")
    return float(match.group(1)), float(match.group(2))


def _parse_exported_cif(path: Path) -> tuple[float, float, list[_ParsedLayer]]:
    """
    Parse a CIF file written by write_cif (single-line 'L ...;' and
    'P ...;' commands — not a general CIF parser, matches only what this
    module itself writes).
    """
    text = path.read_text()
    ds_scale, ds_length = _parse_ds(text)

    layers: list[_ParsedLayer] = []
    current: _ParsedLayer | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        layer_match = re.match(r"L\s+L(\d+);", line)
        if layer_match:
            current = _ParsedLayer(layer_cmd=line, polygons=[])
            layers.append(current)
            continue
        if current is not None and line.startswith("P"):
            poly_match = re.match(r"P\s+([\d\s\-.]+);", line)
            if poly_match:
                nums = [float(v) for v in poly_match.group(1).split()]
                if nums and len(nums) % 2 == 0:
                    current.polygons.append(list(zip(nums[0::2], nums[1::2])))

    return ds_scale, ds_length, layers


def place_in_cif(
    exported_cif_path: str | Path,
    target_cif_path: str | Path,
    calibration: Calibration | None = None,
) -> None:
    """
    Splice an already-exported per-image CIF's polygons into the master
    template CIF (truncate just before the target's closing 'DF;' and insert
    the converted layer/polygon statements, then re-append the original
    ending — same surgical-splice approach as GDS_Code.m's
    UI_Act6_place_in_cif).

    calibration is accepted only for logging/traceability by callers (e.g.
    run.py); the placement math itself needs no shift, since exported
    vertices already encode an absolute chip position.
    """
    exported_cif_path = Path(exported_cif_path)
    target_cif_path = Path(target_cif_path)

    export_ds_scale, export_ds_length, layers = _parse_exported_cif(exported_cif_path)
    export_um_per_unit = _ds_um_per_unit(export_ds_scale, export_ds_length)

    target_text = target_cif_path.read_text()
    target_ds_scale, target_ds_length = _parse_ds(target_text)
    target_um_per_unit = _ds_um_per_unit(target_ds_scale, target_ds_length)

    ds_conversion = export_um_per_unit / target_um_per_unit  # exported_unit * this -> target_unit

    df_pos = target_text.rfind("DF;")
    if df_pos == -1:
        raise ValueError(f"target CIF {target_cif_path} has no DF; marker")

    new_lines: list[str] = []
    for layer in layers:
        if not layer.polygons:
            continue
        new_lines.append(layer.layer_cmd)
        for pts in layer.polygons:
            coords = " ".join(f"{round(x * ds_conversion)} {round(y * ds_conversion)}" for x, y in pts)
            new_lines.append(f"P {coords};")
    new_content = "\n".join(new_lines) + "\n"

    spliced = target_text[:df_pos] + new_content + target_text[df_pos:]

    # Geometry above is now spliced in, but place_in_cif only ever touches
    # the BODY -- the exported CIF's own layer-color header declarations
    # never otherwise reach the target. Without this, any spliced-in layer
    # number the target doesn't already declare a color for renders with
    # whatever default a CAD viewer falls back to for an undeclared layer
    # (confirmed empirically: this, combined with the layer-number-collision
    # fix in LayerPalette.reserved_layers, is what caused flakes to render
    # in colors unrelated to their real cluster color).
    export_colors = _parse_header_colors(exported_cif_path.read_text())
    target_colors = _parse_header_colors(target_text)
    missing = {num: hexcode for num, hexcode in export_colors.items() if num not in target_colors}
    if missing:
        insertion = "".join(
            f"L L{num:03d}; (CleWin: {num:03d} Flake/{hexcode} {hexcode});\n"
            for num, hexcode in sorted(missing.items())
        )
        header_matches = list(_HEADER_COLOR_RE.finditer(spliced))
        if header_matches:
            insert_at = spliced.index('\n', header_matches[-1].start()) + 1
        else:
            marker_pos = spliced.find("(Layer names);")
            if marker_pos == -1:
                raise ValueError(f"target CIF {target_cif_path} has no '(Layer names);' section")
            insert_at = spliced.index('\n', marker_pos) + 1
        spliced = spliced[:insert_at] + insertion + spliced[insert_at:]

    target_cif_path.write_text(spliced)
