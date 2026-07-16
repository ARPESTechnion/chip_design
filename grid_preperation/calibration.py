"""
Filename-driven, 2-marker pixel<->um calibration.

No OCR anywhere: the left (filename-derived) marker's grid ID is decoded via
the placement grid formula (see config.grid_pitch_um / placement_x_offset_um
/ placement_y_offset_um) to anchor the image's absolute position. The right
marker's grid ID is inferred as the next column letter — same row, one grid
step over. Scale and rotation both come from the two detected marker pixel
positions: the real-world distance between them is exactly config.grid_pitch_um
(one shared grid — confirmed by the user that adjacent markers, e.g. AA to
AB, are 200um apart, and the master template uses that same grid pitch).

This intentionally does NOT use any burned-in scale-bar overlay that might be
present in an image — an earlier version of this module did, which was a
mistake (confirmed against real output: it silently used the wrong scale).
The pipeline is meant to work for any grid pitch and any image scale, as
supplied via config.grid_pitch_um — never assumed from image content.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .config import CifExportConfig
from .markers import MarkerPair

_ALPHABET_SIZE = 25  # grid letters A..Y


class CalibrationError(Exception):
    """Raised when a filename or marker geometry can't be turned into a valid
    calibration — callers should alert and skip the image, never guess."""


def _grid_id_to_indices(grid_id: str) -> tuple[int, int]:
    """'QK' -> (row_idx, col_idx), both in [0, 24] for A..Y."""
    if len(grid_id) != 2 or not grid_id.isalpha():
        raise CalibrationError(f"grid ID {grid_id!r} is not a 2-letter A-Y code")
    row_idx = ord(grid_id[0]) - ord('A')
    col_idx = ord(grid_id[1]) - ord('A')
    if not (0 <= row_idx < _ALPHABET_SIZE and 0 <= col_idx < _ALPHABET_SIZE):
        raise CalibrationError(f"grid ID {grid_id!r} outside A-Y range")
    return row_idx, col_idx


def _placement_um(grid_id: str, config: CifExportConfig) -> tuple[float, float]:
    """Absolute (X_um, Y_um) of a named grid cell on the master template."""
    row_idx, col_idx = _grid_id_to_indices(grid_id)
    x_um = config.placement_x_offset_um + col_idx * config.grid_pitch_um
    y_um = config.placement_y_offset_um - row_idx * config.grid_pitch_um
    return x_um, y_um


def left_id_from_filename(image_path: str | Path) -> str:
    """The filename stem *is* the top-left marker's grid ID, by convention."""
    stem = Path(image_path).stem.upper()
    _grid_id_to_indices(stem)  # validates shape; raises CalibrationError if not 2 letters A-Y
    return stem


def right_id_from_left(left_id: str) -> str:
    """Same row, next column letter."""
    row, col = left_id[0], left_id[1]
    if col == 'Y':
        raise CalibrationError(
            f"left marker {left_id!r} is at the rightmost grid column (Y); "
            "the 'next column to the right' assumption breaks at this edge"
        )
    return row + chr(ord(col) + 1)


@dataclass
class Calibration:
    left_id: str
    right_id: str
    left_px: tuple[float, float]
    right_px: tuple[float, float]
    left_um: tuple[float, float]
    rotation_rad: float
    scale_um_per_px: float

    def pixel_to_um(self, x_px: float, y_px: float) -> tuple[float, float]:
        """Transform an image-pixel coordinate to absolute (X_um, Y_um)."""
        dx_px = x_px - self.left_px[0]
        dy_px = y_px - self.left_px[1]

        cos_t, sin_t = math.cos(self.rotation_rad), math.sin(self.rotation_rad)
        # Rotate the pixel offset by -rotation_rad to align it with the grid.
        grid_dx_px = dx_px * cos_t + dy_px * sin_t
        grid_dy_px = -dx_px * sin_t + dy_px * cos_t

        # +x (grid-aligned) -> +X_um; +y (grid-aligned, i.e. downward in the
        # image) -> -Y_um, matching the placement formula's row convention
        # (Y_um decreases as row index / image row increases).
        d_x_um = grid_dx_px * self.scale_um_per_px
        d_y_um = -grid_dy_px * self.scale_um_per_px

        return self.left_um[0] + d_x_um, self.left_um[1] + d_y_um


def calibrate(
    image_path: str | Path,
    markers: MarkerPair,
    config: CifExportConfig,
) -> Calibration:
    """
    Build a pixel<->um Calibration from the filename and the two detected
    marker pixel positions. Raises CalibrationError on any base-case
    violation (bad filename, grid edge, etc.) — callers should alert & skip.
    """
    left_id = left_id_from_filename(image_path)
    right_id = right_id_from_left(left_id)
    left_um = _placement_um(left_id, config)

    dx_px = markers.right_px[0] - markers.left_px[0]
    dy_px = markers.right_px[1] - markers.left_px[1]
    pixel_distance = math.hypot(dx_px, dy_px)
    if pixel_distance == 0:
        raise CalibrationError("left and right marker pixel positions are identical")

    rotation_rad = math.atan2(dy_px, dx_px)
    scale_um_per_px = config.grid_pitch_um / pixel_distance

    calibration = Calibration(
        left_id=left_id,
        right_id=right_id,
        left_px=markers.left_px,
        right_px=markers.right_px,
        left_um=left_um,
        rotation_rad=rotation_rad,
        scale_um_per_px=scale_um_per_px,
    )

    # Self-consistency check: transforming the two marker pixels back through
    # the calibration should reproduce left_um exactly and place the right
    # marker grid_pitch_um away along +X — regression guard against
    # implementation bugs in pixel_to_um, not an independent verification.
    check_left = calibration.pixel_to_um(*markers.left_px)
    check_right = calibration.pixel_to_um(*markers.right_px)
    if math.hypot(check_left[0] - left_um[0], check_left[1] - left_um[1]) > 1e-6:
        raise CalibrationError("internal error: left marker did not map back to left_um")
    expected_right = (left_um[0] + config.grid_pitch_um, left_um[1])
    if math.hypot(check_right[0] - expected_right[0], check_right[1] - expected_right[1]) > 1e-6:
        raise CalibrationError("internal error: right marker did not map back to expected offset")

    return calibration
