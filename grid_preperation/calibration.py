"""
Filename-driven, 2-marker pixel<->um calibration.

No OCR anywhere: the left (filename-derived) marker's grid ID is decoded via
the placement grid formula (see config.grid_pitch_um / placement_x_offset_um
/ placement_y_offset_um) to anchor the image's absolute position. The right
marker's grid ID is inferred from `MarkerPair.direction` -- by default the
next column letter, same row, one grid step over ('col+1'), but see
second_marker_id for the other supported directions (e.g. a vertically-
arranged pair, set via grid_preperation.ui's marker-direction toggle for a
real image where the two crosshairs are stacked rather than side by side).
Scale and rotation both come from the two detected marker pixel positions:
the real-world distance between them is exactly config.grid_pitch_um (one
shared grid — confirmed by the user that adjacent markers, e.g. AA to AB,
are 200um apart, and the master template uses that same grid pitch).

This intentionally does NOT use any burned-in scale-bar overlay that might be
present in an image — an earlier version of this module did, which was a
mistake (confirmed against real output: it silently used the wrong scale).
The pipeline is meant to work for any grid pitch and any image scale, as
supplied via config.grid_pitch_um — never assumed from image content.
"""

from __future__ import annotations

import cmath
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


_DIRECTION_STEPS: dict[str, tuple[int, int]] = {
    'col+1': (0, 1),    # default: right marker one column over, same row (the original assumption)
    'col-1': (0, -1),   # right marker one column to the LEFT, same row
    'row+1': (1, 0),    # right marker one row BELOW the left marker (left marker above)
    'row-1': (-1, 0),   # right marker one row ABOVE the left marker -- i.e. the left/filename
                        # marker sits BELOW the right one; row index increases downward (see
                        # _placement_um), so "one row above" is index - 1
}
"""(row_step, col_step) applied to the left marker's own grid indices to
reach the right marker's, per MarkerPair.direction. Only these 4 axis-
aligned single-step directions are supported -- a real crosshair pair is
always adjacent on the grid along one axis, never diagonal."""


def second_marker_id(first_id: str, direction: str = 'col+1') -> str:
    """
    The second (right_px) marker's grid ID, one grid step from `first_id`
    (the first/left_px marker's own ID) in `direction` -- generalizes the
    original hardcoded "always one column to the right, same row"
    assumption (still the default) to also support a vertically-arranged
    pair. See _DIRECTION_STEPS for the full set and grid_preperation.ui's
    marker-direction toggle for where a non-default value comes from in
    practice (a real image where automatic detection's own horizontal-
    alignment requirement can never be met, because the two crosshairs are
    genuinely stacked one below the other, not side by side).
    """
    if direction not in _DIRECTION_STEPS:
        raise CalibrationError(
            f"unknown marker direction {direction!r} (expected one of {sorted(_DIRECTION_STEPS)})"
        )
    row_idx, col_idx = _grid_id_to_indices(first_id)
    row_step, col_step = _DIRECTION_STEPS[direction]
    new_row, new_col = row_idx + row_step, col_idx + col_step
    if not (0 <= new_row < _ALPHABET_SIZE and 0 <= new_col < _ALPHABET_SIZE):
        raise CalibrationError(
            f"marker {first_id!r} plus direction {direction!r} steps outside the A-Y grid"
        )
    return chr(ord('A') + new_row) + chr(ord('A') + new_col)


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
        """
        Transform an image-pixel coordinate to absolute (X_um, Y_um).

        Image-pixel Y increases downward while um-space Y increases upward
        (see _placement_um's row convention), so this is a rotation+scale
        applied AFTER flipping the pixel offset's Y component -- not a pure
        rotation+scale alone, which could never capture that axis flip (a
        single complex multiplication is always orientation-preserving).
        `rotation_rad`/`scale_um_per_px` are derived in calibrate() via the
        exact inverse of this same flip-then-rotate-then-scale construction,
        from whichever grid direction the right marker actually represents
        (see MarkerPair.direction / second_marker_id) -- this method itself
        doesn't need to know that direction, it's already baked into these
        two numbers.
        """
        dx_px = x_px - self.left_px[0]
        dy_px = y_px - self.left_px[1]

        k = cmath.rect(self.scale_um_per_px, self.rotation_rad)
        z_um = k * complex(dx_px, -dy_px)

        return self.left_um[0] + z_um.real, self.left_um[1] + z_um.imag


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
    right_id = second_marker_id(left_id, markers.direction)
    left_um = _placement_um(left_id, config)
    right_um = _placement_um(right_id, config)

    dx_px = markers.right_px[0] - markers.left_px[0]
    dy_px = markers.right_px[1] - markers.left_px[1]
    pixel_distance = math.hypot(dx_px, dy_px)
    if pixel_distance == 0:
        raise CalibrationError("left and right marker pixel positions are identical")

    # Solve for the (scale, rotation) that maps the left->right PIXEL vector
    # (Y-flipped first, see Calibration.pixel_to_um's own docstring) onto the
    # left->right UM vector implied by right_id -- generalizes the old
    # col+1-only atan2/grid_pitch_um formula to any MarkerPair.direction; it
    # reduces to exactly that formula when direction == 'col+1' (the target
    # UM vector is then (grid_pitch_um, 0), same as before).
    z_px_flipped = complex(dx_px, -dy_px)
    z_target_um = complex(right_um[0] - left_um[0], right_um[1] - left_um[1])
    k = z_target_um / z_px_flipped
    scale_um_per_px = abs(k)
    rotation_rad = cmath.phase(k)

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
    # marker at exactly right_um — regression guard against implementation
    # bugs in pixel_to_um, not an independent verification.
    check_left = calibration.pixel_to_um(*markers.left_px)
    check_right = calibration.pixel_to_um(*markers.right_px)
    if math.hypot(check_left[0] - left_um[0], check_left[1] - left_um[1]) > 1e-6:
        raise CalibrationError("internal error: left marker did not map back to left_um")
    if math.hypot(check_right[0] - right_um[0], check_right[1] - right_um[1]) > 1e-6:
        raise CalibrationError("internal error: right marker did not map back to right_um")

    return calibration
