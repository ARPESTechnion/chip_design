"""
Interactive marker-alignment correction UI for grid_preperation.

Launched by grid_preperation.run.run() between marker DETECTION (Phase A --
channel selection + automatic crosshair finding, see run._detect_markers_for_image)
and everything that consumes it (Phase B -- calibration, polygon extraction,
CIF write, template placement, see run._finish_image), when
CifExportConfig.interactive_alignment is True.

One reconstructed image at a time, with the two calibration crosshairs drawn
red (left_px -- the filename/grid marker) and green (right_px -- one grid
column over), same colors as markers.draw_marker_centers's own debug output.
DRAG either cross to correct its position -- including an image where
automatic detection failed outright, which starts with two placeholder
crosses positioned for you to drag onto the real markers from scratch. A
"Drop Image" toggle excludes the image from this run entirely (regardless of
any position) -- for an image with no usable markers at all.

Per-image "Save Changes" writes marker_overrides/<stem>.json (see
markers.write_marker_override) -- a PERMANENT override: every future run of
grid_preperation.run, interactive or not, uses this saved position/drop
decision and skips automatic detection for that image entirely (see
run._finish_image, which always re-checks this file itself rather than
trusting whatever this UI mutated in memory). "Reset Changes" reverts the
current image to whatever was last saved there (or, if never saved, back to
the original starting position, never dropped). "Save All & Continue" saves
every image's current state, then ends the review (confirms first). "Skip
All" ends the review outright without saving (confirms first) -- any image
whose edits were never explicitly saved is left exactly as it started
(automatic detection's own result, an earlier override, or still alerting if
detection failed and was never corrected).

Deliberately never mutates a MarkerStageResult's own `marker_pair` field --
all per-image working state (current drag position, drop flag, last-saved
snapshot, original starting position) lives entirely in this UI's own
arrays, so an un-Saved edit has zero effect once the window closes, exactly
like flake_detection.ui's chosen/rejected toggle.

Two precision aids, since mouse placement alone is coarser than the
sub-pixel accuracy calibration actually wants (marker separation feeds
scale_um_per_px for the WHOLE image -- see calibration.calibrate):
  - A magnifier loupe (top-right inset) shows a zoomed crop of the FULL-
    RESOLUTION image around the cursor while dragging -- the fit-to-window
    display scale is often well under 1.0, so a screen pixel there is
    already several real image pixels; the loupe is built from the
    pre-resize frame so it isn't itself degraded by that same scale-down.
  - A live numeric readout (both crosses' exact pixel coordinates, their
    separation, angle, and the scale_um_per_px that separation implies via
    config.grid_pitch_um) lets you verify placement objectively instead of
    only by eye.
"""

from __future__ import annotations

import math
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .markers import MarkerPair, MarkerStageResult, load_marker_override, write_marker_override

_RED = (0, 0, 255)      # BGR -- left_px (filename/grid marker), matches draw_marker_centers
_GREEN = (0, 255, 0)    # BGR -- right_px, matches draw_marker_centers
_FALLBACK_W = 1200
_FALLBACK_H = 900
_MARKER_LEN = 50
_MARKER_THICKNESS = 3
_GRAB_RADIUS_SCREEN_PX = 25  # converted to image-space via /scale at grab time

_MAGNIFIER_SRC_HALF = 130    # crop half-size, in ORIGINAL (full-res) image px, around the cursor --
                              # crosshair_template.png is 230x170px, so this comfortably covers a
                              # whole real crosshair (core + all 4 arms), not just its central core
_MAGNIFIER_ZOOM = 1.5        # upscale factor applied to that crop -- modest on purpose: the crop
                              # above is already generously sized, this just fills the inset nicely
_MAGNIFIER_MARGIN = 16       # inset margin from the canvas's top-right corner, screen px
_MAGNIFIER_MARKER_LEN = 24   # reticle arm length drawn IN the magnifier, sized for its own zoom --
_MAGNIFIER_MARKER_THICKNESS = 2  # deliberately NOT _MARKER_LEN/_MARKER_THICKNESS (those are sized
                              # for the full, un-zoomed image and would swallow this small a crop)
_MAGNIFIER_GUIDE_COLOR = (0, 255, 255)  # BGR yellow -- full-span 1px alignment guide lines

_NUDGE_STEP_PX = 1
_NUDGE_STEP_PX_FAST = 10  # with Shift held

Point = tuple[float, float]
State = tuple[Point, Point, bool]  # (left_px, right_px, dropped)

_SOURCE_LABELS = {
    'auto': 'auto-detected',
    'override': 'previously saved override',
    'failed': 'auto-detection FAILED — drag both crosses into place',
    'dropped': 'previously dropped',
}


def _seed_positions(chan_sel) -> tuple[Point, Point]:
    """Starting crosses for an image where automatic detection failed --
    positioned within the marker search band, spaced apart, so both are
    independently grabbable immediately rather than needing to be dragged
    apart from an overlapping starting point first."""
    h, w = chan_sel.reconstructed_bgr.shape[:2]
    y = h * 0.08
    return (w * 0.35, y), (w * 0.65, y)


class AlignmentReviewUI:
    def __init__(self, root: tk.Tk, stages: list[MarkerStageResult], overrides_dir: Path, grid_pitch_um: float):
        self.root = root
        self.stages = stages
        self.overrides_dir = Path(overrides_dir)
        self.grid_pitch_um = grid_pitch_um
        self.idx = 0
        self._dragging: str | None = None  # 'left' | 'right' | None -- set only while the mouse button is down
        self._selected: str | None = None  # 'left' | 'right' | None -- persists after a drag ends, so
                                            # arrow keys can keep fine-tuning the same cross; see
                                            # _on_key_left/_on_key_right for the navigate-vs-nudge split

        # Per-image state, entirely separate from stage.marker_pair -- see
        # module docstring. _original is this image's starting point before
        # any edit this session (never dropped); _saved mirrors whatever's
        # actually on disk (None if never saved); _current is the live
        # working state.
        self._original: list[State] = []
        self._saved: list[State | None] = []
        self._current: list[State] = []
        for stage in self.stages:
            base_pos = (stage.marker_pair.left_px, stage.marker_pair.right_px) if stage.marker_pair \
                else _seed_positions(stage.chan_sel)
            base: State = (base_pos[0], base_pos[1], False)
            self._original.append(base)
            override = load_marker_override(self.overrides_dir, stage.img_path.stem)
            if override is not None:
                pos = (override.marker_pair.left_px, override.marker_pair.right_px) \
                    if override.marker_pair else base_pos
                saved: State | None = (pos[0], pos[1], override.dropped)
            else:
                saved = None
            self._saved.append(saved)
            self._current.append(saved if saved else base)

        self._photo: ImageTk.PhotoImage | None = None
        self._scale = 1.0
        self._offset = (0, 0)

        root.title("Grid Preparation — Alignment Review")
        try:
            root.state('zoomed')
        except tk.TclError:
            root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")

        style = ttk.Style(root)
        style.configure('Arrow.TButton', font=('TkDefaultFont', 20, 'bold'))
        style.configure('Big.TButton', font=('TkDefaultFont', 13, 'bold'), padding=(24, 14))

        top = ttk.Frame(root, padding=6)
        top.pack(side='top', fill='x')
        self.title_label = ttk.Label(top, text="", font=('TkDefaultFont', 10, 'bold'))
        self.title_label.pack(side='left')
        self.dirty_label = ttk.Label(top, text="", foreground='#b00000')
        self.dirty_label.pack(side='left', padx=10)

        readout_row = ttk.Frame(root, padding=(6, 0))
        readout_row.pack(side='top', fill='x')
        self.readout_label = ttk.Label(readout_row, text="", font=('TkFixedFont', 9))
        self.readout_label.pack(side='left')

        middle = ttk.Frame(root)
        middle.pack(side='top', fill='both', expand=True)
        ttk.Button(middle, text="◀", style='Arrow.TButton', command=self._prev).pack(
            side='left', fill='y', padx=4, pady=4)
        ttk.Button(middle, text="▶", style='Arrow.TButton', command=self._next).pack(
            side='right', fill='y', padx=4, pady=4)
        self.canvas = tk.Canvas(middle, bg='black', highlightthickness=0)
        self.canvas.pack(side='left', fill='both', expand=True, padx=4, pady=4)
        self.canvas.bind('<ButtonPress-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.canvas.bind('<Configure>', lambda e: self._render())

        # Arrow keys are dual-purpose: with no cross selected they navigate
        # images (matching flake_detection.ui's own convention); once a
        # cross is selected (by clicking it, including at the end of a
        # drag -- see _on_press/_on_release) they nudge that cross instead,
        # by _NUDGE_STEP_PX (Shift: _NUDGE_STEP_PX_FAST) -- see
        # _on_key_left/_on_key_right/_on_key_up/_on_key_down.
        root.bind_all('<Left>', self._on_key_left)
        root.bind_all('<Right>', self._on_key_right)
        root.bind_all('<Up>', self._on_key_up)
        root.bind_all('<Down>', self._on_key_down)
        root.bind_all('<Return>', lambda e: self._save())

        bottom = ttk.Frame(root, padding=(6, 8))
        bottom.pack(side='top', fill='x')
        action_row = ttk.Frame(bottom)
        action_row.pack(side='top', pady=(0, 6))
        ttk.Button(action_row, text="Reset Changes", style='Big.TButton', command=self._reset).pack(
            side='left', padx=10)
        ttk.Button(action_row, text="Save Changes", style='Big.TButton', command=self._save).pack(
            side='left', padx=10)
        self.drop_btn = ttk.Button(action_row, text="Drop Image", command=self._toggle_drop)
        self.drop_btn.pack(side='left', padx=10)
        skip_row = ttk.Frame(bottom)
        skip_row.pack(side='top')
        ttk.Button(skip_row, text="Save All & Continue", command=self._save_all).pack(side='left', padx=4)
        ttk.Button(skip_row, text="Skip All", command=self._skip_all).pack(side='left', padx=4)

        self.selection_label = ttk.Label(root, text="", padding=(6, 0), font=('TkDefaultFont', 9, 'bold'))
        self.selection_label.pack(side='top', fill='x')

        self.status = ttk.Label(
            root,
            text="Click a cross to select it (red = left/filename marker, green = right marker), "
                 "then drag or use arrow keys to move it (Shift = 10px steps); click empty space to "
                 "deselect.  Enter: save changes.",
            padding=(6, 0),
        )
        self.status.pack(side='top', fill='x')

        self._render()

    # -- helpers --------------------------------------------------------

    def _baseline(self, i: int) -> State:
        return self._saved[i] if self._saved[i] is not None else self._original[i]

    def _is_dirty(self, i: int) -> bool:
        return self._current[i] != self._baseline(i)

    # -- rendering --------------------------------------------------------

    def _render(self) -> None:
        stage = self.stages[self.idx]
        left, right, dropped = self._current[self.idx]
        raw_bgr = stage.chan_sel.reconstructed_bgr  # magnifier crops from this -- see below
        image_bgr = raw_bgr.copy()
        for pos, color in ((left, _RED), (right, _GREEN)):
            cv2.drawMarker(
                image_bgr, (int(round(pos[0])), int(round(pos[1]))), color,
                cv2.MARKER_CROSS, _MARKER_LEN, _MARKER_THICKNESS,
            )
        if dropped:
            cv2.putText(
                image_bgr, "DROPPED — excluded from this run", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3, cv2.LINE_AA,
            )

        h, w = image_bgr.shape[:2]
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        canvas_w = canvas_w if canvas_w > 1 else _FALLBACK_W
        canvas_h = canvas_h if canvas_h > 1 else _FALLBACK_H
        scale = min(canvas_w / w, canvas_h / h)
        disp_w, disp_h = max(1, int(w * scale)), max(1, int(h * scale))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(image_bgr, (disp_w, disp_h), interpolation=interp)
        self._scale = scale
        self._offset = ((canvas_w - disp_w) // 2, (canvas_h - disp_h) // 2)

        # Magnifier loupe: while a cross is selected (mid-drag OR just
        # keyboard-nudged -- see _selected), crop from raw_bgr -- the
        # FULL-RESOLUTION frame, BEFORE either the main marker-cross overlay
        # (drawn at _MARKER_LEN=50, sized for the whole image -- far too big
        # for this small a crop) or the fit-to-window resize above (whose
        # scale is often well under 1.0) -- so what's shown is the real
        # underlying image detail at full precision, with its OWN
        # appropriately small cross drawn back in at the exact position
        # (color-matched to which marker this is). Pasted into the
        # top-right corner of `resized` (screen space) AFTER resizing, so
        # its own zoom is independent of the window's size.
        if self._selected is not None:
            mag_center = left if self._selected == 'left' else right
            mag_color = _RED if self._selected == 'left' else _GREEN
            self._draw_magnifier(resized, raw_bgr, mag_center, mag_color)

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.canvas.delete('all')
        self.canvas.create_image(self._offset[0], self._offset[1], anchor='nw', image=self._photo)

        name = stage.img_path.name
        status_bit = "DROPPED" if dropped else _SOURCE_LABELS[stage.source]
        self.title_label.config(text=f"[{self.idx + 1}/{len(self.stages)}] {name}  ({status_bit})")
        self.dirty_label.config(text="unsaved changes" if self._is_dirty(self.idx) else "")
        self.drop_btn.config(text="Include Image" if dropped else "Drop Image")

        if self._selected == 'left':
            self.selection_label.config(text="Selected: LEFT marker (red) — arrow keys nudge it", foreground='#b00000')
        elif self._selected == 'right':
            self.selection_label.config(text="Selected: RIGHT marker (green) — arrow keys nudge it", foreground='#008000')
        else:
            self.selection_label.config(text="No marker selected — ←/→ navigate images", foreground='#000000')

        dx, dy = right[0] - left[0], right[1] - left[1]
        separation = math.hypot(dx, dy)
        angle_deg = math.degrees(math.atan2(dy, dx))
        scale_txt = f"{self.grid_pitch_um / separation:.4f}" if separation > 0 else "n/a"
        self.readout_label.config(
            text=f"left=({left[0]:8.2f}, {left[1]:8.2f})   right=({right[0]:8.2f}, {right[1]:8.2f})   "
                 f"separation={separation:8.2f}px   angle={angle_deg:7.2f}°   implied scale={scale_txt} um/px"
        )

    def _draw_magnifier(self, resized: np.ndarray, full_res_bgr: np.ndarray, center: Point, color: tuple) -> None:
        """Paste a zoomed crop of full_res_bgr around `center` (image-pixel
        space) into the top-right corner of `resized` (already at screen/
        canvas scale), in place. No-op if it wouldn't fit. Two alignment
        aids drawn on top: a full-span, 1px-wide yellow guide cross --
        edge-to-edge lines through the exact position, for judging fine
        angular/positional alignment against the real underlying feature at
        a glance -- and a small `color`-matched cross exactly like the ones
        on the main image, marking which marker (left/right) this is."""
        h, w = full_res_bgr.shape[:2]
        cx, cy = int(round(center[0])), int(round(center[1]))
        x0, y0 = max(0, cx - _MAGNIFIER_SRC_HALF), max(0, cy - _MAGNIFIER_SRC_HALF)
        x1, y1 = min(w, cx + _MAGNIFIER_SRC_HALF), min(h, cy + _MAGNIFIER_SRC_HALF)
        crop = full_res_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return
        zoomed = cv2.resize(crop, None, fx=_MAGNIFIER_ZOOM, fy=_MAGNIFIER_ZOOM, interpolation=cv2.INTER_NEAREST)
        zh, zw = zoomed.shape[:2]
        # Cross center in zoomed space -- NOT necessarily the exact middle
        # of `zoomed` if the crop was clipped by the image edge, so
        # recompute from the actual crop offset.
        rx = int(round((cx - x0) * _MAGNIFIER_ZOOM))
        ry = int(round((cy - y0) * _MAGNIFIER_ZOOM))
        cv2.line(zoomed, (0, ry), (zw - 1, ry), _MAGNIFIER_GUIDE_COLOR, 1)
        cv2.line(zoomed, (rx, 0), (rx, zh - 1), _MAGNIFIER_GUIDE_COLOR, 1)
        # Small color-matched cross drawn on top, sized for the magnifier's
        # own (modest) zoom, not the full image's _MARKER_LEN.
        cv2.drawMarker(zoomed, (rx, ry), color, cv2.MARKER_CROSS, _MAGNIFIER_MARKER_LEN, _MAGNIFIER_MARKER_THICKNESS)
        cv2.rectangle(zoomed, (0, 0), (zw - 1, zh - 1), (255, 255, 255), 2)

        rh, rw = resized.shape[:2]
        if zw > rw or zh > rh:
            return
        x_off = rw - zw - _MAGNIFIER_MARGIN
        y_off = _MAGNIFIER_MARGIN
        resized[y_off:y_off + zh, x_off:x_off + zw] = zoomed

    # -- drag handling -----------------------------------------------------

    def _to_image_px(self, event: tk.Event) -> Point:
        ox, oy = self._offset
        return (event.x - ox) / self._scale, (event.y - oy) / self._scale

    def _on_press(self, event: tk.Event) -> None:
        px, py = self._to_image_px(event)
        left, right, _ = self._current[self.idx]
        grab_r = _GRAB_RADIUS_SCREEN_PX / self._scale
        dl = math.hypot(px - left[0], py - left[1])
        dr = math.hypot(px - right[0], py - right[1])
        if dl > grab_r and dr > grab_r:
            # Clicked empty canvas -- deselect, handing arrow keys back to
            # image navigation.
            self._dragging = None
            self._selected = None
        elif dl <= dr:
            self._dragging = self._selected = 'left'
        else:
            self._dragging = self._selected = 'right'
        self._render()

    def _on_drag(self, event: tk.Event) -> None:
        if self._dragging is None:
            return
        px, py = self._to_image_px(event)
        left, right, dropped = self._current[self.idx]
        if self._dragging == 'left':
            self._current[self.idx] = ((px, py), right, dropped)
        else:
            self._current[self.idx] = (left, (px, py), dropped)
        self._render()

    def _on_release(self, event: tk.Event) -> None:
        self._dragging = None
        self._render()

    # -- keyboard nudge -----------------------------------------------------

    def _nudge(self, dx: int, dy: int) -> None:
        left, right, dropped = self._current[self.idx]
        if self._selected == 'left':
            left = (left[0] + dx, left[1] + dy)
        else:
            right = (right[0] + dx, right[1] + dy)
        self._current[self.idx] = (left, right, dropped)
        self._render()

    @staticmethod
    def _step(event: tk.Event) -> int:
        shift_held = bool(event.state & 0x0001)
        return _NUDGE_STEP_PX_FAST if shift_held else _NUDGE_STEP_PX

    def _on_key_left(self, event: tk.Event) -> None:
        self._nudge(-self._step(event), 0) if self._selected else self._prev()

    def _on_key_right(self, event: tk.Event) -> None:
        self._nudge(self._step(event), 0) if self._selected else self._next()

    def _on_key_up(self, event: tk.Event) -> None:
        if self._selected:
            self._nudge(0, -self._step(event))

    def _on_key_down(self, event: tk.Event) -> None:
        if self._selected:
            self._nudge(0, self._step(event))

    # -- navigation / actions -----------------------------------------------

    def _goto(self, new_idx: int) -> None:
        if 0 <= new_idx < len(self.stages):
            self.idx = new_idx
            self._selected = None
            self._render()

    def _prev(self) -> None:
        self._goto(self.idx - 1)

    def _next(self) -> None:
        self._goto(self.idx + 1)

    def _reset(self) -> None:
        self._current[self.idx] = self._baseline(self.idx)
        self._render()

    def _toggle_drop(self) -> None:
        left, right, dropped = self._current[self.idx]
        self._current[self.idx] = (left, right, not dropped)
        self._render()

    def _save_index(self, i: int) -> None:
        stage = self.stages[i]
        left, right, dropped = self._current[i]
        write_marker_override(self.overrides_dir, stage.img_path.stem, MarkerPair(left_px=left, right_px=right), dropped=dropped)
        self._saved[i] = self._current[i]

    def _save(self) -> None:
        self._save_index(self.idx)
        self.status.config(text=f"Saved {self.stages[self.idx].img_path.stem}.")
        self._render()

    def _save_all(self) -> None:
        dirty = sum(1 for i in range(len(self.stages)) if self._is_dirty(i))
        if dirty == 0:
            msg = "Nothing has changed since the last save. Finish reviewing and continue?"
        else:
            msg = f"Save changes on {dirty} image(s) and finish reviewing?"
        if not messagebox.askyesno("Save All & Continue — are you sure?", msg):
            return
        for i in range(len(self.stages)):
            self._save_index(i)
        self.root.destroy()

    def _skip_all(self) -> None:
        dirty = sum(1 for i in range(len(self.stages)) if self._is_dirty(i))
        if dirty:
            msg = (
                f"{dirty} image(s) have unsaved changes that will be discarded "
                f"(each proceeds with whatever it already had -- automatic detection, an "
                f"earlier override, or an alert if detection failed and was never corrected).\n\n"
                f"Finish review anyway?"
            )
        else:
            msg = "Finish reviewing and continue?"
        if messagebox.askyesno("Skip All — are you sure?", msg):
            self.root.destroy()


def review_alignment(stages: list[MarkerStageResult], overrides_dir: str | Path, grid_pitch_um: float) -> None:
    """
    Launch the interactive alignment window over every image whose channel
    selection succeeded (an image with no reconstructed image has nothing to
    draw crosses on or calibrate at all, regardless of markers). Blocks
    until the window is closed (Save All & Continue, Skip All, or the
    window's own close button). Only ever writes marker_overrides/<stem>.json
    on explicit Save -- see module docstring for why this never mutates
    `stages` itself.

    grid_pitch_um (CifExportConfig.grid_pitch_um) is used only for the live
    readout's implied-scale figure -- calibration.calibrate computes the
    real thing later from whatever's actually saved.
    """
    reviewable = [s for s in stages if s.chan_sel is not None]
    if not reviewable:
        print("No images available to review — skipping interactive alignment review.")
        return
    root = tk.Tk()
    AlignmentReviewUI(root, reviewable, Path(overrides_dir), grid_pitch_um)
    root.mainloop()


def maybe_review_alignment(stages: list[MarkerStageResult], overrides_dir: str | Path, config) -> None:
    """
    Launch review_alignment only if config.interactive_alignment is set --
    the single gate grid_preperation.run.run() checks, so a headless/
    scripted run never pops a GUI unless the config explicitly asked for it.
    """
    if getattr(config, 'interactive_alignment', False):
        review_alignment(stages, overrides_dir, config.grid_pitch_um)
