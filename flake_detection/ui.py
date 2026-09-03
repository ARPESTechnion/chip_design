"""
Interactive post-detection review UI for flake_detection.

Launched by flake_detection.run (and, if PipelineConfig.interactive also
allows it, all_code.py's orchestrator) immediately after detect_flakes()
finishes, when DetectionConfig.interactive_review is True -- see that
field's own docstring for the full flag semantics.

One chip image at a time, showing EVERY contour that cleared the size
filter for that image (FlakeMetrics.chosen, see detector.py -- includes
candidates rejected for touching the image border, low solidity/shape,
wrong color, or a low final score -- not just ones that already passed
everything) outlined in green (chosen) or red (rejected). Clicking a flake
toggles it -- reject a false positive, or rescue any near-miss regardless of
which stage it originally failed. Only contours that failed the size filter
(or the even smaller _MIN_DISPLAY_AREA noise floor above it) never become a
FlakeMetrics at all -- those are dust/noise/grid-marker debris in bulk, with
no geometry here to bring them back with.

Per-image "Save Changes" rewrites that image's own contours/<stem>.json
(the exact file grid_preperation reads) and scores/<stem>.txt to the
currently-chosen flakes only, and rewrites the shared results.json (which
keeps every flake regardless of chosen, so a rejection is still visible
later). "Reset Changes" reverts the current image to whatever was last
saved to disk (or, if never saved this session, back to the all-chosen
state detect_flakes() originally wrote). "Skip All" ends the review outright
-- any image whose edits were never explicitly saved is left exactly as
detect_flakes() wrote it.

Deliberately no standalone disk-reload entry point: once a flake is
rejected and saved, its contour_px is gone from contours.json for good (by
design -- see write_contours_json), so re-opening a review from results.json
+ contours.json alone could not fully reconstruct every flake's geometry.
Within a single run this module only ever mutates the in-memory
FlakeMetrics objects detect_flakes() itself just produced, so no geometry
is ever actually lost during one review session.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .detector import ImageResult, write_contours_json, write_results_json, write_scores_txt

_GREEN = (0, 210, 0)    # BGR -- chosen
_RED = (0, 0, 230)      # BGR -- rejected
_FALLBACK_W = 1200      # used only for the very first render, before the
_FALLBACK_H = 900       # canvas widget has been mapped and has a real size


class FlakeReviewUI:
    def __init__(
        self,
        root: tk.Tk,
        all_results: list[ImageResult],
        reviewable: list[ImageResult],
        output_folder: Path,
    ):
        self.root = root
        self.all_results = all_results
        self.images = reviewable
        self.output_folder = output_folder
        self.contours_dir = output_folder / 'contours'
        self.scores_dir = output_folder / 'scores'
        self.results_json_path = output_folder / 'results.json'

        self.idx = 0
        # Snapshot of {contour_index: chosen} last written to disk for each
        # image -- initially matches what detect_flakes() itself wrote
        # (every flake chosen), since that's the on-disk state before any
        # review edit happens.
        self.saved = [self._chosen_snapshot(i) for i in range(len(self.images))]

        self._photo: ImageTk.PhotoImage | None = None  # keep a live ref, Tk drops GC'd images
        self._scale = 1.0
        self._offset = (0, 0)
        self._bgr_cache: dict[int, np.ndarray] = {}

        root.title("Flake Detection — Review")
        try:
            root.state('zoomed')  # Windows/most X11 WMs: maximize, keep window chrome
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

        middle = ttk.Frame(root)
        middle.pack(side='top', fill='both', expand=True)
        ttk.Button(
            middle, text="◀", style='Arrow.TButton', command=self._prev,
        ).pack(side='left', fill='y', padx=4, pady=4)
        ttk.Button(
            middle, text="▶", style='Arrow.TButton', command=self._next,
        ).pack(side='right', fill='y', padx=4, pady=4)
        self.canvas = tk.Canvas(middle, bg='black', highlightthickness=0)
        self.canvas.pack(side='left', fill='both', expand=True, padx=4, pady=4)
        self.canvas.bind('<Button-1>', self._on_click)
        self.canvas.bind('<Configure>', lambda e: self._render())

        # bind_all (not bind on a specific widget) so these fire regardless
        # of which widget currently has keyboard focus -- e.g. right after
        # clicking a flake on the canvas, or after clicking a button.
        root.bind_all('<Left>', lambda e: self._prev())
        root.bind_all('<Right>', lambda e: self._next())
        root.bind_all('<Return>', lambda e: self._save())

        bottom = ttk.Frame(root, padding=(6, 8))
        bottom.pack(side='top', fill='x')
        action_row = ttk.Frame(bottom)
        action_row.pack(side='top', pady=(0, 6))
        ttk.Button(
            action_row, text="Reset Changes", style='Big.TButton', command=self._reset,
        ).pack(side='left', padx=10)
        ttk.Button(
            action_row, text="Save Changes", style='Big.TButton', command=self._save,
        ).pack(side='left', padx=10)
        skip_row = ttk.Frame(bottom)
        skip_row.pack(side='top')
        ttk.Button(skip_row, text="Skip All", command=self._skip_all).pack()

        self.status = ttk.Label(
            root,
            text="Click a flake to toggle chosen (green) / rejected (red).  "
                 "←/→: prev/next image.  Enter: save changes.",
            padding=(6, 0),
        )
        self.status.pack(side='top', fill='x')

        self._render()

    # -- current-image helpers ------------------------------------------

    @property
    def result(self) -> ImageResult:
        return self.images[self.idx]

    def _chosen_snapshot(self, idx: int) -> dict[int, bool]:
        return {f.contour_index: f.chosen for f in self.images[idx].flakes}

    def _is_dirty(self, idx: int) -> bool:
        return self._chosen_snapshot(idx) != self.saved[idx]

    # -- rendering --------------------------------------------------------

    def _load_image(self, idx: int) -> np.ndarray:
        if idx not in self._bgr_cache:
            img = cv2.imread(self.images[idx].image_path)
            if img is None:
                img = np.zeros((_FALLBACK_H, _FALLBACK_W, 3), dtype=np.uint8)
            self._bgr_cache[idx] = img
        return self._bgr_cache[idx]

    def _render(self) -> None:
        result = self.result
        image_bgr = self._load_image(self.idx).copy()

        for f in result.flakes:
            color = _GREEN if f.chosen else _RED
            contour = np.array(f.contour_px, dtype=np.int32).reshape(-1, 1, 2)
            cv2.drawContours(image_bgr, [contour], -1, color, 2)

        h, w = image_bgr.shape[:2]
        # winfo_width/height report 1 before the window is first mapped --
        # fall back to a sane default for that one frame; every subsequent
        # render (including the one <Configure> fires once real geometry is
        # known) uses the canvas's actual, current size, so resizing the
        # window rescales the image to match.
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

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.canvas.delete('all')
        self.canvas.create_image(self._offset[0], self._offset[1], anchor='nw', image=self._photo)

        name = Path(result.image_path).name
        n_chosen = sum(1 for f in result.flakes if f.chosen)
        self.title_label.config(
            text=f"[{self.idx + 1}/{len(self.images)}] {name}  ({n_chosen}/{len(result.flakes)} chosen)"
        )
        self.dirty_label.config(text="unsaved changes" if self._is_dirty(self.idx) else "")

    # -- click handling -----------------------------------------------------

    def _on_click(self, event: tk.Event) -> None:
        ox, oy = self._offset
        px = (event.x - ox) / self._scale
        py = (event.y - oy) / self._scale

        best, best_area = None, None
        for f in self.result.flakes:
            contour = np.array(f.contour_px, dtype=np.int32).reshape(-1, 1, 2)
            if cv2.pointPolygonTest(contour, (px, py), False) >= 0:
                # Smallest-area match wins so a click inside two overlapping
                # / nested flakes hits the one actually on top visually.
                if best_area is None or f.area_px < best_area:
                    best, best_area = f, f.area_px
        if best is not None:
            best.chosen = not best.chosen
            self._render()

    # -- navigation / actions -----------------------------------------------

    def _goto(self, new_idx: int) -> None:
        if 0 <= new_idx < len(self.images):
            self.idx = new_idx
            self._render()

    def _prev(self) -> None:
        self._goto(self.idx - 1)

    def _next(self) -> None:
        self._goto(self.idx + 1)

    def _reset(self) -> None:
        snapshot = self.saved[self.idx]
        for f in self.result.flakes:
            f.chosen = snapshot.get(f.contour_index, f.chosen)
        self._render()

    def _save(self) -> None:
        result = self.result
        self.contours_dir.mkdir(parents=True, exist_ok=True)
        self.scores_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(result.image_path).stem
        write_contours_json(result, self.contours_dir / f"{stem}.json")
        write_scores_txt(result, self.scores_dir / f"{stem}.txt")
        write_results_json(self.all_results, self.results_json_path)
        self.saved[self.idx] = self._chosen_snapshot(self.idx)
        self.status.config(text=f"Saved {stem}.")
        self._render()

    def _skip_all(self) -> None:
        dirty = sum(1 for i in range(len(self.images)) if self._is_dirty(i))
        if dirty:
            msg = (
                f"{dirty} image(s) have unsaved changes that will be discarded "
                f"(files stay as flake_detection originally wrote them).\n\nFinish review anyway?"
            )
        else:
            msg = "Finish reviewing and continue?"
        if messagebox.askyesno("Skip All — are you sure?", msg):
            self.root.destroy()


def review_results(results: list[ImageResult], output_folder: str | Path) -> None:
    """
    Launch the interactive review window over every image that has at least
    one scored candidate (chosen or not -- results only ever includes an
    image at all if it had a chosen flake or a dark spot, see detect_flakes,
    so this is effectively every image already flagged). Mutates
    FlakeMetrics.chosen in place; only images the user explicitly Saves get
    their on-disk files rewritten (see FlakeReviewUI). Blocks until the
    window is closed (Skip All, or the window's own close button).
    """
    reviewable = [r for r in results if r.flakes]
    if not reviewable:
        print("No flakes to review — skipping interactive review.")
        return
    root = tk.Tk()
    FlakeReviewUI(root, results, reviewable, Path(output_folder))
    root.mainloop()


def maybe_review(results: list[ImageResult], output_folder: str | Path, config) -> None:
    """
    Launch review_results only if config.interactive_review is set. The
    single gate both flake_detection.run and all_code.py's orchestrator
    check, so a headless/scripted detect_flakes() call never pops a GUI
    unless the caller's own config explicitly asked for it.
    """
    if getattr(config, 'interactive_review', False):
        review_results(results, output_folder)
