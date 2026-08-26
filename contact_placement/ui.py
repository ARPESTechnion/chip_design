"""
Tkinter parameter form + preview for contact_placement's "popping UI" (per
prompt.docx's own UI section: "This part will start by popping UI to insert
variables, allowing the user to first inspect the results, change them and
choosing the right parameters"). Exposes exactly the fields each mode's spec
section calls out as ranges to search -- Hall: number of halls to try, min
flake size for hall; Ring: number of rings to try, inner_ring and outer_ring
(each an explicit min/max/step range per the spec's updated "Ring:"
measurements section, e.g. "Inner ring: 0.7 - 1 with 0.05 jumps" -- see
config.frange) -- plus min_flake_score (this stage's own "chosen flake"
gate) and the I/O paths. Ring's other 3 named measurements
(distance_between_arms, arm_width, finger_width) are spec-fixed constants,
not exposed here, same as every other fixed numeric constant (pad widths,
taper widths, spacing minimums, ...).

Preview runs a full placement pass (layout.place_all) and renders every
successfully placed flake on a canvas, color-coded by CIF layer (blue =
L041 cut, red = L042 internal, green = L040 external), before Export
commits anything to disk -- matches the spec's own "inspect... then choose"
sequencing. Export writes a standalone CIF (cif_export.write_placement_cif)
and splices it into the SAME working master template grid_preperation.run()
already produced (found by locating the one .cif file sitting directly in
grid_preperation_output_dir -- see cif_export.find_working_template).

Deliberately NOT wired into any other pipeline entry point -- this is the
interactive checkpoint itself. Run directly: python -m contact_placement.ui
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from shapely.geometry.base import BaseGeometry

from .cif_export import find_working_template, place_in_cif, write_placement_cif
from .config import ContactPlacementConfig
from .layout import PlacementResult, place_all

_LAYER_COLORS = {'cut': '#3050ff', 'internal': '#ff3020', 'external': '#20c030'}


class ContactPlacementUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Contact Placement")
        self.results: list[PlacementResult] = []
        self.config: ContactPlacementConfig | None = None

        self.vars = {
            'grid_preperation_output_dir': tk.StringVar(),
            'output_dir': tk.StringVar(),
            'min_flake_score': tk.StringVar(value='0.3'),
            'num_halls_to_try': tk.StringVar(value='10'),
            'min_flake_size_for_hall': tk.StringVar(value='10.0'),
            'num_rings_to_try': tk.StringVar(value='10'),
            'inner_ring_min': tk.StringVar(value='0.7'),
            'inner_ring_max': tk.StringVar(value='1.0'),
            'inner_ring_step': tk.StringVar(value='0.05'),
            'outer_ring_min': tk.StringVar(value='0.25'),
            'outer_ring_max': tk.StringVar(value='0.3'),
            'outer_ring_step': tk.StringVar(value='0.025'),
        }

        form = ttk.Frame(root, padding=10)
        form.grid(row=0, column=0, sticky='nsew')

        def add_row(r: int, label: str, key: str, browse: bool = False) -> None:
            ttk.Label(form, text=label).grid(row=r, column=0, sticky='w')
            ttk.Entry(form, textvariable=self.vars[key], width=40).grid(row=r, column=1, sticky='ew')
            if browse:
                ttk.Button(form, text="Browse...", command=lambda: self._browse(key)).grid(row=r, column=2)

        add_row(0, "grid_preperation output dir:", 'grid_preperation_output_dir', browse=True)
        add_row(1, "output dir (optional):", 'output_dir', browse=True)
        add_row(2, "Min flake score:", 'min_flake_score')
        add_row(3, "Number of halls to try:", 'num_halls_to_try')
        add_row(4, "Min flake size for hall:", 'min_flake_size_for_hall')
        add_row(5, "Number of rings to try:", 'num_rings_to_try')
        add_row(6, "Inner ring min:", 'inner_ring_min')
        add_row(7, "Inner ring max:", 'inner_ring_max')
        add_row(8, "Inner ring step:", 'inner_ring_step')
        add_row(9, "Outer ring min:", 'outer_ring_min')
        add_row(10, "Outer ring max:", 'outer_ring_max')
        add_row(11, "Outer ring step:", 'outer_ring_step')

        btns = ttk.Frame(form)
        btns.grid(row=12, column=0, columnspan=3, pady=8, sticky='w')
        ttk.Button(btns, text="Preview", command=self.preview).grid(row=0, column=0, padx=4)
        self.export_btn = ttk.Button(btns, text="Export to CIF", command=self.export, state='disabled')
        self.export_btn.grid(row=0, column=1, padx=4)

        self.status = ttk.Label(form, text="")
        self.status.grid(row=13, column=0, columnspan=3, sticky='w')

        self.canvas = tk.Canvas(root, width=900, height=700, bg='white')
        self.canvas.grid(row=0, column=1, sticky='nsew', padx=10, pady=10)

        form.columnconfigure(1, weight=1)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

    def _browse(self, key: str) -> None:
        path = filedialog.askdirectory()
        if path:
            self.vars[key].set(path)

    def _build_config(self) -> ContactPlacementConfig | None:
        try:
            config = ContactPlacementConfig(
                grid_preperation_output_dir=self.vars['grid_preperation_output_dir'].get() or None,
                output_dir=self.vars['output_dir'].get() or None,
                min_flake_score=float(self.vars['min_flake_score'].get()),
                num_halls_to_try=int(self.vars['num_halls_to_try'].get()),
                min_flake_size_for_hall=float(self.vars['min_flake_size_for_hall'].get()),
                num_rings_to_try=int(self.vars['num_rings_to_try'].get()),
            )
            config.ring.inner_ring_min = float(self.vars['inner_ring_min'].get())
            config.ring.inner_ring_max = float(self.vars['inner_ring_max'].get())
            config.ring.inner_ring_step = float(self.vars['inner_ring_step'].get())
            config.ring.outer_ring_min = float(self.vars['outer_ring_min'].get())
            config.ring.outer_ring_max = float(self.vars['outer_ring_max'].get())
            config.ring.outer_ring_step = float(self.vars['outer_ring_step'].get())
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))
            return None
        if not config.grid_preperation_output_dir:
            messagebox.showerror("Missing input", "grid_preperation output dir is required")
            return None
        return config

    def preview(self) -> None:
        config = self._build_config()
        if config is None:
            return
        self.status.config(text="Running placement...")
        self.root.update_idletasks()
        try:
            self.results = place_all(config)
        except Exception as e:  # noqa: BLE001 - surfaced to the user, not a crash
            messagebox.showerror("Placement failed", str(e))
            self.status.config(text="")
            return
        self.config = config
        placed = [r for r in self.results if r.mode is not None]
        num_hall = sum(1 for r in placed if r.mode == 'hall')
        num_ring = sum(1 for r in placed if r.mode == 'ring')
        self.status.config(text=f"{len(placed)}/{len(self.results)} flakes placed "
                                 f"({num_hall} hall, {num_ring} ring)")
        self.export_btn.config(state='normal' if placed else 'disabled')
        self._render(placed)

    def _render(self, placed: list[PlacementResult]) -> None:
        self.canvas.delete('all')
        if not placed:
            return
        all_geoms: list[BaseGeometry] = []
        for r in placed:
            all_geoms.append(r.result.cut_mask)
            all_geoms += r.result.internal_contacts
            all_geoms += [p.pad for p in r.result.external] + [p.arm for p in r.result.external]
            if r.mode == 'ring':
                all_geoms += r.result.arrows

        xs: list[float] = []
        ys: list[float] = []
        for g in all_geoms:
            minx, miny, maxx, maxy = g.bounds
            xs += [minx, maxx]
            ys += [miny, maxy]
        margin = 20.0
        minx, maxx = min(xs) - margin, max(xs) + margin
        miny, maxy = min(ys) - margin, max(ys) + margin
        w = self.canvas.winfo_width() or 900
        h = self.canvas.winfo_height() or 700
        scale = min(w / max(maxx - minx, 1e-6), h / max(maxy - miny, 1e-6))

        def to_px(pt: tuple[float, float]) -> tuple[float, float]:
            x, y = pt
            return (x - minx) * scale, h - (y - miny) * scale

        def draw(geom: BaseGeometry, color: str) -> None:
            # tkinter Canvas polygons can't express holes either -- fill the
            # exterior, then paint each interior ring (hole) back over in
            # the canvas background color, so a donut shape (e.g. Ring's
            # ring-arms cut-mask frame) renders as an actual ring on the
            # preview instead of a solid box (same underlying issue as
            # cif_export._keyhole_path, different fix since Canvas has no
            # keyhole-path equivalent -- just draws shapes, not booleans).
            polys = geom.geoms if hasattr(geom, 'geoms') else [geom]
            for poly in polys:
                if poly.is_empty:
                    continue
                pts = [coord for xy in poly.exterior.coords for coord in to_px(xy)]
                self.canvas.create_polygon(pts, fill=color, outline='black')
                for interior in poly.interiors:
                    hole_pts = [coord for xy in interior.coords for coord in to_px(xy)]
                    self.canvas.create_polygon(hole_pts, fill=self.canvas['bg'], outline='black')

        for r in placed:
            draw(r.result.cut_mask, _LAYER_COLORS['cut'])
            for c in r.result.internal_contacts:
                draw(c, _LAYER_COLORS['internal'])
            for p in r.result.external:
                draw(p.pad, _LAYER_COLORS['external'])
                draw(p.arm, _LAYER_COLORS['external'])
            if r.mode == 'ring':
                for a in r.result.arrows:
                    draw(a, _LAYER_COLORS['external'])

    def export(self) -> None:
        if self.config is None or not self.results:
            return
        grid_dir = Path(self.config.grid_preperation_output_dir)
        template = find_working_template(grid_dir)
        if template is None:
            messagebox.showerror(
                "No working template found",
                f"Expected exactly one .cif file directly inside {grid_dir} "
                f"(the working master template grid_preperation.run() copies there). "
                f"Found none or more than one -- resolve manually.",
            )
            return

        output_dir = Path(self.config.output_dir) if self.config.output_dir \
            else grid_dir / 'contact_placement_results'
        output_dir.mkdir(parents=True, exist_ok=True)
        export_cif = output_dir / 'contacts.cif'
        write_placement_cif(self.results, self.config, export_cif)
        try:
            place_in_cif(export_cif, template)
        except Exception as e:  # noqa: BLE001 - surfaced to the user, not a crash
            messagebox.showerror("Export failed", str(e))
            return
        messagebox.showinfo("Exported", f"Wrote {export_cif}\nSpliced into {template}")


def main() -> None:
    root = tk.Tk()
    ContactPlacementUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
