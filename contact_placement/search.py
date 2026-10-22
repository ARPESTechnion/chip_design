"""
Contact placement as a search: choose, for every candidate flake, one Hall or
Ring option (or none) so the total objective is as high as possible, instead
of layout.place_all's greedy one-pass placement in score order.

Options per flake: Hall (one option -- its own adaptive pad search), and Ring
at 2 exit sides x SearchConfig.ring_distance_levels pad-distance BANDS
(farthest first): band k starts at distance level k and slides inward in
1 um steps up to level k+1, so together the bands cover every distance the
greedy pass could pick.

What is maximised (SearchConfig.objective) and what steers the search
(SearchConfig.flake_order / option_order / tie_break_by_score) are separate:
the objective gives each option a value -- 'count' = 1, 'score' = the flake's
score, 'weighted' = count_weight + score_weight * score - distance penalty
(0 at the farthest Ring distance, distance_penalty_max at the closest) -- and
the heuristic settings only change the ORDER things are tried in, never the
value or the bound, so the answer's quality doesn't depend on them (only how
fast good layouts are found within the step budget).

The search is a depth-first branch-and-bound:

- Flakes are decided one at a time in a fixed order (flake_order); at each
  flake the children are its options (option_order) then "skip".
  A fixed order means every SET of placements is built exactly once
  (A-then-B, never also B-then-A).
- A child's placement is computed GIVEN everything its parents placed: the
  option's alone placement is reused when it doesn't clash with what's
  placed nearby; otherwise the option is re-placed against the actual
  neighbours (Ring: other ring sizes at the same side/distance; Hall: its
  own pad-distance search) and cached.
- Bound, in the objective's own units: value so far + every undecided
  flake's best objective value ALONE (for 'count': placed so far + undecided
  flakes that fit alone). A branch whose bound can't beat the best complete
  layout found so far is cut. With tie_break_by_score, a branch whose bound
  only TIES the best is still explored if its score bound (score so far +
  undecided feasible flakes' scores) could beat the best layout's score.
- An option infeasible alone is infeasible in every state (more obstacles
  never help), so those are dropped up front.
- A flake whose possible footprint can't reach any later flake doesn't
  branch: its best-objective feasible option is taken (skipping it can't
  help anyone).
- Flakes split into independent clusters (footprints that can't touch);
  each cluster is searched separately, sharing the step budget.
- The search starts from the greedy layout (layout._place_greedy) as its
  incumbent, plus two quick first-feasible passes, so it can only return a
  layout at least as good as greedy's, and a step budget that runs out still
  returns a complete layout.

One step = evaluating one option of one flake in one search state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.geometry import Polygon as ShapelyPolygon

from .config import ContactPlacementConfig
from .geometry import ChipBounds, collides, largest_inscribed_rectangle
from .hall import HallResult, place_hall
from .ring import RingResult, place_ring


@dataclass(frozen=True)
class Option:
    mode: Literal['hall', 'ring']
    side: float | None = None   # Ring exit side (+1 / -1)
    level: int | None = None    # Ring pad distance level, 0 = farthest


@dataclass
class _Placement:
    option: Option
    result: HallResult | RingResult
    geom: BaseGeometry
    pads: BaseGeometry
    bounds: tuple[float, float, float, float]


@dataclass
class _Flake:
    idx: int
    candidate: object                 # layout.FlakeCandidate
    local_bodies: BaseGeometry        # other flakes' outlines within reach
    options: list[Option]             # feasible alone, in option_order (heuristic)
    alone: dict[Option, _Placement]
    values: dict[Option, float]       # objective value of each feasible option
    reach: tuple[float, float, float, float]

    @property
    def best_value(self) -> float:
        return max(self.values.values(), default=0.0)

    @property
    def score(self) -> float:
        return self.candidate.final_score

    @property
    def by_value(self) -> list[Option]:
        """Feasible options, best objective value first (heuristic order breaks ties)."""
        return sorted(self.options, key=lambda o: -self.values[o])


def _bbox_overlap(a, b) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _grow(b, m):
    return (b[0] - m, b[1] - m, b[2] + m, b[3] + m)


class _Searcher:
    def __init__(self, config: ContactPlacementConfig, candidates, chip_bounds: ChipBounds | None):
        self.cfg = config
        self.s = config.search
        self.chip_bounds = chip_bounds
        self.spacing = max(config.hall.min_pad_spacing, config.ring.min_pad_spacing)
        self.n_levels = max(1, int(self.s.ring_distance_levels))
        self.steps = 0
        self.cache: dict = {}
        self.objective = {'heuristic': 'weighted'}.get(self.s.objective, self.s.objective)
        checks = {
            'objective': (self.objective, ('count', 'score', 'weighted')),
            'flake_order': (self.s.flake_order, ('score', 'loaded')),
            'option_order': (self.s.option_order, ('distance', 'score', 'weighted')),
        }
        for name, (val, allowed) in checks.items():
            if val not in allowed:
                raise ValueError(f"search.{name} must be one of {allowed}, got {val!r}")

        bodies = [ShapelyPolygon(c.vertices_um) for c in candidates]
        bodies = [b if b.is_valid else b.buffer(0) for b in bodies]
        tree = STRtree(bodies)
        # Everything a contact can reach: pad centres stay in the work-area
        # square, the pads themselves up to their half-diagonal beyond it.
        pad_reach = max(math.hypot(m.external_pad_width, m.external_pad_height) / 2 for m in (config.hall, config.ring))
        half = max(config.hall.work_area, config.ring.work_area) / 2 + pad_reach + self.spacing + 10.0
        self.flakes: list[_Flake] = []
        for i, (c, body) in enumerate(zip(candidates, bodies)):
            cx, cy = body.centroid.x, body.centroid.y
            near = [bodies[j] for j in tree.query(box(cx - half, cy - half, cx + half, cy + half)) if j != i]
            local = unary_union(near) if near else ShapelyPolygon()
            f = _Flake(i, c, local, [], {}, {}, (cx - half, cy - half, cx + half, cy + half))
            self._precompute_alone(f, (cx, cy))
            self.flakes.append(f)

    # -- options and values --------------------------------------------------

    def weighted(self, f: _Flake, opt: Option) -> float:
        penalty = 0.0
        if opt.mode == 'ring' and self.n_levels > 1:
            penalty = self.s.distance_penalty_max * opt.level / (self.n_levels - 1)
        return self.s.count_weight + self.s.score_weight * f.score - penalty

    def value(self, f: _Flake, opt: Option) -> float:
        """The option's value under the OBJECTIVE (what's maximised); a Hall's
        is multiplied by hall_weight."""
        if self.objective == 'count':
            base = 1.0
        elif self.objective == 'score':
            base = f.score
        else:
            base = self.weighted(f, opt)
        return base * self.s.hall_weight if opt.mode == 'hall' else base

    def _option_key(self, f: _Flake, opt: Option):
        """Sort key for option_order (heuristic only). Distance level: Hall
        counts as the farthest (it has no distance penalty)."""
        level = 0 if opt.mode == 'hall' else opt.level
        hall_first = 0 if opt.mode == 'hall' else 1
        if self.s.option_order == 'distance':
            return (level, hall_first)
        if self.s.option_order == 'score':
            return (-f.score, level, hall_first)  # same for every option of one flake -> by distance
        w = self.weighted(f, opt) * (self.s.hall_weight if opt.mode == 'hall' else 1.0)
        return (-w, level, hall_first)

    def _all_options(self, f: _Flake) -> list[Option]:
        opts: list[Option] = []
        if self.cfg.num_halls_to_try > 0:
            try:
                nf = largest_inscribed_rectangle(f.candidate.vertices_um)
                length, width = max(nf.width, nf.height), min(nf.width, nf.height)
                if length >= self.cfg.min_flake_length_for_hall and width >= self.cfg.min_flake_width_for_hall:
                    opts.append(Option('hall'))
            except ValueError:
                pass
        if self.cfg.num_rings_to_try > 0:
            opts += [Option('ring', side, level) for level in range(self.n_levels) for side in (1.0, -1.0)]
        return opts

    def _place(self, f: _Flake, opt: Option, occupied: BaseGeometry, other_pads: BaseGeometry | None):
        verts = f.candidate.vertices_um
        if opt.mode == 'hall':
            res = place_hall(verts, self.cfg.hall, occupied, self.chip_bounds, other_pads)
        else:
            res = place_ring(
                verts, self.cfg.ring, occupied, self.chip_bounds, other_pads,
                exit_sides=(opt.side,), radial_level=(opt.level, self.n_levels),
                stop_at_first_inner_fit=True,
            )
        if res is None:
            return None
        from .layout import _committed_geometry
        geom = _committed_geometry(opt.mode, res)
        pads = unary_union([p.pad for p in res.external])
        return _Placement(opt, res, geom, pads, geom.bounds)

    def _precompute_alone(self, f: _Flake, center) -> None:
        for opt in self._all_options(f):
            pl = self._place(f, opt, f.local_bodies, None)
            if pl is not None:
                f.alone[opt] = pl
                f.values[opt] = self.value(f, opt)
        f.options = sorted(f.alone, key=lambda o: self._option_key(f, o))
        # Where this flake's contacts can ever be: Ring options are fixed in
        # side + distance (only the tiny ring size varies on a re-place), so
        # their alone footprints bound them; Hall re-runs its own pad-distance
        # search, so it may land anywhere in its work area.
        boxes = [_grow(pl.bounds, self._band_width(pl)) for o, pl in f.alone.items() if o.mode == 'ring']
        if any(o.mode == 'hall' for o in f.alone):
            h = self.cfg.hall.work_area / 2
            boxes.append((center[0] - h, center[1] - h, center[0] + h, center[1] + h))
        if boxes:
            f.reach = _grow((min(b[0] for b in boxes), min(b[1] for b in boxes),
                             max(b[2] for b in boxes), max(b[3] for b in boxes)), self.spacing + 5.0)

    def _band_width(self, pl: _Placement) -> float:
        """How far a Ring option's pads may slide inward from where they landed
        (its distance band), so its footprint can move by at most this."""
        lo, hi = pl.result.external[0].radial_bounds
        return (hi - lo) / max(1, self.n_levels - 1)

    def add_option(self, f: _Flake, opt: Option, pl: _Placement) -> None:
        """Make `opt` (with alone placement `pl`) one of f's options -- used to
        seed the greedy layout's choices that the alone pass didn't produce."""
        if opt in f.alone:
            return
        f.alone[opt] = pl
        f.values[opt] = self.value(f, opt)
        f.options = sorted(f.alone, key=lambda o: self._option_key(f, o))
        b = pl.bounds if opt.mode == 'hall' else _grow(pl.bounds, self._band_width(pl))
        b = _grow(b, self.spacing + 5.0)
        r = f.reach
        f.reach = (min(r[0], b[0]), min(r[1], b[1]), max(r[2], b[2]), max(r[3], b[3]))

    # -- placement in a search state ----------------------------------------

    def _compatible(self, pl: _Placement, placed: list[_Placement]) -> bool:
        grown = _grow(pl.bounds, self.spacing)
        for q in placed:
            if not _bbox_overlap(grown, q.bounds):
                continue
            if collides(pl.geom, q.geom) or pl.pads.distance(q.pads) < self.spacing - 1e-9:
                return False
        return True

    def _place_in_state(self, f: _Flake, opt: Option, placed: list[_Placement]):
        alone = f.alone[opt]
        if self._compatible(alone, placed):
            return alone
        near = tuple(sorted(id(q) for q in placed if _bbox_overlap(f.reach, q.bounds)))
        key = (f.idx, opt, near)
        if key not in self.cache:
            near_pl = [q for q in placed if _bbox_overlap(f.reach, q.bounds)]
            occupied = unary_union([f.local_bodies] + [q.geom for q in near_pl])
            pads = unary_union([q.pads for q in near_pl])
            pl = self._place(f, opt, occupied, pads)
            # The fresh placement avoided `near_pl`; anything farther can't touch it.
            self.cache[key] = pl
        return self.cache[key]

    # -- clusters ------------------------------------------------------------

    def clusters(self) -> list[list[_Flake]]:
        live = [f for f in self.flakes if f.options]
        parent = {f.idx: f.idx for f in live}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        tree = STRtree([box(*f.reach) for f in live])
        for a, f in enumerate(live):
            for b in tree.query(box(*f.reach)):
                if b > a:
                    parent[find(f.idx)] = find(live[b].idx)
        groups: dict[int, list[_Flake]] = {}
        for f in live:
            groups.setdefault(find(f.idx), []).append(f)
        if self.s.flake_order == 'loaded':
            key = lambda f: (f.candidate.image, f.candidate.flake_index)  # file order, as loaded
        else:
            key = lambda f: (-f.score, f.idx)
        return [sorted(g, key=key) for g in groups.values()]

    # -- the search ----------------------------------------------------------

    def solve_cluster(self, flakes: list[_Flake], budget: int, caps: dict[str, int],
                      seed: list[tuple[_Flake, _Placement]] = ()):
        n = len(flakes)
        tie = bool(self.s.tie_break_by_score)
        suffix = [0.0] * (n + 1)        # objective bound for the undecided flakes
        suffix_score = [0.0] * (n + 1)  # tie-break bound: their scores
        for i in range(n - 1, -1, -1):
            suffix[i] = suffix[i + 1] + flakes[i].best_value
            suffix_score[i] = suffix_score[i + 1] + flakes[i].score
        # flake i doesn't branch when its reach can't touch any later flake's
        independent = [
            not any(_bbox_overlap(flakes[i].reach, flakes[j].reach) for j in range(i + 1, n))
            for i in range(n)
        ]

        placed: list[_Placement] = []
        chosen: list[tuple[_Flake, _Placement]] = []
        counts = {'hall': 0, 'ring': 0}
        best = {'value': -1.0, 'score': -1.0, 'layout': []}
        stop_at = self.steps + budget  # the incumbent pass below counts too
        state = {'value': 0.0, 'score': 0.0, 'exhausted': False}
        eps = 1e-9

        def better(value, score) -> bool:
            if value > best['value'] + eps:
                return True
            return tie and value > best['value'] - eps and score > best['score'] + eps

        def record():
            if better(state['value'], state['score']):
                best['value'], best['score'] = state['value'], state['score']
                best['layout'] = list(chosen)

        def push(f, pl):
            placed.append(pl)
            chosen.append((f, pl))
            counts[pl.option.mode] += 1
            state['value'] += f.values[pl.option]
            state['score'] += f.score

        def pop(f, pl):
            placed.pop()
            chosen.pop()
            counts[pl.option.mode] -= 1
            state['value'] -= f.values[pl.option]
            state['score'] -= f.score

        # Incumbents (no budget limit -- at most one step per option per flake),
        # so a complete layout always exists before the DFS starts:
        #  1. first-feasible pass in the search's own flake/option order;
        #  2. greedy's order -- every flake's Hall first, then Rings for the
        #     rest -- so the result is never worse than layout.place_all's
        #     greedy pass, which a large, never-completed cluster could
        #     otherwise miss (its Halls get crowded out by earlier Rings).
        def first_feasible(order):
            for f, opts in order:
                if any(c is f for c, _ in chosen):
                    continue
                for opt in opts:
                    if counts[opt.mode] >= caps[opt.mode]:
                        continue
                    self.steps += 1
                    pl = self._place_in_state(f, opt, placed)
                    if pl is not None:
                        push(f, pl)
                        break

        # 0. the greedy layout itself (a valid layout: greedy checked it)
        for f, pl in seed:
            push(f, pl)
        record()
        while chosen:
            pop(*chosen[-1])

        passes = [
            [[(f, f.by_value if independent[i] else f.options) for i, f in enumerate(flakes)]],
            [[(f, [o for o in f.options if o.mode == 'hall']) for f in flakes],
             [(f, [o for o in f.options if o.mode == 'ring']) for f in flakes]],
        ]
        for phases in passes:
            for order in phases:
                first_feasible(order)
            record()
            while chosen:
                pop(*chosen[-1])
        if all(independent):
            return best['layout'], best['value'], True  # no interactions: best-value pass is optimal

        def dfs(i: int) -> None:
            if i == n:
                record()
                return
            if not better(state['value'] + suffix[i], state['score'] + suffix_score[i]):
                return
            f = flakes[i]
            # An independent flake takes its best-objective feasible option
            # (it can't affect later flakes), so try those in objective order.
            for opt in (f.by_value if independent[i] else f.options):
                if self.steps >= stop_at:
                    state['exhausted'] = True
                    return
                if counts[opt.mode] >= caps[opt.mode]:
                    continue
                self.steps += 1
                pl = self._place_in_state(f, opt, placed)
                if pl is None:
                    continue
                push(f, pl)
                dfs(i + 1)
                pop(f, pl)
                if independent[i] or state['exhausted']:
                    return  # best feasible option taken; skipping can't help later flakes
            dfs(i + 1)  # skip this flake

        dfs(0)
        return best['layout'], best['value'], not state['exhausted']


def place_all_search(config: ContactPlacementConfig, candidates, chip_bounds: ChipBounds | None):
    """Search-based replacement for layout.place_all's greedy loop. Returns
    one layout.PlacementResult per candidate (mode None = no contacts)."""
    from .contacts import distance_level
    from .layout import PlacementResult, _committed_geometry, _place_greedy

    searcher = _Searcher(config, candidates, chip_bounds)

    # The greedy layout is the search's starting point (see module docstring).
    greedy = _place_greedy(config, candidates, chip_bounds)
    seeds: dict[int, _Placement] = {}
    for f, g in zip(searcher.flakes, greedy):
        if g.mode is None:
            continue
        if g.mode == 'hall':
            opt = Option('hall')
        else:
            ext = g.result.external[0]
            opt = Option('ring', g.result.exit_sign, distance_level(ext.radial_dist, ext.radial_bounds, searcher.n_levels))
        geom = _committed_geometry(g.mode, g.result)
        pl = _Placement(opt, g.result, geom, unary_union([p.pad for p in g.result.external]), geom.bounds)
        searcher.add_option(f, opt, pl)
        seeds[f.idx] = pl

    clusters = sorted(searcher.clusters(), key=len)
    caps = {'hall': config.num_halls_to_try, 'ring': config.num_rings_to_try}
    results = {id(c): PlacementResult(candidate=c, mode=None, result=None) for c in candidates}

    remaining_steps = config.search.max_steps
    remaining_size = sum(len(c) for c in clusters if len(c) > 1)
    total_value, total_score, proven = 0.0, 0.0, 0
    for cl in clusters:
        if len(cl) > 1:
            budget = max(1, int(remaining_steps * len(cl) / remaining_size)) if remaining_size else 0
            remaining_size -= len(cl)
        else:
            budget = 1
        before = searcher.steps
        seed = [(f, seeds[f.idx]) for f in cl if f.idx in seeds]
        layout, value, complete = searcher.solve_cluster(cl, budget, caps, seed)
        remaining_steps = max(0, remaining_steps - (searcher.steps - before))
        total_value += max(value, 0.0)
        proven += complete
        for f, pl in layout:
            total_score += f.score
            caps[pl.option.mode] -= 1
            results[id(f.candidate)] = PlacementResult(
                candidate=f.candidate, mode=pl.option.mode, result=pl.result,
                exit_side=pl.option.side, distance_level=pl.option.level, value=f.values[pl.option],
                weighted_value=searcher.weighted(f, pl.option),
            )

    n_feasible = sum(1 for f in searcher.flakes if f.options)
    print(f"Search: {len(candidates)} candidates, {n_feasible} fit alone, greedy start {len(seeds)}, {len(clusters)} clusters "
          f"({proven} searched to completion), {searcher.steps} steps; objective={searcher.objective} "
          f"value={total_value:.3f}, total score={total_score:.3f}")
    return [results[id(c)] for c in candidates]
