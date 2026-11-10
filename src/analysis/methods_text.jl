# The METHODS_meanstd.md body, written verbatim into each results dir by
# write_methods_doc. Kept as a plain const so the documentation and the code that
# produces it never drift apart.

const _METHODS_TEXT = """
# Methods — the two-level mean/std model

This benchmark compares value-iteration variants computing reachability probabilities
in interval MDPs. The convergence metric is the **sound gap** `U − L ∈ [0, 1]`,
monotonically non-increasing. Each run records `(wall_time, bellman_updates, gap)` over
time. Each (algorithm, problem) pair is run with **n repeated trials** (seeds).

The gap is reduced over the states the sweep actually converged on: the **initial
states** when the sweep set `[property].restrict_to_initial = true` (the quantity its
solvers terminate on), and **all states** otherwise. Every figure and every derived
scalar below uses that one basis, which `manifest.txt` names in its COMPLETENESS block
and which is stamped onto the figures whenever it is not the all-states gap.

The **reference algorithm** is `gsrdp_exhaustive_state`. Every relative analysis
normalizes against it. The convergence target `ε` is **absolute** on the gap (the gap
is already in `[0,1]`); no relative normalization is applied to it.

Derived per-run scalars: **`solve_time`** = the `wall_time` at which `gap` first drops
to ≤ ε (and **`solve_updates`** on the Bellman-update axis). If a run never reaches ε
its solve scalar is **missing** (right-censored). When the sweep declared
`[benchmark].timeout_s`, that is the censoring budget. Otherwise the solver has no
explicit iteration/time budget — it terminates on ε — so censored runs are entered,
where a finite stand-in is needed, at the largest value observed for that problem.

Note that "right-censored" here covers two different things, which `manifest.txt`
reports separately: a run that produced data but never crossed the *analysis* ε (which
includes runs that met their own, looser, `convergence_eps`), and a run the harness
stopped mid-solve at `timeout_s`. A cell missing from the sweep entirely — killed by a
cluster wall clock — is in neither count and appears in no figure; the COMPLETENESS
block of `manifest.txt` is the only place it is recorded.

## Two distinct sources of variation

- **Level 1 — trial-repeat (seed) variation.** Within a *fixed* (algorithm, problem),
  the repeats differ mostly in `wall_time` (JIT/GC/OS noise). `bellman_updates` is
  typically deterministic per algorithm unless sweep order is randomized, so its
  seed-level std is often ~0. We reduce the repeats to one per-problem value, the
  **mean ± std** (the median is also computed and kept in the CSVs). The Level-1 std is
  **run-to-run measurement variability**, shown as error bars/bands on per-problem
  objects.
- **Level 2 — problem-set variation.** Across *different* problems, the per-problem
  (Level-1-reduced) values form a distribution. The **mean ± band** (analysis 1) lives
  here. The Level-2 spread is **benchmark heterogeneity**, NOT measurement noise.

Analyses **1–2 aggregate over the problem set** (Level-1 collapsed first, then
Level-2). Analyses **3–5 show the problem set as the spread of points/lines itself**
(no Level-2 aggregation), using Level-1 std only as error bars.

## Per-analysis usage

| # | Plot | each point/line is | Level-1 (over seeds) | Level-2 (over problems) | error bar / band shows |
|---|---|---|---|---|---|
| 1 | Normalized collapsed gap curves | one line per algorithm (wall-clock + Bellman-updates panels) | per-(alg,problem) gap curve, mean over seeds on common τ grid | geometric mean gap at each normalized-time gridpoint | ±1 geometric-std band over problems (Level-2) |
| 2 | Cactus / quantile plot | one line per algorithm | mean solve_time per (alg,problem) | sort per-problem times ascending (no averaging across problems) | per-point seed-std whiskers (Level-1) |
| 3 | Per-problem gap-vs-time | one line per algorithm, one figure per problem | per-(alg,problem) gap curve, mean over seeds | none (per-problem) | ± seed-std band (Level-1) |
| 4 | alg vs reference time scatter | one point per problem | mean solve_time over seeds (x=ref, y=alg) | none (each point is a problem) | seed-std whiskers in x and y (Level-1) |
| 5 | time / updates vs num_transitions scatter | one point per (alg,problem) | mean solve_time / Bellman updates over seeds | none (all problems shown) | vertical seed-std whiskers (Level-1) |

## Figure captions

- **normalized_gap_curves** — Wall-clock and Bellman-update panels side by side (shared
  legend). Lines are the Level-2 **geometric mean** gap over problems at each
  `τ = x/X_ref(problem)` (x = wall_time / Bellman updates; log τ axis); the band is ±1 Level-2
  **geometric std** (`gmean/GSD … gmean·GSD`, multiplicative, so it may exceed the gap's
  [0,1] range). Per (alg,problem) the gap is the Level-1 mean over seeds. The reference
  crosses gap ≈ ε at τ ≈ 1.
- **cactus** — Wall-clock and Bellman-update panels side by side (shared legend):
  sorted Level-1 mean solve_time / Bellman updates over solved instances (no
  cross-problem averaging), with ± Level-1 seed-std whiskers (low end clamped on the
  log axis when the std exceeds the mean). Line length = number solved.
- **per_problem/&lt;problem&gt;_gap** — 2×2 (shared legend): columns are the
  gap-vs-wall-time and gap-vs-Bellman-updates panels, rows are linear and log y.
  Level-1 **mean** gap over seeds with a ± Level-1 seed-std band; one line per
  algorithm. No cross-problem aggregation.
- **per_problem/&lt;problem&gt;_bounds** — the same two panels, but the bounds themselves
  rather than the gap between them: lower bound L (solid) and upper bound U (dashed)
  per algorithm, each the mean over the model's initial states S₀, then the Level-1
  **mean** over samples with a ± Level-1 sample-std band. Linear y only. Emitted only
  for a sweep that recorded value traces (`[output].traces` includes `"value"` or
  `"initial_value"`).
- **scatter_vs_reference** — 2×2: columns are solve_time / Bellman-update panels, rows
  are linear and log–log axes (shared legend). One point per problem: x = reference,
  y = algorithm, Level-1 **mean**; all non-reference algorithms overlaid (color).
  Whiskers are Level-1 seed-std in both axes. `y=x` drawn.
- **vs_transitions** — 2×2: columns are solve_time / Bellman-update panels, rows are
  linear and log–log axes (shared legend). One point per (algorithm, problem): Level-1
  **mean** vs num_transitions, vertical Level-1 seed-std whiskers; dashed per-algorithm
  power-law fit `y ≈ c·n^α` (OLS on logs, solved points only).

  On the log rows of both scatter figures a whisker whose low end would land at or below
  zero — which happens whenever the seed-std exceeds the mean, as it can for a problem
  that solves in milliseconds — is floored at 10% of the value, since a log axis cannot
  draw it otherwise. The statistic is unchanged (the high end still carries the full
  seed-std, and the linear rows are untouched); the figure caption says how many whisker
  ends were floored, so a lower end measured off a log panel may be a floor rather than
  `mean − std`.
"""
