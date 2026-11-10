# Benchmark Experiment Config Reference

Experiment configs are TOML files passed to `run.jl`. They define which pre-built problems to solve, which algorithm variants to run, timing/sampling settings, and what to save.

```
cd IntervalMDP-GSRDP-Benchmark.jl   # the repository root
julia --project run.jl configs/smoke.toml
julia --project run.jl configs/a.toml configs/b.toml     # sequential experiments
```

Problems live under `problems/`, one subdirectory per problem. Each subdirectory holds either a `.nc` + `.json` pair (IntervalMDP native format) or a PRISM `.sta/.tra/.lab/.pctl` set. The benchmark's 38 problems are built by the scripts listed in [problems/README.md](../problems/README.md).

Results land in `results/<name>_<timestamp>/` and contain `results.csv`, per-cell `trace.csv`, `meta.json`, `resolved_config.toml`, and `plots/`.

---

## Top-level fields

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `name` | string | filename stem | Label used in the results directory name and plot titles |

---

## `[benchmark]`

Timing and repetition settings.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `samples` | int | `5` | Number of benchmark samples per (problem × algorithm) cell, used only when `seeds` is omitted; each sample draws a fresh random seed |
| `warmup` | bool | `true` | One untimed solve (with a throwaway callback) before timing to prime the JIT |
| `seeds` | int array | `[]` (random) | Explicit per-sample seeds. If given, the cell runs one sample per seed (reproducible) and `samples` is ignored; if omitted, `samples` samples each get a fresh random seed (recorded per row). Seeds are shared across all problem × algorithm pairs (paired comparison) |
| `timeout_s` | float | `0.0` (off) | Per-**solve** wall-clock budget. A sample that exceeds it is stopped, recording a real `status="timeout"` row plus the partial convergence curve it had reached. The untimed warm-up gets **10×** this, because its cost is dominated by JIT rather than by the algorithm — a cell with 0.5 s samples can easily take seconds to warm up, and censoring on that would discard cells whose real runs fit comfortably |
| `wall_budget_s` | float | `0.0` (auto) | Total sweep budget. The loop stops cleanly **between** cells once it is spent. `0` derives the deadline from SLURM's `SLURM_JOB_END_TIME` when running inside a job allocation, else runs unlimited |
| `wall_reserve_s` | float | `300.0` | Slack subtracted from the sweep deadline, left for the post-loop writes (`problems.csv`, `validation.csv`, the final `meta.json`) |

Each cell is solved as the samples of a **single** BenchmarkTools benchmark: per sample, `setup` seeds and builds a fresh `Trace` callback, the timed solve runs with that callback attached, and `teardown` harvests the sample's solution + trace. There is no separate seed axis and no within-sample repetition — each sample has a different seed, so it is its own data point. One results row is emitted per sample (`evals=1`); the across-sample spread carries the variance. The callback runs inside the timed region (so it can be harvested), so timing includes callback overhead — uniform across algorithms, so relative speed-ups are preserved.

### Censoring and wall-clock kills

Left to itself a solve runs until it converges — the solver has no iteration cap, and BenchmarkTools' own `seconds` budget is deliberately not used for censoring (it is only checked *between* samples, so it cannot stop a solve in flight, and it fires silently). The three keys above exist because of what that means on a cluster: **a cell killed by SLURM's wall clock leaves no row at all**, and the analysis has no way to tell "missing" from "never ran". It silently reduces around the hole — the cactus plot renders a killed cell exactly like an instance the algorithm failed to solve.

In order of how much they buy you:

1. **The sweep deadline** (`wall_budget_s` / `SLURM_JOB_END_TIME`) stops between cells with the post-loop writes still to come, so the results directory is a *complete description of a partial sweep*. Nothing to configure inside a SLURM job.
2. **`timeout_s`** censors an individual solve. The censored run keeps its partial trace, so it still contributes a convergence curve, and the analysis uses `timeout_s` as the censoring budget instead of the max-observed fallback. Set it to something a healthy solve comfortably fits inside.
3. **A SIGTERM** (e.g. SLURM's `--signal=TERM@120`) leaves an `aborted.json` marker; a SIGKILL or the kernel OOM-killer leaves only `progress.json`, which is rewritten atomically after every cell.

Whatever happens, `meta.json` records the full expected cell list up front, so `run_analyses.jl` can name exactly which cells are absent. `manifest.txt` leads with a `SWEEP INCOMPLETE` banner and a COMPLETENESS block, and every figure carries a caption saying how much data is missing. `datetime_end` in `meta.json` is the positive signal that a sweep reached its own end; its absence means a hard kill.

---

## `[output]`

Controls what is written to the results directory.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `results_root` | string | `"results"` | Base directory for result subdirectories. Relative paths resolve against the repository root; absolute paths are used as-is. `~`, `$VAR` and `${VAR}` are expanded (see [Portable paths](#portable-paths)) |
| `save_raw` | bool | `true` | Save raw per-sample timing arrays to `results.csv` |
| `save_trace` | bool | `true` | Save per-iteration value function trace to `trace.csv` |
| `save_policy` | bool | `true` | Save the solved policy to `runs/<cell>/sample<k>/policy.csv` (`state,action`, one row per state, in flat state order). Drives the `plots/per_problem/<problem>_policy.*` grids, which colour each state by the action the solver selected. Cheap — kilobytes per sample, not the gigabytes of a heavy trace — so it is on by default. Action `0` means the solver never assigned one, which on a sampled variant is every state the sampler never reached |
| `track_initial` | bool | `true` | Add the `initial_gap_max` / `initial_gap_mean` columns to `trace.csv`, i.e. the value gap restricted to the model's initial states. Requires the problem to declare initial states — for a model with `AllStates()` both columns are `NaN` on every row (see [`[property]`](#property)) |
| `traces` | list | `["trace"]` | Which per-iteration traces to record. `"trace"` / `"initial"` are the cheap scalar gap curves that feed `trace.csv`; the rest are **heavy** wide matrices (a row per iteration, a column per state) written to their own file under `runs/<cell>/sample<k>/`: `"gap"` → `gap_trace.csv`, `"value"` → `value_trace_{lower,upper}.csv`, `"updates"` → `update_trace.csv` (per-state Bellman-backup count). `"initial_gap"` / `"initial_value"` are the same restricted to the model's initial states. A single heavy trace can reach several GB per sample (see [State-space videos](../README.md#state-space-videos)). With `traces = []` and `track_initial = true`, only the initial-state curve is recorded and the gap is computed at the initial states alone each iteration (not O(\|S\|)), which matters for fast samplers on large models; `trace.csv` then has no all-states `gap_max`/`gap_mean`, and `final_gap_*` in `results.csv` is over the initial states |
| `gap_videos` | bool | `false` | Render `plots/gap_videos/*.mp4` after the sweep — one mp4 per `(problem, algorithm)` cell, with one panel per heavy trace recorded. Off by default because of the trace sizes above; `render_gap_videos.jl` does the same job on a finished directory |
| `gap_video_max_frames` | int | `600` | Iterations per video, counted from the first. The early iterations are where the spatial propagation happens |
| `gap_video_framerate` | int | `15` | Video frames per second |
| `reference_label` | string | `"gsrdp_exhaustive_state"` | `alg_label` of the baseline variant the analysis uses as the reference (gap curves, cactus, scatter); should match a configured algorithm's label, else the analysis falls back to the first algorithm |
| `image_formats` | string or list | `["png", "svg"]` | Figure file types written to `plots/` by the analysis suite. Any of `png`, `svg`, `pdf`, `eps`; a leading dot is tolerated |
| `analysis` | bool | `true` | Run the report/plot suite after the sweep. Set `false` for raw results only — no `plots/`, no `METHODS_meanstd.md`, no `manifest.txt` (see [Skipping the analysis](#skipping-the-analysis)) |

### Skipping the analysis

`analysis = false` runs the sweep and stops before the report stage. The run directory still
gets everything the sweep itself produces — `results.csv`, `runs/*/trace.csv`, `raw/`,
`meta.json`, `resolved_config.toml`, `provenance.json`, `problems.csv`, `validation.csv` and
`validation_summary.json` — but no `plots/`, `METHODS_meanstd.md`, `manifest.txt`, or
`*_fit.csv`.

This is for batch and cluster runs, where plotting wastes allocated walltime. Produce the
figures afterwards from the finished directory:

```
julia --project run_analyses.jl results/<run-dir>
```

`run_analyses.jl` deliberately **ignores** this flag — that is the whole point of it — and
reads `image_formats` back from the run's `resolved_config.toml`, so the figures come out as
the experiment configured them. A run directory is relocatable (trace paths in `results.csv`
are stored relative to it), so the analysis can be run on a machine other than the one that
produced it.

Note that `problems.csv` and the validation report are *not* skipped. Both are cheap, produce
no figures, and are built from state that only exists in memory during the sweep, so they
cannot be reconstructed from the results directory afterwards — `problems.csv` in particular
supplies the `num_transitions` join the analysis needs.

---

## `[property]`

Optional. Overrides fields of each problem's stored specification before it is solved. An
omitted key (or an omitted section) leaves the problem's own value untouched.

| Key | Type | Default | Description |
|---|---|---|---|
| `convergence_eps` | float | from spec | Value-iteration tolerance. The solve stops once the gap falls below it |
| `restrict_to_initial` | bool | from spec | Check convergence on the model's initial states only, rather than on all states |

### `restrict_to_initial` requires initial states on the model

The flag lives on the *property*, but the states it refers to live on the *model*. Initial
states are baked into a problem's `.nc` at generation time; they cannot be supplied from a
config.

If the model declares `AllStates()` — i.e. it was generated without an initial region — then
`restrict_to_initial = true` has nothing to restrict to. IntervalMDP's
`apply_initial_restriction` silently falls back to the all-states criterion in that case, so
the solve would converge over every state while the results claimed otherwise. The harness
therefore **skips such a cell with a warning** rather than reporting a misleading speed-up,
and `initial_gap_*` in `trace.csv` would be `NaN` regardless.

To fix, regenerate the problem with an initial region — each generator under
`tools/IntervalMDPAbstractions/` defines an `INITIAL_REGION` constant for this. The initial
states are recorded unconditionally, independently of this flag, so once a problem has them
`restrict_to_initial` can be flipped in a config without regenerating anything.

---

## `[problems]`

Optional. If this section is omitted entirely, every subdirectory under `problems/` is used.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `root` | string | `"problems"` | Path to the problems root. Relative paths resolve against the repository root; absolute paths are used as-is. `~`, `$VAR` and `${VAR}` are expanded (see [Portable paths](#portable-paths)) |
| `names` | string array | all subdirs | Explicit list of problem subdirectory names to run; omit to run all |

```toml
# run all problems
# (omit [problems] section entirely)

# or restrict to a subset:
[problems]
names = ["coin2_K4", "pd2_10x20"]
```

---

## Portable paths

`results_root` and `[problems].root` are the only path-valued keys. Both go through
`_expand_path` (`src/config.jl`) at load time, which expands `~`, `$VAR` and `${VAR}` from
the environment before the relative/absolute decision is made. This lets a config avoid
hardcoding a username:

```toml
[output]
results_root = "/scratch/$USER/gsrdp/results"
```

An **undefined** variable is deliberately left as the literal `$VAR` rather than expanding
to an empty string, so a typo fails loudly (`problems root not found: /scratch/$NOPE/...`)
instead of silently writing to a plausible-looking wrong path. The expanded value is what
gets recorded in the run's `resolved_config.toml`.

Note that relative paths resolve against the repository root — **not** the config file's
directory and not the current working directory — because `run.jl` passes `root=@__DIR__`
into `Bench.main`.

---

## `[[algorithms]]`

One entry per algorithm variant. Repeat with different `[algorithms.params]` to compare variants. The auto-generated label is used as the column name in `results.csv` and plot legends.

| Key | Type | Description |
|-----|------|-------------|
| `name` | string | Registered builder name: `"rvi"` or `"gsrdp"` |
| `label` | string | Optional display/CSV label override; auto-generated as `<name>_<sampling>_k<k>` if absent |
| `[algorithms.params]` | subtable | Builder-specific parameters (see below) |

### `name = "rvi"` — Robust Value Iteration (baseline)

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `bellman` | `"default"`, `"omax"`, `"vertex"`, `"lp"` | `"default"` | Bellman operator; `"default"` auto-selects based on model type |

### `name = "gsrdp"` — Generalized Sampling-based Robust Dynamic Programming

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `bellman` | same as rvi | `"default"` | Bellman operator |
| `sampling` | see [Sampling strategies](#sampling-strategies) | `"exhaustive_state"` | State/action sampling strategy per iteration |
| `k` | int | — | Sample count per iteration; required by every `sampling` except the exhaustive sweeps and non-Gauss-Seidel `trajectory` sampling |
| `term` | `"gap"`, `"gap_initial"`, `"auto"` | `"auto"` | Termination criterion; `"auto"` derives gap from the spec's `convergence_eps` |
| `tol` | float | from spec | Explicit tolerance for `gap`/`gap_initial`; overrides spec value |
| `drive` | `"follow"`, `"both"` | `"follow"` | Which bound runs the optimizing Bellman update; see [Bound drive](#bound-drive-drive) |

#### Bound drive (`drive`)

Each iteration updates both bounds of the interval value function. By default
(`drive = "follow"`) one bound optimizes over the actions and the other is evaluated
under the action it picked. The driver is the controller-optimistic bound: the **upper**
bound under `Maximize`, the **lower** bound under `Minimize`. The returned strategy is the
driver's. Driving with the other bound is deliberately not offered: under `Maximize`,
evaluating U under L's action can pull it below the optimal value (and symmetrically
under `Minimize`), so the bracket closes on a wrong answer while reporting convergence.

`drive = "both"` lets each bound optimize on its own, and the returned strategy is the
upper bound's. It pays a full action sweep on both bounds, so `bellman_updates` grows by
`2·|A|` per relaxed state instead of `|A| + 1`.

The label gets a `_drive<value>` suffix (`gsrdp_round_robin_kf0.05_driveboth`) only
for a non-default value. A follow arm keeps its old label and pools with runs that
predate `drive`; those were all `Maximize`, where follow is the old upper-driven update.

#### Sampling strategies

`sampling` picks which part of the model each GSRDP iteration relaxes. The
values fall into five families, mirroring the `SamplingStrategy` categories in
IntervalMDP.jl — they are alternatives to the same question, not layers.

Where a family has both a `_state` and a `_state_action` form, the difference is
what the sampler yields: a state form yields bare states and the Bellman update
then sweeps every action of each one, while a state-action form yields
`(action, state)` pairs, so a state can be relaxed for one action without the
others.

| Family | Values |
|--------|--------|
| [Exhaustive sweeps](#exhaustive-sweeps) | `"exhaustive_state"`, `"exhaustive_state_action"` |
| [Random subset](#random-subset) | `"random_state"`, `"random_state_action"` |
| [Round-robin](#round-robin) | `"round_robin"`, `"round_robin_state_action"` |
| [Priority-queue sampling](#priority-queue-sampling) | `"priority_gap"`, `"priority_upper"`, `"priority_action_uncertainty"`, `"value_function_ordered"` |
| [Trajectory sampling](#trajectory-sampling) | `"trajectory"`, preset `"trajectory_brtdp"` |

##### Exhaustive sweeps

Relax the whole model every iteration — no sampling, so `k` is ignored.
`"exhaustive_state"` is the default and is what makes GSRDP reproduce RVI.

| Value | Description |
|-------|-------------|
| `"exhaustive_state"` | Full Bellman sweep over all states each iteration — same work as RVI |
| `"exhaustive_state_action"` | Sweep all state-action pairs each iteration |

The pre-rename spellings `"all_states_sweep"` and `"all_sampling"` are still
accepted, so a banked config keeps resolving. They are no longer the names to
write: `alg_label` is built from the raw `sampling` string, so a config using an
old spelling produces `gsrdp_all_states_sweep` rows that will not line up with
the `gsrdp_exhaustive_state` rows a current config produces.

##### Random subset

Draw `k` entries uniformly at random each iteration, independently and with
replacement — so the same state can come up twice in one iteration. The
cheapest way to trade per-iteration cost against iteration count, and the
baseline any targeted strategy has to beat.

| Value | Description |
|-------|-------------|
| `"random_state"` | Sample `k` states uniformly at random each iteration |
| `"random_state_action"` | Sample `k` state-action pairs uniformly at random each iteration |

##### Round-robin

Walk a cursor through the model in the same order the exhaustive sweeps
enumerate it, taking the next `k` entries each iteration and wrapping around at
the end. Same per-iteration cost as random subset, but every entry is visited on
a fixed cycle rather than only in expectation, so no state is starved. The
cursor is rewound at the start of each run, so repeated samples of one cell are
identical.

| Value | Description |
|-------|-------------|
| `"round_robin"` | Cycle through the next `k` states in a fixed order each iteration, wrapping around |
| `"round_robin_state_action"` | Cycle through the next `k` state-action pairs in a fixed order each iteration, wrapping around |

##### Size-relative `k` (`k_frac`)

`k` is an absolute count, so one value means very different things across
problems — `k = 512` is more than a full sweep of a 400-state abstraction and a
sliver of a 330k-state coin model. For the random-subset, round-robin,
priority-queue and Gauss-Seidel trajectory samplers, give `k_frac` in `(0, 1]`
instead of `k`: each problem then runs with `k = max(1, ceil(k_frac * n))`, where
`n` is its number of states (or of state-action pairs, for the `_state_action`
forms). Giving both `k` and `k_frac` is an error. For trajectory sampling `k` is
the batch size — how many states of the active trajectory are relaxed per
iteration — so `k_frac` requires `gauss_seidel = true`.

For trajectory sampling, `k_chain_outdeg = d` (with `k_frac`) uses `k = 1` instead on models
whose mean successor support -- transitions per state-action pair -- is below `d`. On
near-deterministic chains (brp, crowds, the coin protocols: support 1.2-1.7) single-state
Gauss-Seidel batches carry a rollout's values end to end, while every Jacobi batch of a
size-relative `k` moves them one step; wider-support models keep the size-relative batch.

```toml
[[algorithms]]
name = "gsrdp"
[algorithms.params]
sampling = "round_robin"
k_frac   = 0.25          # a quarter of the states per iteration, on every problem
```

The label records the fraction (`gsrdp_round_robin_kf0.25`), so it is the same
on every problem; the `sampling_k` column of `results.csv` records the absolute
`k` each problem actually ran with.

##### Priority-queue sampling

Rank every state by a priority and relax the top `k`, then repair only the
priorities that batch could have made stale. Covers the whole state space, which
is what `term = "gap"` stops on unless the property sets `restrict_to_initial`.

| Value | Description |
|-------|-------------|
| `"priority_gap"` | Incremental priority queue; top `k` states by value gap `U(s) - L(s)`, priorities only recomputed where the previous iteration's relaxations could have changed them |
| `"priority_upper"` | Same incremental priority queue; ranked by upper (optimistic) value `U(s)` |
| `"priority_action_uncertainty"` | Same incremental priority queue; ranked by action-selection uncertainty `U^{-a_L(s)}(s) - L(s)` |
| `"value_function_ordered"` | The non-incremental case: re-rank all states by the current value function each iteration and take the top `k` |

##### Trajectory sampling

Roll out one trajectory per iteration — from an initial state, pick an action,
realize a concrete transition out of the ambiguity set, draw a successor — and
relax the states visited along the way. Concentrates on the states reachable
from the initial state, which suits `term = "gap_initial"`. Unlike the families
above it is one strategy, `sampling = "trajectory"`, configured by a product of
orthogonal choices.

Every knob below is an optional key in `[algorithms.params]` and defaults to the
library default. An unrecognized key is an error, so a typo in a sweep config
fails rather than quietly running the wrong thing. The action and state keys are
hierarchical (marked ↳): each applies only under its parent's value, and setting
it under any other value is an error too.

###### Preset `trajectory_brtdp`

`sampling = "trajectory_brtdp"` is `sampling = "trajectory"` with the keys below
filled in. Any key you give explicitly overrides the preset's, and the merged keys
are validated as usual. Its Gauss-Seidel batching needs `k` or `k_frac`.

| Key | Preset value |
|-----|--------------|
| `state_policy` | `"boltzmann"` |
| `state_score`, `state_temperature` | `"log_mixture"`, `1.0` |
| `state_explore` | `"gap"` |
| `state_alpha_e_schedule`, `state_alpha_e` | `"fixed"`, `1.0` |
| `state_alpha_p_schedule`, `state_alpha_p` | `"fixed"`, `0.0` |
| `terminate`, `expected_gap_tau` | `["max_steps", "expected_gap"]`, `10.0` |
| `gauss_seidel`, `reverse` | `true`, `true` |

This is BRTDP's successor rule (McMahan et al., 2005). With α_e = 1 and α_p = 0,
the mixture is `f_S = p·(U − L) / Σ p·(U − L)`, where `p` is the realized
distribution (`p_U` under the default transition). The power rule at T = 1 then
draws `s'` with probability proportional to `p·(U − L)`, whatever the scale of the
gaps. Rollouts stop on the expected-gap rule. Each rollout is handed out goal-end
first. The action side and the successor filters stay at their defaults.

It is the closest point to the slow-tail branch's `trajectory_brtdp` (the
brtdp38 runs), but **not equal to it**:

1. **Successor weight.** slow-tail weighs successors by `max(p_U, p_L)·(U − L)`
   over `supp p_U ∪ supp p_L`. That is an upper bound on the gap at `(s, a)`. The
   preset uses `p_U·(U − L)` on `supp p_U`, which misses the share of the gap
   that only the lower-bound adversary loads. The two agree wherever
   `p_U = p_L`.
2. **Stop rule.** slow-tail sums `max(p_U, p_L)·(U − L)` in `ExpectedGapStop`
   (`max_adversary = true`). The preset sums `p_U·(U − L)`. Its sum is never
   larger, so it stops at least as early and its rollouts are at most as long.
3. **`dedup` is off.** slow-tail removes repeated states from each k-batch. The
   preset leaves `dedup = false`, so a repeat is backed up twice. Under GSRDP's
   Jacobi backup the second backup writes the same value, so values are
   unaffected, but the update count rises. Set `dedup = true` to match slow-tail
   here. `state_exclude_visited` is not a substitute: it changes where rollouts go.
4. **All-zero fallback.** When every weight is 0, the draw is uniform over
   `supp p_U` rather than over the union.

Give every arm its own `label`: `combine_results.jl` refuses one label that means two
different arms.

###### Action configs

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `action_policy` | `"epsilon_greedy"`, `"boltzmann"` | `"epsilon_greedy"` | How the action is drawn given its score |
| ↳ `action_epsilon` | float in [0,1] | `0.1` | Exploration probability; only under `action_policy = "epsilon_greedy"` |
| ↳ `action_temperature_schedule` | `"fixed"`, `"gap_based"` | `"fixed"` | How `T_a` is obtained at the start of each rollout; only under `action_policy = "boltzmann"` |
| &nbsp;&nbsp;↳ `action_temperature` | float > 0 | `1.0` | `T_a = t`; only under `action_temperature_schedule = "fixed"` |
| &nbsp;&nbsp;↳ `action_temperature_k` | float > 0 | `1.0` | `T_a = max(t_min, k·δ / (1 − δ))`, with δ the initial gap; only under `action_temperature_schedule = "gap_based"` |
| &nbsp;&nbsp;↳ `action_temperature_min` | float ≥ 0 | `0.1` | `t_min`, the exploration floor of `"gap_based"`; `0` removes it. Only under `action_temperature_schedule = "gap_based"` |
| `action_score` | `"upper"`, `"lower"`, `"weighted"`, optionally `log_`-prefixed | `"upper"` | `f_A(s,a)`: the O-max Q-value under the upper bound, the lower bound, or `L + β(U−L)`, oriented to the controller: under a `Minimize` spec the bracket is complemented (`U' = 1−L`, `L' = 1−U`), so `"upper"` is always the controller-optimistic choice (argmin L for a minimizer). `log_` wraps it in a log, so `"boltzmann"` at temperature 1 draws proportionally to the inner score |
| `action_beta` | float in [0,1] | `0.5` | β for `action_score = "weighted"` |

###### Transition configs

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `transition_bound` | `"upper"`, `"lower"` | `"upper"` | Which value function the adversary optimizes `p(·\|s,a)` against |
| `transition_adversary` | `"spec"`, `"optimistic"`, `"pessimistic"` | `"spec"` | Adversary direction. `"spec"` takes it from the specification's satisfaction mode, which is what keeps the rollout aligned with the Bellman update — override deliberately |

###### State configs

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `state_exploit` | `"upper"`, `"lower"` | `"upper"` | `f_exploit(s')`, which value function the exploit term reads. Applies under both policies |
| `state_policy` | `"epsilon_greedy"`, `"boltzmann"` | `"epsilon_greedy"` | How the successor is drawn |
| ↳ `state_epsilon` | float in [0,1] | `0.1` | Exploration probability; only under `state_policy = "epsilon_greedy"`, which draws `argmax p(s'\|s,a)·f_exploit(s')`, or uniformly from the support with this probability |
| ↳ `state_temperature` | float > 0 | `1.0` | `T_s`, fixed; only under `state_policy = "boltzmann"`. Under the default `"log_mixture"`, `1` draws exactly from `f_S`, larger flattens towards uniform and smaller sharpens towards argmax, at every support size. Under `"mixture"` its effect depends on the support size — see the note below |
| ↳ `state_score` | `"log_mixture"`, `"mixture"` | `"log_mixture"` | How the successor is drawn from `f_S(s',s,a) = (1−α_e)·f'_exploit + α_e·f'_explore`, the mixture of the normalized exploit and explore terms (see the note below). `"log_mixture"` is the power rule, `P ∝ f_S^(1/T_s)`; `"mixture"` is the plain softmax, `P ∝ exp(f_S / T_s)`. Only under `state_policy = "boltzmann"` |
| ↳ `state_explore` | `"gap"`, `"upper"`, `"action_uncertainty"`, `"bellman_error_upper"` (alias `"bellman_error"`), `"bellman_error_lower"`, `"max_bellman_error"`, `"vpi"` | `"gap"` | `f_explore(s')`, the exploration term: `U − L`; `U`; `U^{−a_L} − L`, where `a_L` is the lower-bound-optimal action; `\|U_{k−1} − U_k\|`, `\|L_{k−1} − L_k\|`, or their max; `maxₐ VPI(s',a)`, the value of perfect information about the Q-values at `s'` (how likely learning them is to change which action is best there). The Bellman errors are measured over each state's own last backup, and a state that hasn't been backed up yet falls back to `U − L`. `"action_uncertainty"` and `"vpi"` each cost about one Bellman backup per successor per step. Only under `state_policy = "boltzmann"` |
| ↳ `state_alpha_e_schedule` | `"fixed"`, `"power"` | `"fixed"` | How α_e, the explore share of `f_S`, is obtained at the start of each rollout; only under `state_policy = "boltzmann"` |
| &nbsp;&nbsp;↳ `state_alpha_e` | float in [0,1] | `0.5` | α_e; `0` exploits, `1` explores. Only under `state_alpha_e_schedule = "fixed"` |
| &nbsp;&nbsp;↳ `state_alpha_e_k` | float > 0 | `1.0` | α_e = δᵏ, with δ the initial gap; only under `state_alpha_e_schedule = "power"` |
| &nbsp;&nbsp;↳ `state_alpha_e_min`, `state_alpha_e_max` | 0 ≤ min ≤ max ≤ 1 | `0.0`, `1.0` | Optional clip of δᵏ, to keep some exploration (min) or some exploitation (max) whatever the gap; only under `state_alpha_e_schedule = "power"` |
| ↳ `state_alpha_p_schedule` | `"fixed"`, `"power"` | `"fixed"` | How α_p, which flattens `p(·\|s,a)` towards uniform inside `f_S`, is obtained; only under `state_policy = "boltzmann"` |
| &nbsp;&nbsp;↳ `state_alpha_p` | float in [0,1] | `0.0` | α_p; `0` keeps the realized `p`, `1` is uniform over the support. Only under `state_alpha_p_schedule = "fixed"` |
| &nbsp;&nbsp;↳ `state_alpha_p_k`, `state_alpha_p_min`, `state_alpha_p_max` | as for α_e | `1.0`, `0.0`, `1.0` | As for α_e; only under `state_alpha_p_schedule = "power"` |
| `state_exclude_visited` | bool | `false` | Within a rollout, never move to a state that rollout has already visited (the current one included), so a trajectory has no self-loops or cycles. Applies under both policies |
| `state_exclude_converged` | bool | `false` | Never move to a successor whose gap `U − L` is below `state_converged_eps`. Terminal states (goal, avoid, implicit sinks) are exempt, so a rollout still ends by reaching one. Applies under both policies. **Needs `restrict_to_initial = true`** — see the note below |
| ↳ `state_converged_eps` | float ≥ 0 | the property's `convergence_eps` | The threshold for `state_exclude_converged`; the default is the tolerance GSRDP terminates on. Only under `state_exclude_converged = true` |
| `terminate` | list of `"max_steps"`, `"expected_gap"` | `["max_steps"]` | Extra stopping rules, ORed together |
| ↳ `max_steps` | int ≥ 0 | number of states | Cap on transitions per rollout; only when `terminate` contains `"max_steps"` |
| ↳ `expected_gap_tau` | float > 0 | `10.0` | τ in BRTDP's expected-gap stopping rule; only when `terminate` contains `"expected_gap"` |

###### TrajectorySampling configs

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `gauss_seidel` | bool | `false` | Dole one trajectory out `k` states at a time, one batch per iteration, instead of relaxing all of it at once |
| `k` | int > 0 | — | Batch size; **required** when `gauss_seidel = true` (or give `k_frac` instead — see [Size-relative `k`](#size-relative-k-k_frac)), and an error without it: the whole trajectory is then one batch |
| `reverse` | bool | `true` | Return each trajectory goal-first. Only observable when `gauss_seidel = true` |
| `dedup` | bool | `false` | Drop repeated states from each batch (the whole trajectory when `gauss_seidel = false`), keeping the first. Each backup in a batch reads the previous iterate, so a repeat recomputes the same value and only adds a Bellman update. Unlike `state_exclude_visited`, it does not change where rollouts go |
| ↳ `dedup_scope` | `"batch"`, `"rollout"`, `"consecutive"` | `"batch"` | What `dedup` removes repeats from: each batch; the whole rollout before it is doled out (keeping each state's first occurrence in the returned order, the goal end first under `reverse`; with `k = 1` that is Gauss-Seidel over the rollout's distinct states); or only runs of the same state (self-loop steps). Only with `dedup = true` |
| `initial_state` | `"random"`, `"max"` | `"random"` | Where each rollout starts: a uniform draw from the declared initial states `S₀` (every state when the problem declares none), or the state of `S₀` with the widest gap `U − L`, ties broken uniformly at random |

Worth knowing:

- **The successor score `f_S` is a mixture of two distributions** over the
  support of `p(·|s,a)`:

  ```
  f_S        = (1 − α_e)·f'_exploit + α_e·f'_explore
  f'_exploit = p'·f_exploit / Σ p'·f_exploit      (likewise f'_explore)
  p'         = (1 − α_p)·p + α_p / |supp(s,a)|
  ```

  Each term is clamped at zero before normalizing, and a term that is zero
  over the whole support counts as uniform.
- **Why the power rule is the default.** `f_S` sums to 1, so its values scale
  like `1/|supp|`: write `f_S = r/|supp|`, with `r` how many times more likely
  than uniform a successor is. The plain softmax is then
  `softmax(r / (|supp|·T_s))`, whose effective temperature grows with the
  support, so one `T_s` is near-argmax on a 2-successor support and
  near-uniform on a 1000-successor one. Under the power rule the `1/|supp|`
  cancels in the normalization, `P ∝ r^(1/T_s)`, and the odds between two
  successors, `(f_S(x)/f_S(y))^(1/T_s)`, do not depend on how many others there
  are. A successor with `f_S = 0` is never drawn. With the defaults
  (`state_exploit = "upper"`, `state_explore = "gap"`, `0 < α_e < 1`) that means
  `U = 0` there, a converged dead end; but with `α_e = 0` and
  `state_exploit = "lower"`, every successor whose lower bound is still 0 is
  excluded.
- **α_e and α_p are resolved once per rollout** against the initial gap
  `δ = max_{s ∈ S₀} Diff(s)`, clamped to `[0,1]` (all states when the model
  declares no initial states). `"power"` gives `α = δᵏ`: uniform exploration
  while the gap is wide open, fading to pure exploitation as it closes. A
  `state_alpha_e_min` above 0 keeps some exploration until the end.
- **The successor filters run before any scoring.** `state_exclude_visited` and
  `state_exclude_converged` remove successors from the support, so they hold
  under ε-greedy (both its argmax and its uniform branch) and Boltzmann alike,
  and the mixture renormalizes over what remains. If every successor is
  filtered out, the rollout ends at the current state. They change only which
  states rollouts reach, not the backups.
- **Use `state_exclude_converged` only with `restrict_to_initial = true`.** A
  state can count as converged while a low-probability successor's gap is still
  open, and once every route to an open state passes through a converged one,
  no rollout reaches it again. Converging on the initial states is unaffected,
  but with `restrict_to_initial = false` every state's gap must close, and the
  solve stalls on the cut-off states (seen on `linear_stochastically_switched_S10`
  and `car_parking_S10_I3`). The library logs a warning once per solve in that
  case, or when the model declares no initial states.
- **ε-greedy successor selection ignores `f_S`.** It exploits on
  `p(s'|s,a)·f_exploit(s')` and reads only `state_exploit`, which is why the
  other state keys are Boltzmann-only.
- **Do not set both epsilons to 0.** That makes the rollout a deterministic
  function of the current bounds, which can lock onto a cycle and relax the
  same few states forever while the rest keep their initial values — the gap
  criterion then never fires and the solve does not terminate.
- **The action temperature is resolved once per rollout**, before the first
  action is selected, and held for the whole trajectory. In
  Gauss-Seidel mode the temperature belongs to the rollout, not the batch: a
  trajectory doled out over several iterations keeps the temperature it was
  sampled at.
  - The action policy's `"gap_based"` is `T = max(t_min, k·δ / (1 − δ))`,
    `k` times the odds of the initial gap `δ = max_{s ∈ S₀} Diff(s)` (clamped
    to `[0,1]`; all states when the model declares no initial states, and the
    same δ the state mixture weights follow), floored at `t_min`: `δ = ½` gives
    `T = k`, wide-open initial states draw uniformly and converged ones cool to
    `t_min`.
    The odds alone would end a `gap_initial` solve at `T ≈ k·ε`, i.e. argmax.
  - **Keep `action_temperature_min` well away from zero.** A cold Boltzmann policy
    is argmax in all but name, so annealing all the way down hits exactly the
    failure above: on a three-state IMDP, `t_min = 0.01` stalls where
    `t_min = 0.5` converges.

---

#### Removed sampling strategies

`"trajectory_transition"`, `"trajectory_gap"`, `"trajectory_action_uncertainty"`
and `"trajectory_reach"` were removed when trajectory sampling was reworked.
Each drew the next state *proportionally* to a weight, which the new argmax /
softmax policies cannot express, so no configuration reproduces them exactly.
A config naming one of them now fails with a migration error.

**Results from the replacements are not directly comparable to banked runs of
the old names** — re-run the baseline for any figure that spans the change. The
nearest replacements are `"trajectory"` with `state_policy = "boltzmann"`,
`state_score = "log_mixture"`, `state_temperature = 1` and `state_alpha_e = 1`
(successors drawn in proportion to `p·(U−L)`) for `"trajectory_gap"`, and
`"trajectory"` (its defaults) for `"trajectory_transition"`.

## Examples

### Smoke test (fast end-to-end check)

```toml
name = "smoke"

[benchmark]
samples = 2
warmup  = true
seeds   = [1, 2]      # pin seeds for a reproducible run (one sample per seed)

[output]
results_root    = "results"
save_raw        = true
save_trace      = true
track_initial   = true
reference_label = "gsrdp_exhaustive_state"
image_formats   = ["png", "svg"]

[problems]
names = ["coin2_K4"]

[[algorithms]]
name = "gsrdp"                    # full sweep (reference baseline)
[algorithms.params]
sampling = "exhaustive_state"

[[algorithms]]
name = "gsrdp"                    # sampled variant, compared on S0
[algorithms.params]
sampling = "random_state"
k        = 64
```

### Full sweep (all problems, multiple randomly-seeded runs, several sampling strategies)

```toml
name = "sampling_sweep"

[benchmark]
samples = 5          # 5 independent runs per problem×algorithm, each a fresh random seed
warmup  = true
# seeds = [1, 2, 3]  # optional: pin the per-sample seeds (overrides `samples`)

[output]
results_root    = "results"
save_raw        = true
save_trace      = true
track_initial   = true
reference_label = "gsrdp_exhaustive_state"

# [problems] omitted => every subdirectory under problems/ is included

[[algorithms]]
name = "gsrdp"                    # full sweep (reference baseline)
[algorithms.params]
sampling = "exhaustive_state"

[[algorithms]]
name = "gsrdp"
[algorithms.params]
sampling = "random_state"
k        = 128

[[algorithms]]
name = "gsrdp"
[algorithms.params]
sampling = "random_state_action"
k        = 256
```

---

## How results are structured

Each run produces `results/<name>_<timestamp>/`:

```
results/sampling_sweep_2026-06-01_14-23-00/
├── resolved_config.toml   # fully-expanded config for exact reproduction
├── meta.json              # run metadata (timing, seeds, git hash)
├── results.csv            # one row per sample (timing + quality + ref check)
├── runs/<cell>/sample<k>/trace.csv   # per-sample per-iteration convergence trace
├── raw/<cell>.json        # serialized BenchmarkTools.Trial (one per cell)
└── plots/
    ├── speedup_median_time.png
    └── convergence_*.png
```

`results.csv` columns include `problem`, `algorithm`, `sample_index`, `seed`, `time_median_s`, convergence quality, and the reference-agreement fields. The `samples` rows of one cell share a `cell_id`; aggregating them per (problem, algorithm) gives the timing and convergence spread.
