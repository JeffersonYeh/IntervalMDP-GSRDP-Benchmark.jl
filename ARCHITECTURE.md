# Architecture & Implementation Notes

This document explains how the benchmarking harness is built: what each file
does, the key types and functions, and the end-to-end flow from a TOML config to
the results on disk. For *usage* see [README.md](README.md).

---

## 1. Mental model

The harness turns a declarative TOML config into a **run matrix** — the cartesian
product `{problems} × {algorithms}`, one cell per pair — and solves each cell
`samples` times as the samples of a single benchmark (each sample its own seed):

```
                                  configs/foo.toml
                                        │
                          load_experiment_config()         (config.jl)
                                        │
                            ┌───────────▼───────────┐
                            │   ExperimentConfig     │  problems / algorithms / benchmark / output
                            └───────────┬───────────┘
                          build_run_matrix()                (runmatrix.jl)
                                        │
                              Vector{RunUnit}                one per (problem, algorithm) cell
                                        │
                          run_experiment()                  (benchmark.jl)  ── loops over cells
                                        │
            ┌───────────────────────────┼───────────────────────────────┐
            │ per problem instance (cached, built once):                  │
            │     build_problem(name, params)  ──► BenchmarkProblem       │  (registry.jl → problems/*.jl)
            │ per cell:                                                    │
            │     build_algorithm(name, bp, params) ──► BuiltAlgorithm    │  (registry.jl → algorithms.jl)
            │     1. warm-up solve         (untimed, callback attached)   │
            │     2. run_sampled_solve: `samples` benchmark samples, each │
            │        seeded; timed solve w/ Trace callback; teardown      │
            │        harvests (solution, trace) per sample          ──────┼─► Trial + N (sol, trace)  (benchmark.jl)
            │     3. reference solve        (once/problem, cached)        │
            │     per sample j: extract_quality + value_agreement + row   │  (quality.jl, results.jl)
            └───────────────────────────┬───────────────────────────────┘
                       assemble_row()×N + append_row!()×N   (results.jl)  ── one row per sample, appended per cell
                                        │
              results/<name>_<timestamp>/results.csv  + runs/<cell>/sample<k>/trace.csv + raw/<cell>.json
                                        │
                          generate_report()                  (analysis/BenchmarkAnalysis.jl)
                                        │
                              plots/  +  summary_table.csv
```

Two design choices make this **easy to extend**:

- Everything funnels through two uniform types — `BenchmarkProblem` and
  `BuiltAlgorithm` — so the engine never knows what a problem/algorithm actually
  *is*.
- Two registries (`PROBLEM_REGISTRY`, `ALGORITHM_REGISTRY`) map config **names**
  to **builder functions**. Adding a source/algorithm is one `register_*!` call.

---

## 2. File-by-file

| File | Role |
|------|------|
| `run.jl` | CLI entry point |
| `src/Bench.jl` | module root: dependencies, include order, `main()` |
| `src/problem_provider.jl` | `BenchmarkProblem` container + size metadata helper |
| `src/registry.jl` | the two registries + config→object resolvers |
| `src/trace.jl` | convergence `Trace` callback (dual GSRDP/RVI) + CSV writer |
| `src/quality.jl` | quality scalars + reference-agreement |
| `src/config.jl` | TOML → spec structs (`ExperimentConfig`), `problems/` discovery, resolved-config dump |
| `src/runmatrix.jl` | cartesian product → `RunUnit`s; finite-horizon predicate |
| `src/results.jl` | results.csv (incremental), meta.json, Manifest snapshot |
| `src/benchmark.jl` | the driver: warm-up / instrumented / reference / timed |
| `src/analysis/` | `BenchmarkAnalysis`: loaders, reductions, the generic report (`run_analyses.jl`) and `report-plots/` read results through it |
| `src/combine.jl` | merge several results dirs into one (`combine_results`) |
| `src/file_sources.jl` | `prism` / `bmdp_tool` / `intervalmdp_file` loaders (used by `run.jl`) |
| `src/analysis/gap_video.jl` | grid geometry, reach/avoid/S0 overlays, per-state gap mp4s (opt-in) |
| `src/algorithms.jl` | `rvi` + `gsrdp` builders |
| `tools/IntervalMDPAbstractions/*.jl` | one generator per abstracted system (car_parking, patrol_robot, bas_4d, IMPaCT's impact_robot_2d/room_temp, …) → `problems/<name>/`; `run.jl` lists every instance, local vs cluster-only, and the systems dropped as degenerate |
| `tools/prism/` | `convert_prism.sh` + `prism_to_imdp.jl`: PRISM IMDP case studies (robot, coin) → infinite-time reach-avoid; `imdp_writer.jl` is the explicit-model → `.nc`/`.json` back end shared with `tools/storm/` |
| `tools/end_components.jl` | read-only check: maximal end components (sure / possible) of every problem in `problems/` — ECs can stall GSRDP (storm_tiny04); run it when adding problems |
| `tools/storm/` | `convert_storm.sh` + `drn_to_imdp.jl`: Storm's interval-MDP test models (`models/*.drn`: tiny-*, brp, crowds) → infinite-time reach-avoid |
| `tools/repro_package/` | `fetch_lfs.sh` + `import_repro.jl`: IntervalMDP.jl reproducibility-package models (multiObj_robotIMDP, pimdp_*) → infinite-time reach-avoid |

### `run.jl`
Activates the project, includes `src/Bench.jl`, and calls
`Bench.main(ARGS; root=@__DIR__)`. `root` is the harness directory so dataset
globs and the Manifest snapshot resolve regardless of the current working dir.

### `combine_results.jl`
Merges N results directories into one new directory that behaves exactly like a single
sweep's, then analyses it — see `src/combine.jl` below.

### `src/Bench.jl` — module root
`using IntervalMDP` (its accessors stay unqualified everywhere), but
`import IntervalMDPAbstractions / ArchCompStochasticModels / LazySets` because
several of their exports **clash** with IntervalMDP's (notably `system`). Includes the
framework files in dependency order, then the self-registering `file_sources.jl` (the file
readers `run.jl` uses) and `algorithms.jl`. `main(args; root)` runs the sweep and then the
report.

### `src/problem_provider.jl`
```julia
struct BenchmarkProblem
    name::String                 # instance id, e.g. "grid_world_24x24"
    problem::Any                 # ControlSynthesisProblem ready for solve()
    source::Symbol               # :grid_world | :builtin | :prism | ...
    metadata::Dict{String,Any}   # num_states, num_actions, dims, source_file, ...
    builder_params::Dict{String,Any}
end
```
`safe_size_metadata(sys)` best-effort-extracts `num_states`/`num_actions` (never
throws). The materialized `problem` is what `solve` consumes; it is built **once**
per instance and reused across all algorithms/samples.

### `src/registry.jl`
Two `Dict{String,Function}` registries plus:
- `register_problem!(name, builder)` — `builder(params::Dict) → BenchmarkProblem`.
- `register_algorithm!(name, builder)` — `builder(bp, params) → BuiltAlgorithm`.
- `BuiltAlgorithm(alg, label, kind, descriptor)` — `alg` is the
  `ModelCheckingAlgorithm`; `kind` is `"rvi"`/`"gsrdp"` (used for grouping +
  finite-horizon skip); `descriptor` holds resolved knobs for the CSV columns.
- **Resolvers** turn flat config values into Julia objects:
  `resolve_bellman(bp, params)` (`"default"` is problem-dependent) and
  `resolve_sampling(params)` (`random_*` read `k`). Sampling types are referenced
  via the qualified `IntervalMDP.` path because they are **not exported**. GSRDP
  termination is no longer configured here — the fork auto-derives it from the
  specification's property (gap criterion from `convergence_eps`, optionally
  restricted to the model's initial states via the property's `restrict_to_initial`).
- `algo_label(name, params)` builds a unique label like `gsrdp_random_state_k128`.

### `src/trace.jl`
A `TraceExecutor` is the callable struct passed to `solve` on the **instrumented**
run only. It fans one GSRDP callback out to a tuple of traces, computing the gap
once per iteration into a reused buffer and stamping a single wall-clock, then
handing both to each trace's `record!(tr, vf, gap, wall, bellman_updates, seq)`.
It also enforces the per-solve deadline (`CellTimeout`), which is why it fires
even when it holds no traces at all.

The executor accepts **both** GSRDP callback arities:
- `(::TraceExecutor)(vf, bellman_updates)` — the long-standing contract.
- `(::TraceExecutor)(vf, bellman_updates, update_sequence)` — additionally reports
  which states that iteration relaxed. The fork probes for this form
  (`IntervalMDP/src/gsrdp.jl`, `_invoke_callback`); only `UpdateTrace` consumes it.

Trace kinds (`TRACE_BUILDERS`, selected by `[output].traces`) and the file each
writes into `runs/<cell>/sample<k>/`:

| kind | records | file |
|---|---|---|
| `trace` / `initial` | scalar `gap_max`/`gap_mean` | folded into `trace.csv` |
| `gap` / `initial_gap` | per-state gap vector | `gap_trace.csv` |
| `value` / `initial_value` | both bound vectors | `value_trace_{lower,upper}.csv` |
| `updates` | per-state Bellman-backup count | `update_trace.csv` |

`updates` counts occurrences in the sequence the solver actually swept, so
duplicates are kept — a trajectory sampler that revisits a state really does relax
it twice — and `sum(row) * (num_actions + 1)` reconciles with the solver's own
`bellman_updates` delta.

`write_trace_csv` zeroes wall-clock to the run start and drops NaN rows. Scalar
columns: `wallclock_s, bellman_updates, gap_max, gap_mean[, initial_gap_max,
initial_gap_mean]`; the heavy kinds are wide matrices, one column per state.

### `src/quality.jl`
- `extract_quality(trace, sol, bp) → QualityResult` — `num_iterations`,
  `final_gap_max`/`final_gap_mean` (real gap for GSRDP via `trace.saw_interval`;
  `missing` for RVI), `converged` (`< convergence_eps`), and the primary value
  vector for the reference check.
- `value_agreement(V_alg, V_ref, initial_states, eps; tol_factor=10) →
  ReferenceAgreement` — max/mean |difference| over all states and over initial
  states; `agrees` if within `tol_factor*eps`.
- `problem_initial_states(bp)` / `problem_convergence_eps(bp)` read the spec's
  property (guarded).

### `src/config.jl`
Spec structs (`ProblemSpec`, `AlgorithmSpec`, `BenchmarkSettings`,
`OutputSettings`, `ExperimentConfig`). `load_experiment_config(path; base_dir)`:
- parses `[benchmark]` / `[output]` with defaults,
- turns each `[[algorithms]]` into an `AlgorithmSpec`,
- **discovers problems from files**: `discover_problems(root, names)` scans the
  `problems/` directory (`[problems]` `root`, default `"problems"`), one
  subdirectory per problem; `_problem_spec_from_dir` dispatches on the files
  inside — a `*.nc` + same-basename `.json` pair → the `intervalmdp_file`
  builder, a `.sta/.tra/.lab/.pctl` set → the `prism` builder. `[problems]`
  `names` selects a subset; omit it to run every subdirectory.
- **validates** every builder name against the registries (fail fast).
`write_resolved_config` dumps the fully-expanded config back to TOML for an
exactly reproducible record.

### `src/runmatrix.jl`
`build_run_matrix(cfg)` = `{problems} × {algorithms}`, each a `RunUnit` (one cell)
with a deterministic id `"<instance>__<alg_label>"` (also the per-cell output dir
name). `resolve_sample_seeds(cfg)` picks the per-sample seeds once for
the whole sweep, shared across all cells (paired comparison): the explicit
`cfg.benchmark.seeds` if given, otherwise `cfg.benchmark.samples` seeds drawn from
OS entropy via `draw_run_seeds` (a fresh set per invocation). `is_finite_horizon(bp)`
(via `IntervalMDP.isfinitetime`) lets the driver skip GSRDP×finite-horizon cells.
The product is full here; the skip happens at run time because finiteness is only
known after construction.

### `src/results.jl`
- `make_results_dir(cfg, base_dir)` → `results/<name>_<timestamp>/{runs,raw,plots}`.
- `empty_timing()` / `sample_timing(trial, k)` → timing NamedTuple
  (`time_min/median/mean/max/std_s`, `n_samples`, `allocs`, `memory_bytes`,
  `gctime_min_s`). `sample_timing` pulls sample `k`'s time from `trial.times[k]`;
  since each sample is its own data point, min=median=mean=max and std=0 (n=1).
- `assemble_row(...)` builds one results row (per sample) with a **fixed column
  order** (so the incremental CSV stays consistent), taking a precomputed `timing`
  NamedTuple and the sample's `sample_index`/`seed`.
  `merge(head, timing, quality, reference, tail)`.
- `append_row!` writes the header once, then appends rows. Note the granularity: a
  cell's rows are all appended together once `_run_cell` returns, so a kill mid-cell
  loses that cell's completed samples too. Completed *cells* are durable.
- `write_progress` / `write_aborted` → `progress.json` (rewritten atomically after
  every cell) and `aborted.json` (best-effort, from an `atexit` hook on SIGTERM). These
  are what survives a hard kill, when the post-loop writes never run.
- `write_meta` → `meta.json` (Julia + IntervalMDP + BenchmarkTools versions via
  `Pkg.dependencies()`, hostname, CPU, **`Threads.nthreads()`** — load-bearing for
  timing — git commit/dirty, ok/error/skip counts). `snapshot_manifest` copies the
  active `Manifest.toml` (pins the local IntervalMDP fork).

### `src/benchmark.jl` — the driver
`run_sampled_solve(prob, alg, seeds, S0, track_initial, upi)` solves the cell as
`length(seeds)` samples of one `@benchmarkable solve($prob,$alg; callback=__cb)`:
`setup` seeds with the next entry of `seeds` and builds a fresh `Trace`; the timed
`core` keeps the returned solution in a setup-declared variable (reassigning a
setup local in `core` is visible in `teardown`); `teardown` harvests `(solution,
trace)`. Run with `evals=1`, an effectively-unbounded `seconds` budget,
`gctrial=true`, and **`warmup=false`** — the default warmup pass would fire the
sample function one extra time and over-harvest, so disabling it makes the harvest
exactly `length(seeds)` long (our own untimed warm-up pays the JIT). Returns the
`Trial` (per-sample times in `trial.times`) + the harvested solutions/traces, all
in sample order. The Trace callback runs inside the timed region (required to be
harvestable), so timing includes callback overhead — uniform across algorithms.

`run_experiment(cfg; base_dir, project_dir)`:
1. makes the results dir, writes resolved config + meta + Manifest snapshot,
2. resolves the shared per-sample seeds once,
3. caches `BenchmarkProblem`s per instance (`get!`), catching build errors so a
   bad instance becomes a per-cell error row rather than a crash,
4. for each cell: builds the algorithm; if `gsrdp` × finite-horizon → record
   `skipped:finite_horizon`; else `_run_cell` (returns one row per sample);
   everything in `try/catch` → a single `error` row,
5. appends every row, logs the cell; finally rewrites meta with counts.

`_run_cell` warms up (with a throwaway callback, to JIT the callback solve path),
calls `run_sampled_solve`, saves the raw Trial once, then for each sample emits a
row (`extract_quality` + its own `sample<k>/trace.csv`). The agreement columns are
currently left empty (`empty_agreement`); `reference_label` only selects the
baseline the analysis suite plots against.

### `src/combine.jl`
`combine_results(dirs; ...)` unions N results directories into a new one. Everything in
`src/analysis/` takes exactly ONE `results_dir` and resolves `trace_path` relative to it,
so sweeps that were split to fit a wall clock, or the separate configs of `configs/thesis/`,
cannot otherwise be analysed together.

The merge is a union over `{problems} × {algorithms}` cells. A cell present in several
sources — the shared reference arm those split configs re-run in every part — has its
samples **pooled**: rows concatenated, `sample_index` renumbered `1..N` in (source order,
original index) order, `runs/<cell>/sample<k>/` renamed and `trace_path` rewritten to
match. Rows with no `sample_index` (`error:build`, `skipped:finite_horizon`) pass through
untouched, so a cell that failed in one source and ran in another keeps both records.

**A merged directory absorbs.** When one of the sources was itself produced by this tool
(`meta.json` carries `combined`), it becomes the target and the merge happens inside it:
its rows and run directories stay exactly where they are (`_plan_samples`'s `pinned` path
gives them `new_index == old_index`) and the others are added on top, counting from the
highest index each cell already has. With several, the first wins; an explicit `out` forces
a fresh directory. Re-adding a run already in the target is blocked via `_underlying_runs`,
which reads `meta.json`'s `combined_from` — itself flattened to the underlying *sweeps*, so
merged sources never nest. `_transfer` no-ops when source and destination are the same file,
which an in-place merge hits constantly.

`COMBINED_FROM.md` is *rendered* from `combine_log.json` rather than written directly,
because a second merge must still describe what the first folded in — long after those
directories may have moved or been deleted. The log is append-only, one entry per merge;
an added source that is itself merged contributes its own log entries as history, and a
target predating the log gets one entry reconstructed from `meta.json` (the runs survive;
the per-sample moves do not, and the prose says so rather than printing an empty list).
`rerender_sources_file(dir)` rewrites the prose from the log alone. The
additive work (run directories) happens first and the records are rewritten atomically
last, so an in-place merge that dies part-way leaves the target's own data intact and only
orphan sample directories that nothing references.

Tables are read with `types=String` and written back verbatim — the tool only reshuffles
rows, and inferring types would silently reformat the numbers the sweep wrote. Run and raw
files are placed by `_transfer`, which hardlinks and falls back to `cp` across filesystems
(`gap_trace.csv` reaches several GB, and a sweep never rewrites its outputs, so sharing an
inode is both safe and free).

Everything a consumer reads is rebuilt rather than copied: `meta.json`'s `cells` is the
UNION of the sources' expected-cell lists (`load_completeness` derives the SWEEP INCOMPLETE
banner from it) with `datetime_end` always set; `resolved_config.toml` is synthesized by
hand, since `load_experiment_config` cannot read one back (it parses `problems` as a table
while `write_resolved_config` emits an array of tables); `validation_summary.json` is
recomputed per instance. `plots/`, `manifest.txt` and `METHODS_meanstd.md` are NOT copied —
the analysis regenerates them — but already-rendered `plots/gap_videos/*.mp4` are carried
over under their new sample number.

Blocking checks (downgraded to recorded caveats by `force`): a differing `convergence_eps`
/ `restrict_to_initial` / `timeout_s`, one `alg_label` meaning two different algorithms, a
problem whose structural counts changed, and — unless `dedupe_seeds` is set — a
`(cell_id, seed)` pair measured twice. That last one blocks rather than warns because
`run_scalars` (`analysis/reduce.jl`) reduces by `(algorithm, problem, seed)` with no sample
identity, and `load_canonical`'s `sample_idx` is the row index *within a trace file*, so it
cannot disambiguate either: two runs sharing a seed merge into one group, their traces sort
together by wall-clock, and `first_crossing` reads a solve time off the interleaved mixture.
It lands on precisely the arm a split sweep re-runs in every part, which is the reference
every normalized figure divides by. Recorded as caveats instead: a differing host/thread
count, git commit, or `[output].traces`. Every caveat lands in `COMBINED_FROM.md` alongside
a per-source summary and the sample-provenance table. Sources are opened read-only.

### `src/file_sources.jl`
Thin wrappers over the library's existing readers (the `Data` submodule):
`IntervalMDP.Data.read_prism_file` (`file` = path without extension),
`read_bmdp_tool_file`, `read_intervalmdp_jl`. `_wrap_file_problem` extracts size
metadata. **Extension point** for the deferred continuous-PRISM→IMDP conversion
(a marked TODO in `build_prism`). Errors here are isolated per-cell.

### `src/algorithms.jl`
`_descriptor(params; kind)` records the resolved knobs for the CSV. Registers:
- `"rvi"` → `RobustValueIteration(resolve_bellman(...))` (single-bound; registered but unused by current configs).
- `"gsrdp"` → `GeneralizedSamplingbasedRobustDynamicProgramming(bellman;
  sampling_strategy=resolve_sampling)`. Only the sampling strategy lives on the
  **constructor**; termination is auto-derived from the specification's property
  (the fork removed `term_criteria` from the GSRDP struct).

---

## 3. Data structures (quick reference)

| Type | Where | Fields |
|------|-------|--------|
| `BenchmarkProblem` | problem_provider.jl | `name, problem, source, metadata, builder_params` |
| `BuiltAlgorithm` | registry.jl | `alg, label, kind, descriptor` |
| `ProblemSpec` | config.jl | `name, instance_name, params` |
| `AlgorithmSpec` | config.jl | `name, label, params` |
| `BenchmarkSettings` | config.jl | `samples, warmup, seeds` |
| `OutputSettings` | config.jl | `save_raw, save_trace, track_initial, traces, reference_label, results_root, image_formats, analysis, gap_videos, gap_video_max_frames, gap_video_framerate, trace_thin_rtol` |
| `GridGeometry` | analysis/gap_video.jl | `shape, extra, extent, provenance` (flat state index → grid cell) |
| `ExperimentConfig` | config.jl | `name, problems, algorithms, benchmark, output, raw` |
| `RunUnit` | runmatrix.jl | `problem, algorithm, id` (one cell; samples/seeds resolved per-sweep) |
| `Trace` | trace.jl | wallclock / bellman_updates / gap_* vectors + bookkeeping |
| `QualityResult` | quality.jl | `num_iterations, final_gap_max, final_gap_mean, converged, primary_value` |
| `ReferenceAgreement` | quality.jl | `ref_max_diff, ref_mean_diff, ref_max_diff_initial, agrees` |

---

## 4. Output layout

```
results/<name>_<timestamp>/
├── results.csv            one row per sample (see columns below)
├── resolved_config.toml   fully-expanded config that ran
├── meta.json              versions, git, host, thread counts, ok/error/skip
├── Manifest.toml          dependency snapshot (pins the IntervalMDP fork)
├── runs/<cell_id>/sample<k>/trace.csv   per-sample convergence trace
├── runs/<cell_id>/sample<k>/gap_trace.csv    per-state gap matrix ([output].traces ∋ "gap")
├── runs/<cell_id>/sample<k>/value_trace_{lower,upper}.csv   bound matrices (∋ "value")
├── runs/<cell_id>/sample<k>/update_trace.csv   per-state backup counts (∋ "updates")
├── raw/<cell_id>.json          serialized BenchmarkTools.Trial (one per cell)
├── plots/                     speed-up + convergence pngs, summary_table.csv
├── plots/gap_videos/<cell_id>__sample<k>.mp4   state-space panels ([output].gap_videos)
├── COMBINED_FROM.md        only in a dir built by combine_results.jl: its source runs
└── combine_log.json        ... and the machine-readable log it is rendered from
```

`plots/gap_videos/` is opt-in (`[output].gap_videos` or `run_analyses(...; gap_videos=true)`): one mp4 per `(problem, algorithm)` cell — NOT per sample,
since a cell's samples differ only by random seed — animating the heavy trace matrices
over the 2-D state space, as up to four synchronized panels (lower bound, upper bound,
gap, per-iteration backup count; `_PANEL_ORDER` / `_load_panels`). A panel whose trace
the sweep did not record is dropped, so a dir holding only `gap_trace.csv` renders the
single gap panel it always did, and the gap is derived as `U − L` when only the bounds
are present. `_select_one_per_cell` takes the lowest sample index present
(comparing sample *numbers*, not `readdir` order) and the filename records which.
Every frame outlines reach (white) / avoid (red) from the spec JSON and the initial states
S0 (magenta dashed) from the `initial_states` column of problems.csv, falling back to the
model's netCDF variable for dirs written before that column existed. `gap_video.jl` maps abstract state `s` to grid cell
`(i, j)` by `s = (j-1)*n + i` — the column-major order `LazySets.split` produces — and
refuses to draw any problem that is not a registered uniform-grid family, recording the
reason in `manifest.txt`. Nothing there reads a whole trace file: `_read_matrix_head`
streams and stops at `gap_video_max_frames` rows, because these files reach several GB.

**`results.csv` columns** (grouped): identity (`cell_id, problem, instance_name,
source, sample_index, seed`); problem (`num_states, num_actions`); algorithm (`algorithm,
alg_label, alg_kind, bellman, sampling, sampling_k, convergence_eps`); timing
(`time_min/median/mean/max/std_s, n_samples, allocs, memory_bytes, gctime_min_s`);
quality (`num_iterations, final_gap_max, final_gap_mean, converged, trace_path`);
reference (`ref_max_diff, ref_mean_diff, ref_max_diff_initial, agrees`); bookkeeping
(`status, error, error_type, wall_total_s, timestamp`).

`status` ∈ `ok` | `error` | `error:build` | `skipped:finite_horizon`.

---

## 5. Information flow, end to end (numbered)

1. **Parse** — `load_experiment_config` reads the TOML, applies defaults,
   discovers `ProblemSpec`s from the `problems/` directory, builds `AlgorithmSpec`s, and
   validates names against the registries.
2. **Matrix** — `build_run_matrix` forms `{problems}×{algorithms}` as `RunUnit`s
   (one cell each) with deterministic ids; `resolve_sample_seeds` picks the shared
   per-sample seeds once (explicit, or `samples` random).
3. **Provenance** — the driver writes `resolved_config.toml`, `meta.json`, and a
   `Manifest.toml` snapshot before doing any work.
4. **Construct (cached)** — `build_problem(name, params)` runs the registered
   builder once per instance → `BenchmarkProblem` (expensive abstraction not
   repeated across algorithms/samples).
5. **Algorithm** — `build_algorithm(name, bp, params)` resolves bellman/sampling/
   termination → `BuiltAlgorithm`.
6. **Measure (per cell)** — warm-up (callback attached); `run_sampled_solve` runs
   `samples` seeded samples of one benchmark, each with a `Trace` callback whose
   `(solution, trace)` is harvested via `teardown`; reference solve (once per
   problem) → per-sample `QualityResult` + `value_agreement`.
7. **Record** — one `assemble_row` + `append_row!` per sample; each sample's trace
   goes to `runs/<cell>/sample<k>/trace.csv`, the raw `Trial` to `raw/<cell>.json`.
8. **Report** — `generate_report` reads `results.csv` (+ the trace files) and
   writes the speed-up and convergence plots and `summary_table.csv`.
