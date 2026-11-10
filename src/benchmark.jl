# The benchmark driver. Each (problem × algorithm) cell is solved `samples` times
# as the samples of a SINGLE BenchmarkTools benchmark:
#   1. warm-up solve (untimed, callback attached) — pays JIT for the callback path
#   2. one benchmark of `samples` samples: each sample seeds with its own seed,
#      attaches a fresh Trace callback to the timed solve, and `teardown` harvests
#      that sample's (solution, trace). One results row is emitted per sample.
# Because each sample has a different seed it is a genuine independent data point,
# not a repeat of the same computation; the across-sample spread (timing AND
# convergence) is recovered by aggregating the per-sample rows in the report.
# Problems are constructed once per instance and reused; the reference value
# function is solved once per problem and cached. Every cell is isolated by
# try/catch and its rows are appended as soon as the cell finishes. Note the
# granularity: a kill mid-cell loses that cell's already-completed samples, because
# their rows and traces are only written once every sample has returned. Completed
# cells are durable, and `progress.json` records how far the sweep got.
#
# Note: the Trace callback runs inside the timed region (it must, to be harvested
# via teardown), so timing includes callback overhead — uniform across algorithms,
# so relative speed-ups are preserved.

# A solve is expensive: evals=1, and BenchmarkTools' own time budget is left
# effectively unbounded. That budget is deliberately NOT used for censoring: it is
# only checked BETWEEN samples (so it cannot stop a solve in flight) and it fires
# silently, yielding fewer samples than requested with no record anywhere. Censoring
# is enforced instead by `[benchmark].timeout_s` inside the trace callback, which
# produces a real `status="timeout"` row (see `CellTimeout` in trace.jl).
const _UNBOUNDED_SECONDS = 1.0e9

# The warm-up gets a far looser deadline than a timed sample. `timeout_s` is a budget for
# a MEASURED solve, but the warm-up's cost is dominated by JIT for the whole solver stack
# -- measured here, a cell with 0.5 s samples took over 2 s to warm up. Censoring the
# warm-up at `timeout_s` would therefore discard cells whose real runs fit comfortably,
# on the basis of compilation time. It still needs *some* bound, so that a genuinely
# divergent solve cannot hang the sweep before a single row is written.
const _WARMUP_TIMEOUT_FACTOR = 2.0

"""
    run_sampled_solve(prob, alg, seeds, S0, trace_kinds, track_initial; timeout_s, thin_rtol)
        -> (trial, sols, traces, walls, aborted)

Run `solve(prob, alg; callback)` as `length(seeds)` samples of one benchmark.
Per sample, `setup` seeds with the next entry of `seeds` and builds a fresh
`Trace`; the timed `core` keeps the returned solution in a setup-declared variable
(reassigning a setup local in `core` is visible in `teardown`); `teardown` harvests
the solution. `run(...; warmup=false)` makes the sample function fire exactly
`length(seeds)` times — the default warmup pass would add one extra leading
invocation and over-harvest. `evals=1` + unbounded `seconds` ⇒ no early stop.

Returns the `Trial` (per-sample times in `trial.times`, sample order) plus the
harvested solutions and traces in the same order, our own per-sample wall times, and
the `CellTimeout` that stopped the run (`nothing` when every sample completed). Holds
all `length(seeds)` solutions/traces in memory at once — fine for the small `samples`
counts used here.

Censoring details. `timeout_s` bounds each SAMPLE, not the cell: that gives every
(problem, algorithm, seed) run the same budget, so the censoring point is comparable
across cells, whereas a cell-wide budget would let sample 1 consume all of it.
When a sample throws `CellTimeout`:

- the executor is pushed in `setup` rather than `teardown`, so the in-flight sample's
  PARTIAL trace survives and `length(traces) == length(sols) + 1` identifies it;
- `trial` is lost — BenchmarkTools' `teardown` is not in a `finally` and its `Trial`
  is a local of its runner, so the completed samples' times go with it. `walls` is our
  own stamp around the same region (accurate to a `time_ns()` call), which is why
  completed samples still get real timings on the timeout path.

`sweep_deadline` (absolute `time()`) additionally caps each sample at the time left when
it starts, so a cell that begins near the end of a job still stops before the job does.
"""
function run_sampled_solve(prob, alg, seeds, S0, trace_kinds, track_initial::Bool;
                           timeout_s::Real=Inf, sweep_deadline::Real=Inf, thin_rtol::Real=0.0)
    sols    = Vector{Any}()
    traces  = TraceExecutor[]
    walls   = Float64[]
    k       = Ref(0)
    t_start = Ref(UInt64(0))
    bench = @benchmarkable begin
            __sol = solve($prob, $alg; callback=__cb)
        end setup = (begin
            __i   = ($k[] += 1)
            Random.seed!($seeds[__i])
            __cb  = build_trace_executor($trace_kinds, $S0; track_initial=$track_initial,
                                         timeout_s=_clamp_to_deadline($timeout_s, $sweep_deadline),
                                         thin_rtol=$thin_rtol)
            # Pushed HERE, not in teardown: on a CellTimeout teardown never runs, and
            # this partial trace is the censored convergence curve we want to keep.
            push!($traces, __cb)
            __sol = nothing
            $t_start[] = time_ns()
        end) teardown = (begin
            push!($sols, __sol)
            push!($walls, (time_ns() - $t_start[]) / 1e9)
        end)
    bench.params.samples = length(seeds)
    bench.params.evals   = 1
    bench.params.seconds = _UNBOUNDED_SECONDS
    # Inert: BenchmarkTools only honours gctrial under `warmup=true`, which we pass as false.
    bench.params.gctrial = true
    trial = nothing
    aborted = nothing
    try
        trial = run(bench; warmup=false)
    catch e
        e isa CellTimeout || rethrow()
        aborted = e
    end
    return trial, sols, traces, walls, aborted
end

"""
    run_experiment(config_path; base_dir, project_dir) -> results_dir

Run the full benchmark sweep defined by a TOML config and return the results
directory.
"""
function run_experiment(config_path::AbstractString;
                        base_dir::AbstractString=dirname(abspath(config_path)),
                        project_dir::AbstractString=base_dir)
    cfg = load_experiment_config(config_path; base_dir=base_dir)
    return run_experiment(cfg; base_dir=base_dir, project_dir=project_dir)
end

"""
    _sweep_deadline(cfg) -> Float64

Absolute `time()` past which the sweep stops starting new cells, minus a reserve for
the post-loop writes. From `[benchmark].wall_budget_s`, else SLURM's `SLURM_JOB_END_TIME`
(a unix timestamp set inside a job allocation), else `Inf`.

This is the primary defence against a cluster wall-clock kill, and it is strictly better
than any signal handling: stopping cleanly BETWEEN cells means `problems.csv`,
`validation.csv` and the final `meta.json` still get written, so the results directory
is a complete description of a partial sweep rather than a truncated one.
"""
function _sweep_deadline(cfg::ExperimentConfig)
    reserve = cfg.benchmark.wall_reserve_s
    if cfg.benchmark.wall_budget_s > 0
        return time() + cfg.benchmark.wall_budget_s - reserve
    end
    slurm_end = tryparse(Float64, get(ENV, "SLURM_JOB_END_TIME", ""))
    slurm_end === nothing && return Inf
    return slurm_end - reserve
end

"""
    _clamp_to_deadline(timeout_s, sweep_deadline) -> Float64

Per-solve budget for a solve starting NOW: `timeout_s`, cut to the time left before the
absolute `sweep_deadline` (a `time()` stamp). Evaluated when each solve starts, not once
per cell -- otherwise every sample would get the full time left at the start of the cell
and a later sample could outlive the job. `0` censors immediately rather than disabling
the deadline (see `build_trace_executor`).
"""
_clamp_to_deadline(timeout_s::Real, sweep_deadline::Real) =
    Float64(min(timeout_s, max(sweep_deadline - time(), 0.0)))

function run_experiment(cfg::ExperimentConfig;
                        base_dir::AbstractString,
                        project_dir::AbstractString=base_dir)
    dir = make_results_dir(cfg, base_dir)
    csv_path = joinpath(dir, "results.csv")
    matrix = build_run_matrix(cfg)

    write_resolved_config(cfg, joinpath(dir, "resolved_config.toml"))
    snapshot_manifest(project_dir, dir)

    header_written = Ref(false)

    # Per-sample seeds: chosen once and shared across every cell (paired comparison).
    sample_seeds = resolve_sample_seeds(cfg)

    # The expected-cell list is recorded UP FRONT so the analysis can name exactly which
    # cells are absent from a partial run, instead of re-deriving ids from the config and
    # drifting from `run_unit_id`'s escaping.
    write_meta(joinpath(dir, "meta.json"), cfg, length(matrix); git_dir=base_dir,
               extra=Dict("cells" => [u.id for u in matrix], "samples" => length(sample_seeds)))

    timeout_s = cfg.benchmark.timeout_s > 0 ? cfg.benchmark.timeout_s : Inf
    sweep_deadline = _sweep_deadline(cfg)

    @info "benchmark sweep" experiment=cfg.name cells=length(matrix) samples=length(sample_seeds) results=dir timeout_s=cfg.benchmark.timeout_s

    problem_cache = Dict{String,Any}()    # instance_name => BenchmarkProblem | Exception
    # Per-trial value bounds, grouped by problem instance, for the cross-trial
    # consistency check run once the whole sweep has completed.
    bounds_by_instance = Dict{String,Vector{TrialBounds}}()
    # Built lazily per cell, and `problems.csv` is rewritten on every cache miss so the
    # structural facts (num_transitions, num_states) survive a hard kill. Without this the
    # file only lands after the loop, and a killed sweep loses the scaling-fit x-axis
    # entirely. Building eagerly up front would instead front-load peak memory — the worst
    # thing to do under an OOM-killer, the one failure we cannot record.
    function get_problem(spec)
        haskey(problem_cache, spec.instance_name) && return problem_cache[spec.instance_name]
        bpx = try build_problem(spec.name, spec.params) catch e; e end
        problem_cache[spec.instance_name] = bpx
        try
            write_problems_csv(joinpath(dir, "problems.csv"), cfg.problems, problem_cache)
        catch e
            @warn "failed to write problems.csv" exception=(e, catch_backtrace())
        end
        return bpx
    end

    # Rewritten after every cell, atomically, and read back by the analysis: this is the
    # only record that survives a SIGKILL or the OOM-killer.
    progress = Ref((cells_done=0, n_cells=length(matrix), last_cell_id="",
                    n_ok=0, n_error=0, n_skipped=0, n_timeout=0))
    completed = Ref(false)
    # Best-effort marker for SIGTERM (Julia runs atexit hooks on it). Deliberately a
    # single small write: the main task may hold an open file handle at signal time, and
    # a deadlock inside an atexit hook hangs the process until SIGKILL.
    atexit() do
        completed[] && return
        try
            write_aborted(joinpath(dir, "aborted.json"), progress[],
                          "process exited before the sweep completed")
        catch
        end
    end

    n_ok = n_err = n_skip = n_timeout = 0
    stopped_early = false
    stop_reason = ""
    for (i, unit) in enumerate(matrix)
        if time() > sweep_deadline
            stopped_early = true
            stop_reason = "wall_budget"
            @warn "sweep deadline reached; stopping before cell $i of $(length(matrix))" cell=unit.id
            break
        end
        t0 = time_ns()
        bpx = get_problem(unit.problem)

        # A cell yields one row per sample on success; a single row on skip/error.
        local rows
        if bpx isa Exception
            rows = [assemble_row(unit, nothing, nothing, empty_timing(), empty_quality(), empty_agreement();
                                 status="error:build", error_msg=_short(bpx),
                                 error_type=string(typeof(bpx)), wall_total_s=(time_ns() - t0) / 1e9)]
            n_err += 1
        else
            bp = bpx::BenchmarkProblem
            built = nothing
            try
                ok, bp = override_specification(bp, cfg)
                if !ok
                    rows = [assemble_row(unit, bp, built, empty_timing(), empty_quality(), empty_agreement();
                                         status="skipped:not_reach_avoid", wall_total_s=(time_ns() - t0) / 1e9)]
                    n_skip += 1
                else
                    built = build_algorithm(unit.algorithm.name, bp, unit.algorithm.params)
                    # `AlgorithmSpec.label` is what names run directories and
                    # results rows, and it honours an explicit `label` key that the
                    # builder never sees. Keep the two from drifting apart.
                    built = with_label(built, unit.algorithm.label)

                    if built.kind == "gsrdp" && is_finite_horizon(bp)
                        rows = [assemble_row(unit, bp, built, empty_timing(), empty_quality(), empty_agreement();
                                             status="skipped:finite_horizon", wall_total_s=(time_ns() - t0) / 1e9)]
                        n_skip += 1
                    else
                        # Never let a cell outlive the sweep deadline: censoring it
                        # ourselves keeps its partial curve, a SLURM kill would not.
                        # `_run_cell` clamps EACH solve to the time left when it starts.
                        rows = _run_cell(unit, bp, built, cfg, dir, sample_seeds, t0,
                                         bounds_by_instance; timeout_s=timeout_s,
                                         sweep_deadline=sweep_deadline)
                        # A cell counts as timed out if ANY of its samples was censored,
                        # so `n_ok` never overstates how much data the sweep produced.
                        if any(r -> startswith(r.status, "timeout"), rows)
                            n_timeout += 1
                        else
                            n_ok += 1
                        end
                    end
                end
            catch e
                rows = [assemble_row(unit, bp, built, empty_timing(), empty_quality(), empty_agreement();
                                     status="error", error_msg=_short(e),
                                     error_type=string(typeof(e)), wall_total_s=(time_ns() - t0) / 1e9)]
                n_err += 1
                @warn "cell failed" cell=unit.id exception=(e, catch_backtrace())
            end
        end

        for r in rows
            append_row!(csv_path, r, header_written)
        end
        progress[] = (cells_done=i, n_cells=length(matrix), last_cell_id=unit.id,
                      n_ok=n_ok, n_error=n_err, n_skipped=n_skip, n_timeout=n_timeout)
        try
            write_progress(joinpath(dir, "progress.json"), progress[])
        catch e
            @warn "failed to write progress.json" exception=e
        end
        _log_cell(i, length(matrix), rows)
    end

    # Final rewrite with the complete cache (a cell that errored before its problem was
    # reached leaves a `not_built` row from the incremental writes above).
    try
        write_problems_csv(joinpath(dir, "problems.csv"), cfg.problems, problem_cache)
    catch e
        @warn "failed to write problems.csv" exception=(e, catch_backtrace())
    end

    # Solution-based validation: per-trial convergence + cross-trial bound overlap.
    # Best-effort — a validation error must never fail the (already complete) sweep.
    try
        tvals, pvals = validate_bounds(bounds_by_instance)
        write_validation_report(dir, tvals, pvals)
    catch e
        @warn "failed to write validation report" exception=(e, catch_backtrace())
    end

    write_meta(joinpath(dir, "meta.json"), cfg, length(matrix); git_dir=base_dir,
               extra=Dict("cells" => [u.id for u in matrix], "samples" => length(sample_seeds),
                          "n_ok" => n_ok, "n_error" => n_err, "n_skipped" => n_skip,
                          "n_timeout" => n_timeout, "n_cells_run" => progress[].cells_done,
                          "stopped_early" => stopped_early, "stop_reason" => stop_reason,
                          "datetime_end" => _now_string()))
    # The sweep reached its own end, so the atexit marker must not fire. `datetime_end`
    # in meta.json is the positive signal; its absence means a hard kill.
    completed[] = true
    @info "sweep complete" ok=n_ok error=n_err skipped=n_skip timeout=n_timeout stopped_early=stopped_early results=dir
    return dir
end

# One cell: warm-up, then `length(sample_seeds)` independently-seeded solves run as
# benchmark samples (trace + solution harvested per sample). Returns one assembled row
# per sample, in sample order — ALWAYS `length(sample_seeds)` rows, so a censored cell
# is a hole in the data with a name rather than an absence.
#
# `timeout_s` bounds each individual solve (0/Inf disables). Rows come in three classes:
#   status="ok"              — the sample completed
#   status="timeout"         — the sample was censored mid-solve; carries its partial
#                              trace, so it still contributes a censored gap curve
#   status="timeout:not_run" — a later sample that never started because of the above
function _run_cell(unit, bp, built, cfg, dir, sample_seeds, t0, bounds_by_instance;
                   timeout_s::Real=Inf, sweep_deadline::Real=Inf)

    S0 = problem_initial_states(bp)
    track = cfg.output.track_initial
    kinds = cfg.output.traces
    nsamples = length(sample_seeds)
    elapsed() = (time_ns() - t0) / 1e9

    # 1. warm-up (untimed) — solve WITH a throwaway callback to JIT the callback path
    #    that the timed samples below exercise. Deadlined at `_WARMUP_TIMEOUT_FACTOR ×
    #    timeout_s` (see the const): loose enough that JIT cannot trip it, tight enough
    #    that a divergent solve cannot hang the sweep. Blowing even that budget means no
    #    sample would finish either, so censor the whole cell rather than burn
    #    `samples × timeout_s` proving it. Also cut to the sweep deadline, like every sample.
    warmup_timeout = isfinite(timeout_s) ? timeout_s * _WARMUP_TIMEOUT_FACTOR : Inf
    if cfg.benchmark.warmup
        Random.seed!(sample_seeds[1])
        warmup_budget = _clamp_to_deadline(warmup_timeout, sweep_deadline)
        try
            solve(bp.problem, built.alg;
                  callback=build_trace_executor(kinds, S0; track_initial=track,
                                                timeout_s=warmup_budget,
                                                thin_rtol=cfg.output.trace_thin_rtol))
        catch e
            e isa CellTimeout || rethrow()
            @warn "cell censored during warm-up" cell=unit.id warmup_timeout_s=warmup_budget
            why = warmup_budget < warmup_timeout ? "the sweep deadline" :
                  "$(_WARMUP_TIMEOUT_FACTOR)x timeout_s"
            msg = "warm-up solve exceeded $why ($(round(warmup_budget, digits=3))s) " *
                  "after $(round(e.elapsed_s, digits=3))s; no sample was timed"
            # `empty_timing`, not the warm-up's elapsed time: the timing columns describe
            # a timed sample, and no sample ran. Reporting the warm-up duration there --
            # identically across every sample row -- would read as a measured solve time.
            # The elapsed time is in `wall_total_s` and in the message.
            return [assemble_row(unit, bp, built, empty_timing(), empty_quality(),
                                 empty_agreement(); status="timeout", sample_index=j,
                                 seed=sample_seeds[j], error_msg=msg,
                                 error_type="CellTimeout", wall_total_s=elapsed())
                    for j in 1:nsamples]
        end
    end

    # 2. timed sampled solve: `length(sample_seeds)` samples, each its own seed;
    #    teardown harvests the solution, setup the trace.
    trial, sols, traces, walls, aborted =
        run_sampled_solve(bp.problem, built.alg, sample_seeds, S0, kinds, track;
                          timeout_s=timeout_s, sweep_deadline=sweep_deadline,
                          thin_rtol=cfg.output.trace_thin_rtol)
    if cfg.output.save_raw && trial !== nothing
        try
            BenchmarkTools.save(joinpath(dir, "raw", unit.id * ".json"), trial)
        catch e
            @warn "failed to save raw trial" cell=unit.id exception=e
        end
    end

    # 3. one row per sample. `traces` holds one entry per STARTED sample and `sols` one
    #    per COMPLETED sample, so indices past `length(sols)` are the censored sample.
    rows = Vector{Any}(undef, nsamples)
    restrict = cfg.property.restrict_to_initial === true
    ncomplete = length(sols)
    for j in 1:nsamples
        if j > length(traces)
            # Never started: the cell was already censored. No trace, no timing.
            rows[j] = assemble_row(unit, bp, built, empty_timing(), empty_quality(),
                                   empty_agreement(); status="timeout:not_run",
                                   sample_index=j, seed=sample_seeds[j],
                                   error_msg="cell censored at sample $(ncomplete + 1) of $(nsamples)",
                                   error_type="CellTimeout", wall_total_s=elapsed())
            continue
        end

        executor  = traces[j]
        censored  = j > ncomplete
        # A censored run has no solution: read its gap from the partial trace, and keep
        # it out of the cross-trial bound check, where an unconverged pair would look
        # like a genuine inconsistency.
        if censored
            q = censored_quality(executor)
            timing = wall_timing(aborted === nothing ? NaN : aborted.elapsed_s)
        else
            q = extract_quality(executor, sols[j], bp, S0, restrict)
            # Prefer BenchmarkTools' timing; fall back to our own stamp when a censored
            # sample took the Trial down with it.
            timing = trial === nothing ? wall_timing(walls[j]) : sample_timing(trial, j)

            # Harvest value bounds for the post-sweep cross-trial consistency check.
            tb = extract_trial_bounds(sols[j], bp, S0, restrict; cell_id=unit.id,
                                      alg_label=unit.algorithm.label, sample_index=j,
                                      seed=sample_seeds[j])
            tb === nothing || push!(get!(() -> TrialBounds[], bounds_by_instance, bp.name), tb)

            # The policy GSRDP converged on. Outside the timed region, and independent of
            # `save_trace`: a sweep can want the spatial policy picture without paying for
            # the per-iteration matrices.
            if cfg.output.save_policy
                write_policy_csv(sols[j], bp,
                                 joinpath(dir, "runs", unit.id, "sample$(j)", "policy.csv"))
            end
        end

        trace_path = missing
        if cfg.output.save_trace && has_records(executor)
            sample_dir = joinpath(dir, "runs", unit.id, "sample$(j)")
            # Report-facing combined CSV (gap_max/gap_mean [+ initial_*]); becomes trace_path.
            rp = write_report_trace_csv(executor, joinpath(sample_dir, "trace.csv"))
            rp === nothing || (trace_path = relpath(rp, dir))
            # Heavy traces also write their own file alongside (scalar Trace/InitialTrace
            # are already captured by the report CSV above).
            for tr in executor.traces
                (tr isa Trace || tr isa InitialTrace) && continue
                write_trace_csv(tr, joinpath(sample_dir, trace_filename(tr)))
            end
        end

        agr = empty_agreement()

        rows[j] = assemble_row(unit, bp, built, timing, q, agr;
                               status=(censored ? "timeout" : "ok"),
                               sample_index=j, seed=sample_seeds[j], trace_path=trace_path,
                               error_msg=(censored && aborted !== nothing ? _short(aborted) : missing),
                               error_type=(censored ? "CellTimeout" : missing),
                               wall_total_s=elapsed())
    end
    return rows
end

_short(e) = first(sprint(showerror, e), 500)

function _log_cell(i, n, rows)
    r = first(rows)
    # A cell can now mix statuses (some samples completed, a later one was censored), so
    # log the histogram rather than the first row's status alone.
    counts = sort(collect(_status_counts(rows)); by=first)
    hist = join(("$(s)=$(c)" for (s, c) in counts), " ")
    # Summarise only the COMPLETED samples. A censored row's time is the timeout
    # budget, not a measured solve, and averaging it in would drag the cell's time
    # toward `timeout_s` while reading as a real measurement. The histogram above
    # already reports the censoring.
    ok = [row for row in rows if row.status == "ok"]
    if !isempty(ok)
        _agg(vals) = isempty(vals) ? "—" :
            string(round(mean(vals); sigdigits=4), "±",
                   round(length(vals) > 1 ? std(vals) : 0.0; sigdigits=4))
        ts = [row.time_median_s for row in ok if row.time_median_s !== missing]
        tstr = isempty(ts) ? "—" : _agg(ts) * "s"
        # `bellman_updates` and `iters` are different units and both are worth seeing:
        # their ratio is the per-iteration batch size, which is the whole point of a
        # sampled strategy. Printing the iteration count under the `bellman_updates`
        # label understated a full sweep by `num_states × (num_actions + 1)`.
        bustr = _agg([row.bellman_updates for row in ok if row.bellman_updates !== missing])
        itstr = _agg([row.num_iterations for row in ok if row.num_iterations !== missing])
        @info "[$i/$n] $(r.cell_id)" status=hist samples=length(rows) time=tstr bellman_updates=bustr iters=itstr agrees=r.agrees
    else
        @info "[$i/$n] $(r.cell_id)" status=hist
    end
    return nothing
end

function _status_counts(rows)
    counts = Dict{String,Int}()
    for row in rows
        counts[String(row.status)] = get(counts, String(row.status), 0) + 1
    end
    return counts
end
