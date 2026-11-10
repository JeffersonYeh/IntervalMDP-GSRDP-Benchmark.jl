# Data adapter: map the harness's on-disk results into the canonical long-format
# table the analyses consume. This is the ONLY file that knows the real storage
# layout (results.csv / problems.csv / runs/<cell>/sample<k>/trace.csv); everything
# downstream is decoupled from it.
#
# Canonical long table — one row per recorded trace sample:
#   algorithm        :: String   (the alg_label, e.g. "gsrdp_exhaustive_state")
#   problem          :: String
#   n_trans          :: Int|Missing  (num_transitions, joined from problems.csv)
#   num_states       :: Int|Missing
#   seed             :: Int
#   sample_idx       :: Int      (1-based row index within the trace file — the ORIGINAL
#                                 row, so it has gaps once the trace is thinned)
#   wall_time        :: Float64  (seconds since solve start, == wallclock_s)
#   bellman_updates  :: Int
#   gap              :: Float64  (sound gap U-L; see the gap-basis note below)
#   run_converged    :: Bool     (did this run reach its own convergence_eps)
#   convergence_eps  :: Float64  (the run's ε)
#   censored         :: Bool     (the harness stopped this run at [benchmark].timeout_s)
#   time_budget      :: Float64  (per-problem censoring budget — see note)
#   update_budget    :: Float64  (per-problem max observed bellman_updates)
#
# Gap-basis note: the `gap` column follows how the sweep actually converged, because
# that is the quantity its ε refers to. A sweep with `[property].restrict_to_initial =
# true` terminates on the gap reduced over the model's initial states, so `gap` is read
# from the trace's `initial_gap_max` column; otherwise it is `gap_max`, the all-states
# gap. The flag is read from `resolved_config.toml`, so a results dir without that file
# (or one whose config left the key unset) degrades to the all-states gap. Which basis
# was used is recorded in `load_diagnostics().gap_basis` and reported by the manifest and
# the figure note — it must never be invisible.
#
# The initial-state gap is not always there to be had: `initial_gap_max` exists only when
# the sweep ran with `[output].track_initial = true`, and it is all-`NaN` for a model that
# declares no initial states (`InitialTrace` records `NaN` for an empty `S0`). Either way
# that trace falls back to `gap_max`, exactly as the harness's own `extract_quality` falls
# back to all states for an empty `S0`. The basis then reports `:mixed` when only some
# traces fell back, or `:global_fallback` when every one of them did.
#
# Budget note: when the sweep ran with `[benchmark].timeout_s`, that IS the censoring
# budget and `time_budget` is set from it. Without it the solver has no time or
# iteration cap — it terminates only on convergence_eps — so there is no recorded
# budget, and the censoring-aware analyses fall back to treating "did not reach ε" as
# right-censored at the largest value observed for that problem across all runs. That
# fallback is contaminated by a killed run (it removes exactly the largest observations),
# which is one more reason to set `timeout_s` on any sweep that might not finish.
#
# Robustness contract: every function here tolerates a results directory produced by a
# sweep that was killed part-way. A missing results.csv, a missing problems.csv, and a
# trace.csv truncated mid-line by a SIGKILL all degrade to fewer rows plus a counted
# diagnostic — never an exception. A missing or unparseable resolved_config.toml degrades
# to the all-states gap basis. The analyses are individually try/caught downstream,
# but this loader is not, so a throw here loses the entire report.
#
# Thinning note: a sampled solver can record tens of millions of trace rows per run (a
# single coin2_K4 trajectory trace is 6.6 GB), and materialising every one of them in
# the long table is what used to OOM the machine. Each trace is therefore STREAMED and
# thinned on the fly with the relative-gap rule of `../thinning.jl` (`thin_rtol`,
# default 1e-3), keyed on both `gap_max` and `initial_gap_max` so the table is faithful
# on either gap basis. Every analysis reads the table as first crossings of gap levels
# or as curves on a ~100-point grid, both of which the rule preserves to within a
# relative `thin_rtol` in the gap. `thin_rtol = 0` loads every row. The rows read and
# kept are reported by the manifest.

# Diagnostics from the most recent `load_canonical`, surfaced by the manifest. A Ref
# rather than a second return value so the many existing call sites keep working
# (same pattern as `_IMAGE_FORMATS` in analyses.jl).
const _LOAD_DIAGNOSTICS = Ref((traces_read=0, traces_missing=0, traces_unreadable=0,
                               gap_basis=:global, rows_read=0, rows_kept=0,
                               thin_rtol=0.0))

# Default relative-gap tolerance for thinning traces at load time (see the note above).
const DEFAULT_THIN_RTOL = 1e-3

load_diagnostics() = _LOAD_DIAGNOSTICS[]

# The canonical schema in one place, so the empty frame returned on a missing
# results.csv carries every column the downstream reductions index — including
# time_budget/update_budget, which are otherwise only appended after the row loop.
function _empty_canonical()
    long = DataFrame(algorithm=String[], problem=String[], n_trans=Union{Int,Missing}[],
                     num_states=Union{Int,Missing}[], seed=Int[], sample_idx=Int[],
                     wall_time=Float64[], bellman_updates=Int[], gap=Float64[],
                     run_converged=Bool[], convergence_eps=Float64[], censored=Bool[])
    long.time_budget   = Float64[]
    long.update_budget = Float64[]
    return long
end

# Statuses whose rows carry usable trace data. "timeout" rows are censored runs: they
# stopped short of ε but everything they recorded up to that point is real.
const _USABLE_STATUS = ("ok", "timeout")

# Defensive numeric coercion for trace fields. A trace truncated by a SIGKILL yields a
# short final row, which CSV.jl fills with `missing` (`Float64(missing)` throws), and a
# column CSV.jl typed as Float64 can hold a non-integral value where an Int is expected
# (`Int(2.5)` throws). Returning `nothing` lets the caller drop just that row.
_as_float(x) = x isa Real && !ismissing(x) ? Float64(x) : nothing
function _as_int(x)
    x isa Real && !ismissing(x) || return nothing
    v = Float64(x)
    isfinite(v) && isinteger(v) || return nothing
    return Int(v)
end

"""
    _stream_trace(path, thin_rtol) -> NamedTuple | nothing

Read one `trace.csv` a line at a time, keeping only the rows the relative-gap thinning
of `../thinning.jl` selects on `gap_max` and `initial_gap_max`, so memory is O(kept rows)
however large the file is. `nothing` when the header lacks `wallclock_s` or
`bellman_updates` (a truncated header, an empty file).

Rows with the wrong field count, or a `wallclock_s` / `bellman_updates` that is not a
number (an integral one, for the latter), are dropped and counted in `nbad` — what a
SIGKILL mid-write leaves. An unparseable gap is kept as `NaN` and counted per column
(`nbad_g`, `nbad_i`), since whether it matters depends on the gap basis the caller picks.
`have_init` is whether ANY row, kept or not, carries a finite `initial_gap_max`.
"""
function _stream_trace(path::AbstractString, thin_rtol::Real)
    open(path, "r") do io
        header = readline(io)
        ncol = count(==(','), header) + 1
        cols = trace_header_columns(header)
        jw = get(cols, "wallclock_s", 0); jb = get(cols, "bellman_updates", 0)
        jg = get(cols, "gap_max", 0);     ji = get(cols, "initial_gap_max", 0)
        (jw > 0 && jb > 0) || return nothing

        idx = Int[]; w = Float64[]; bu = Int[]; g = Float64[]; gi = Float64[]
        emit(r) = (push!(idx, r[1]); push!(w, r[2]); push!(bu, r[3]); push!(g, r[4]);
                   push!(gi, r[5]); nothing)
        tb = TailBuffer{Tuple{Int,Float64,Int,Float64,Float64}}(GapThinner(thin_rtol, 2))
        fields = SubString{String}[]
        nrows = nbad = nbad_g = nbad_i = 0
        have_init = false
        for line in eachline(io)
            isempty(line) && continue
            nrows += 1
            split_fields!(fields, line)
            length(fields) == ncol || (nbad += 1; continue)
            wv = _field_float(fields, jw); bv = _field_float(fields, jb)
            (wv === nothing || bv === nothing || !isfinite(bv) || !isinteger(bv)) &&
                (nbad += 1; continue)
            gv = jg > 0 ? _field_float(fields, jg) : NaN
            gv === nothing && (nbad_g += 1; gv = NaN)
            iv = ji > 0 ? _field_float(fields, ji) : NaN
            iv === nothing && (nbad_i += 1; iv = NaN)
            isfinite(iv) && (have_init = true)
            offer!(emit, tb, (nrows, wv, Int(bv), gv, iv), (gv, iv))
        end
        flush_tail!(emit, tb)
        return (; idx, w, bu, g, gi, nrows, nbad, nbad_g, nbad_i, have_init,
                has_g=jg > 0, has_i=ji > 0)
    end
end

"""
    load_canonical(results_dir; thin_rtol=DEFAULT_THIN_RTOL) -> DataFrame

Read `results.csv`, join `num_transitions` from `problems.csv`, keep the rows with
usable trace data (`ok` and censored `timeout`), read every `trace_path`, and explode
into the canonical long-format table documented at the top of this file. The `gap`
column follows the sweep's `restrict_to_initial` — see the gap-basis note at the top.
Each trace is streamed and thinned to relative gap tolerance `thin_rtol` (`0` keeps
every row) — see the thinning note at the top. Pure with respect to the filesystem
snapshot.

Returns an empty canonical frame (not an error) when the directory has no
`results.csv` — the shape a sweep leaves when it is killed before its first cell
completes, since the header is written lazily with the first row.
"""
function load_canonical(results_dir::AbstractString; thin_rtol::Real=DEFAULT_THIN_RTOL)
    thin_rtol >= 0 || throw(ArgumentError("thin_rtol must be >= 0, got $thin_rtol"))
    n_read = n_missing = n_unreadable = 0
    rows_read = rows_kept = 0
    # How the sweep converged decides which gap the analyses measure; `n_no_initial`
    # counts the traces that could not honour it, which is what makes the basis `:mixed`.
    restrict = _restrict_to_initial_from_dir(results_dir)
    n_no_initial = 0
    n_used_initial = 0
    _finish(long) = begin
        basis = !restrict            ? :global :
                n_used_initial == 0  ? :global_fallback :
                n_no_initial == 0    ? :initial : :mixed
        _LOAD_DIAGNOSTICS[] = (traces_read=n_read, traces_missing=n_missing,
                               traces_unreadable=n_unreadable, gap_basis=basis,
                               rows_read=rows_read, rows_kept=rows_kept,
                               thin_rtol=Float64(thin_rtol))
        long
    end

    csv = joinpath(results_dir, "results.csv")
    if !isfile(csv)
        @warn "load_canonical: no results.csv (sweep killed before its first cell completed?)" results_dir
        return _finish(_empty_canonical())
    end
    res = try
        CSV.read(csv, DataFrame)
    catch e
        @warn "load_canonical: results.csv is unreadable" results_dir exception=e
        return _finish(_empty_canonical())
    end
    ("status" in names(res) && "trace_path" in names(res)) || begin
        @warn "load_canonical: results.csv has no status/trace_path columns" results_dir
        return _finish(_empty_canonical())
    end

    # join n_trans / num_states from problems.csv when present
    trans = Dict{String,Union{Int,Missing}}()
    states = Dict{String,Union{Int,Missing}}()
    pcsv = joinpath(results_dir, "problems.csv")
    if isfile(pcsv)
        try
            pdf = CSV.read(pcsv, DataFrame)
            for r in eachrow(pdf)
                trans[String(r.problem)] = :num_transitions in propertynames(pdf) ? r.num_transitions : missing
                states[String(r.problem)] = :num_states in propertynames(pdf) ? r.num_states : missing
            end
        catch e
            @warn "load_canonical: problems.csv is unreadable; n_trans will be missing" exception=e
        end
    end

    long = _empty_canonical()

    usable = res[coalesce.(in.(res.status, Ref(_USABLE_STATUS)), false), :]
    for row in eachrow(usable)
        ismissing(row.trace_path) && continue
        tp = String(row.trace_path)
        path = isabspath(tp) ? tp : joinpath(results_dir, tp)
        if !isfile(path)
            n_missing += 1
            continue
        end
        # One unreadable trace must never take the whole report down with it.
        try
            tr = _stream_trace(path, thin_rtol)
            # Honour the sweep's basis when this trace has an initial-state gap to offer:
            # a run without `track_initial` has no such column, and one whose model
            # declares no initial states has a column of nothing but NaN.
            gapcol = tr === nothing              ? nothing :
                     restrict && tr.have_init    ? :initial_gap_max :
                     tr.has_g                    ? :gap_max : nothing
            gapcol === nothing && (n_unreadable += 1; continue)
            restrict && (gapcol === :initial_gap_max ? (n_used_initial += 1) :
                                                       (n_no_initial += 1))
            # the unique problem instance lives in `instance_name`; the `problem` column
            # holds the source/builder name (e.g. "intervalmdp_file"), shared across instances.
            alg  = String(row.alg_label)
            prob = String(coalesce(row.instance_name, row.problem))
            seed = ismissing(row.seed) ? 0 : Int(row.seed)
            eps  = ismissing(row.convergence_eps) ? NaN : Float64(row.convergence_eps)
            conv = coalesce(row.converged, false)
            cens = coalesce(row.status, "") == "timeout"
            gaps = gapcol === :initial_gap_max ? tr.gi : tr.g
            # A truncated final line (dropped by the reader) or an unparseable gap in the
            # column we measure makes the trace count as unreadable; its other rows stay.
            nbad = tr.nbad + (gapcol === :initial_gap_max ? tr.nbad_i : tr.nbad_g)
            kept = 0
            for k in eachindex(tr.idx)
                g = gaps[k]
                # A NaN gap is a recorded non-measurement, not a defect — it carries no
                # value for any curve or crossing, so drop the row without calling the
                # trace unreadable. (An unparseable gap is also NaN here; it was counted
                # in `nbad` above.)
                isnan(g) && continue
                kept += 1
                # Budgets are per-problem and computed once the whole table is known;
                # NaN placeholders keep the row arity equal to the full schema.
                push!(long, (alg, prob,
                             get(trans, prob, missing), get(states, prob, missing),
                             seed, tr.idx[k], tr.w[k], tr.bu[k], g, Bool(conv), eps, cens,
                             NaN, NaN))
            end
            nbad > 0 && (n_unreadable += 1)
            rows_read += tr.nrows
            rows_kept += kept
            n_read += 1
        catch e
            n_unreadable += 1
            @warn "load_canonical: skipping unreadable trace" path exception=e
        end
    end

    # per-problem censoring budgets
    long.time_budget   = Vector{Float64}(undef, nrow(long))
    long.update_budget = Vector{Float64}(undef, nrow(long))
    if nrow(long) > 0
        # `timeout_s`, when the sweep declared one, is the real budget; max-observed is
        # only the fallback for sweeps that ran uncapped.
        timeout_s = _timeout_from_dir(results_dir)
        tb = Dict{String,Float64}(); ub = Dict{String,Float64}()
        for g in groupby(long, :problem)
            tb[g.problem[1]] = timeout_s === nothing ? maximum(g.wall_time) : timeout_s
            ub[g.problem[1]] = maximum(g.bellman_updates)
        end
        long.time_budget   .= getindex.(Ref(tb), long.problem)
        long.update_budget .= getindex.(Ref(ub), long.problem)
    end
    return _finish(long)
end

# --- value bounds ----------------------------------------------------------
# A second, narrower table: the lower and upper bounds themselves, rather than the gap
# between them. They come from the heavy per-iteration value traces, which a sweep only
# writes when `[output].traces` asked for "value"/"initial_value" — so this loader is
# independent of `load_canonical` and returns an empty frame for a dir that has none.

function _empty_bounds()
    return DataFrame(algorithm=String[], problem=String[], seed=Int[], sample_idx=Int[],
                     wall_time=Float64[], bellman_updates=Int[],
                     lower=Float64[], upper=Float64[])
end

# One bound reduced to a scalar per iteration: `(wallclock, bellman_updates, mean)`,
# where the mean is over the `v[...]` columns selected by `cols` (every column when
# `cols === nothing`). `nothing` when the file is absent or has nothing to reduce.
# Incomplete rows are dropped, so a trace truncated mid-line costs its last iteration
# rather than the whole sample.
function _bound_curve(path::AbstractString, cols::Union{Nothing,Vector{Int}})
    isfile(path) || return nothing
    df = CSV.read(path, DataFrame)
    vcols = [n for n in names(df) if startswith(n, "v[")]
    isempty(vcols) && return nothing
    use = cols === nothing ? vcols : [vcols[c] for c in cols if 1 <= c <= length(vcols)]
    isempty(use) && return nothing
    df = df[completecases(df, [["wallclock_s", "bellman_updates"]; use]), :]
    nrow(df) == 0 && return nothing
    return (df.wallclock_s, df.bellman_updates, vec(mean(Matrix{Float64}(df[!, use]), dims=2)))
end

"""
    load_bounds(results_dir; problems=nothing) -> DataFrame

The lower and upper value bounds, each reduced to the **mean over the model's initial
states S₀**, one row per recorded iteration:
`algorithm, problem, seed, sample_idx, wall_time, bellman_updates, lower, upper`.

`initial_value_trace_{lower,upper}.csv` already holds only the S₀ slice and is used as
written; `value_trace_{lower,upper}.csv` holds every state and is sliced with the S₀
indices from problems.csv, falling back to the model's netCDF. A sample with neither
pair of files, and a problem whose model declares no initial states, contribute no rows
— which is the whole table for a sweep that recorded no value traces.

`problems` restricts the load to those instances. These files are one row per iteration
by one column per state, so reading every one of them costs gigabytes on a real sweep;
callers pass the subset they are about to plot.

Robustness contract as `load_canonical`: an unreadable trace costs its own rows and
nothing else.
"""
function load_bounds(results_dir::AbstractString;
                     problems::Union{Nothing,Vector{String}}=nothing)
    out = _empty_bounds()
    csv = joinpath(results_dir, "results.csv")
    isfile(csv) || return out
    res = try
        CSV.read(csv, DataFrame)
    catch e
        @warn "load_bounds: results.csv is unreadable" results_dir exception=e
        return out
    end
    ("status" in names(res) && "trace_path" in names(res)) || return out

    s0cache = Dict{String,Vector{Int}}()
    usable = res[coalesce.(in.(res.status, Ref(_USABLE_STATUS)), false), :]
    for row in eachrow(usable)
        ismissing(row.trace_path) && continue
        prob = String(coalesce(row.instance_name, row.problem))
        (problems === nothing || prob in problems) || continue
        tp = String(row.trace_path)
        sdir = dirname(isabspath(tp) ? tp : joinpath(results_dir, tp))
        try
            lo = _bound_curve(joinpath(sdir, "initial_value_trace_lower.csv"), nothing)
            hi = _bound_curve(joinpath(sdir, "initial_value_trace_upper.csv"), nothing)
            if lo === nothing || hi === nothing
                s0 = get!(s0cache, prob) do
                    c = _initial_states_from_csv(results_dir, prob)
                    c === nothing && (c = _initial_states_from_model(model_path_for(results_dir, prob)))
                    c
                end
                isempty(s0) && continue
                lo = _bound_curve(joinpath(sdir, "value_trace_lower.csv"), s0)
                hi = _bound_curve(joinpath(sdir, "value_trace_upper.csv"), s0)
            end
            (lo === nothing || hi === nothing) && continue
            ws, bus, L = lo
            _, _, U = hi
            alg = String(row.alg_label)
            seed = ismissing(row.seed) ? 0 : Int(row.seed)
            for i in 1:min(length(L), length(U))
                w = _as_float(ws[i]); bu = _as_int(bus[i])
                (w === nothing || bu === nothing) && continue
                (isfinite(L[i]) && isfinite(U[i])) || continue
                push!(out, (alg, prob, seed, i, w, bu, L[i], U[i]))
            end
        catch e
            @warn "load_bounds: skipping unreadable value trace" sdir exception=e
        end
    end
    return out
end

# The sweep's resolved config, or an empty Dict when it is absent or unparseable — the
# two shapes a killed or hand-assembled results dir leaves behind.
function _resolved_config(results_dir::AbstractString)
    path = joinpath(results_dir, "resolved_config.toml")
    isfile(path) || return Dict{String,Any}()
    try
        return TOML.parsefile(path)
    catch
        return Dict{String,Any}()
    end
end

# `[benchmark].timeout_s` from the sweep's resolved config, or `nothing` when the sweep
# ran uncapped / the file is absent.
function _timeout_from_dir(results_dir::AbstractString)
    try
        t = Float64(get(get(_resolved_config(results_dir), "benchmark", Dict()), "timeout_s", 0.0))
        return t > 0 ? t : nothing
    catch
        return nothing
    end
end

# Did the sweep converge on the model's initial states only? `false` whenever the config
# is absent or left the key unset, which is also the harness default.
function _restrict_to_initial_from_dir(results_dir::AbstractString)
    try
        return get(get(_resolved_config(results_dir), "property", Dict()),
                   "restrict_to_initial", false) === true
    catch
        return false
    end
end

"""
    default_eps(long) -> Float64

The convergence target ε used when the caller does not specify one: the unique
`convergence_eps` recorded in the data if all runs agree, else `1e-6`.
"""
function default_eps(long::DataFrame)
    isempty(long) && return 1e-6
    eps = unique(skipmissing(long.convergence_eps))
    eps = filter(!isnan, eps)
    return length(eps) == 1 ? float(eps[1]) : 1e-6
end

# --- completeness -----------------------------------------------------------
# Everything above filters down to what survived. This reads the sweep's own record of
# what it INTENDED to run, so the report can say "31 of 40 cells are missing" rather
# than silently describing a subset. A killed cell leaves no row anywhere — it is
# invisible to every count derived from results.csv — so the expected-cell list in
# meta.json is the only thing that can detect it.

# A bare `catch` here would silently turn any bug in this function into "the file isn't
# there", which is indistinguishable from the condition it exists to detect — so warn.
function _read_json(path)
    isfile(path) || return nothing
    try
        return JSON.parsefile(path)
    catch e
        @warn "unreadable JSON; treating as absent" path exception=e
        return nothing
    end
end

"""
    load_completeness(results_dir) -> NamedTuple

Compare what the sweep declared it would run against what is actually on disk. Fields:

  - `complete`        — the sweep wrote `datetime_end`, i.e. it reached its own end
  - `stopped_early`   — it stopped cleanly at the sweep deadline (still `complete`)
  - `aborted`         — an `aborted.json` marker is present (SIGTERM)
  - `n_cells` / `cells_present` / `missing_cells` — expected vs observed cell ids
  - `status_counts`   — rows per `results.csv` status
  - `problems_status` — `nothing`, or counts of the `problems.csv` status column
  - `traces`          — the `load_diagnostics()` counters and `gap_basis` from the last load

Every field degrades to a safe default when its source file is absent, because absent
files are exactly the case this is here to report on.
"""
function load_completeness(results_dir::AbstractString)
    meta     = _read_json(joinpath(results_dir, "meta.json"))
    progress = _read_json(joinpath(results_dir, "progress.json"))
    aborted  = _read_json(joinpath(results_dir, "aborted.json"))

    expected = meta === nothing ? String[] : String[String(c) for c in get(meta, "cells", [])]
    n_cells  = meta === nothing ? length(expected) : Int(get(meta, "n_cells", length(expected)))
    samples  = meta === nothing ? nothing : get(meta, "samples", nothing)

    status_counts = Dict{String,Int}()
    present = String[]
    csv = joinpath(results_dir, "results.csv")
    if isfile(csv)
        try
            res = CSV.read(csv, DataFrame)
            if "status" in names(res)
                for s in res.status
                    k = ismissing(s) ? "<missing>" : String(s)
                    status_counts[k] = get(status_counts, k, 0) + 1
                end
            end
            "cell_id" in names(res) && (present = unique(String.(skipmissing(res.cell_id))))
        catch e
            @warn "load_completeness: results.csv unreadable" exception=e
        end
    end

    # Only meaningful when meta.json recorded the expected list; an older results dir
    # without it can still report row counts, just not missing cells.
    missing_cells = isempty(expected) ? String[] : setdiff(expected, present)

    problems_status = nothing
    pcsv = joinpath(results_dir, "problems.csv")
    if isfile(pcsv)
        problems_status = Dict{String,Int}()
        try
            pdf = CSV.read(pcsv, DataFrame)
            if "status" in names(pdf)
                for s in pdf.status
                    k = ismissing(s) ? "<missing>" : String(s)
                    problems_status[k] = get(problems_status, k, 0) + 1
                end
            else
                problems_status["<no status column>"] = nrow(pdf)
            end
        catch e
            @warn "load_completeness: problems.csv unreadable" exception=e
        end
    end

    return (complete        = meta !== nothing && haskey(meta, "datetime_end"),
            stopped_early   = meta !== nothing && get(meta, "stopped_early", false) === true,
            stop_reason     = meta === nothing ? "" : String(get(meta, "stop_reason", "")),
            aborted         = aborted !== nothing,
            abort_reason    = aborted === nothing ? "" : String(get(aborted, "reason", "")),
            n_cells         = n_cells,
            samples         = samples,
            expected_cells  = expected,
            cells_present   = present,
            missing_cells   = missing_cells,
            cells_done      = progress === nothing ? nothing : get(progress, "cells_done", nothing),
            status_counts   = status_counts,
            problems_csv    = isfile(pcsv),
            problems_status = problems_status,
            traces          = load_diagnostics())
end

# One human-readable phrase per gap basis, so the manifest and the figure note can never
# disagree about which gap the report describes.
_gap_basis_phrase(basis) =
    basis === :initial         ? "initial-state gap (restrict_to_initial = true)" :
    basis === :mixed           ? "initial-state gap where recorded, all-states gap for the " *
                                 "runs that recorded none (restrict_to_initial = true)" :
    basis === :global_fallback ? "all-states gap (restrict_to_initial = true, but no run " *
                                 "recorded an initial-state gap — no track_initial, or the " *
                                 "models declare no initial states)" :
                                 "all-states gap"

"""
    completeness_lines(c) -> Vector{String}

Render `load_completeness` as the manifest's COMPLETENESS block. Returns a leading
banner line when the sweep did not finish, so an incomplete run cannot be mistaken for
a complete one at a glance.
"""
function completeness_lines(c)
    lines = String[]
    incomplete = !c.complete || !isempty(c.missing_cells) || c.aborted
    incomplete && push!(lines, "*** SWEEP INCOMPLETE — the figures below describe a SUBSET of the run matrix ***", "")
    push!(lines, "COMPLETENESS")

    state = c.complete ? (c.stopped_early ? "STOPPED EARLY ($(c.stop_reason))" : "complete") :
            c.aborted  ? "ABORTED ($(c.abort_reason))" :
                         "INCOMPLETE (meta.json has no datetime_end — hard kill?)"
    push!(lines, "  sweep status  : $state")

    if isempty(c.expected_cells)
        push!(lines, "  cells         : $(length(c.cells_present)) present (meta.json lists no expected cells)")
    else
        push!(lines, "  cells         : $(c.n_cells) expected, $(length(c.cells_present)) present, " *
                     "$(length(c.missing_cells)) MISSING")
        for id in first(sort(c.missing_cells), 20)
            push!(lines, "    missing     : $id")
        end
        length(c.missing_cells) > 20 &&
            push!(lines, "    missing     : ... and $(length(c.missing_cells) - 20) more")
    end

    counts = sort(collect(c.status_counts); by=first)
    push!(lines, "  rows by status: " * (isempty(counts) ? "(none)" :
                                         join(("$(s)=$(n)" for (s, n) in counts), "  ")))
    if c.problems_csv
        ps = c.problems_status === nothing ? "(unreadable)" :
             join(("$(s)=$(n)" for (s, n) in sort(collect(c.problems_status); by=first)), " ")
        push!(lines, "  problems.csv  : present ($ps)")
    else
        push!(lines, "  problems.csv  : ABSENT — no num_transitions, so the scaling fits are empty")
    end
    t = c.traces
    push!(lines, "  traces        : $(t.traces_read) read, $(t.traces_unreadable) unreadable, " *
                 "$(t.traces_missing) missing")
    # The table is a thinned view of the traces; say how thin, so it is never invisible.
    rtol = get(t, :thin_rtol, 0.0)
    push!(lines, "  trace rows    : $(get(t, :rows_read, 0)) read, $(get(t, :rows_kept, 0)) kept " *
                 (rtol > 0 ? "(relative-gap thinning, rtol = $rtol; see adapter.jl)" : "(unthinned)"))
    # Which gap every figure and every solve_time below is measured against.
    push!(lines, "  gap basis     : $(_gap_basis_phrase(get(t, :gap_basis, :global)))")
    return lines
end

"""
    figure_note(c) -> String

A one-line provenance note stamped onto every figure, so a plot that leaves the results
directory still carries the fact that it describes a partial sweep, or a gap that is not
the all-states one. Empty for the ordinary case: a complete sweep on the all-states gap.
"""
function figure_note(c)
    parts = String[]
    isempty(c.missing_cells) || push!(parts, "$(length(c.missing_cells))/$(c.n_cells) cells MISSING (killed or never run)")
    n_to = get(c.status_counts, "timeout", 0) + get(c.status_counts, "timeout:not_run", 0)
    n_to > 0 && push!(parts, "$n_to run(s) censored at timeout_s")
    n_err = get(c.status_counts, "error", 0) + get(c.status_counts, "error:build", 0)
    n_err > 0 && push!(parts, "$n_err errored")
    c.problems_csv || push!(parts, "problems.csv absent")
    note = isempty(parts) ? "" : "incomplete data — " * join(parts, " · ")

    # A non-default gap basis changes what the y axis means, so it travels with the
    # figure too — but it is not a defect, so it stays outside the "incomplete data"
    # clause and joins it only when both apply.
    basis = get(c.traces, :gap_basis, :global)
    (basis === :global || basis === :global_fallback) && return note
    bnote = "gap basis: " * _gap_basis_phrase(basis)
    return isempty(note) ? bnote : note * " · " * bnote
end
