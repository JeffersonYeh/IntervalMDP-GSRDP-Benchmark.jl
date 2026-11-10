# BenchmarkAnalysis — the statistical analysis & plotting suite that replaces the
# old ad-hoc report. Driven by `generate_report` (kept signature-compatible with the
# harness) and runnable standalone via `run_analyses` on any finished results dir.
#
# See METHODS_meanstd.md (written into each results dir) for the two-level mean/std
# model. The five analyses live in analyses.jl; the adapter and Level-1 reduction are
# split across adapter.jl / reduce.jl.
module BenchmarkAnalysis

using DataFrames
using CSV
using Statistics
using Interpolations
using CairoMakie
using LaTeXStrings
using Printf
using Dates
using Pkg
using TOML
using JSON
# For `initial_states` in a problem's netCDF model — the only place S0 is recorded for a
# results dir written before problems.csv carried it (see gap_video.jl).
using NCDatasets
using Random

export generate_report, run_analyses, load_canonical, load_bounds, render_gap_videos,
       analysis_per_problem_policy, REFERENCE, parse_group_spec, load_group_file,
       resolve_groups

# The reference algorithm every relative analysis normalizes against. The spec writes
# "gsrdp+all_state_sweep"; the real harness label is "gsrdp_exhaustive_state".
const REFERENCE = "gsrdp_exhaustive_state"

# The label the same arm carried before `sampling` was renamed from
# "all_states_sweep" to "exhaustive_state". Results banked under the old name are
# still the reference arm, so `run_analyses` falls back to it by name rather than
# dropping to whichever algorithm happens to sort first.
const LEGACY_REFERENCE = "gsrdp_all_states_sweep"

include(joinpath("..", "thinning.jl"))   # GapThinner / streaming trace reader (also in Bench)
include("adapter.jl")
include("reduce.jl")
include("analyses.jl")
include("gap_video.jl")
include("policy_plot.jl")
include("groups.jl")

"""
    run_analyses(results_dir; reference_label=REFERENCE, eps=nothing,
                 problems=nothing) -> NamedTuple

Load the canonical table from `results_dir`, run all five analyses, and write
figures (`<results_dir>/plots/`, named by analysis), result tables, `METHODS_meanstd.md`
and `manifest.txt` (into `<results_dir>`). `eps` defaults to the data's convergence
target. `problems` selects which per-problem gap figures to emit (default: every
problem in the dir); `all_problems=false` narrows that to a
smallest/median/largest-by-`n_trans` subset. `image_formats` chooses the figure file
types (e.g. `["png"]`, `["svg"]`, or both); when `nothing`, it is read from
`[output].image_formats` in the dir's `resolved_config.toml`, defaulting to png + svg.
The gap the analyses measure follows the sweep's `[property].restrict_to_initial` — the
initial-state gap when it converged on the initial states, the all-states gap otherwise
(see the gap-basis note in `adapter.jl`). Each trace is streamed and thinned to relative
gap tolerance `thin_rtol` (`0` loads every row; see the thinning note in `adapter.jl`).
Each analysis is isolated by try/catch so one plotting hiccup never loses the rest.

One policy figure per grid-world problem is always emitted into
`<results_dir>/plots/per_problem/<instance>_policy.*` from the `policy.csv` files the
sweep recorded (see policy_plot.jl) — a grid of the solved policy with one row per
algorithm and one column per sample. It needs no flag: the files are small, and a dir
holding none simply records that in `manifest.txt`.

`gap_videos` additionally renders one mp4 per run per sample of the gap over the state
space (see gap_video.jl); `nothing` reads `[output].gap_videos` from the dir's
`resolved_config.toml`, which defaults to off. It is opt-in because a results dir can
hold a hundred-plus multi-gigabyte `gap_trace.csv` files, which a routine re-analysis
should not pay for. `gap_video_max_frames` / `gap_video_framerate` likewise fall back to
`[output].gap_video_max_frames` / `[output].gap_video_framerate`.

`groups` (a vector of `name => [label or glob, ...]`, see groups.jl) additionally redraws
cactus, normalized_gap_curves, scatter_vs_reference and vs_transitions for each named
subset of the algorithms into `<results_dir>/plots/groups/<name>/`, the reference always
included. `only_groups=true` draws only those, skipping the full suite, the policy
figures and the videos; its manifest goes to `plots/groups/manifest.txt`, leaving the
full suite's `manifest.txt` untouched.
"""
function run_analyses(results_dir::AbstractString; reference_label::AbstractString=REFERENCE,
                      eps::Union{Nothing,Real}=nothing, problems::Union{Nothing,Vector{String}}=nothing,
                      all_problems::Bool=true,
                      image_formats::Union{Nothing,Vector{String}}=nothing,
                      gap_videos::Union{Nothing,Bool}=nothing,
                      gap_video_max_frames::Union{Nothing,Integer}=nothing,
                      gap_video_framerate::Union{Nothing,Integer}=nothing,
                      groups=nothing, only_groups::Bool=false,
                      thin_rtol::Real=DEFAULT_THIN_RTOL)
    only_groups && groups === nothing && error("run_analyses: only_groups=true needs groups")
    _IMAGE_FORMATS[] = image_formats === nothing ? _image_formats_from_dir(results_dir) : image_formats
    long = load_canonical(results_dir; thin_rtol=thin_rtol)
    # Read AFTER load_canonical, which fills the trace diagnostics it reports.
    completeness = load_completeness(results_dir)
    _FIG_NOTE[] = figure_note(completeness)

    # Rendered BEFORE the `isempty(long)` bail-out on purpose: the videos are read
    # straight from `runs/**/gap_trace.csv` and owe nothing to the canonical table, so a
    # sweep whose results.csv never made it to disk can still produce every one of them.
    # Isolated like the analyses — a broken video must not cost the manifest.
    videos = nothing
    if !only_groups && _gap_videos_enabled(results_dir, gap_videos)
        try
            videos = render_gap_videos(results_dir;
                max_frames=_gap_video_setting(results_dir, gap_video_max_frames,
                                              "gap_video_max_frames", _GAP_VIDEO_MAX_FRAMES),
                framerate=_gap_video_setting(results_dir, gap_video_framerate,
                                             "gap_video_framerate", _GAP_VIDEO_FRAMERATE))
        catch e
            @warn "gap video step failed" exception=(e, catch_backtrace())
        end
    end

    # Same reasoning as the videos above, and the same placement: policies are read
    # straight from `runs/**/policy.csv` and owe nothing to the canonical table, so a
    # sweep whose results.csv never made it to disk still gets its policy figures. Cheap
    # enough (one small CSV per sample) that it is not gated on a config flag the way the
    # videos are -- only on the grid geometry the figure needs to exist at all.
    policies = nothing
    if !only_groups
        try
            policies = analysis_per_problem_policy(results_dir, joinpath(results_dir, "plots"))
        catch e
            @warn "policy figure step failed" exception=(e, catch_backtrace())
        end
    end

    if isempty(long)
        # The fully-partial case: no usable data at all. Still write the manifest — this
        # is exactly when a reader most needs to be told the sweep never produced any,
        # and an early return with only a @warn is how that used to go unrecorded.
        @warn "run_analyses: no usable trace data" results_dir
        try
            write_manifest(results_dir, nothing, NaN, reference_label; completeness=completeness,
                           gap_videos=videos, policies=policies)
        catch e
            @warn "manifest write failed" exception=(e, catch_backtrace())
        end
        return (; results_dir, completeness, videos, policies)
    end
    ε = eps === nothing ? default_eps(long) : float(eps)
    fellback = false
    if !(reference_label in long.algorithm)
        if reference_label == REFERENCE && LEGACY_REFERENCE in long.algorithm
            @info "run_analyses: using pre-rename reference label" reference_label=LEGACY_REFERENCE
            reference_label = LEGACY_REFERENCE
        else
            @warn "run_analyses: reference not present, falling back to first algorithm" reference_label
            reference_label = first(sort(unique(long.algorithm)))
            fellback = true
        end
    end

    # Resolved against the reference actually in use (legacy label / fallback above), and
    # before any figure is drawn, so a malformed spec fails fast rather than after the suite.
    resolved = groups === nothing ? nothing :
               resolve_groups(groups, unique(long.algorithm), reference_label)

    rs = run_scalars(long, ε)
    l1 = level1(rs)
    plots = joinpath(results_dir, "plots"); mkpath(plots)
    subset = problems !== nothing ? problems :
             all_problems       ? sort(unique(String.(l1.problem))) :
                                  default_problem_subset(l1)

    results = Dict{String,Any}()
    for (name, f) in (only_groups ? () : (
        ("normalized_gap_curves",  () -> analysis_normalized_gap_curves(long, l1, plots; reference=reference_label, eps=ε)),
        ("cactus",                 () -> analysis_cactus(l1, plots; reference=reference_label)),
        ("per_problem_gap",        () -> analysis_per_problem_gap(long, plots; problems=subset)),
        ("per_problem_bounds",     () -> analysis_per_problem_bounds(load_bounds(results_dir; problems=subset), plots; problems=subset)),
        ("scatter_vs_reference",   () -> analysis_scatter_vs_reference(rs, plots; reference=reference_label)),
        ("vs_transitions",         () -> analysis_vs_transitions(rs, plots)),
    ))
        try
            results[name] = f()
        catch e
            @warn "analysis step failed" step=name exception=(e, catch_backtrace())
        end
    end

    for g in something(resolved, ())
        _run_group!(results, g, long, l1, rs, plots; reference=reference_label, eps=ε)
    end

    try
        write_methods_doc(results_dir)
        write_manifest(results_dir, rs, ε, reference_label;
                       completeness=completeness, reference_fallback=fellback,
                       gap_videos=videos, policies=policies, groups=resolved,
                       # a groups-only pass must not clobber the full suite's manifest,
                       # whose policy/video sections it has no data to rewrite
                       path=joinpath(only_groups ? joinpath(plots, "groups") : results_dir,
                                     "manifest.txt"))
    catch e
        @warn "doc/manifest write failed" exception=(e, catch_backtrace())
    end
    @info "analysis report written" plots ε reference=reference_label
    return (; results_dir, eps=ε, reference=reference_label, results, completeness, videos,
            policies, groups=resolved)
end

# The cross-algorithm figures for one resolved group, into `plots/groups/<name>/`. The
# analyses take their algorithm list (and colours, by position) from the frames they are
# handed, so filtering the frames is all a group needs. The group is prepended to the
# provenance caption, since a figure copied out of its directory loses the dir name.
function _run_group!(results, g, long, l1, rs, plots; reference, eps)
    keep(df) = df[in(g.algorithms).(df.algorithm), :]
    glong, gl1, grs = keep(long), keep(l1), keep(rs)
    dir = joinpath(plots, "groups", g.name)
    note = _FIG_NOTE[]
    _FIG_NOTE[] = join(filter(!isempty, ["group $(g.name): $(join(g.algorithms, ", "))", note]),
                       "  ·  ")
    try
        for (name, f) in (
            ("normalized_gap_curves", () -> analysis_normalized_gap_curves(glong, gl1, dir; reference=reference, eps=eps)),
            ("cactus",                () -> analysis_cactus(gl1, dir; reference=reference)),
            ("scatter_vs_reference",  () -> analysis_scatter_vs_reference(grs, dir; reference=reference)),
            ("vs_transitions",        () -> analysis_vs_transitions(grs, dir)),
        )
            try
                results["group:$(g.name)/$name"] = f()
            catch e
                @warn "group analysis step failed" group=g.name step=name exception=(e, catch_backtrace())
            end
        end
    finally
        _FIG_NOTE[] = note
    end
    return nothing
end

"""
    generate_report(results_dir; reference_label=REFERENCE)

Harness entry point (kept signature-compatible). Delegates to [`run_analyses`](@ref).
"""
function generate_report(results_dir::AbstractString; reference_label::AbstractString=REFERENCE)
    csv = joinpath(results_dir, "results.csv")
    isfile(csv) || (@warn "no results.csv to report" results_dir; return results_dir)
    run_analyses(results_dir; reference_label=reference_label)
    return results_dir
end

# Read `[output].image_formats` from the resolved config in a finished results dir,
# so a standalone `run_analyses` honours what the experiment was configured to emit.
# Falls back to the current default (png + svg) when absent or unreadable.
function _image_formats_from_dir(results_dir::AbstractString)
    cfg_path = joinpath(results_dir, "resolved_config.toml")
    isfile(cfg_path) || return _IMAGE_FORMATS[]
    try
        fmts = get(get(TOML.parsefile(cfg_path), "output", Dict()), "image_formats", nothing)
        fmts === nothing && return _IMAGE_FORMATS[]
        return String[lowercase(strip(String(f))) for f in fmts]
    catch
        return _IMAGE_FORMATS[]
    end
end

# Gap-video defaults, used when neither the caller nor the resolved config says
# otherwise. 600 frames at 15 fps is a 40-second video; see gap_video.jl for why the
# window is the FIRST N iterations rather than a subsample of the whole run.
const _GAP_VIDEO_MAX_FRAMES = 600
const _GAP_VIDEO_FRAMERATE  = 15

# `[output].gap_videos` from the resolved config, unless the caller passed an explicit
# value. Defaults to off — the expensive thing is never the default.
function _gap_videos_enabled(results_dir::AbstractString, explicit::Union{Nothing,Bool})
    explicit === nothing || return explicit
    try
        return get(get(_resolved_config(results_dir), "output", Dict()), "gap_videos", false) === true
    catch
        return false
    end
end

# Same fallback chain for the numeric video knobs: caller > `[output].<key>` > `default`.
function _gap_video_setting(results_dir::AbstractString, explicit::Union{Nothing,Integer},
                            key::AbstractString, default::Int)
    explicit === nothing || return Int(explicit)
    try
        v = get(get(_resolved_config(results_dir), "output", Dict()), key, nothing)
        v isa Real && isfinite(v) && v > 0 && return round(Int, v)
    catch
    end
    return default
end

# --- manifest.txt ----------------------------------------------------------
# The manifest is the single authority for cell and row accounting. The figures can only
# show what was loaded; a cell that was killed leaves nothing to plot, and a censored run
# with no trace at all (a warm-up timeout, or `timeout:not_run`) is not even in `rs`. So
# the counts below come from two different places on purpose: `completeness` from the raw
# results.csv and meta.json, `counts` from the reduced per-run table.
#
# `rs === nothing` is the fully-partial case — no usable trace data — where only the
# completeness block can be written.
function write_manifest(results_dir, rs::Union{DataFrame,Nothing}, ε::Real,
                        reference::AbstractString; completeness=nothing,
                        reference_fallback::Bool=false, gap_videos=nothing,
                        policies=nothing, groups=nothing,
                        path=joinpath(results_dir, "manifest.txt"))
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "BenchmarkAnalysis manifest")
        println(io, "date           : ", Dates.now())
        println(io, "julia          : ", VERSION)
        println(io, "reference      : ", reference,
                reference_fallback ? "  (FALLBACK — the configured reference has no usable runs)" : "")
        println(io, "eps (target)   : ", ε)
        println(io, "results_dir    : ", results_dir)

        if completeness !== nothing
            println(io)
            for line in completeness_lines(completeness)
                println(io, line)
            end
        end

        if rs === nothing
            println(io, "\nNo usable trace data — no figures were produced.")
        else
            counts = combine(groupby(rs, :algorithm),
                             :status => (s -> count(==(:converged), s)) => :solved,
                             :status => (s -> count(==(:timeout), s)) => :timed_out,
                             :censored => (c -> count(identity, c)) => :censored)
            println(io, "\nper-(alg,problem,seed) counts over the runs that produced data:")
            println(io, "  solved   = reached the analysis ε")
            println(io, "  timeout  = DERIVED: has data but never crossed ε (includes runs that met")
            println(io, "             their own, looser, convergence_eps)")
            println(io, "  censored = stopped by the harness at [benchmark].timeout_s")
            for r in eachrow(counts)
                @printf(io, "  %-32s solved=%-4d timeout=%-4d censored=%-4d\n",
                        r.algorithm, r.solved, r.timed_out, r.censored)
            end
            if completeness !== nothing
                notes = String[]
                isempty(completeness.missing_cells) ||
                    push!(notes, "$(length(completeness.missing_cells)) cell(s) missing from the sweep entirely")
                # A censored run with no trace (warm-up timeout, or a sample that never
                # started) has nothing to plot and nothing to reduce, so it reaches
                # neither the counts above nor any figure. Without this line it looks
                # like those runs simply did not happen.
                n_to = get(completeness.status_counts, "timeout", 0) +
                       get(completeness.status_counts, "timeout:not_run", 0)
                invisible = n_to - count(rs.censored)
                invisible > 0 &&
                    push!(notes, "$invisible censored run(s) produced no trace at all")
                if !isempty(notes)
                    println(io, "\n  NOTE: the counts above describe only the runs that produced data.")
                    for n in notes
                        println(io, "        - $n")
                    end
                    println(io, "        These contribute to NO row above and appear in NO figure — do not")
                    println(io, "        read a per-algorithm count as the number of runs attempted.")
                end
            end
        end

        # Gap videos are opt-in and per-sample; the manifest is where a reader finds out
        # which traces got one and, more importantly, which did not and why (a problem
        # with no grid geometry is skipped on purpose — see gap_video.jl).
        if gap_videos !== nothing
            println(io, "\ngap videos (plots/gap_videos/): rendered=", gap_videos.rendered,
                    " skipped=", gap_videos.skipped, " failed=", gap_videos.failed)
            for line in gap_videos.log
                println(io, "  ", line)
            end
        end

        # Same accounting for the policy grids. A problem with no grid geometry is skipped
        # on purpose (see policy_plot.jl), and so is a results dir written before
        # `[output].save_policy` existed — both need to say so somewhere a reader looks.
        if policies !== nothing
            println(io, "\npolicy figures (plots/per_problem/*_policy): rendered=", policies.rendered,
                    " skipped=", policies.skipped)
            for line in policies.log
                println(io, "  ", line)
            end
        end

        # Which labels each group's patterns actually selected — a glob that matched nothing
        # (a typo, or an arm this sweep did not run) is otherwise invisible in the figures.
        if groups !== nothing
            println(io, "\nalgorithm groups (plots/groups/<name>/), reference always included:")
            isempty(groups) && println(io, "  none — every group was empty (see the warnings)")
            for g in groups
                println(io, "  ", g.name, "  patterns: ", join(g.patterns, ", "))
                println(io, "    algorithms: ", join(g.algorithms, ", "))
                isempty(g.unmatched) || println(io, "    UNMATCHED:  ", join(g.unmatched, ", "))
            end
        end

        println(io, "\nkey package versions:")
        for (name, ver) in _pkg_versions(("CairoMakie", "DataFrames", "Interpolations"))
            @printf(io, "  %-20s %s\n", name, ver)
        end
    end
    return nothing
end

function _pkg_versions(names)
    out = Tuple{String,String}[]
    try
        deps = Pkg.dependencies()
        byname = Dict(info.name => info.version for (_, info) in deps)
        for n in names
            push!(out, (n, string(get(byname, n, "?"))))
        end
    catch
        for n in names; push!(out, (n, "?")); end
    end
    return out
end

# --- METHODS_meanstd.md ----------------------------------------------------
function write_methods_doc(results_dir)
    open(joinpath(results_dir, "METHODS_meanstd.md"), "w") do io
        write(io, _METHODS_TEXT)
    end
    return nothing
end

include("methods_text.jl")  # defines const _METHODS_TEXT

end # module
