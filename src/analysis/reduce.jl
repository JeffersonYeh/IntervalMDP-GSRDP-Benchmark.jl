# Per-run scalar extraction (solve_time / solve_updates) with explicit right-censoring,
# Level-1 (over-seed) reduction, and trace interpolation onto a common grid.
#
# Two-level mean/std model (see METHODS_meanstd.md):
#   Level 1 = trial-repeat (seed) variation within a fixed (algorithm, problem).
#   Level 2 = problem-set variation across different problems.
# This file owns Level-1 only; Level-2 aggregation lives in the individual analyses.

# ---------------------------------------------------------------------------
# solve_time / solve_updates: first crossing of the gap target ε
# ---------------------------------------------------------------------------

"""
    first_crossing(xs, gaps, eps) -> Union{Float64,Missing}

The `x` value at the first sample whose `gap ≤ eps` (gap is monotone non-increasing,
so the first crossing is the solve point). `missing` if the run never reaches `eps`
(right-censored). `xs` and `gaps` must be in trace order.
"""
function first_crossing(xs, gaps, eps::Real)
    @inbounds for i in eachindex(gaps)
        gaps[i] <= eps && return Float64(xs[i])
    end
    return missing
end

"""
    run_scalars(long, eps) -> DataFrame

One row per (algorithm, problem, seed) run with `solve_time` (wall-clock) and
`solve_updates` (Bellman-update axis) = the first crossing of `eps`, `missing` when
censored. Also carries `n_trans`, `num_states`, and per-problem `time_budget` /
`update_budget`.

Two different notions of "timed out" are kept apart on purpose:

  - `status == :timeout` is DERIVED here — this run has data but never crossed the
    ANALYSIS ε. It also fires for a run that converged perfectly well to its own,
    looser, `convergence_eps`.
  - `censored` is reported by the harness — the run was stopped mid-solve at
    `[benchmark].timeout_s` and genuinely never finished.

Conflating them makes a stricter analysis ε look like an algorithmic failure, so every
consumer that reports "timed out" should say which one it means.
"""
function run_scalars(long::DataFrame, eps::Real)
    out = DataFrame(algorithm=String[], problem=String[], seed=Int[],
                    n_trans=Union{Int,Missing}[], num_states=Union{Int,Missing}[],
                    solve_time=Union{Float64,Missing}[], solve_updates=Union{Float64,Missing}[],
                    time_budget=Float64[], update_budget=Float64[], status=Symbol[],
                    censored=Bool[])
    has_censored = "censored" in names(long)
    for g in groupby(long, [:algorithm, :problem, :seed])
        gs = sort(g, :wall_time)
        st = first_crossing(gs.wall_time, gs.gap, eps)
        su = first_crossing(gs.bellman_updates, gs.gap, eps)
        push!(out, (gs.algorithm[1], gs.problem[1], gs.seed[1],
                    gs.n_trans[1], gs.num_states[1], st, su,
                    gs.time_budget[1], gs.update_budget[1],
                    ismissing(st) ? :timeout : :converged,
                    has_censored ? any(gs.censored) : false))
    end
    return out
end

# ---------------------------------------------------------------------------
# Level-1 reduction: collapse the seed repeats to one value per (alg, problem)
# ---------------------------------------------------------------------------

_med(x) = (v = collect(skipmissing(x)); isempty(v) ? missing : median(v))
_mn(x)  = (v = collect(skipmissing(x)); isempty(v) ? missing : mean(v))
_sd(x)  = (v = collect(skipmissing(x)); length(v) > 1 ? std(v) : 0.0)
_nsolved(x) = count(!ismissing, x)

"""
    level1(run_scalars) -> DataFrame

Collapse the seed repeats: one row per (algorithm, problem) carrying both the
**median** (wall-clock robust) and **mean** per-problem solve_time/solve_updates,
their seed-level std (Level-1 measurement variability), the number of seeds, and the
number that reached ε (`n_solved`). Reductions run over solved seeds only; a problem
the algorithm never solved yields `missing` central values with `n_solved == 0`.
"""
function level1(rs::DataFrame)
    combine(groupby(rs, [:algorithm, :problem]),
            :solve_time    => _med => :time_median,
            :solve_time    => _mn  => :time_mean,
            :solve_time    => _sd  => :time_std,
            :solve_updates => _med => :upd_median,
            :solve_updates => _mn  => :upd_mean,
            :solve_updates => _sd  => :upd_std,
            :solve_time    => _nsolved => :n_solved,
            :solve_time    => length   => :n_seeds,
            # How many of those seeds were stopped by the harness rather than by
            # convergence — the difference between "the algorithm is slow" and "we ran
            # out of wall clock", which the figures otherwise render identically.
            :censored      => count    => :n_censored,
            :n_trans       => first => :n_trans,
            :num_states    => first => :num_states,
            :time_budget   => first => :time_budget,
            :update_budget => first => :update_budget)
end

# ---------------------------------------------------------------------------
# Trace interpolation onto a common grid (ported from the old report.jl
# _aggregate_traces; gap is monotone so last-value-carried-forward via Flat()).
# ---------------------------------------------------------------------------

"""
    interp_gap(trace_df, grid; xcol=:wall_time, ycol=:gap) -> Vector{Float64}

Interpolate `ycol` onto `grid` along `xcol` with linear interpolation and flat
extrapolation (LVCF beyond the last sample — valid because gap is monotone, as are the
individual bounds it is the difference of).
"""
function interp_gap(tr::AbstractDataFrame, grid; xcol::Symbol=:wall_time, ycol::Symbol=:gap)
    ds = sort(tr, xcol)
    xs = Float64.(ds[!, xcol]); ys = Float64.(ds[!, ycol])
    if length(xs) == 1
        return fill(ys[1], length(grid))
    end
    itp = linear_interpolation(xs, ys; extrapolation_bc=Flat())
    return itp.(grid)
end

"""
    seed_mean_curve(long, alg, prob, grid; xcol=:wall_time, ycol=:gap) -> (mean, std)

Level-1 mean ± std curve of `ycol` over the seeds of one (algorithm, problem) on a
common `grid`. Returns `(mean, std)` vectors aligned with `grid`.
"""
function seed_mean_curve(long::DataFrame, alg::AbstractString, prob::AbstractString,
                         grid; xcol::Symbol=:wall_time, ycol::Symbol=:gap)
    sub = long[(long.algorithm .== alg) .& (long.problem .== prob), :]
    isempty(sub) && return (fill(NaN, length(grid)), fill(NaN, length(grid)))
    curves = [interp_gap(g, grid; xcol=xcol, ycol=ycol) for g in groupby(sub, :seed)]
    M = reduce(hcat, curves)
    return (vec(mean(M, dims=2)), size(M, 2) > 1 ? vec(std(M, dims=2)) : zeros(length(grid)))
end
