# The seven analyses. Each is a pure-ish function `analysis_*(...) -> (figure, table)`
# that also saves `<dir>/<name>.png` and `.svg`. Figures are named by analysis (not
# numbered). The two-level mean/std model is documented in METHODS_meanstd.md; each
# docstring states which mean and which std the figure shows and over what.

const _PALETTE = [:steelblue, :darkorange, :forestgreen, :crimson, :purple,
                  :teal, :brown, :magenta, :goldenrod, :navy]
_color(i) = _PALETTE[((i - 1) % length(_PALETTE))+1]

# add a legend only when the axis actually has labeled plots (Makie throws otherwise)
function _legend!(ax; kwargs...)
    try
        axislegend(ax; kwargs...)
    catch
        # no labeled series — skip the legend rather than lose the figure
    end
    return nothing
end

# place a single shared legend (built from `ax`'s labeled series) into a figure cell
# outside the plot area — used by the side-by-side time/updates figures. Skips silently
# when the axis has no labeled series (Makie throws otherwise).
function _shared_legend!(cell, ax; kwargs...)
    try
        Legend(cell, ax; kwargs...)
    catch
        # no labeled series — skip rather than lose the figure
    end
    return nothing
end

# `errorbars!(ax, x, y, e)` puts the whisker's low end at `y - e`. With a handful of seeds
# a fast problem can have a seed-std larger than its mean, so `y - e < 0` — harmless on a
# linear axis, fatal on a log one: Makie takes log10 of the plot's bounding box when it
# autoscales, and a negative corner throws a DomainError that takes down the ENTIRE figure
# (and, in `vs_transitions`, the fit CSVs written after it). Clamp the low end to a
# fraction of the centre: the whisker stays visible, stays positive, and the high end
# still carries the true spread.
const _LOG_WHISKER_FLOOR = 0.1   # lowest whisker end, as a fraction of the value

_is_log(scale) = scale === log10 || scale === log || scale === log2

# (low, high) offsets for `errorbars!`, plus the number of low ends the log axis clamped.
# `max(centre, 0)` also covers a centre of exactly 0 — a run already below ε in the
# trace's first row, at `wallclock_s = 0` — whose whisker then collapses to zero length
# rather than reaching into negative territory.
function _whiskers(centre, err, scale)
    _is_log(scale) || return (err, err, 0)
    cap = (1 - _LOG_WHISKER_FLOOR) .* max.(centre, 0.0)
    return (min.(err, cap), err, count(err .> cap))
end

# caption text for the clamped whiskers, empty when nothing was clamped
_clamp_note(n) = n == 0 ? "" :
    "$n whisker low end$(n == 1 ? "" : "s") clamped on the log panels (seed-std exceeds the mean)"

# figure file types to emit; set by `run_analyses` from `[output].image_formats`.
# Defaults to raster + vector so a standalone call still produces both.
const _IMAGE_FORMATS = Ref{Vector{String}}(["png", "svg"])

# Provenance note stamped onto every figure, set by `run_analyses` from
# `figure_note(load_completeness(dir))`. Empty for a complete sweep. A figure regularly
# outlives the results directory it came from — it ends up in a slide or a thesis
# chapter — so the fact that it describes a PARTIAL run has to travel with the image
# itself, not just sit in manifest.txt.
const _FIG_NOTE = Ref("")

# Append the provenance note (plus any figure-specific `extra`) as a caption row.
function _note!(fig, extra::AbstractString="")
    parts = filter(!isempty, [extra, _FIG_NOTE[]])
    isempty(parts) && return nothing
    try
        Label(fig[end+1, :], join(parts, "  ·  "); fontsize=10, color=:gray40,
              halign=:left, tellwidth=false, justification=:left)
    catch
        # a caption must never cost us the figure
    end
    return nothing
end

# save a figure in each configured format, with the provenance caption attached
function _save(fig, dir, name; note::AbstractString="")
    _note!(fig, note)
    mkpath(dir)
    for ext in _IMAGE_FORMATS[]
        CairoMakie.save(joinpath(dir, name * "." * ext), fig)
    end
    return nothing
end

# ===========================================================================
# Analysis 1 — Normalized, collapsed gap curves
# ===========================================================================

# Gaps are floored before the log so a converged (gap = 0) problem doesn't send the
# geometric mean to -Inf/NaN; with the floor it collapses cleanly toward the floor.
const _GAP_FLOOR = 1e-9

# Rowwise geometric mean and ±1 geometric-std band over the columns (problems) of `M`.
# Returns `(gmean, lo, hi)` with `lo = gmean / GSD`, `hi = gmean * GSD` (multiplicative).
function _geo_stats(M::AbstractMatrix)
    L = log.(max.(M, _GAP_FLOOR))
    μ = vec(mean(L, dims=2))
    σ = size(L, 2) > 1 ? vec(std(L, dims=2)) : zeros(size(L, 1))
    return exp.(μ), exp.(μ .- σ), exp.(μ .+ σ)
end

# Draw one normalized-gap panel (`xcol` scaled by the reference's `refcol`) onto `ax`.
# Returns `(info::DataFrame, excluded::Int)` where `info` is (algorithm, n_problems).
function _normalized_panel!(ax, long::DataFrame, l1::DataFrame, reference::AbstractString,
                            xcol::Symbol, refcol::Symbol, ngrid::Int; τfloor::Real=1e-3)
    refmap = Dict(r.problem => r[refcol] for r in eachrow(l1[l1.algorithm .== reference, :]))
    usable = [p for (p, t) in refmap if !ismissing(t) && t > 0]
    excluded = length(unique(long.problem)) - length(usable)
    algs = sort(unique(long.algorithm))
    # Log-spaced τ grid from the earliest recorded progress (but no earlier than `τfloor`)
    # to the latest row of any run. τ spans many decades — a run that pays JIT or times
    # out lands at τ ~ 1e2…1e6 — so a linear grid puts every convergence into its first
    # cell. Rows at τ = 0 still anchor the interpolation; they just fall left of the axis.
    τs = [Float64(r[xcol]) / refmap[r.problem] for r in eachrow(long) if r.problem in usable]
    τpos = filter(>(0), τs)
    τmax = max(maximum(τs; init=3.0), 3.0)
    τmin = min(max(minimum(τpos; init=τfloor), τfloor), τmax / 10)
    grid = exp10.(range(log10(τmin), log10(τmax), length=ngrid))
    info = DataFrame(algorithm=String[], n_problems=Int[])
    for (ai, a) in enumerate(algs)
        curves = Vector{Vector{Float64}}()
        for p in usable
            Tref = refmap[p]
            sub = long[(long.algorithm .== a) .& (long.problem .== p), :]
            isempty(sub) && continue
            scaled = copy(sub); scaled[!, xcol] = Float64.(scaled[!, xcol]) ./ Tref
            m, _ = seed_mean_curve(scaled, a, p, grid; xcol=xcol)
            push!(curves, m)
        end
        isempty(curves) && continue
        gmean, lo, hi = _geo_stats(reduce(hcat, curves))
        band!(ax, grid, lo, hi; color=(_color(ai), 0.15))
        lines!(ax, grid, gmean; color=_color(ai), linewidth=2.5, label=a)
        push!(info, (a, length(curves)))
    end
    return info, excluded
end

"""
    analysis_normalized_gap_curves(long, l1, dir; reference, eps, ngrid=200) -> (fig, info)

Wall-clock and Bellman-update panels side by side (shared legend). One line per
algorithm: gap (linear y) vs. dimensionless `τ = x / X_ref(p)`, where `x` is `wall_time`
(left) or `bellman_updates` (right) and `X_ref(p)` is the reference's Level-1 **mean**
solve_time / solve_updates on problem `p`. Every algorithm on a problem is normalized by
the **same** `X_ref(p)`, so the reference reaches gap = ε at τ ≈ 1. τ is on a log axis
over a log-spaced grid (see `_normalized_panel!`). Level-1: mean gap over seeds on that
common τ grid. Level-2: **geometric mean** gap over problems with a ±1
**geometric-std** band (`gmean / GSD … gmean · GSD`); gaps are floored at `_GAP_FLOOR`
before the log. Problems where the reference times out are excluded (count returned).
Vertical line at τ = 1, horizontal dotted line at gap = ε.
"""
function analysis_normalized_gap_curves(long::DataFrame, l1::DataFrame, dir;
                                        reference::AbstractString, eps::Real, ngrid::Int=200)
    fig = Figure(size=(1180, 1000))
    info = DataFrame(algorithm=String[], n_problems=Int[]); excluded = 0; legend_ax = nothing
    cols = ((:wall_time,       :time_mean, "τ = wall_time / T_ref(problem)"),
            (:bellman_updates, :upd_mean,  "τ = Bellman updates / U_ref(problem)"))
    # rows: linear-y then log-y (log exposes the sub-ε tail the linear axis flattens to 0)
    for (i, (scale, yscale)) in enumerate((("linear", identity), ("log", log10)))
        for (j, (xcol, refcol, xlabel)) in enumerate(cols)
            ax = Axis(fig[i, j]; xlabel=xlabel, ylabel="sound gap U − L", xscale=log10, yscale=yscale,
                      title="Normalized gap curves — $(scale) y (geomean ± GSD over problems)")
            info_j, excluded = _normalized_panel!(ax, long, l1, reference, xcol, refcol, ngrid)
            vlines!(ax, [1.0]; color=:gray, linestyle=:dash)
            hlines!(ax, [Float64(eps)]; color=:black, linestyle=:dot)
            yscale === log10 && ylims!(ax, Float64(eps) / 10, 1.5)
            (i == 1 && j == 1) && (info = info_j)
            legend_ax = ax
        end
    end
    _shared_legend!(fig[1:2, 3], legend_ax)
    # Each algorithm's geometric mean is taken over the problems IT has curves for, so
    # two curves on these axes can describe different problem sets — and a wall-clock
    # kill removes exactly the largest problems, biasing the survivor optimistic. That
    # makes the per-algorithm problem count part of the result, not a diagnostic.
    n_all = length(unique(long.problem))
    per_alg = join(("$(r.algorithm)=$(r.n_problems)" for r in eachrow(info)), ", ")
    note = "problems per algorithm (of $n_all): " * (isempty(per_alg) ? "none" : per_alg)
    excluded > 0 && (note *= "  ·  $excluded problem(s) excluded (no reference solve time)")
    _save(fig, dir, "normalized_gap_curves"; note=note)
    return (fig, (curves=info, excluded=excluded))
end

# ===========================================================================
# Analysis 2 — Cactus / quantile plot
# ===========================================================================

"""
    analysis_cactus(l1, dir; reference) -> fig

For each algorithm, Level-1 **mean** solve_time per solved problem, sorted ascending,
plotted as a cumulative line: point `(x, y)` ⇒ the x-th fastest solved instance took
`y` seconds. Timed-out instances are not plotted (line length = number solved). Log y.
Each point carries ± Level-1 **seed-std** whiskers (`time_std` / `upd_std`); where the
std exceeds the mean the low end is clamped by [`_whiskers`](@ref) to keep the log axis
drawable, and the caption records how many were. The reference is styled distinctly (dashed black). The wall-clock and Bellman-updates
(`solve_updates`) variants are drawn side by side on one figure (`cactus`) sharing a
single legend placed outside the plot area.

Reading a shortened line: this figure cannot by itself distinguish "the algorithm did
not solve that instance" from "that cell never ran". Each legend entry therefore carries
`solved/total` against the problems present in the data, and the caption carries the
count of cells missing from the sweep entirely.
"""
function analysis_cactus(l1::DataFrame, dir; reference::AbstractString)
    algs = sort(unique(l1.algorithm))
    nprob = length(unique(l1.problem))
    fig = Figure(size=(1180, 520))
    # returns the number of whisker low ends `_whiskers` clamped on this (log) panel
    function draw!(ax, col, stdcol)
        nclamped = 0
        for (ai, a) in enumerate(algs)
            # sort rows (not the bare column) so each seed-std stays paired with its mean
            sub = sort(l1[(l1.algorithm .== a) .& .!ismissing.(l1[!, col]), :], col)
            isempty(sub) && continue
            ys = Float64.(sub[!, col]); es = Float64.(coalesce.(sub[!, stdcol], 0.0))
            xs = 1:length(ys)
            isref = a == reference
            c = isref ? :black : _color(ai)
            lo, hi, n = _whiskers(ys, es, log10)
            nclamped += n
            errorbars!(ax, xs, ys, lo, hi; color=(c, 0.4), whiskerwidth=6)
            scatterlines!(ax, xs, ys; color=c, linestyle=isref ? :dash : :solid,
                          label="$a ($(length(ys))/$nprob)", markersize=8, linewidth=2)
        end
        return nclamped
    end
    axt = Axis(fig[1, 1]; xlabel="instances solved (sorted)",
               ylabel="mean solve time (s) ± seed-std",
               title="Cactus plot — wall-clock", yscale=log10)
    axu = Axis(fig[1, 2]; xlabel="instances solved (sorted)",
               ylabel="mean Bellman updates ± seed-std",
               title="Cactus plot — Bellman updates", yscale=log10)
    nclamped = draw!(axt, :time_mean, :time_std) + draw!(axu, :upd_mean, :upd_std)
    _shared_legend!(fig[1, 3], axt)
    ncens = "n_censored" in names(l1) ? sum(l1.n_censored) : 0
    note = join(filter(!isempty, [
        "legend shows solved/$nprob problems present in the data",
        ncens > 0 ? "$ncens run(s) stopped at timeout_s, not by the algorithm" : "",
        _clamp_note(nclamped)]), "  ·  ")
    _save(fig, dir, "cactus"; note=note)
    return fig
end

# ===========================================================================
# Analysis 3 — Per-problem gap vs. time
# ===========================================================================

"""
    analysis_per_problem_gap(long, dir; problems, ngrid=120) -> Vector{String}

One figure per problem, 2×2 with a single legend: columns are gap-vs-`wall_time` and
gap-vs-`bellman_updates`, rows are linear and log y (the log row exposes the sub-ε tail
the linear axis flattens onto zero). Each line per algorithm is the Level-1 **mean** gap
curve over seeds on a common grid, with a ± **seed-std** band (Level-1 only; no
cross-problem aggregation). On the log row the curve and its band are floored at
`_GAP_FLOOR`, since a converged gap of 0 has no place on a log axis. Saved under
`<dir>/per_problem/`. Returns the list of problems plotted.
"""
function analysis_per_problem_gap(long::DataFrame, dir; problems::Vector{String}, ngrid::Int=120)
    sub = joinpath(dir, "per_problem")
    algs = sort(unique(long.algorithm))
    done = String[]
    for prob in problems
        pl = long[long.problem .== prob, :]
        isempty(pl) && continue
        fig = Figure(size=(1180, 1000))
        legend_ax = nothing
        for (i, (scale, yscale)) in enumerate((("linear", identity), ("log", log10)))
            for (j, (xcol, xlabel)) in enumerate(((:wall_time, "wall time (s)"),
                                                  (:bellman_updates, "Bellman updates")))
                xmax = maximum(pl[!, xcol]); xmax > 0 || continue
                grid = range(0, Float64(xmax), length=ngrid)
                ax = Axis(fig[i, j]; xlabel=xlabel, ylabel="sound gap U − L", yscale=yscale,
                          title="vs $(xlabel) — $(scale) y")
                # Smallest gap any curve actually reaches before flooring, which is what
                # the log panel's lower limit follows: a band edge sitting on the floor
                # would otherwise stretch the axis down over decades that hold no curve.
                ymin = Inf
                for (ai, a) in enumerate(algs)
                    (a in pl.algorithm) || continue
                    mu, sd = seed_mean_curve(pl, a, prob, grid; xcol=xcol)
                    all(isnan, mu) && continue
                    lo = mu .- sd; hi = mu .+ sd
                    if yscale === log10
                        ymin = min(ymin, minimum(v for v in mu if v > 0; init=Inf))
                        mu = max.(mu, _GAP_FLOOR)
                        lo = max.(lo, _GAP_FLOOR); hi = max.(hi, _GAP_FLOOR)
                    end
                    band!(ax, grid, lo, hi; color=(_color(ai), 0.18))
                    lines!(ax, grid, mu; color=_color(ai), linewidth=2.5, label=a)
                end
                yscale === log10 && isfinite(ymin) &&
                    ylims!(ax, max(ymin, _GAP_FLOOR) / 3, 1.5)
                legend_ax = ax
            end
        end
        legend_ax === nothing && continue
        _shared_legend!(fig[1:2, 3], legend_ax)
        # After the panels and the legend: `:` spans the columns that exist when the
        # cell is created, so a title added first would land in column 1 alone.
        Label(fig[0, :], "$(prob) — sound gap (mean ± seed-std)";
              fontsize=18, font=:bold, tellwidth=false)
        _save(fig, sub, "$(prob)_gap")
        push!(done, prob)
    end
    return done
end

"""
    analysis_per_problem_bounds(bounds, dir; problems, ngrid=120) -> Vector{String}

One figure per problem with the bound-vs-`wall_time` and bound-vs-`bellman_updates`
panels drawn side by side, sharing a single legend. Two curves per algorithm in its
colour — the lower bound L (solid) and the upper bound U (dashed), each reduced to the
mean over the model's initial states S₀ (see [`load_bounds`](@ref)) and then taken as
the Level-1 **mean** over samples on a common grid, with a ± **sample-std** band.

Linear y only, unlike the gap figure: L starts at exactly 0 and U at a constant, neither
of which a log axis can draw, and the bounds do not decay toward zero the way the gap
between them does. Saved under `<dir>/per_problem/`. Returns the problems plotted.
"""
function analysis_per_problem_bounds(bounds::DataFrame, dir; problems::Vector{String},
                                     ngrid::Int=120)
    isempty(bounds) && return String[]
    sub = joinpath(dir, "per_problem")
    algs = sort(unique(bounds.algorithm))
    done = String[]
    for prob in problems
        pl = bounds[bounds.problem .== prob, :]
        isempty(pl) && continue
        fig = Figure(size=(1180, 540))
        legend_ax = nothing
        for (j, (xcol, xlabel)) in enumerate(((:wall_time, "wall time (s)"),
                                              (:bellman_updates, "Bellman updates")))
            xmax = maximum(pl[!, xcol]); xmax > 0 || continue
            grid = range(0, Float64(xmax), length=ngrid)
            ax = Axis(fig[1, j]; xlabel=xlabel, ylabel="value bound (mean over S₀)",
                      title="vs $(xlabel)")
            for (ai, a) in enumerate(algs)
                (a in pl.algorithm) || continue
                for (ycol, style) in ((:lower, :solid), (:upper, :dash))
                    mu, sd = seed_mean_curve(pl, a, prob, grid; xcol=xcol, ycol=ycol)
                    all(isnan, mu) && continue
                    band!(ax, grid, mu .- sd, mu .+ sd; color=(_color(ai), 0.18))
                    # Only L carries the label, so the legend names each algorithm once;
                    # the solid/dashed convention is in the figure note.
                    lines!(ax, grid, mu; color=_color(ai), linewidth=2.5, linestyle=style,
                           label=(ycol === :lower ? a : nothing))
                end
            end
            legend_ax = ax
        end
        legend_ax === nothing && continue
        _shared_legend!(fig[1, 3], legend_ax)
        Label(fig[0, :], "$(prob) — value bounds (mean ± sample-std)";
              fontsize=18, font=:bold, tellwidth=false)
        _save(fig, sub, "$(prob)_bounds"; note="solid = lower bound L  ·  dashed = upper bound U")
        push!(done, prob)
    end
    return done
end

# default representative subset: smallest / median / largest by n_trans
function default_problem_subset(l1::DataFrame)
    d = unique(select(l1, [:problem, :n_trans]))
    d = d[.!ismissing.(d.n_trans), :]
    if isempty(d)
        # No problems.csv (the usual shape of a killed sweep) means nothing can be ranked
        # by size, and the fallback quietly emits a figure for EVERY problem instead of
        # the intended three. Worth saying out loud — it changes what the report contains.
        @warn "default_problem_subset: no num_transitions data; emitting a per-problem figure for every problem"
        return unique(l1.problem)
    end
    d = sort(d, :n_trans)
    n = nrow(d)
    idx = unique([1, cld(n, 2), n])
    return String.(d.problem[idx])
end

# ===========================================================================
# Analysis 4 — Algorithm-vs-reference time scatter
# ===========================================================================

"""
    analysis_scatter_vs_reference(rs, dir; reference) -> fig

A scatter with x = reference, y = algorithm, **one point per problem**, with
**all non-reference algorithms overlaid** on a single axis (distinct color). Uses
Level-1 **mean** values and **seed-std whiskers in both x and y** (Level-1); on the log
row a whisker low end that would fall at or below zero (seed-std > mean) is floored at
10% of the value, which the caption records. The `y = x`
diagonal is drawn (points below ⇒ algorithm faster). 2×2 layout: columns are the
solve_time and Bellman-updates variants, rows are linear and log–log axes; a single
legend is placed outside the plot area.
"""
# Draw one alg-vs-reference scatter panel onto `ax` (points + seed-std whiskers + y=x).
# `scale` is the panel's axis scale (both axes share it); returns the number of whisker
# low ends `_whiskers` had to clamp to keep a log panel drawable.
function _scatter_vs_ref_panel!(ax, agg::DataFrame, refmap::Dict, algs::Vector{<:AbstractString},
                                scale)
    allx = Float64[]; ally = Float64[]; nclamped = 0
    for (ai, a) in enumerate(algs)
        sub = agg[agg.algorithm .== a, :]
        xs = Float64[]; ys = Float64[]; xe = Float64[]; ye = Float64[]
        for r in eachrow(sub)
            haskey(refmap, r.problem) || continue
            rr = refmap[r.problem]
            (ismissing(r.v_mean) || ismissing(rr.v_mean)) && continue
            push!(xs, rr.v_mean); push!(ys, r.v_mean)
            push!(xe, ismissing(rr.v_std) ? 0.0 : rr.v_std); push!(ye, ismissing(r.v_std) ? 0.0 : r.v_std)
        end
        isempty(xs) && continue
        append!(allx, xs); append!(ally, ys)
        xlo, xhi, nx = _whiskers(xs, xe, scale)
        ylo, yhi, ny = _whiskers(ys, ye, scale)
        nclamped += nx + ny
        errorbars!(ax, xs, ys, xlo, xhi; direction=:x, color=(_color(ai), 0.4), whiskerwidth=6)
        errorbars!(ax, xs, ys, ylo, yhi; direction=:y, color=(_color(ai), 0.4), whiskerwidth=6)
        scatter!(ax, xs, ys; color=_color(ai), markersize=10, label=a)
    end
    if !isempty(allx)
        lo = min(minimum(allx), minimum(ally)) * 0.7; hi = max(maximum(allx), maximum(ally)) * 1.4
        lines!(ax, [lo, hi], [lo, hi]; color=:black, linestyle=:dash)  # y = x
    end
    return nclamped
end

function analysis_scatter_vs_reference(rs::DataFrame, dir; reference::AbstractString)
    fig = Figure(size=(1180, 1000))
    legend_ax = nothing
    nclamped = 0
    cols = ((:solve_time,    "solve_time (s)",  "wall-clock"),
            (:solve_updates, "Bellman updates", "Bellman updates"))
    for (j, (col, unit, short)) in enumerate(cols)
        agg = combine(groupby(rs, [:algorithm, :problem]),
                      col => _mn => :v_mean, col => _sd => :v_std)
        refmap = Dict(r.problem => r for r in eachrow(agg[agg.algorithm .== reference, :]))
        algs = sort(filter(!=(reference), unique(agg.algorithm)))
        isempty(algs) && continue
        # row 1: linear axes, row 2: log-log
        for (i, (scale, sc)) in enumerate((("linear", identity), ("log", log10)))
            ax = Axis(fig[i, j]; xlabel="$(reference) $(unit)", ylabel="algorithm $(unit)",
                      title="$(short) — $(scale) (per-problem mean ± seed-std)",
                      xscale=sc, yscale=sc)
            nclamped += _scatter_vs_ref_panel!(ax, agg, refmap, algs, sc)
            legend_ax = ax
        end
    end
    legend_ax === nothing || _shared_legend!(fig[1:2, 3], legend_ax)
    _save(fig, dir, "scatter_vs_reference"; note=_clamp_note(nclamped))
    return fig
end

# ===========================================================================
# Analysis 5 — Solve-time vs. num_transitions scatter (+ power-law fit)
# ===========================================================================

"""
    analysis_vs_transitions(rs, dir) -> (fig, fit_time, fit_updates)

2×2 layout on one figure (`vs_transitions`): x = `n_trans`, y = solve_time / Bellman
updates, **one point per (algorithm, problem)**, colored by algorithm, Level-1 **mean**
with vertical seed-std whiskers (on the log row a whisker low end that would fall at or
below zero — seed-std > mean — is floored at 10% of the value, which the caption
records). Columns are the solve_time and Bellman-updates
variants, rows are linear and log–log axes. Each overlays a per-algorithm OLS power-law
fit `y ≈ c · n_trans^α` (on logs regardless of axis scale; censored points not fitted)
and shares a single legend placed outside the plot area. Both
`(algorithm, α, c, R², n_used)` fit tables are written (`time_vs_transitions_fit.csv`,
`updates_vs_transitions_fit.csv`) and returned.
"""
function analysis_vs_transitions(rs::DataFrame, dir)
    fig = Figure(size=(1200, 1000))
    cols = ((:solve_time,    "solve_time (s)",  "Solve time"),
            (:solve_updates, "bellman updates", "Bellman updates"))
    fits = Dict{Symbol,DataFrame}()
    legend_ax = nothing
    nclamped = 0
    # rows: linear axes then log-log (the fit is computed on logs regardless of scale)
    for (i, (scale, sc)) in enumerate((("linear", identity), ("log", log10)))
        for (j, (ycol, ylabel, short)) in enumerate(cols)
            ax = Axis(fig[i, j]; xlabel="num transitions", ylabel=ylabel,
                      title="$(short) vs transitions — $(scale) (per-problem mean ± seed-std)",
                      xscale=sc, yscale=sc)
            fit, nc = _draw_vs_transitions!(ax, rs, ycol, sc)
            fits[ycol] = fit   # identical across scales; last write wins
            nclamped += nc
            legend_ax = ax
        end
    end
    _shared_legend!(fig[1:2, 3], legend_ax)
    # `n_trans` comes from problems.csv. When that file is absent — which used to be the
    # case for every killed sweep — every point is dropped and this figure silently
    # became a set of empty axes with header-only fit CSVs, losing the scaling exponent
    # with no warning at all. Say so instead.
    n_missing = count(ismissing, rs.n_trans)
    if n_missing == nrow(rs) && nrow(rs) > 0
        @warn "vs_transitions: no num_transitions data (problems.csv missing or unreadable); no scaling fit"
        note = "NO num_transitions DATA — problems.csv is missing or unreadable, so no scaling fit could be made"
    else
        note = n_missing > 0 ? "$n_missing of $(nrow(rs)) runs have no num_transitions and are not fitted" : ""
    end
    fit_t = fits[:solve_time]; fit_u = fits[:solve_updates]
    # Carried in the CSV so a header-only fit table is self-explaining.
    fit_t.n_missing_ntrans .= n_missing
    fit_u.n_missing_ntrans .= n_missing
    # Written BEFORE the figure: rendering is the fragile step (one bad data point used to
    # throw out of `_save` and take the scaling exponents down with the image), and the fit
    # tables do not depend on it.
    mkpath(dir)
    CSV.write(joinpath(dir, "time_vs_transitions_fit.csv"), fit_t)
    CSV.write(joinpath(dir, "updates_vs_transitions_fit.csv"), fit_u)
    _save(fig, dir, "vs_transitions"; note=join(filter(!isempty, [note, _clamp_note(nclamped)]), "  ·  "))
    return (fig, fit_t, fit_u)
end

# Draw a single `y vs n_trans` scatter (+ power-law fit) onto `ax`. `scale` is the panel's
# axis scale; returns `(fit table, number of whisker low ends clamped)`.
function _draw_vs_transitions!(ax, rs::DataFrame, ycol::Symbol, scale)
    agg = combine(groupby(rs, [:algorithm, :problem]),
                  ycol => _mn => :y_mean, ycol => _sd => :y_std,
                  :n_trans => first => :n_trans)
    algs = sort(unique(agg.algorithm))
    fit = DataFrame(algorithm=String[], alpha=Float64[], alpha_se=Float64[], c=Float64[],
                    R2=Float64[], n_used=Int[], n_missing_ntrans=Int[])
    nclamped = 0
    for (ai, a) in enumerate(algs)
        sub = agg[(agg.algorithm .== a) .& .!ismissing.(agg.y_mean) .& .!ismissing.(agg.n_trans), :]
        isempty(sub) && continue
        x = Float64.(sub.n_trans); y = Float64.(sub.y_mean)
        ye = [ismissing(s) ? 0.0 : s for s in sub.y_std]
        ylo, yhi, ny = _whiskers(y, ye, scale)
        nclamped += ny
        errorbars!(ax, x, y, ylo, yhi; color=(:gray, 0.5), whiskerwidth=6)
        scatter!(ax, x, y; color=_color(ai), markersize=10, label=a)
        if length(x) >= 2 && length(unique(x)) >= 2   # OLS on logs
            lx = log.(x); ly = log.(y)
            X = hcat(ones(length(lx)), lx)
            β = X \ ly
            resid = ly .- X * β
            n = length(lx); dof = n - 2
            s2 = dof > 0 ? sum(abs2, resid) / dof : NaN
            se_alpha = dof > 0 ? sqrt(s2 * inv(X'X)[2, 2]) : NaN
            sst = sum(abs2, ly .- mean(ly)); R2 = sst > 0 ? 1 - sum(abs2, resid) / sst : NaN
            xs = sort(x)
            lines!(ax, xs, exp(β[1]) .* xs .^ β[2]; color=_color(ai), linestyle=:dash)
            push!(fit, (a, β[2], se_alpha, exp(β[1]), R2, n, 0))
        else
            push!(fit, (a, NaN, NaN, NaN, NaN, length(x), 0))
        end
    end
    return fit, nclamped
end
