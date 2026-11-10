# Shared by the experiments-*.jl scripts (not run on its own: `_`-prefixed scripts are skipped by
# make_all.jl): load the runs of a set of arms, split them by the dataset clusters, and draw the
# convergence figures of the Experiments chapter into output/experiments/.
#
#   E = load_experiment(THESIS_DIR, ARMS)   ARMS: (label, legend, colour, width, style), the
#                                           reference first
#   pair_figure(E, name, ps; ...)           wall-clock | updates for the problems `ps`
#   cluster_grid_figure(E, kind, name; ...) one panel per cluster, the legend in the spare cell
#   problem_curves_figure(E, g, ps, name)   one panel per problem of a cluster
#   speedup_histogram(E, arm, refs, name)   speed-up to ε per problem, stacked by cluster
#
# Curves: per problem the seed-mean initial-state gap against τ = cost / (the reference's mean
# cost to ε), then the geometric mean (±1 geometric std) over the group's problems. Clusters come
# from the dataset clustering in cache/dataset_cluster_{problems,excluded}.csv (run
# scripts/dataset-cluster.jl first): C1–C5, then the extremes set aside before the clustering as
# C6*, C7*.

include(joinpath(@__DIR__, "..", "common.jl"))
using CSV

const EPS = 1e-5
const REF = "gsrdp_exhaustive_state"
const FIRST_SEED = 20260922      # compiled in the first sample of runs without warm-up

const OUT = "experiments"         # output/experiments/
isfile(CLUSTER_PROBLEMS_CSV) || error("run scripts/dataset-cluster.jl first (cache/dataset_cluster_problems.csv missing)")

# kind => (x column, level-1 cost column, per-seed cost column, title, x label)
const COSTS = Dict(:wall => (:wall_time, :time_mean, :solve_time, "wall-clock (8 threads)",
                             L"\tau_t = t \,/\, T_{\mathrm{ref}}"),
                   :upd => (:bellman_updates, :upd_mean, :solve_updates, "Bellman updates",
                            L"\tau_u = u \,/\, U_{\mathrm{ref}}"))

struct Experiment
    dirs::Vector{String}
    arms::Vector{Tuple}
    long::Dict{Symbol,DataFrame}        # :upd all seeds, :wall without FIRST_SEED
    rs::Dict{Symbol,DataFrame}
    l1::Dict{Symbol,DataFrame}
    problems::Vector{String}            # solved on every seed by every arm
    excluded::Vector{String}
    clusters::Vector{Tuple{String,Vector{String}}}
    nseed::Int
    short_seeds::Vector{Tuple{String,String,Int}}   # (arm, problem, seeds) with fewer than nseed
end

algs(E) = first.(E.arms)
style(E, a) = only(filter(x -> x[1] == a, E.arms))

load_experiment(name::AbstractString, arms; kw...) = load_experiment([name], arms; kw...)

# `rename`: data label => arm label, e.g. "gsrdp_exhaustive_state_driveboth" => REF to make another
# arm the reference (the data's own REF runs are then left out).
# `fill`: arm label => (source arm label, problems): the source arm's runs on those problems stand in
# for the arm, which must not have run them (e.g. a knob that does nothing on those problems).
function load_experiment(names::AbstractVector, arms; rename=Dict{String,String}(),
                         fill=Dict{String,Tuple{String,Vector{String}}}())
    dirs = results_dir.(names)
    as = first.(arms)
    @assert first(as) == REF "the reference arm comes first"
    data_label = Dict(v => k for (k, v) in rename)
    src = Set(get(data_label, a, a) for a in as)
    parts = map(dirs) do dir
        long = load_long(dir)
        long = long[in.(long.algorithm, Ref(src)), :]
        long.algorithm = [get(rename, a, a) for a in long.algorithm]
        long
    end
    # a run measured in two sources would be pooled into one (algorithm, problem, seed) group
    # (combine_results.jl's --dedupe-seeds note), so the sources must not overlap
    runs = [Set(zip(l.algorithm, l.problem, l.seed)) for l in parts]
    for i in eachindex(runs), j in i+1:lastindex(runs)
        both = intersect(runs[i], runs[j])
        isempty(both) || error("load_experiment: $(basename(dirs[i])) and $(basename(dirs[j])) " *
                               "share runs, e.g. $(first(both))")
    end
    long_all = reduce(vcat, parts)
    for (a, (from, ps)) in fill
        @assert a in as && from in as "fill: $a and $from must both be arms"
        @assert !any((long_all.algorithm .== a) .& in.(long_all.problem, Ref(Set(ps)))) "fill: $a ran some of its fill problems"
        stand_in = long_all[(long_all.algorithm .== from) .& in.(long_all.problem, Ref(Set(ps))), :]
        stand_in.algorithm .= a
        long_all = vcat(long_all, stand_in)
    end
    l1_all =BenchmarkAnalysis.level1(BenchmarkAnalysis.run_scalars(long_all, EPS))
    solved = Set((r.algorithm, r.problem) for r in eachrow(l1_all) if r.n_solved == r.n_seeds)
    all_problems = sort(unique(long_all.problem))
    problems = [p for p in all_problems if all(a -> (a, p) in solved, as)]
    excluded = setdiff(all_problems, problems)
    # problems no arm has run at all (e.g. a model left out of a new sweep) are excluded too
    dc = CSV.read(CLUSTER_PROBLEMS_CSV, DataFrame)
    ex = CSV.read(CLUSTER_EXCLUDED_CSV, DataFrame)
    clusters = [("C$c $cname", sort(dc.problem[dc.cluster .== c]))
                for (c, cname) in sort(unique(zip(dc.cluster, dc.cluster_name)))]
    append!(clusters, [("C$(length(clusters) + i)* $g extremes", sort(ex.problem[ex.group .== g]))
                       for (i, g) in enumerate(unique(ex.group))])
    @assert issubset(all_problems, reduce(vcat, last.(clusters))) "a problem has no cluster"
    excluded = sort(union(excluded, setdiff(reduce(vcat, last.(clusters)), all_problems)))
    clusters = [(g, filter(in(Set(problems)), ps)) for (g, ps) in clusters]
    filter!(c -> !isempty(last(c)), clusters)

    long_upd = long_all[in.(long_all.problem, Ref(Set(problems))), :]
    long_wall = long_upd[long_upd.seed .!= FIRST_SEED, :]
    rs = Dict(:upd => BenchmarkAnalysis.run_scalars(long_upd, EPS),
              :wall => BenchmarkAnalysis.run_scalars(long_wall, EPS))
    l1 = Dict(k => BenchmarkAnalysis.level1(v) for (k, v) in rs)
    nseed = maximum(l1[:upd].n_seeds)
    short = [(r.algorithm, r.problem, r.n_seeds) for r in eachrow(l1[:upd]) if r.n_seeds < nseed]
    println("  $(length(problems)) problems in $(length(clusters)) groups; excluded: ",
            isempty(excluded) ? "none" : join(excluded, ", "))
    return Experiment(dirs, collect(arms), Dict(:upd => long_upd, :wall => long_wall), rs, l1,
                      problems, excluded, clusters, nseed, short)
end

refmap(E, kind) = Dict(r.problem => r[COSTS[kind][2]] for r in eachrow(E.l1[kind]) if r.algorithm == REF)
function cost_ratio(E, kind, a, p)
    l1 = E.l1[kind]
    return only(l1[(l1.algorithm .== a) .& (l1.problem .== p), COSTS[kind][2]]) / refmap(E, kind)[p]
end
group_of(E, p) = only(g for (g, ps) in E.clusters if p in ps)

# The curves stop where the runs ended: the grid cut at τend, with τend as the last point.
run_end(tr, xcol) = maximum(maximum(g[!, xcol]) for g in groupby(tr, :seed))
upto(grid, τend) = [filter(<(τend), grid); τend]

# Geometric mean over `ps` of the seed-mean gap curve against τ = x / X_ref(p), up to where the
# last run of any problem ended. Returns (x, geomean, lo, hi).
function gap_curve(E, kind, ps, alg, grid)
    xcol = COSTS[kind][1]
    long = E.long[kind]
    ref = refmap(E, kind)
    trs = map(ps) do p
        tr = long[(long.algorithm .== alg) .& (long.problem .== p), :]
        tr[!, xcol] = Float64.(tr[!, xcol]) ./ ref[p]
        tr
    end
    x = upto(grid, maximum(run_end(tr, xcol) for tr in trs))
    curves = [first(BenchmarkAnalysis.seed_mean_curve(tr, alg, p, x; xcol)) for (tr, p) in zip(trs, ps)]
    return (x, BenchmarkAnalysis._geo_stats(reduce(hcat, curves))...)
end

gm(v) = exp(mean(log.(v)))

# One panel: every arm's geometric-mean curve for problems `ps`, with its own τ range (to 1.25×
# the latest arm's τ at ε, or the end of the runs; from TAU_LOG_MIN on a log axis, `xlog`).
const TAU_LOG_MIN = 1e-2
function convergence_panel!(ax, E, kind, gname, ps; xlog=false)
    rs = E.rs[kind][in.(E.rs[kind].problem, Ref(Set(ps))), :]
    ref = refmap(E, kind)
    τmax = maximum(r[COSTS[kind][3]] / ref[r.problem] for r in eachrow(rs))
    grid = xlog ? 10 .^ collect(range(log10(TAU_LOG_MIN), log10(1.05 * τmax); length=3000)) :
                  collect(range(0.0, 1.05 * τmax; length=3000))
    curves = Dict(a => gap_curve(E, kind, ps, a, grid) for a in algs(E))
    τreach(a) = (k = findfirst(<=(EPS), curves[a][2])) === nothing ? NaN : curves[a][1][k]
    for (a, label, c, lw, ls) in reverse(E.arms)   # reference drawn last, on top
        lw > 0 || continue                         # width 0: sets the τ scale, not drawn
        x, g, lo, hi = curves[a]
        band!(ax, x, lo, hi; color=(c, 0.08))
        lines!(ax, x, g; color=c, linewidth=lw, linestyle=ls, label=label)
    end
    hlines!(ax, [EPS]; color=C_CONTEXT, linestyle=:dot, linewidth=0.8)
    vlines!(ax, [1.0]; color=C_CONTEXT, linestyle=:dot, linewidth=0.8)
    ylims!(ax, EPS / 10, 1.5)
    reached = filter(!isnan, τreach.(algs(E)))
    xhi = isempty(reached) ? τmax : min((xlog ? 1.5 : 1.25) * maximum(reached), 1.05 * τmax)
    xlims!(ax, xlog ? TAU_LOG_MIN : 0, xhi)
    if xlog   # whole decades only (Makie labels short ranges with 10^-1.5 otherwise)
        es = round(Int, log10(TAU_LOG_MIN)):floor(Int, log10(xhi))
        ax.xticks = (10.0 .^ es, [L"10^{%$e}" for e in es])
    end
    return
end

const YTICKS = (10.0 .^ (-6:2:0), [L"10^{%$e}" for e in -6:2:0])

"""
    pair_figure(E, name, ps; title="", logwall=false)

Wall-clock | updates for the problems `ps`, the legend on top (and `title` above it, if given);
`logwall` puts the wall-clock τ on a log axis.
"""
function pair_figure(E, name, ps; title="", gname=title, logwall=false)
    fig = report_figure(width=1.0, aspect=2.3)
    axs = map(enumerate((:wall, :upd))) do (j, kind)
        _, _, _, ptitle, xlabel = COSTS[kind]
        xlog = logwall && kind == :wall
        ax = Axis(fig[2, j]; yscale=log10, xscale=xlog ? log10 : identity, title=ptitle, xlabel,
                  ylabel=j == 1 ? GAP_LABEL : "")
        convergence_panel!(ax, E, kind, isempty(gname) ? "all" : gname, ps; xlog)
        ax
    end
    plots, labels = Makie.get_labeled_plots(axs[end]; merge=false, unique=false)
    Legend(fig[1, 1:2], reverse(plots), reverse(labels); orientation=:horizontal,
           patchsize=(18, 8), framevisible=false, labelsize=8, tellwidth=false)
    rowgap!(fig.layout, 1, 3)
    if !isempty(title)
        Label(fig[0, 1:2], title; fontsize=9, font=:bold, halign=:left, tellwidth=false)
        rowgap!(fig.layout, 1, 2)
    end
    save_figure(fig, name, OUT)
end

"""
    cluster_grid_figure(E, kind, name; xlog=false)

One panel per cluster (2 columns, own τ range each) for cost `kind`, the legend in the spare cell.
"""
# `clusters` picks the panels (default: every cluster of E).
function cluster_grid_figure(E, kind, name; ncol=2, xlog=false, clusters=E.clusters)
    _, _, _, _, xlabel = COSTS[kind]
    # The legend takes the free cell after the last panel; when the panels fill every row it
    # goes below them as one horizontal strip instead of a row of its own.
    strip = length(clusters) % ncol == 0
    nrow_ = strip ? length(clusters) ÷ ncol : cld(length(clusters) + 1, ncol)
    fig = report_figure(height=135 * nrow_ + (strip ? 45 : 25))
    local ax
    for (q, (gname, ps)) in enumerate(clusters)
        r, c = fldmod1(q, ncol)
        n = length(ps)
        ax = Axis(fig[r, c]; yscale=log10, title="$gname ($n problem$(n == 1 ? "" : "s"))",
                  titlesize=8, titlealign=:left, ylabel=c == 1 ? GAP_LABEL : "",
                  yticks=YTICKS, xticklabelsize=7, yticklabelsize=7, xscale=xlog ? log10 : identity)
        convergence_panel!(ax, E, kind, gname, ps; xlog)
        c == 1 || hideydecorations!(ax; grid=false, ticks=false)
    end
    plots, labels = Makie.get_labeled_plots(ax; merge=false, unique=false)
    if strip
        Label(fig[nrow_ + 1, 1:ncol], xlabel; fontsize=9, tellwidth=false)
        Legend(fig[nrow_ + 2, 1:ncol], reverse(plots), reverse(labels); framevisible=false, labelsize=8,
               patchsize=(18, 8), orientation=:horizontal, tellwidth=false, tellheight=true)
        rowgap!(fig.layout, nrow_ + 1, 2)
    else
        r, c = fldmod1(length(clusters) + 1, ncol)
        Legend(fig[r, c], reverse(plots), reverse(labels); framevisible=false, labelsize=8,
               patchsize=(18, 8), tellwidth=false, tellheight=false)
        Label(fig[nrow_ + 1, 1:ncol], xlabel; fontsize=9, tellwidth=false)
    end
    rowgap!(fig.layout, nrow_, 4)
    save_figure(fig, name, OUT)
end

# File-name tag of a cluster: "C1 long near-deterministic chains" → "c1", "C6* …" → "c6".
cluster_tag(gname) = lowercase(replace(first(split(gname)), "*" => ""))

const SHORTEN = ["linear_stochastically_switched" => "lss", "almost_identity" => "almost_id",
                 "package_delivery" => "pkg_delivery", "impact_robot_2d_ra" => "impact_robot"]
short_name(p) = replace(p, SHORTEN...)

# Seed curves of one (arm, problem) against the raw Bellman updates, up to where the arm's last
# seed ended: (x, seed mean, min over seeds, max over seeds, number of seeds).
function seed_curves(E, a, p, grid)
    tr = E.long[:upd][(E.long[:upd].algorithm .== a) .& (E.long[:upd].problem .== p), :]
    x = upto(grid, run_end(tr, :bellman_updates))
    M = reduce(hcat, [BenchmarkAnalysis.interp_gap(g, x; xcol=:bellman_updates) for g in groupby(tr, :seed)])
    return x, vec(mean(M, dims=2)), vec(minimum(M, dims=2)), vec(maximum(M, dims=2)), size(M, 2)
end

"""
    problem_curves_figure(E, gname, ps, name; ncol=3)

One panel per problem of a cluster: every arm's seed-mean gap against the actual Bellman updates
(not normalised; each panel its own range, to the end of the slowest run), shaded from the
minimum to the maximum over the seeds.
"""
function problem_curves_figure(E, gname, ps, name; ncol=3)
    nrow_ = cld(length(ps), ncol)
    fig = report_figure(height=128 * nrow_ + 45)
    local ax
    for (q, p) in enumerate(ps)
        r, c = fldmod1(q, ncol)
        grid0 = collect(range(0.0, 1.0; length=2500))
        ends = Dict(a => run_end(E.long[:upd][(E.long[:upd].algorithm .== a) .& (E.long[:upd].problem .== p), :],
                                 :bellman_updates) for a in algs(E))
        xmax = maximum(values(ends))
        e = floor(Int, log10(xmax))
        ax = Axis(fig[r + 1, c]; yscale=log10, title=short_name(p), titlesize=8, titlealign=:left,
                  xlabel=L"updates ($\times 10^{%$e}$)", xlabelsize=7, xlabelpadding=1,
                  ylabel=c == 1 ? GAP_LABEL : "", yticks=YTICKS,
                  xticklabelsize=7, yticklabelsize=7)
        for (a, label, col, lw, ls) in reverse(E.arms)   # reference drawn last, on top
            x, m, lo, hi, n = seed_curves(E, a, p, grid0 .* xmax)
            fl(v) = max.(v, EPS / 20)                    # a gap of exactly 0 at the bottom
            n > 1 && any(hi .> lo) && band!(ax, x ./ 10.0^e, fl(lo), fl(hi); color=(col, 0.18))
            lines!(ax, x ./ 10.0^e, fl(m); color=col, linewidth=lw, linestyle=ls, label=label)
        end
        hlines!(ax, [EPS]; color=C_CONTEXT, linestyle=:dot, linewidth=0.6)
        ylims!(ax, EPS / 10, 1.5); xlims!(ax, 0, 1.03 * xmax / 10.0^e)
        c == 1 || hideydecorations!(ax; grid=false, ticks=false)
    end
    plots, labels = Makie.get_labeled_plots(ax; merge=false, unique=false)
    Legend(fig[1, 1:ncol], reverse(plots), reverse(labels); orientation=:horizontal,
           nbanks=length(E.arms) > 3 ? 2 : 1,   # four or more entries overflow one row
           patchsize=(18, 8), framevisible=false, labelsize=8, tellwidth=false)
    rowgap!(fig.layout, 1, 2)
    save_figure(fig, name, OUT)
end

# Cluster colours of dataset-cluster.jl (C1–C5); the hand-set-aside extremes C6*, C7* in greys.
const CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4d4d4d", "#a8a8a8"]

"""
    speedup_histogram(E, arm, refs, name; kind=:upd, binw=0.125)

One panel per reference arm in `refs`: histogram over the problems of the speed-up
cost(ref) / cost(arm) to ε (`kind` :upd or :wall), on a log axis in bins of `binw` decades
(1/40 when the speed-ups span less than a decade), stacked by cluster. Dotted: no speed-up; dashed: geometric mean. Returns the speed-ups.
"""
function speedup_histogram(E, arm, refs, name; kind=:upd, binw=0.125)
    sp = Dict(r => Dict(p => cost_ratio(E, kind, r, p) / cost_ratio(E, kind, arm, p) for p in E.problems)
              for r in refs)
    lx = [log10(v) for d in values(sp) for v in values(d)]
    # a narrow range (under a decade) gets finer bins
    maximum(lx) - minimum(lx) < 1 && (binw = 1 / 40)
    lo = binw * (floor(minimum(lx) / binw) - 1); hi = binw * (ceil(maximum(lx) / binw) + 1)   # one spare bin each side
    edges = collect(lo:binw:hi + 1e-9)
    mids = edges[1:end-1] .+ binw / 2
    fig = report_figure(width=1.0, aspect=2.4)
    cand = hi - lo < 1.2 ? [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5] : [0.5, 1, 2, 5, 10, 20, 50, 100]
    tickv = filter(v -> lo - 1e-9 <= log10(v) <= hi + 1e-9, cand)
    local ax
    for (j, r) in enumerate(refs)
        ax = Axis(fig[1, j]; title="$(plain_title(arm)) vs $(plain_title(r))", titlesize=9,
                  xlabel="speed-up in " * (kind == :upd ? "Bellman updates" : "wall-clock time") *
                         " (" * plain_title(r) * " ÷ " * plain_title(arm) * ")",
                  xlabelsize=8, ylabel=j == 1 ? "problems" : "",
                  xticks=(log10.(tickv), [string(v isa Integer || v == round(v) ? Int(v) : v) * "×" for v in tickv]),
                  xticklabelsize=7, yticklabelsize=7)
        base = zeros(length(mids))
        for (ci, (gname, ps)) in enumerate(E.clusters)
            counts = [count(p -> edges[b] <= log10(sp[r][p]) < edges[b+1] + (b == length(mids) ? 1e-9 : 0), ps)
                      for b in 1:length(mids)]
            any(>(0), counts) || continue
            barplot!(ax, mids, counts; offset=base, width=binw * 0.92, color=CLUSTER_COLORS[ci],
                     label=gname)
            base .+= counts
        end
        g = gm(collect(values(sp[r])))
        vlines!(ax, [0.0]; color=C_CONTEXT, linestyle=:dot, linewidth=0.9)
        vlines!(ax, [log10(g)]; color="#0b0b0b", linestyle=:dash, linewidth=1.0)
        text!(ax, log10(g), 1.2 * maximum(base); text=@sprintf(" geometric mean %.1f×", g), fontsize=7,
              align=(:left, :top), color="#0b0b0b")
        xlims!(ax, lo, hi); ylims!(ax, 0, 1.22 * maximum(base))
        j == 1 || hideydecorations!(ax; grid=false, ticks=false, ticklabels=false)
    end
    linkyaxes!(contents(fig[1, :])...)
    elems = [PolyElement(color=CLUSTER_COLORS[ci]) for ci in 1:length(E.clusters)]
    Legend(fig[2, 1:length(refs)], elems, first.(E.clusters); orientation=:horizontal, nbanks=3,
           framevisible=false, labelsize=7, patchsize=(8, 8), tellwidth=false)
    rowgap!(fig.layout, 1, 4)
    save_figure(fig, name, OUT)
    return sp
end
# an arm's short name for figure text
const PLAIN_TITLES = Dict(REF => "RIVI", "gsrdp_round_robin_k1" => "Async. RIVI", "brtdp_rw6_k1" => "BRTDP",
                          "brtdp_rw6_k1_s0max" => "BRTDP (max-gap s₀)",
                          "gsrdp_round_robin_k1_driveboth" => "Async. RIVI",
                          "brtdp_rw6_k1_s0max_driveboth" => "BRTDP")
plain_title(a) = get(PLAIN_TITLES, a, a)
