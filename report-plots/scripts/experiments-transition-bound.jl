# Experiments chapter, ablation "bound of the adversary": what does the bound used for the concrete
# transition in the trajectory sampling change? BRTDP with transition_bound = "upper"
# (p = argmin_p p.U) against "lower" (p = argmin_p p.L), everything else equal.
#
#   output/experiments/experiments_tb_convergence_upd.{pdf,png}
#       per cluster: BRTDP (upper), BRTDP (lower) and round robin (k = 1); gap against tau, the
#       Bellman updates relative to the exhaustive sweep of each problem (log axis for C3, C5).
#       Clusters where upper and lower take the same updates on every problem are left out.
#
# Runs: configs/thesis/ablation-transition-bound.toml (upper), brtdp.toml (lower), rivi.toml and
# arivi-k1.toml. load_experiment is not used: it drops every problem an arm did not solve on all
# seeds, which would remove pimdp_0 and pimdp_1, where upper is censored at 7200 s on all seeds.

include(joinpath(@__DIR__, "_experiments_common.jl"))
using Printf, Statistics

const DIR = THESIS_DIR
const EX = "gsrdp_exhaustive_state_driveboth"
const RR = "gsrdp_round_robin_k1_driveboth"
const LOWER = "brtdp_rw6_k1_s0max_driveboth"
const UPPER = "brtdp_rw6_k1_s0max_driveboth_tbupper"
const C_RR, C_UPPER, C_LOWER = "#2a78d6", "#eb6834", "#1baf7a"   # slots 1-3, validated all-pairs
const INK, INK2 = "#0b0b0b", "#52514e"
const ARMS = [(REF, "exhaustive", INK, 1.2, :solid),
              (RR, "Async. RIVI", "#808080", 1.2, :dash),
              (LOWER, "BRTDP (lower)", C_LOWER, 1.3, :solid),
              (UPPER, "BRTDP (upper)", C_UPPER, 1.3, :solid)]

# --- load: like load_experiment, but censored runs stay in -------------------------------------
results = CSV.read(joinpath(DIR, "results.csv"), DataFrame)
long_all = load_long(DIR)
long_all = long_all[in.(long_all.algorithm, Ref(Set([EX, RR, LOWER, UPPER]))), :]
long_all.algorithm = [a == EX ? REF : a for a in long_all.algorithm]
const PROBLEMS = sort(unique(results.instance_name[results.alg_label .== UPPER]))
@assert length(PROBLEMS) == 37 && !("coin2_K64" in PROBLEMS)
long_all = long_all[in.(long_all.problem, Ref(Set(PROBLEMS))), :]
long_wall = long_all[long_all.seed .!= FIRST_SEED, :]
rs = Dict(:upd => BenchmarkAnalysis.run_scalars(long_all, EPS),
          :wall => BenchmarkAnalysis.run_scalars(long_wall, EPS))
l1 = Dict(k => BenchmarkAnalysis.level1(v) for (k, v) in rs)
dc = CSV.read(CLUSTER_PROBLEMS_CSV, DataFrame)
ex = CSV.read(CLUSTER_EXCLUDED_CSV, DataFrame)
clusters = [("C$c $cname", sort(dc.problem[dc.cluster .== c]))
            for (c, cname) in sort(unique(zip(dc.cluster, dc.cluster_name)))]
append!(clusters, [("C$(length(clusters) + i)* $g extremes", sort(ex.problem[ex.group .== g]))
                   for (i, g) in enumerate(unique(ex.group))])
clusters = [(g, filter(in(Set(PROBLEMS)), ps)) for (g, ps) in clusters]
filter!(c -> !isempty(last(c)), clusters)
@assert sort(reduce(vcat, last.(clusters))) == PROBLEMS "every problem belongs to one cluster"
const E = Experiment([DIR], collect(ARMS), Dict(:upd => long_all, :wall => long_wall), rs, l1,
                     PROBLEMS, ["coin2_K64"], clusters, 5, Tuple{String,String,Int}[])

# --- per (problem, seed) and per problem table ----------------------------------------------------
arm_rows(a) = results[results.alg_label .== a, [:instance_name, :seed, :status, :bellman_updates, :time_median_s, :final_gap_max]]
paired = innerjoin(arm_rows(UPPER), arm_rows(LOWER); on=[:instance_name, :seed], renamecols="_u" => "_l")
@assert nrow(paired) == 185 && all(paired.status_l .== "ok")
paired.upd_ratio = paired.bellman_updates_u ./ paired.bellman_updates_l
paired.wall_ratio = paired.time_median_s_u ./ paired.time_median_s_l
tab = combine(groupby(paired, :instance_name),
    :upd_ratio => median => :upd_ratio, :upd_ratio => minimum => :upd_min, :upd_ratio => maximum => :upd_max,
    :status_u => (s -> any(!=("ok"), s)) => :censored,
    :final_gap_max_u => maximum => :final_gap_upper,
    :bellman_updates_l => median => :updates_lower, :bellman_updates_u => median => :updates_upper,
    [:seed, :time_median_s_l] => ((s, t) -> mean(t[s .!= FIRST_SEED])) => :wall_lower,
    [:seed, :time_median_s_u] => ((s, t) -> mean(t[s .!= FIRST_SEED])) => :wall_upper)
tab.wall_ratio = tab.wall_upper ./ tab.wall_lower
rename!(tab, :instance_name => :problem)
tab.cluster = [only(g for (g, ps) in clusters if p in ps) for p in tab.problem]
row_of(p) = only(eachrow(tab)[tab.problem .== p])

# --- convergence per cluster -------------------------------------------------------------------------
function tb_panel!(ax, kind, gname, ps; xlog=false)
    ref = refmap(E, kind)
    xcol = COSTS[kind][1]
    long = E.long[kind]
    arms = E.arms[2:end]
    τends = [maximum(run_end(long[(long.algorithm .== a[1]) .& (long.problem .== p), :], xcol) / ref[p] for p in ps)
             for a in arms]
    τmax = maximum(τends)
    grid = xlog ? 10 .^ collect(range(log10(TAU_LOG_MIN), log10(1.02 * τmax); length=1500)) :
                  collect(range(0.0, 1.02 * τmax; length=1500))
    curves = [gap_curve(E, kind, ps, a[1], grid) for a in arms]
    τreach = [(k = findfirst(<=(EPS), c[2])) === nothing ? NaN : c[1][k] for c in curves]
    for ((a, label, c, lw, ls), (x, g, lo, hi)) in zip(arms, curves)   # upper last: on top where they coincide
        band!(ax, x, lo, hi; color=(c, 0.10))
        lines!(ax, x, g; color=c, linewidth=lw, linestyle=ls, label=label)
    end
    hlines!(ax, [EPS]; color=C_CONTEXT, linestyle=:dot, linewidth=0.8)
    vlines!(ax, [1.0]; color=C_CONTEXT, linestyle=:dot, linewidth=0.8)
    if any(p -> row_of(p).censored, ps)   # the runs stop here without having reached eps
        i = findfirst(==(UPPER), first.(arms))
        scatter!(ax, [curves[i][1][end]], [curves[i][2][end]]; marker=:xcross, markersize=6, color=C_UPPER)
        text!(ax, curves[i][1][end], curves[i][2][end]; text="timed out", fontsize=6.5, color=INK2,
              align=(:right, :top), offset=(-3, -4))
    end
    ylims!(ax, EPS / 10, 1.5)
    reached = filter(!isnan, τreach)
    xhi = max(1.5 * maximum(reached; init=1.0), any(isnan, τreach) ? 1.1 * τmax : 0.0)
    xlims!(ax, xlog ? TAU_LOG_MIN : 0.0, min(xhi, 1.1 * τmax))
    if xlog
        es = round(Int, log10(TAU_LOG_MIN)):floor(Int, log10(min(xhi, 1.1 * τmax)))
        ax.xticks = (10.0 .^ es, [L"10^{%$e}" for e in es])
    end
end

# Clusters whose arms run far past tau = 1 (BRTDP upper): logarithmic tau axis from TAU_LOG_MIN.
const XLOG_CLUSTERS = ("C3", "C5")
xlog_cluster(g) = any(c -> startswith(g, c * " "), XLOG_CLUSTERS)

# Clusters where upper and lower take the same updates on every problem (within 2 %) are left out.
coincide(ps) = all(p -> abs(row_of(p).upd_ratio - 1) <= 0.02, ps)
const TB_CLUSTERS = [(g, ps) for (g, ps) in E.clusters if !coincide(ps)]

function tb_convergence(kind)
    ncol = 2
    nrow_ = cld(length(TB_CLUSTERS), ncol)
    fig = report_figure(height=135 * nrow_ + 45)
    local ax
    for (q, (gname, ps)) in enumerate(TB_CLUSTERS)
        r, c = fldmod1(q, ncol)
        n = length(ps)
        xlog = xlog_cluster(gname)
        ax = Axis(fig[r, c]; yscale=log10, xscale=xlog ? log10 : identity, title="$gname ($n problem$(n == 1 ? "" : "s"))",
                  titlesize=8, titlealign=:left, ylabel=c == 1 ? GAP_LABEL : "",
                  yticks=YTICKS, xticklabelsize=7, yticklabelsize=7)
        tb_panel!(ax, kind, gname, ps; xlog)
        c == 1 || hideydecorations!(ax; grid=false, ticks=false)
    end
    elems = [LineElement(color=a[3], linewidth=a[4], linestyle=a[5]) for a in E.arms[2:end]]
    Label(fig[nrow_ + 1, 1:ncol], COSTS[kind][5]; fontsize=9, tellwidth=false)
    Legend(fig[nrow_ + 2, 1:ncol], reverse(elems), reverse(getindex.(E.arms[2:end], 2)); framevisible=false,
           labelsize=8, patchsize=(18, 8), orientation=:horizontal, tellwidth=false, tellheight=true)
    rowgap!(fig.layout, nrow_, 4)
    rowgap!(fig.layout, nrow_ + 1, 2)
    save_figure(fig, "experiments_tb_convergence_$kind", OUT)
end
tb_convergence(:upd)
