# Benchmark chapter: the model metrics of the 38 problems as box plots, one panel per metric,
# problems sorted per panel.
#
#   output/benchmark/dataset_box_size_metrics.{pdf,png}
#       size and property sets: non-terminal states, actions, non-zero transitions, reach and
#       avoid share; one dot per problem
#   output/benchmark/dataset_box_stoch_metrics.{pdf,png}
#       stochasticity of the transitions: effective successors N_eff (dot: mean), largest lower
#       bound, interval width
#   output/benchmark/dataset_box_horizon_metrics.{pdf,png}
#       horizon and spectrum: depth to G ∪ O (dot: maximum), reach probability from S0,
#       guaranteed exit probability, eigenvalue magnitudes of the midpoint chain (dot: 90th
#       percentile)
#
# Box: lower to upper quartile over the problem's (s, a) pairs, states or initial states; line:
# median; whiskers: minimum and maximum; dot: the value the clustering (dataset-cluster.jl) uses
# where it is not the median. Log panels put zeros on a separate 0 row below a dashed break.
# Inputs: results/mixing/ (tools/mixing_stats.jl, s0_horizon.jl, stochasticity_stats.jl,
# spectral_density.jl; see configs/thesis/README.md) and the problems.csv of THESIS_DIR.

include(joinpath(@__DIR__, "_dataset_common.jl"))

const OUT = "benchmark"   # output/benchmark/

# Source colours: categorical slots 1–6 in fixed order.
const CATEGORY_COLOR = Dict(zip(CATEGORY_NAMES,
                                ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))

# --- inputs -----------------------------------------------------------------------------------
need = [joinpath(MIXING_DIR, "mixing_stats.csv"); S0_CSV; joinpath(STOCH_DIR, "summary.csv");
        [joinpath(d, p * ".csv") for d in (DISTS_DIR, STOCH_DIR, SPECTRAL_DIR) for p in problems.problem]]
missing_inputs = filter(!isfile, need)
isempty(missing_inputs) || error("dataset-metrics.jl: $(length(missing_inputs)) mixing files missing, e.g. " *
                                 "$(first(missing_inputs)) (run the tools/ metric scripts, configs/thesis/README.md)")
mrow = CSV.read(joinpath(MIXING_DIR, "mixing_stats.csv"), DataFrame)
s0r = CSV.read(S0_CSV, DataFrame)
ssum = CSV.read(joinpath(STOCH_DIR, "summary.csv"), DataFrame)
dists = Dict(p => CSV.read(joinpath(DISTS_DIR, p * ".csv"), DataFrame) for p in problems.problem)
stoch = Dict(p => CSV.read(joinpath(STOCH_DIR, p * ".csv"), DataFrame) for p in problems.problem)
spectral = Dict(p => CSV.read(joinpath(SPECTRAL_DIR, p * ".csv"), DataFrame) for p in problems.problem)

# --- helpers ----------------------------------------------------------------------------------
# Log-axis ticks: decades over ≥ 2 decades (every 2nd/3rd/5th when there are more than 7, so
# labels do not collide), 1-2-5 steps over 1–2 decades, every integer multiple below one decade.
function log_ticks(lo, hi)
    ks = floor(Int, log10(lo)):ceil(Int, log10(hi))
    step = something(findfirst(st -> cld(length(ks), st) <= 7, [1, 2, 3, 5, 10]), 5)
    step = [1, 2, 3, 5, 10][step]
    ks = filter(k -> k % step == 0, ks)
    span = log10(hi / lo)
    vs = span >= 2 ? 10.0 .^ ks :
         [m * 10.0^k for k in ks for m in (span >= 1 ? (1, 2, 5) : 1:9)]
    # below one decade, keep whole numbers only when there are some (0.6, 0.7, … add nothing)
    span < 1 && count(v -> lo <= v <= hi && isinteger(v), vs) >= 2 && filter!(isinteger, vs)
    return filter(v -> lo <= v <= hi, vs)
end
function logfmt(vs)
    map(vs) do v
        1e-3 <= v < 1e4 && return @sprintf("%.4g", v)
        k = floor(Int, log10(v) + 1e-9); m = round(Int, v / 10.0^k)
        m == 1 ? latexstring("10^{$k}") : latexstring("$m \\times 10^{$k}")
    end
end
# At most three log ticks, decades only over more than two decades, labelled 10^k outside 0.1–100.
function rows_ticks(lo, hi)
    log10(hi / lo) <= 2 && return (tk = log_ticks(lo, hi); (tk, logfmt(tk)))
    ks = ceil(Int, log10(lo) - 1e-9):floor(Int, log10(hi) + 1e-9)
    st = cld(length(ks) - 1, 2)
    ks = filter(k -> (k - last(ks)) % st == 0, ks)
    return 10.0 .^ ks, [-1 <= k <= 2 ? @sprintf("%g", 10.0^k) : latexstring("10^{$k}") for k in ks]
end

# Five-number summary of problem p's `col` read off its stored quantiles (q = 0, 0.25, 0.5,
# 0.75, 1) in `src`.
function box_of(p, col; src)
    qd = src[p]
    ok = .!ismissing.(qd[!, col])
    any(ok) || error("dataset-metrics.jl: no $col data for $p")
    q, v = Float64.(qd.q[ok]), Float64.(qd[ok, col])
    at(level) = v[argmin(abs.(q .- level))]
    return (min=at(0), q1=at(0.25), med=at(0.5), q3=at(0.75), max=at(1))
end
qat(t, col, level) = Float64(t[argmin(abs.(t.q .- level)), col])

# A panel's entry per problem is (min, q1, med, q3, max, dot): box = quartiles, line = median,
# whiskers = min–max, dot = the value the clustering uses where that is not the median; a
# scalar metric is all dot.
five(v) = (minimum(v), quantile(v, 0.25), median(v), quantile(v, 0.75), maximum(v))
dist_box(src, col; dot=nothing) = Dict(map(problems.problem) do p
    s = box_of(p, col; src)
    p => (s.min, s.q1, s.med, s.q3, s.max, dot === nothing ? nothing : dot[p])
end)
s0_box(col) = Dict(map(problems.problem) do p
    v = Float64.(collect(skipmissing(s0r[s0r.problem .== p, col])))
    p => (five(v)..., length(v) > 1 ? mean(v) : nothing)
end)
scalar_box(col) = Dict(p => (v = Float64(mrow[findfirst(==(p), mrow.problem), col]); (ntuple(_ -> v, 5)..., v))
                       for p in problems.problem)

# One panel per metric, stacked. Each panel sorts the problems by increasing median (ties: the
# dot, then the maximum, then the name), or with `bydot` by the dot, and labels them with their
# short names.
function metric_box_figure(panels; height)
    n = nrow(problems)
    xs = collect(1:n)
    fig = report_figure(width=1.0, height=height)
    for (j, pn) in enumerate(panels)
        order = sort(problems.problem;
                     by=p -> (t = pn.val[p]; get(pn, :bydot, false) ?
                              (something(t[6], t[3]), t[3], t[5], p) :
                              (t[3], something(t[6], t[3]), t[5], p)))
        cols = [to_color(CATEGORY_COLOR[CATEGORY_OF[p]]) for p in order]
        st = [pn.val[p] for p in order]
        allv = [x for t in st for x in t[1:5]]
        if pn.log
            lo, hi = extrema(filter(>(0), allv))
            hi / lo < 1.5 && ((lo, hi) = (lo / 2, hi * 2))
            span = max(log10(hi / lo), 1.0)
            haszero = any(<=(0), allv)
            z0 = lo * 10^(-0.14span)
            y = v -> v > 0 ? v : z0
            tk, tl = rows_ticks(lo, hi)
            yticks = haszero ? ([z0; tk], AbstractString["0"; tl]) : (tk, tl)
            ylims = ((haszero ? z0 : lo) * 10^(-0.06span), hi * 10^(0.06span))
        else
            haszero, y = false, identity
            yticks = ([0, 0.5, 1], ["0", "0.5", "1"])
            ylims = (-0.06, 1.06)
        end
        ax = Axis(fig[j, 1]; ylabel=pn.sym, title=replace(pn.name, '\n' => ' '),
                  titlesize=8, titlefont=:regular, titlegap=2, titlealign=:left,
                  yscale=pn.log ? log10 : identity, yticks, yticklabelsize=7, ylabelsize=8,
                  xticks=(xs, short_name.(order)), xticklabelrotation=π / 2,
                  xticklabelsize=6, xgridvisible=false, limits=((0.4, n + 0.6), ylims))
        haszero && hlines!(ax, z0 * 10^(0.07span); color=C_CONTEXT, linestyle=:dash, linewidth=0.6)
        isbox = [t[1] < t[5] || t[2] < t[4] for t in st]
        if any(isbox)
            rangebars!(ax, xs[isbox], [y(t[1]) for t in st[isbox]], [y(t[5]) for t in st[isbox]];
                       color=cols[isbox], linewidth=0.8, whiskerwidth=3)
            wide = isbox .& [t[2] < t[4] for t in st]
            any(wide) && barplot!(ax, xs[wide], [y(t[4]) for t in st[wide]];
                                  fillto=[y(t[2]) for t in st[wide]], color=cols[wide],
                                  width=0.7, gap=0, strokewidth=0)
        end
        # median: black over a box, in the source colour where the quartiles coincide
        flat = [t[2] == t[4] for t in st]
        for (sel, color, lw) in ((.!flat, fill(to_color(INK), n), 1.1), (flat, cols, 2.2))
            sel = sel .& [t[6] === nothing || isbox[i] for (i, t) in enumerate(st)]
            any(sel) || continue
            linesegments!(ax, [Point2f(x + d, y(t[3])) for (x, t) in zip(xs[sel], st[sel])
                               for d in (-0.35, 0.35)];
                          color=repeat(color[sel]; inner=2), linewidth=lw)
        end
        dots = [t[6] !== nothing for t in st]
        any(dots) && scatter!(ax, xs[dots], [y(t[6]) for t in st[dots]]; color=cols[dots],
                              markersize=4.5, strokecolor=INK, strokewidth=0.5)
    end
    cats = [c for c in CATEGORY_NAMES if c in values(CATEGORY_OF)]
    Legend(fig[0, 1], [PolyElement(color=CATEGORY_COLOR[c]) for c in cats], cats;
           orientation=:horizontal, tellheight=true, tellwidth=false, colgap=7,
           labelsize=7, patchsize=(8, 6), halign=:right, padding=(0, 0, 0, 0))
    rowgap!(fig.layout, 4)
    return fig
end

# --- the three figures ------------------------------------------------------------------------
size_panels = [
    (name="non-terminal states", sym=L"|S_{\mathrm{nt}}|", val=scalar_box(:num_maybe), log=true),
    (name=" actions", sym=L"|A|", val=scalar_box(:num_actions), log=true),   # leading space as in the thesis figure
    (name="non-zero transitions", sym=L"|T|", val=scalar_box(:num_transitions), log=true),
    (name="reach share", sym=L"|G|\,/\,|S|", val=scalar_box(:reach_frac), log=true),
    (name="avoid share", sym=L"|O|\,/\,|S|", val=scalar_box(:avoid_frac), log=true)]
save_figure(metric_box_figure(size_panels; height=520), "dataset_box_size_metrics", OUT)

neff_mean = Dict(zip(ssum.problem, Float64.(ssum.neff_mean)))
stoch_panels = [
    (name="effective successors (dot: mean)", sym=L"N_{\mathrm{eff}}",
     val=dist_box(stoch, :neff; dot=neff_mean), log=true, bydot=true),
    (name="largest lower bound", sym=L"\underline{p}_{\max}", val=dist_box(stoch, :lmax), log=true),
    (name="interval width", sym=L"\overline{p} - \underline{p}", val=dist_box(dists, :width), log=true)]
save_figure(metric_box_figure(stoch_panels; height=400), "dataset_box_stoch_metrics", OUT)

horizon_panels = [
    (name="depth to G ∪ O (dot: maximum)", sym=L"d(s)",
     val=dist_box(dists, :depth; dot=Dict(p => qat(dists[p], :depth, 1.0) for p in problems.problem)),
     log=true, bydot=true),
    (name="reach probability", sym=L"h_{\mathrm{reach}}(S_0)", val=s0_box(:h_reach), log=true, bydot=true),
    (name="guaranteed exit probability", sym=L"e(s)", val=dist_box(dists, :exit_best), log=true),
    (name="eigenvalue magnitude (dot: 90th percentile)", sym=L"|\lambda|",
     val=dist_box(spectral, :lambda_abs; dot=Dict(p => qat(spectral[p], :lambda_abs, 0.9)
                                                  for p in problems.problem)),
     log=true, bydot=true)]
save_figure(metric_box_figure(horizon_panels; height=490), "dataset_box_horizon_metrics", OUT)
