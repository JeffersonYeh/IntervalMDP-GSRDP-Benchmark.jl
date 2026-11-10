# Experiments chapter, sensitivity of BRTDP to its two sampler knobs, one knob at a time around the
# selected BRTDP (T = 1, eps = 0.02): the knob's value on a log x axis, Bellman updates to eps
# relative to aRIVI (round robin k = 1) on the y axis, one line per cluster (colour and marker
# shape) and one for all problems.
#
#   output/experiments/experiments_eps_temp_sensitivity_upd_logx.{pdf,png}
#       left: T = 1, eps in {0.002, 0.006, 0.02, 0.06, 0.2, 0.4, 0.7}
#       right: eps = 0.02, T in {0.5, 0.75, 1, 1.25, 1.5, 4, 16}
#       y: geometric mean over the group's problems of (arm / aRIVI), ratio of seed means
#
# Runs: configs/thesis/sensitivity-epsilon.toml and sensitivity-temperature.toml (the 12 arms
# around the selected one), the selected arm and round robin from brtdp.toml and arivi-k1.toml.
# For eps != 0.02 the selected arm stands in on the 11 single-action problems, where eps has no
# effect (load_experiment's `fill`). Clusters: run dataset-cluster.jl first.

include(joinpath(@__DIR__, "_experiments_common.jl"))
using Printf

const RR1 = "gsrdp_round_robin_k1_driveboth"
const CENTRE = "brtdp_rw6_k1_s0max_driveboth"
const SEL = (1.0, 0.02)
const EPS_ROW = [(1.0, e) for e in [0.002, 0.006, 0.06, 0.2, 0.4, 0.7]]
const T_COL = [(t, 0.02) for t in [0.5, 0.75, 1.25, 1.5, 4.0, 16.0]]
const INK = "#0b0b0b"
# cluster → colour (_experiments_common.jl) and marker shape
const CLUSTER_MARKERS = [:circle, :rect, :utriangle, :diamond, :dtriangle, :star5, :xcross]

num(x) = x == round(x) ? string(Int(x)) : string(x)
tag(prefix, x) = prefix * replace(@sprintf("%g", x), "." => "")
cell_tag(t, e) = tag("T", t) * "_" * tag("e", e)
label_of(t, e) = (t, e) == SEL ? CENTRE : "$(CENTRE)_$(cell_tag(t, e))"

const CELLS = [EPS_ROW; T_COL]

# problems with one action per state, from the problem facts every run writes
const SINGLE_ACTION = let
    df = CSV.read(joinpath(THESIS_DIR, "problems.csv"), DataFrame)
    sort(String.(df.problem[df.num_actions .== 1]))
end
@assert length(SINGLE_ACTION) == 11

const ARMS = [(REF, "RIVI (reference)", INK, 0.0, :solid), (RR1, "round robin", INK, 1.0, :dash),
              (CENTRE, "selected", INK, 1.0, :solid),
              [(label_of(c...), "", INK, 1.0, :solid) for c in CELLS]...]
const FILL = Dict(label_of(t, e) => (CENTRE, SINGLE_ACTION) for (t, e) in EPS_ROW)

E = load_experiment(THESIS_DIR, ARMS;
                    rename=Dict("gsrdp_exhaustive_state_driveboth" => REF), fill=FILL)
@assert length(E.clusters) <= length(CLUSTER_MARKERS) <= length(CLUSTER_COLORS)

const EPS_LEVELS = [(1.0, e) for e in [0.002, 0.006, 0.02, 0.06, 0.2, 0.4, 0.7]]
const T_LEVELS = [(t, 0.02) for t in [0.5, 0.75, 1.0, 1.25, 1.5, 4.0, 16.0]]
const GROUPS = [("All problems", E.problems); E.clusters]
const C_SEL = "#e8e6df"   # band behind the selected value
const ALPHA_CL = 0.6      # opacity of the cluster lines and markers (all problems: opaque, on top)
const YTICKS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150]

vs_ref(kind, t, e, p) = cost_ratio(E, kind, label_of(t, e), p) / cost_ratio(E, kind, RR1, p)   # / aRIVI
grp(kind, t, e, ps) = gm([vs_ref(kind, t, e, p) for p in ps])

function sensitivity_figure(kind; logx=false)
    panels = [(L"action $\epsilon_A$ (state temperature $T_s = 1$)", EPS_LEVELS, 2),
              (L"state temperature $T_s$ (action $\epsilon_A = 0.02$)", T_LEVELS, 1)]
    vals = [grp(kind, c..., ps) for (_, lv, _) in panels for c in lv for (_, ps) in GROUPS]
    lo, hi = minimum(vals) / 1.2, max(maximum(vals), 1.0) * 1.2
    yt = filter(v -> lo <= v <= hi, YTICKS)
    fig = report_figure(height=250)
    for (j, (title, levels, ki)) in enumerate(panels)
        n = length(levels)
        xv = [c[ki] for c in levels]
        xs = logx ? xv : collect(1:n)
        ax = Axis(fig[1, j]; title, titlesize=8, yscale=log10, xscale=logx ? log10 : identity,
                  xticklabelsize=logx ? 6.5 : 7, yticklabelsize=7, xticks=(xs, [logx && v == 1.25 ? "" : num(v) for v in xv]),
                  yticks=(yt, [@sprintf("%g", v) for v in yt]),
                  ylabel=j == 1 ? (kind == :upd ? "updates / aRIVI" : "wall-clock / aRIVI") : "",
                  ylabelsize=8, yticklabelsvisible=j == 1, xgridvisible=false)
        i0 = findfirst(==(SEL), levels)
        logx ? vspan!(ax, [xs[i0] / 1.08], [xs[i0] * 1.08]; color=C_SEL) :
               vspan!(ax, [i0 - 0.4], [i0 + 0.4]; color=C_SEL)
        text!(ax, xs[i0], hi; text="selected", fontsize=6.5, color=C_INK2, align=(:center, :top), offset=(0, -2))
        hlines!(ax, [1.0]; color=C_CONTEXT, linestyle=:dot, linewidth=0.9)
        for k in [2:length(GROUPS); 1]   # all problems drawn last, on top
            ps = GROUPS[k][2]
            ys = [grp(kind, c..., ps) for c in levels]
            all_ = k == 1
            col = all_ ? INK : CLUSTER_COLORS[k - 1]
            lines!(ax, xs, ys; color=all_ ? col : (col, ALPHA_CL * 0.85), linewidth=all_ ? 1.6 : 0.9)
            scatter!(ax, xs, ys; color=all_ ? col : (col, ALPHA_CL), marker=all_ ? :circle : CLUSTER_MARKERS[k - 1],
                     markersize=all_ ? 6 : 6.5, strokewidth=0.5, strokecolor=all_ ? :white : (:white, ALPHA_CL))
        end
        ylims!(ax, lo, hi)
        logx ? xlims!(ax, xs[1] / 1.5, xs[end] * 1.5) : xlims!(ax, 0.5, n + 0.5)
    end
    elems = [[[LineElement(color=INK, linewidth=1.6), MarkerElement(marker=:circle, color=INK, markersize=6)]];
             [[LineElement(color=(CLUSTER_COLORS[k], ALPHA_CL * 0.85), linewidth=0.9),
               MarkerElement(marker=CLUSTER_MARKERS[k], color=(CLUSTER_COLORS[k], ALPHA_CL), markersize=6.5)]
              for k in eachindex(E.clusters)]]
    labs = first.(GROUPS)
    Legend(fig[2, 1:2], elems, labs; orientation=:horizontal, nbanks=4, framevisible=false,
           labelsize=7, patchsize=(14, 8), tellwidth=false)
    rowgap!(fig.layout, 1, 6)
    save_figure(fig, "experiments_eps_temp_sensitivity_$kind" * (logx ? "_logx" : ""), OUT)
end

sensitivity_figure(:upd; logx=true)
