# Experiments chapter, BRTDP appendix: the car_parking paragraph (slow mixing towards an almost
# unreachable target).
#
# Two figures, the subfigures of one thesis figure:
#   output/experiments/experiments_brtdp_carparking_bounds.{pdf,png}
#                car_parking S10, S20: upper and lower bound at the initial state for BRTDP, exhaustive
#                states and round robin. S10 has one initial state; S20 has four and shows state 22,
#                the one with the largest gap at the end of all three runs (read at the current
#                largest-gap state, BRTDP's curves jump between the states)
#   output/experiments/experiments_brtdp_carparking_gap.{pdf,png}
#                against the gap at the initial state, for every C5 abstraction run (car_parking in
#                colour, the others grey): left, the effective share of the non-terminal states that
#                carry the gap under exhaustive states; right, car_parking only, per BRTDP rollout, the
#                fraction of BRTDP's own remaining gap held by the states it updates (dashed: the fraction
#                of the non-terminal states it updates, which a gap-blind draw would match)
#
# Source: results/gap_spread_driveboth/trace.csv (DIAG_DRIVE=both tools/gap_spread.jl, seed 20260922,
# the arms of configs/thesis/{rivi,arivi-k1,brtdp}.toml; all eight C5 problems of the clustering,
# round robin on car_parking only). The S20 panel: results/s0_bounds_driveboth/
# trace.csv (tools/s0_bounds.jl car_parking_S20_I3, same seed and arms), where BRTDP has a row at every
# change of a bound at an initial state, so it is drawn as steps. Bottom right: results/
# rollout_states_driveboth/rollouts.csv (tools/rollout_states.jl, same seed, BRTDP as above).

include(joinpath(@__DIR__, "..", "common.jl"))
using CSV, DataFrames, Statistics

const SRC = joinpath(BENCH_DIR, "results", "gap_spread_driveboth")
const S0_SRC = joinpath(BENCH_DIR, "results", "s0_bounds_driveboth")
const RS_SRC = joinpath(BENCH_DIR, "results", "rollout_states_driveboth")
const OUT = "experiments"
const ARMS = [("exhaustive", "RIVI (reference)", "#0b0b0b", 1.6, :solid),
              ("rr1", "Async. RIVI", "#808080", 1.2, :dash),
              ("brtdp", "BRTDP", "#1baf7a", 1.3, :solid)]
const CARS = [("car_parking_S10_I3", "#eb6834"), ("car_parking_S20_I3", "#2a78d6")]
const FLOOR = 1e-12
const S20 = "car_parking_S20_I3"
const S20_STATE = 22
const EPS = 1e-5

tr = CSV.read(joinpath(SRC, "trace.csv"), DataFrame; missingstring="")
s0tr = CSV.read(joinpath(S0_SRC, "trace.csv"), DataFrame)
s0tr = s0tr[(s0tr.problem .== S20) .& (s0tr.s0 .== S20_STATE), :]
ro = CSV.read(joinpath(RS_SRC, "rollouts.csv"), DataFrame)
ro = ro[(ro.gap_sum .> 0) .& (ro.gap0 .> 0), :]

# centred rolling mean over about 1/30 of a run's rollouts (1 to 25), to show the trend through the noise
function rolling(y)
    w = clamp(length(y) ÷ 30, 1, 25); h = w ÷ 2
    [mean(y[max(1, i - h):min(end, i + h)]) for i in eachindex(y)]
end

# the bounds of one arm in a top panel, sorted by updates
function arm_bounds(p, d, a)
    p == S20 || return sort(d[d.alg .== a, :], :updates)
    s = sort(s0tr[s0tr.alg .== a, :], :updates)
    # at a fixed state the bounds are monotone, which is why this panel uses one
    @assert all(diff(s.U) .<= 0) && all(diff(s.L) .>= 0) "non-monotone bounds: $a"
    return DataFrame(updates=s.updates, U0=s.U, L0=s.L)
end
# the bounds at the initial state
fig = report_figure(width=1.0, aspect=2.2)
local axb
for (j, (p, _)) in enumerate(CARS)
    d = tr[tr.problem .== p, :]
    xmax = maximum(d.updates); e = floor(Int, log10(xmax))
    global axb = Axis(fig[2, j]; yscale=log10, title=p, titlesize=8, titlealign=:left,
                      xlabel=L"Bellman updates ($\times 10^{%$e}$)", xlabelsize=7, xlabelpadding=1,
                      ylabel=j == 1 ? "bounds at the initial state" : "", ylabelsize=8,
                      yticks=(10.0 .^ (-12:2:0), [L"10^{%$k}" for k in -12:2:0]),
                      xticklabelsize=7, yticklabelsize=7)
    for (a, label, col, lw, ls) in ARMS
        s = arm_bounds(p, d, a)
        x = s.updates ./ 10.0^e
        if p == S20 && a == "brtdp"
            stairs!(axb, x, max.(s.U0, FLOOR); step=:post, color=col, linewidth=lw, label=label)
            stairs!(axb, x, max.(s.L0, FLOOR); step=:post, color=col, linewidth=lw)
        else
            lines!(axb, x, max.(s.U0, FLOOR); color=col, linewidth=lw, linestyle=ls, label=label)
            lines!(axb, x, max.(s.L0, FLOOR); color=col, linewidth=lw, linestyle=ls)
        end
    end
    # label the bundles where the reference is halfway through its run
    s = arm_bounds(p, d, "exhaustive")
    i = findfirst(>=(0.55xmax), s.updates)
    Lmax = maximum(arm_bounds(p, d, a).L0[end] for a in first.(ARMS))
    below = s.L0[i] > 1e3 * FLOOR
    text!(axb, s.updates[i] / 10.0^e, s.U0[i]; text="upper bound", fontsize=7, align=(:left, :bottom),
          offset=(0, 3), color=C_INK2)
    text!(axb, s.updates[i] / 10.0^e, below ? s.L0[i] : Lmax; text="lower bound", fontsize=7,
          align=(:left, below ? :top : :bottom), offset=(0, below ? -3 : 3), color=C_INK2)
    ylims!(axb, FLOOR / 3, 2.0); xlims!(axb, 0, 1.03 * xmax / 10.0^e)
end
Legend(fig[1, 1:2], axb; orientation=:horizontal, patchsize=(18, 8), framevisible=false, labelsize=8,
       tellwidth=false)
rowgap!(fig.layout, 1, 2)

save_figure(fig, "experiments_brtdp_carparking_bounds", OUT)

# how evenly the gap is spread under exhaustive states (left), and how much of BRTDP's own
# remaining gap each of its rollouts updates (right), against the gap at the initial state
const XT = (-(0:-1:-5), [L"10^{%$k}" for k in 0:-1:-5])
style(p) = (k = findfirst(c -> c[1] == p, CARS);
            k === nothing ? ((C_CONTEXT, 0.6), 0.8, "other C5 abstractions", false) : (CARS[k][2], 1.6, p, true))
fig = report_figure(width=1.0, aspect=2.2)
axl = Axis(fig[2, 1]; xlabel="gap at the initial state", xlabelsize=7, xlabelpadding=1,
           ylabel="ratio of states\ncarrying the gap", ylabelsize=8, xticks=XT, yticks=0:0.25:1,
           xticklabelsize=7, yticklabelsize=7)
for p in sort(unique(tr.problem))
    s = sort(tr[(tr.problem .== p) .& (tr.alg .== "exhaustive"), :], :updates)
    s = s[.!ismissing.(s.gap_share) .& (s.gap0 .> 0), :]   # the axis clips what ends below ε
    isempty(s) && continue
    col, lw, lab, _ = style(p)
    lines!(axl, -log10.(s.gap0), s.gap_share; color=col, linewidth=lw, label=lab)   # gap falls to the right
end
axs = Axis(fig[2, 2]; xlabel="gap at the initial state", xlabelsize=7, xlabelpadding=1,
           ylabel="fraction of the gap updated\nper BRTDP trajectory", ylabelsize=8, xticks=XT, yticks=0:0.25:1,
           xticklabelsize=7, yticklabelsize=7)
for p in first.(CARS)                     # car_parking only
    r = sort(ro[ro.problem .== p, :], :rollout)
    col, lw, _, _ = style(p)
    x = -log10.(r.gap0)
    lines!(axs, x, rolling(r.distinct ./ r.n_nt); color=col, linewidth=0.9, linestyle=:dash)
    lines!(axs, x, rolling(r.gap_updated ./ r.gap_sum); color=col, linewidth=lw)
end
for ax in (axl, axs); ylims!(ax, 0, 1.04); xlims!(ax, 0, 5.2); end
# the left panel carries every problem's label; the dashed style is the right panel's only
plots, labels = Makie.get_labeled_plots(axl; merge=true, unique=true)
plots = Any[plots...]; labels = Any[labels...]
push!(plots, LineElement(color=:black, linestyle=:dash, linewidth=0.9))
push!(labels, rich("fraction of ", rich("S"; font=:italic), subscript("nt"), " states updated"))
Legend(fig[1, 1:2], plots, labels; orientation=:horizontal, nbanks=2, patchsize=(18, 8), rowgap=0,
       framevisible=false, labelsize=8, tellwidth=false)
rowgap!(fig.layout, 1, 2)

save_figure(fig, "experiments_brtdp_carparking_gap", OUT)
