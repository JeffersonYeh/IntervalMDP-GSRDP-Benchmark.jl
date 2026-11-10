# Experiments chapter, BRTDP appendix: the single-seed diagnostic runs behind the thesis text on
# the problems with several initial states (lss S40, package_delivery S50) and on multiObj;
# car_parking has its own figure (experiments-brtdp-carparking.jl).
#
#   output/experiments/experiments_brtdp_diagnostics.{pdf,png}
#       lss S40, package_delivery S50: largest gap over the initial states (solid) and, for BRTDP,
#       the mean gap over the initial states (dotted); the trajectory in which BRTDP's largest gap
#       leaves 1 is marked
#       multiObj: the upper and lower bound at the initial state
#
# Source: results/brtdp_diagnostics_driveboth/trace.csv (DIAG_DRIVE=both tools/brtdp_diagnostics.jl,
# seed 20260922, the arms of configs/thesis/{rivi,arivi-k1,brtdp}.toml; log-spaced rows of updates,
# the initial state with the largest gap, its U and L, the mean U and L over the initial states,
# and the rollouts started so far).

include(joinpath(@__DIR__, "..", "common.jl"))
using CSV, DataFrames

const SRC = joinpath(BENCH_DIR, "results", "brtdp_diagnostics_driveboth")
const OUT = "experiments"
const ARMS = [("exhaustive", "RIVI (reference)", "#0b0b0b", 1.6, :solid),
              ("rr1", "Async. RIVI", "#808080", 1.2, :dash),
              ("brtdp", "BRTDP", "#1baf7a", 1.3, :solid)]
const EPS = 1e-5
const PANELS = [("linear_stochastically_switched_S40", :gap, 1e-6),
                ("package_delivery_S50_I5", :gap, 1e-6),
                ("multiObj_robotIMDP", :bounds_lin, 0.0)]
const V_MULTIOBJ = 0.8946629825788643   # V*(s0) of multiObj (the converged bounds)
const NCOL = 3
const TITLES = Dict("car_parking_S10_I3" => "car_parking_S10_I3", "car_parking_S20_I3" => "car_parking_S20_I3",
                    "linear_stochastically_switched_S20" => "lss_S20 (4 initial states)",
                    "linear_stochastically_switched_S40" => "lss_S40 (16 initial states)",
                    "package_delivery_S50_I5" => "pkg_S50 (32 initial states)",
                    "multiObj_robotIMDP" => "multiObj_robotIMDP")

tr = CSV.read(joinpath(SRC, "trace.csv"), DataFrame)
fig = report_figure(width=1.0, aspect=3.0)
local ax
local ax, ax1
for (q, (p, kind, floor_)) in enumerate(PANELS)
    r, c = fldmod1(q, NCOL)
    d = tr[tr.problem .== p, :]
    xmax = maximum(d.updates)
    if kind == :bounds_lin
        # multiObj: the bounds near 0.9 on a linear axis, the updates on a log axis (BRTDP ends 10x earlier)
        global ax = Axis(fig[r + 1, c]; xscale=log10, title=TITLES[p], titlesize=8, titlealign=:left,
                         xlabel="Bellman updates", xlabelsize=7, xlabelpadding=1,
                         ylabel="bounds at the initial state", ylabelsize=8, yticks=0:0.25:1,
                         xticks=(10.0 .^ (1:2:7), [L"10^{%$k}" for k in 1:2:7]),
                         xticklabelsize=7, yticklabelsize=7)
        for (a, label, col, lw, ls) in ARMS
            s = sort(d[d.alg .== a, :], :updates)
            lines!(ax, s.updates, s.U; color=col, linewidth=lw, linestyle=ls, label=label)
            lines!(ax, s.updates, s.L; color=col, linewidth=lw, linestyle=ls)
        end
        hlines!(ax, [V_MULTIOBJ]; color=C_CONTEXT, linestyle=:dot, linewidth=0.8)
        text!(ax, 3e4, 0.94; text="upper bound", fontsize=7, align=(:left, :bottom), color=C_INK2)
        text!(ax, 15.0, 0.04; text="lower bound", fontsize=7, align=(:left, :bottom), color=C_INK2)
        ylims!(ax, -0.03, 1.08); xlims!(ax, 8, 1.3 * xmax)
        continue
    end
    e = floor(Int, log10(xmax))
    fl(v) = max.(v, floor_)
    lo = round(Int, log10(floor_))
    global ax = Axis(fig[r + 1, c]; yscale=log10, title=TITLES[p], titlesize=8, titlealign=:left,
                     xlabel=L"Bellman updates ($\times 10^{%$e}$)", xlabelsize=7, xlabelpadding=1,
                     ylabel=c == 1 ? GAP_LABEL : "",
                     ylabelsize=8, yticks=(10.0 .^ (lo:2:0), [L"10^{%$k}" for k in lo:2:0]),
                     xticklabelsize=7, yticklabelsize=7)
    q == 1 && (global ax1 = ax)
    for (a, label, col, lw, ls) in ARMS
        s = sort(d[d.alg .== a, :], :updates)
        x = s.updates ./ 10.0^e
        lines!(ax, x, fl(s.gap); color=col, linewidth=lw, linestyle=ls, label=label)
        a == "brtdp" && lines!(ax, x, fl(s.meanU .- s.meanL); color=col, linewidth=1.0, linestyle=:dot,
                               label="BRTDP, mean over the initial states")
    end
    hlines!(ax, [EPS]; color=C_CONTEXT, linestyle=:dot, linewidth=0.6)
    # the trajectory during which BRTDP's largest gap first drops below 1
    s = sort(d[d.alg .== "brtdp", :], :updates)
    i = findfirst(<(1), s.gap)
    vlines!(ax, [s.updates[i] / 10.0^e]; color="#1baf7a", linestyle=:dot, linewidth=0.8)
    text!(ax, s.updates[i] / 10.0^e, 10.0^(lo + 1); text=" trajectory $(s.rollouts[i])",
          fontsize=7, align=(:left, :bottom), color=C_INK2)
    ylims!(ax, floor_ / 3, 2.0); xlims!(ax, 0, 1.03 * xmax / 10.0^e)
end
Legend(fig[1, 1:NCOL], ax1; orientation=:horizontal, nbanks=2, patchsize=(18, 8), framevisible=false,
       labelsize=8, tellwidth=false)
rowgap!(fig.layout, 1, 2)

save_figure(fig, "experiments_brtdp_diagnostics", OUT)
