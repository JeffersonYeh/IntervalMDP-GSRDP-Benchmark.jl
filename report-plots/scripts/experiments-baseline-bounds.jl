# Experiments chapter, baseline appendix: why exhaustive states plateaus on the product IMDPs (C3).
# Per sweep, the upper and lower bound at the initial state, and how far the region with a positive
# lower bound has spread.
#
#   output/experiments/experiments_baseline_bounds_c3.{pdf,png}
#       one column per problem (pimdp_0, pimdp_1):
#       top     U(s0), L(s0), gap and the optimal value against the sweep (log axis)
#       bottom  share of the states with a non-zero value whose lower bound is already positive
#
# Source: results/bound_traces_driveboth/ (BT_DRIVE=both tools/bound_traces.jl, a replica of the
# exhaustive sweep with drive = "both" that reproduces the sweep counts of the rivi.toml runs,
# 73 and 53).

include(joinpath(@__DIR__, "..", "common.jl"))
using CSV, DataFrames

const SRC = joinpath(BENCH_DIR, "results", "bound_traces_driveboth")
const OUT = "experiments"
const PROBLEMS = ["pimdp_0", "pimdp_1"]
const FLOOR = 1e-7                      # values below, including exactly 0, are drawn here
const EPS = 1e-5                        # convergence threshold of the benchmark runs
const C_UPPER = "#eb6834"
const C_LOWER = "#2a78d6"

summ = CSV.read(joinpath(SRC, "summary.csv"), DataFrame)
fig = report_figure(width=1.0, aspect=1.45)
local ax1, ax2
for (j, p) in enumerate(PROBLEMS)
    d = CSV.read(joinpath(SRC, p * ".csv"), DataFrame)
    d = d[1:findfirst(<(EPS), d.gap0), :]   # the benchmark run stops here
    r = summ[summ.problem .== p, :][1, :]
    v = r.value_s0
    fl(x) = max.(x, FLOOR)
    global ax1 = Axis(fig[2, j]; yscale=log10, title=p, titlesize=9, titlealign=:left,
                      ylabel=j == 1 ? "bound at the initial state" : "",
                      yticks=(10.0 .^ (-7:0), [L"10^{%$e}" for e in -7:0]),
                      xticklabelsize=7, yticklabelsize=7)
    hlines!(ax1, [v]; color=C_CONTEXT, linestyle=:dash, linewidth=0.8, label=L"optimal value $V^*(s_0)$")
    hlines!(ax1, [EPS]; color=C_CONTEXT, linestyle=:dot, linewidth=0.6)
    lines!(ax1, d.sweep, fl(d.gap0); color="#0b0b0b", linewidth=1.0, label="gap")
    lines!(ax1, d.sweep, fl(d.U0); color=C_UPPER, linewidth=1.4, label="upper bound")
    lines!(ax1, d.sweep, fl(d.L0); color=C_LOWER, linewidth=1.4, label="lower bound")
    ylims!(ax1, FLOOR / 2, 1.5)
    # sweep in which the lower bound at s0 first reaches 1% of the optimal value
    k1 = d.sweep[findfirst(>=(0.01v), d.L0)]
    global ax2 = Axis(fig[3, j]; xlabel="sweep", ylabel=j == 1 ? "states with\nlower bound > 0" : "",
                      yticks=(0:0.25:1, ["0", "25%", "50%", "75%", "100%"]),
                      xticklabelsize=7, yticklabelsize=7, xlabelsize=8)
    lines!(ax2, d.sweep, d.npos ./ r.ever_positive; color=C_LOWER, linewidth=1.4)
    ylims!(ax2, 0, 1.04)
    for a in (ax1, ax2)
        vlines!(a, [k1]; color=C_CONTEXT, linestyle=:dot, linewidth=0.8)
        xlims!(a, 0, d.sweep[end] + 1)
    end
    text!(ax2, k1 - 1, 0.08; text=L"$L(s_0) \geq 0.01\,V^*(s_0)$", fontsize=7, align=(:right, :bottom),
          color=C_INK2)
    linkxaxes!(ax1, ax2); hidexdecorations!(ax1; grid=false, ticks=false)
    j == 1 || (hideydecorations!(ax1; grid=false, ticks=false); hideydecorations!(ax2; grid=false, ticks=false))
end
Legend(fig[1, 1:2], ax1; orientation=:horizontal, patchsize=(18, 8), framevisible=false, labelsize=8,
       tellwidth=false)
rowsize!(fig.layout, 3, Relative(0.32))
rowgap!(fig.layout, 1, 2)

save_figure(fig, "experiments_baseline_bounds_c3", OUT)
