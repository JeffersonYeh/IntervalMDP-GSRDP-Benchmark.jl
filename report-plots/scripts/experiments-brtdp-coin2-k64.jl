# Experiments chapter, BRTDP section: convergence on coin2_K64, BRTDP (k = 1) against exhaustive-state
# GSRDP (RIVI) and round robin (asynchronous RIVI, k = 1), all with drive = "both".
#
#   output/experiments/experiments_brtdp_coin2_k64.{pdf,png}
#       initial-state gap against wall-clock (left) | Bellman updates (right), both on log axes;
#       seed mean, shaded from the minimum to the maximum over the 5 seeds
#
# load_experiment (_experiments_common.jl) keeps only problems every arm solves, and BRTDP is
# censored at the 24 h timeout on coin2_K64 (configs/thesis/brtdp.toml), so this script
# reads the traces itself. The curves stop where the runs ended (for BRTDP, the censoring point).

include(joinpath(@__DIR__, "..", "common.jl"))
using Statistics

const P = "coin2_K64"
const EPS = 1e-5
const OUT = "experiments"
# (label, legend, colour, width, style); the reference drawn last, on top
const ARMS = [("gsrdp_exhaustive_state_driveboth", "RIVI", "#0b0b0b", 1.6, :solid),
              ("gsrdp_round_robin_k1_driveboth", "Async. RIVI", "#808080", 1.2, :dash),
              ("brtdp_rw6_k1_s0max_driveboth", "BRTDP", "#1baf7a", 1.3, :solid)]

long = load_long(THESIS_DIR)
long = long[(long.problem .== P) .& in.(long.algorithm, Ref(Set(first.(ARMS)))), :]

# Seed curves of one arm on a log grid up to where its last seed ended: (x, mean, min, max, n).
function curves(a, xcol)
    tr = long[long.algorithm .== a, :]
    xpos = filter(>(0), Float64.(tr[!, xcol]))
    x = 10 .^ range(log10(minimum(xpos)), log10(maximum(xpos)); length=600)
    M = reduce(hcat, [BenchmarkAnalysis.interp_gap(g, x; xcol) for g in groupby(tr, :seed)])
    return x, vec(mean(M, dims=2)), vec(minimum(M, dims=2)), vec(maximum(M, dims=2)), size(M, 2)
end

fig = report_figure(width=1.0, aspect=2.3)
axs = map(enumerate(((:wall_time, "wall-clock (s, 8 threads)"), (:bellman_updates, "Bellman updates")))) do (j, (xcol, xlabel))
    ax = Axis(fig[2, j]; xscale=log10, yscale=log10, xlabel, ylabel=j == 1 ? GAP_LABEL : "")
    for (a, label, col, lw, ls) in reverse(ARMS)
        x, m, lo, hi, n = curves(a, xcol)
        n > 1 && any(hi .> lo) && band!(ax, x, lo, hi; color=(col, 0.18))
        lines!(ax, x, m; color=col, linewidth=lw, linestyle=ls, label=label)
        println("  $a [$xcol]: $n seeds, end x = $(round(x[end]; sigdigits=3)), gap $(round(m[end]; sigdigits=3))")
    end
    hlines!(ax, [EPS]; color=C_CONTEXT, linestyle=:dot, linewidth=0.6)
    ylims!(ax, EPS / 3, 1.5)
    j == 1 || hideydecorations!(ax; grid=false, ticks=false)
    ax
end
plots, labels = Makie.get_labeled_plots(axs[end]; merge=false, unique=false)
Legend(fig[1, 1:2], reverse(plots), reverse(labels); orientation=:horizontal,
       patchsize=(18, 8), framevisible=false, labelsize=8, tellwidth=false)
rowgap!(fig.layout, 1, 3)

save_figure(fig, "experiments_brtdp_coin2_k64", OUT)
