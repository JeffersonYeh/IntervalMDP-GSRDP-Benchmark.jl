# Experiments chapter, BRTDP comparison: BRTDP (single-state batches, k = 1) against the two
# baselines, exhaustive-state GSRDP (RIVI, the reference) and round robin (asynchronous RIVI,
# k = 1), over the whole dataset, per cluster and per problem.
#
#   output/experiments/experiments_brtdp_convergence_all.{pdf,png}
#       all problems: wall-clock (log τ) | Bellman updates
#   output/experiments/experiments_brtdp_convergence_clusters_{upd,wall}.{pdf,png}
#       one panel per cluster (C1–C5 and the extremes C6*, C7*)
#   output/experiments/experiments_brtdp_per_problem_<c1…c7>.{pdf,png}
#       one figure per cluster, one panel per problem: gap against the actual Bellman updates,
#       seed mean with min–max shading over the seeds
#   output/experiments/experiments_brtdp_selected.{pdf,png}
#       the six problems on which BRTDP deviates from its cluster, raw updates
#   output/experiments/experiments_brtdp_speedup_updates.{pdf,png}
#       histogram of the per-problem speed-up in Bellman updates, vs RIVI | vs asynchronous RIVI
#
# BRTDP = brtdp_rw6_k1_s0max_driveboth (configs/thesis/brtdp.toml). All three
# arms with drive = "both" (each bound runs its own optimizing Bellman update; 2|A| updates per
# backed-up state). Curves and clusters: scripts/_experiments_common.jl (run dataset-cluster.jl
# first). gsrdp_exhaustive_state_driveboth is renamed to the reference label. coin2_K64 drops
# out: BRTDP is censored at the 24 h timeout on every seed.

include(joinpath(@__DIR__, "_experiments_common.jl"))

const RR1 = "gsrdp_round_robin_k1_driveboth"
const BRTDP = "brtdp_rw6_k1_s0max_driveboth"
const ARMS = [(REF, "RIVI (reference)", "#0b0b0b", 1.6, :solid),
              (RR1, "Async. RIVI", "#808080", 1.2, :dash),
              (BRTDP, "BRTDP", "#1baf7a", 1.3, :solid)]

E = load_experiment(THESIS_DIR, ARMS; rename=Dict("gsrdp_exhaustive_state_driveboth" => REF))

pair_figure(E, "experiments_brtdp_convergence_all", E.problems; logwall=true)

for kind in (:upd, :wall)
    cluster_grid_figure(E, kind, "experiments_brtdp_convergence_clusters_$kind"; xlog=kind == :wall)
end

# per problem, one figure per cluster: actual Bellman updates, min–max over the seeds
for (gname, ps) in E.clusters
    problem_curves_figure(E, gname, ps, "experiments_brtdp_per_problem_$(cluster_tag(gname))")
end

# the six problems on which BRTDP deviates from its cluster
const SELECTED = ["car_parking_S10_I3", "car_parking_S20_I3",
                  "linear_stochastically_switched_S40", "package_delivery_S50_I5",
                  "multiObj_robotIMDP", "coin2_K16"]
problem_curves_figure(E, "selected", SELECTED, "experiments_brtdp_selected"; ncol=2)

# speed-up in Bellman updates of BRTDP over each baseline, a histogram over the problems
speedup_histogram(E, BRTDP, [REF, RR1], "experiments_brtdp_speedup_updates"; kind=:upd)
