# Experiments chapter, baseline: exhaustive-state GSRDP (RIVI, the reference) against round robin
# with single-state batches (asynchronous RIVI, k = 1), over the whole dataset and per cluster of
# the Benchmark chapter.
#
#   output/experiments/experiments_baseline_convergence_all.{pdf,png}
#       initial-state gap curves over all problems: wall-clock | Bellman updates
#   output/experiments/experiments_baseline_convergence_clusters_{upd,wall}.{pdf,png}
#       the same per cluster (C1–C5 and the extremes C6*, C7*), one panel each
#
# Curves and clusters: scripts/_experiments_common.jl (run dataset-cluster.jl first).
# Both arms with drive = "both" (each bound runs its own optimizing Bellman update; 2|A| updates
# per backed-up state). Runs: configs/thesis/rivi.toml and arivi-k1.toml; the exhaustive arm
# gsrdp_exhaustive_state_driveboth is renamed to the reference label.

include(joinpath(@__DIR__, "_experiments_common.jl"))

const RR1 = "gsrdp_round_robin_k1_driveboth"
const ARMS = [(REF, "RIVI (reference)", "#0b0b0b", 1.6, :solid),
              (RR1, "Async. RIVI", "#808080", 1.2, :dash)]

E = load_experiment(THESIS_DIR, ARMS; rename=Dict("gsrdp_exhaustive_state_driveboth" => REF))
@assert isempty(E.excluded) "the figures assume every run reached ε"

pair_figure(E, "experiments_baseline_convergence_all", E.problems)

for kind in (:upd, :wall)
    cluster_grid_figure(E, kind, "experiments_baseline_convergence_clusters_$kind")
end
