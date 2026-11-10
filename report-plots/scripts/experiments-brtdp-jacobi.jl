# Experiments chapter, ablation "asynchronous vs synchronous backups": BRTDP with single-state
# Gauss-Seidel batches (k = 1) against BRTDP without Gauss-Seidel batching (Jacobi: every rollout
# relaxed in one GSRDP iteration), next to the two baselines, exhaustive-state GSRDP (RIVI, the
# reference) and round robin (asynchronous RIVI, k = 1), per cluster of the Benchmark chapter.
#
#   output/experiments/experiments_brtdp_jacobi_convergence_clusters_upd.{pdf,png}
#       Bellman updates, one panel per cluster (C1–C5 and the extreme C6*; C7* left out)
#
# BRTDP (async.) = brtdp_rw6_k1_s0max_driveboth (configs/thesis/brtdp.toml). BRTDP (sync.) =
# brtdp_rw6_jacobi_s0max_driveboth: the same sampler with `k = 1` replaced by gauss_seidel = false
# (configs/thesis/ablation-jacobi.toml). All four arms with drive = "both".
# Curves and clusters: scripts/_experiments_common.jl (run dataset-cluster.jl first).
# gsrdp_exhaustive_state_driveboth is renamed to the reference label. coin2_K64 drops out: BRTDP
# (async.) is censored at the 24 h timeout on every seed.

include(joinpath(@__DIR__, "_experiments_common.jl"))

const RR1 = "gsrdp_round_robin_k1_driveboth"
const BRTDP = "brtdp_rw6_k1_s0max_driveboth"
const JAC = "brtdp_rw6_jacobi_s0max_driveboth"
const ARMS = [(REF, "RIVI", "#0b0b0b", 1.6, :solid),
              (RR1, "Async. RIVI", "#808080", 1.2, :dash),
              (BRTDP, "BRTDP (async.)", "#1baf7a", 1.3, :solid),
              (JAC, "BRTDP (sync.)", "#eb6834", 1.3, :solid)]

E = load_experiment(THESIS_DIR, ARMS; rename=Dict("gsrdp_exhaustive_state_driveboth" => REF))

cluster_grid_figure(E, :upd, "experiments_brtdp_jacobi_convergence_clusters_upd";
                    clusters=filter(c -> !startswith(c[1], "C7"), E.clusters))
