# Experiments chapter, ablation "backup order": BRTDP handing each rollout out goal-end first
# (reverse = true, the default) against initial-state-end first (reverse = false), next to round
# robin (asynchronous RIVI, k = 1), per cluster of the Benchmark chapter.
#
#   output/experiments/experiments_brtdp_no_reverse_convergence_clusters_upd.{pdf,png}
#       Bellman updates, one panel per cluster (C1–C5 and the extreme C6*; C7* left out)
#
# BRTDP (reverse) = brtdp_rw6_k1_s0max_driveboth (configs/thesis/brtdp.toml). BRTDP (no reverse) =
# brtdp_rw6_k1_s0max_driveboth_noreverse: the same sampler with reverse = false
# (configs/thesis/ablation-no-reverse.toml). All arms with drive = "both". The exhaustive sweep is
# not drawn; it only sets the τ scale, so these curves stay comparable with the other figures.
# Curves and clusters: scripts/_experiments_common.jl (run dataset-cluster.jl first).
# gsrdp_exhaustive_state_driveboth is renamed to the reference label. coin2_K64 drops out: the
# reverse arm is censored at the 24 h timeout and the no-reverse arm does not run it.

include(joinpath(@__DIR__, "_experiments_common.jl"))

const RR1 = "gsrdp_round_robin_k1_driveboth"
const BRTDP = "brtdp_rw6_k1_s0max_driveboth"
const NOREV = "brtdp_rw6_k1_s0max_driveboth_noreverse"
const ARMS = [(REF, "RIVI (reference)", "#0b0b0b", 0.0, :solid),   # width 0: not drawn
              (RR1, "Async. RIVI", "#808080", 1.2, :dash),
              (BRTDP, "BRTDP (reverse)", "#1baf7a", 1.3, :solid),
              (NOREV, "BRTDP (no reverse)", "#eb6834", 1.3, :solid)]

E = load_experiment(THESIS_DIR, ARMS; rename=Dict("gsrdp_exhaustive_state_driveboth" => REF))

cluster_grid_figure(E, :upd, "experiments_brtdp_no_reverse_convergence_clusters_upd";
                    clusters=filter(c -> !startswith(c[1], "C7"), E.clusters))
