# Thesis experiments

The configs behind every figure of the thesis, and the full pipeline from problems to figures.
All commands run from the repository root.

## Solver runs

Every config pins the same five seeds (`20260922, 8675309, 424242, 31415926, 27182818`),
`convergence_eps = 1e-5` on the initial states and `drive = "both"` (each bound runs its own
optimizing Bellman update). A config has one per-solve timeout and one problem list, so the arms
are split wherever those differ.

| config | arms (label) | problems | timeout | used by |
|---|---|---|---|---|
| `rivi.toml` | `gsrdp_exhaustive_state_driveboth` (RIVI, the reference) | all 38 | none | every experiments figure |
| `arivi-k1.toml` | `gsrdp_round_robin_k1_driveboth` (asynchronous RIVI, k = 1) | all 38 | none | every experiments figure |
| `brtdp.toml` | `brtdp_rw6_k1_s0max_driveboth` (BRTDP) | all 38 | 24 h | BRTDP section (coin2_K64 is censored), ablations, sensitivity |
| `ablations.toml` | `brtdp_rw6_jacobi_s0max_driveboth` (gauss_seidel = false), `…_noreverse` (reverse = false), `…_tbupper` (transition_bound = "upper") | 37 (all but coin2_K64) | 24 h | ablations |
| `sensitivity-temperature.toml` | `brtdp_rw6_k1_s0max_driveboth_T<T>_e002`, T ∈ {0.5, 0.75, 1.25, 1.5, 4, 16} | 36 (all but coin4_K8, coin2_K64) | 2 h | sensitivity figure |
| `sensitivity-epsilon.toml` | `brtdp_rw6_k1_s0max_driveboth_T1_e<ε>`, ε ∈ {0.002, 0.006, 0.06, 0.2, 0.4, 0.7} | the 25 of those 36 with more than one action | 2 h | sensitivity figure |

Run them, then merge them into the one directory the figures read:

```sh
for c in configs/thesis/*.toml; do julia --project -t 8 run.jl "$c"; done
julia --project combine_results.jl results/thesis_* --out results/thesis_combined --name thesis \
    --reference gsrdp_exhaustive_state_driveboth --force --no-analyses
```

`--force` because the timeouts differ between the configs. The thesis ran every config on 8
threads of one node of a cluster (wall-clock figures are for 8 Julia threads); update counts do
not depend on the machine. In total this is about 3,000 solves: coin2_K64 alone takes 5 × 24 h in
`brtdp.toml`, coin4_K8 a few hours per solve and arm, and the rest from milliseconds to two
hours. On a cluster, split a config by problem or by seed (`names`, `seeds`) into several
configs with different `name`s and combine all of them in the one `combine_results.jl` call.
Per-solve timeouts must stay as listed, or the censored runs change.

## Model metrics (no solve)

The Benchmark-chapter figures and the clusters read per-problem structure metrics from
`results/mixing/`:

```sh
P=$(ls problems | grep -v README)        # the 38 problem names
julia --project tools/mixing_stats.jl --spectral problems $P          # mixing_stats.csv, dists/
julia --project tools/s0_horizon.jl problems $P                       # s0_horizon.csv
julia --project tools/stochasticity_stats.jl problems $P              # dists_stoch/
for p in $P; do julia --project tools/spectral_density.jl --max-block Inf problems $p; done   # dists_spectral/
```

Memory peaks at a few times a problem's `.nc` for `mixing_stats.jl`, and grows with the largest
strongly connected component for `spectral_density.jl` (coin4_K8, pimdp_0 and pimdp_1 need a
large-memory node; run one problem per process).

## Single-seed diagnostics

Five figures come from instrumented single-seed runs (seed 20260922) of the same arms:

```sh
BT_DRIVE=both julia --project -t 8 tools/bound_traces.jl pimdp_0 pimdp_1
C5="linear_stochastically_switched_S10 linear_stochastically_switched_S20 linear_stochastically_switched_S40
    car_parking_S10_I3 car_parking_S20_I3 bas_4d_S5x5x7x7_I4 almost_identity_n3_S8 almost_identity_n3_S16"
DIAG_DRIVE=both DIAG_ALGS=brtdp,exhaustive julia --project tools/gap_spread.jl $C5
DIAG_DRIVE=both DIAG_ALGS=rr1 julia --project tools/gap_spread.jl car_parking_S10_I3 car_parking_S20_I3
julia --project tools/s0_bounds.jl car_parking_S20_I3
julia --project tools/rollout_states.jl car_parking_S10_I3 car_parking_S20_I3
DIAG_DRIVE=both julia --project tools/brtdp_diagnostics.jl \
    linear_stochastically_switched_S40 package_delivery_S50_I5 multiObj_robotIMDP
```

They write `results/{bound_traces,gap_spread,s0_bounds,rollout_states,brtdp_diagnostics}_driveboth/`.
`gap_spread.jl` and `brtdp_diagnostics.jl` append to their trace, so delete it before a rerun.

## Figures

```sh
julia --project report-plots/make_all.jl
```

writes every figure to `report-plots/output/{benchmark,experiments}/`; see
[report-plots/README.md](../../report-plots/README.md) for which script makes which figure.
