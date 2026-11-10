# report-plots

Scripts that turn the benchmark results into the figures of the thesis. Each script writes
fixed file names as PDF (vector) and PNG (3 px per pt), drawn at the report's text width
(`TEXTWIDTH_PT`), so they can be included without scaling.

```
report-plots/
  common.jl          shared setup: environment, theme, sizing, results_dir, save_figure
  make_all.jl        regenerate everything (or a subset)
  scripts/           one script per figure or small group of figures
                     (`_`-prefixed files are shared code, skipped by make_all.jl)
  output/benchmark/   figures of the Benchmark chapter and its appendix   (git-ignored)
  output/experiments/ figures of the Experiments chapter and its appendix (git-ignored)
  cache/             parsed traces and the cluster assignment             (git-ignored)
```

## Inputs

Everything the scripts read is produced by the pipeline in
[configs/thesis/README.md](../configs/thesis/README.md):

| input | produced by |
|---|---|
| `results/thesis_combined/` | the solver runs of `configs/thesis/*.toml`, merged with `combine_results.jl` |
| `results/mixing/` | the model-metric tools: `tools/mixing_stats.jl`, `s0_horizon.jl`, `stochasticity_stats.jl`, `spectral_density.jl` |
| `results/*_driveboth/` | the single-seed diagnostic tools: `tools/bound_traces.jl`, `gap_spread.jl`, `s0_bounds.jl`, `rollout_states.jl`, `brtdp_diagnostics.jl` |

## Figures

| figure | script | data |
|---|---|---|
| `benchmark/dataset_box_{size,stoch,horizon}_metrics` | `dataset-metrics.jl` | results/mixing |
| `benchmark/dataset_cluster_{pc_loadings,dendrogram,k,scatter}` | `dataset-cluster.jl` | results/mixing |
| `experiments/experiments_baseline_convergence_{all,clusters_upd,clusters_wall}` | `experiments-baseline.jl` | rivi, arivi-k1 |
| `experiments/experiments_baseline_bounds_c3` | `experiments-baseline-bounds.jl` | bound_traces_driveboth |
| `experiments/experiments_brtdp_{convergence_all,convergence_clusters_upd,convergence_clusters_wall,selected,speedup_updates,per_problem_c1…c7}` | `experiments-brtdp.jl` | rivi, arivi-k1, brtdp |
| `experiments/experiments_brtdp_carparking_{bounds,gap}` | `experiments-brtdp-carparking.jl` | gap_spread / s0_bounds / rollout_states_driveboth |
| `experiments/experiments_brtdp_diagnostics` | `experiments-brtdp-diagnostics.jl` | brtdp_diagnostics_driveboth |
| `experiments/experiments_brtdp_coin2_k64` | `experiments-brtdp-coin2-k64.jl` | rivi, arivi-k1, brtdp |
| `experiments/experiments_tb_convergence_upd` | `experiments-transition-bound.jl` | + ablation-transition-bound |
| `experiments/experiments_brtdp_no_reverse_convergence_clusters_upd` | `experiments-brtdp-no-reverse.jl` | + ablation-no-reverse |
| `experiments/experiments_brtdp_jacobi_convergence_clusters_upd` | `experiments-brtdp-jacobi.jl` | + ablation-jacobi |
| `experiments/experiments_eps_temp_sensitivity_upd_logx` | `experiments-eps-temp.jl` | + sensitivity-{epsilon,temperature} |

The experiments scripts group the problems by the clusters of `dataset-cluster.jl`, which writes
them to `cache/dataset_cluster_{problems,excluded}.csv`; `make_all.jl` runs it first.

## Running

From the repository root:

```sh
julia report-plots/make_all.jl                 # every script
julia report-plots/make_all.jl baseline        # scripts whose file name contains a pattern
julia report-plots/scripts/<name>.jl           # one script
```

`make_all.jl` loads the analysis code once and then runs each script in its own module. A
failing script is reported and does not stop the others. Parsing the traces of
`results/thesis_combined` takes a while the first time; the parsed frame is cached in `cache/`
and rebuilt when `results.csv` is newer.

Axis labels are rendered by MathTeXEngine, which supports only part of LaTeX; `common.jl`
registers `\underline`, which it lacks.
