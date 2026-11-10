# IntervalMDP-GSRDP-Benchmark.jl

A config-driven benchmark for robust value iteration on interval MDPs, built for comparing GSRDP
(generalised sampling-based robust dynamic programming) variants of
[IntervalMDP.jl](https://github.com/Zinoex/IntervalMDP.jl): exhaustive sweeps (RIVI),
asynchronous round robin, and trajectory sampling (BRTDP). It also holds the 38-problem dataset
recipe, the experiment configs and the figure scripts of the MSc thesis it was written for.

Each (problem × algorithm) cell is solved once per seed. Every solve records:

- **wall-clock time** of the solve (BenchmarkTools), one timed sample per seed;
- **convergence**: the gap between the upper and lower bound at the initial states, against
  wall-clock time and against Bellman updates;
- **quality**: iterations, final gap, whether it converged, and agreement of the value with a
  reference arm (the exhaustive sweep) on the initial states.

## Install

Julia 1.12. From the repository root:

```sh
julia --project -e 'using Pkg; Pkg.instantiate()'
```

`Project.toml` pins the solver to a commit of IntervalMDP.jl's `jy/ps-sampling` branch, and
`Manifest.toml` pins every other dependency.

## Run

```sh
bash tools/storm/convert_storm.sh storm_tiny01             # build a few problems (problems/README.md)
bash tools/prism/convert_prism.sh robot_delta0.1 coin2_K2  # (needs PRISM)
julia --project -t 8 run.jl configs/smoke.toml             # RIVI, async. RIVI and BRTDP on them
```

A run writes `results/<name>_<timestamp>/`: `results.csv` (one row per solve), the per-solve
convergence traces, `problems.csv` (problem facts), `resolved_config.toml` and `meta.json`
(versions, commit, host, threads). `combine_results.jl` merges several such directories into
one; `run_analyses.jl` draws a generic report of one. The config format is
described in [docs/experiment-config.md](docs/experiment-config.md), and the code layout in
[ARCHITECTURE.md](ARCHITECTURE.md).

## Reproducing the thesis

1. Build the 38 problems: [problems/README.md](problems/README.md).
2. Run the configs, the model-metric tools and the diagnostics, then combine:
   [configs/thesis/README.md](configs/thesis/README.md).
3. Draw the figures: `julia --project report-plots/make_all.jl`
   ([report-plots/README.md](report-plots/README.md)).

## Layout

```
run.jl, combine_results.jl      run configs; merge results directories
run_analyses.jl                 generic analysis and report of one results directory
compact_traces.jl               thin the traces of a results directory in place
src/                            the harness (Bench) and the analysis code (BenchmarkAnalysis)
test/                           tests: julia --project test/runtests.jl
configs/                        smoke.toml and the thesis experiments (configs/thesis/)
problems/                       the problem files (generated, git-ignored) and their description
tools/                          problem generators, model-metric tools, single-seed diagnostics
report-plots/                   the thesis figures
```
