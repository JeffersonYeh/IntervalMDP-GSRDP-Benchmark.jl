#!/usr/bin/env julia
# Entry point for the MDP-solver benchmarking harness.
#
#   julia --project -t auto run.jl configs/smoke.toml                        # single run
#   julia --project -t auto run.jl configs/smoke.toml configs/sweep.toml     # sequential
#
# Results land in results/<experiment>_<timestamp>/ (results.csv, per-cell
# trace.csv, resolved_config.toml, meta.json, plots/).

import Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "src", "Bench.jl"))
using .Bench

Bench.main(ARGS; root=@__DIR__)
