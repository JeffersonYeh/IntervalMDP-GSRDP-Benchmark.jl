module Bench

# --- external dependencies (shared by all included files) ------------------
# IntervalMDP is `using` so its accessors (system, specification, value_function,
# num_states, ...) are unqualified everywhere. IntervalMDPAbstractions / LazySets /
# ArchCompStochasticModels are `import`ed (not `using`) because several of their
# exports clash with IntervalMDP's (e.g. `system`); the product_delivery builder
# qualifies them explicitly.
using IntervalMDP
import IntervalMDPAbstractions
import ArchCompStochasticModels
import LazySets
using JSON
using SparseArrays
using LinearAlgebra
using Statistics
using Random
using Dates
using Serialization
using TOML
using Pkg
using InteractiveUtils
using BenchmarkTools
using DataFrames
using CSV
using CairoMakie
using LaTeXStrings
using Interpolations

# --- framework (order matters: later files use earlier definitions) --------
include("problem_provider.jl")   # BenchmarkProblem
include("registry.jl")           # registries + resolvers
include("thinning.jl")           # relative-gap trace thinning (also in BenchmarkAnalysis)
include("trace.jl")              # convergence Trace callback
include("quality.jl")            # quality scalars + reference agreement
include("validation.jl")         # solution-based convergence + cross-trial bound checks
include("config.jl")             # TOML -> ExperimentConfig
include("runmatrix.jl")          # cartesian product -> RunUnit
include("results.jl")            # results.csv / meta.json
include("problem_stats.jl")      # problems.csv (per-problem structural facts)
include("benchmark.jl")          # the BenchmarkTools driver
include("analysis/BenchmarkAnalysis.jl")  # 7-analysis statistical & plotting suite
using .BenchmarkAnalysis: generate_report, run_analyses, render_gap_videos, REFERENCE
include("combine.jl")            # merge several results dirs into one (needs run_analyses)

# --- registered problem sources (self-register at load) --------------------
# Only the file-reader builders (intervalmdp_file / prism / bmdp_tool) load here:
# the benchmark runner (run.jl) consumes pre-built problem files (problems/README.md).
include("file_sources.jl")

# --- registered algorithms -------------------------------------------------
include("algorithms.jl")

export run_experiment, generate_report, run_analyses, render_gap_videos, load_experiment_config, build_run_matrix
export combine_results, rerender_sources_file, COMBINE_SOURCES_FILE, COMBINE_LOG_FILE
export register_problem!, register_algorithm!, build_problem, build_algorithm
export PROBLEM_REGISTRY, ALGORITHM_REGISTRY, BenchmarkProblem, BuiltAlgorithm
export TrialBounds, extract_trial_bounds, validate_bounds, write_validation_report

"""
    main(args)

CLI entry: one or more `.toml` config paths.

    julia --project run.jl configs/smoke.toml
    julia --project run.jl configs/a.toml configs/b.toml ...     # sequential
"""
function main(args::AbstractVector{<:AbstractString}; root::AbstractString=dirname(@__DIR__))
    isempty(args) && error("usage: julia --project run.jl <config.toml> [config2.toml ...]")
    configs = filter(a -> endswith(a, ".toml"), args)
    non_toml = filter(a -> !endswith(a, ".toml"), args)
    isempty(configs) && error("no config .toml files given")
    isempty(non_toml) || error("unexpected non-toml args: $(join(non_toml, ' '))")

    dirs = String[]
    for config_path in configs
        cfg = load_experiment_config(config_path; base_dir=root)
        dir = run_experiment(cfg; base_dir=root, project_dir=root)
        # `[output].analysis = false` yields raw results only (no figures, no
        # METHODS_meanstd.md / manifest.txt) -- intended for batch/cluster runs, where
        # plotting is better done afterwards against the copied-back directory.
        if cfg.output.analysis
            generate_report(dir; reference_label=cfg.output.reference_label)
        else
            @info "analysis skipped ([output].analysis = false); run \
                   `julia --project run_analyses.jl <dir>` to produce figures" results=dir
        end
        push!(dirs, dir)
    end
    return length(dirs) == 1 ? first(dirs) : dirs
end

end # module
