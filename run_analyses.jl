#!/usr/bin/env julia
# Standalone driver for the 7-analysis benchmark suite. Re-runs the full statistical
# analysis & plotting suite on a results directory, without re-benchmarking.
#
# Works on a PARTIAL directory too — one left by a sweep that was killed or hit its
# wall clock. It never hard-errors on missing pieces; instead `manifest.txt` leads with
# a SWEEP INCOMPLETE banner naming the cells that produced no data, and every figure
# carries a caption saying how much is missing.
#
#   julia --project run_analyses.jl "results/<experiment>_<timestamp>"
#   julia --project run_analyses.jl "results/<dir>" gsrdp_exhaustive_state   # override reference
#
# Figures (named by analysis) land in <results_dir>/plots/; METHODS_meanstd.md and
# manifest.txt land in <results_dir>.
#
# ALGORITHM GROUPS. A sweep with tens of arms overlays them all on one axis. `--group`
# redraws cactus, normalized_gap_curves, scatter_vs_reference and vs_transitions for a
# named subset into <results_dir>/plots/groups/<name>/ — the reference is always added.
# Patterns are exact labels or globs (`*`, `?`); groups may overlap:
#
#   julia --project run_analyses.jl "results/<dir>" \
#       --group eps=traj_centre,traj_eps* --group k=traj_centre,traj_k*
#   julia --project run_analyses.jl "results/<dir>" --groups configs/groups.toml --only-groups
#
# where groups.toml holds `name = ["label or glob", ...]` entries, drawn in file order.
#
# Options:
#   --group NAME=PAT[,PAT...]   one group; repeatable
#   --groups FILE.toml          groups from a file; repeatable, appended after --group
#   --only-groups               draw only the group figures (skips the rest of the suite,
#                               the policy figures and the videos); its manifest goes to
#                               plots/groups/manifest.txt, leaving manifest.txt untouched
#   --thin-rtol X               relative gap tolerance the traces are thinned to while
#                               loading (default 1e-3; 0 loads every row — beware, a
#                               sampled arm's trace can hold tens of millions of rows).
#                               See the thinning note in src/analysis/adapter.jl.

import Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "src", "Bench.jl"))
using .Bench

const USAGE = "usage: julia --project run_analyses.jl <results_dir> [reference_label] " *
              "[--group NAME=PAT[,PAT...]]... [--groups FILE.toml]... [--only-groups] [--thin-rtol X]"

function parse_args(args)
    pos = String[]
    group_specs = Pair{String,Vector{String}}[]
    group_files = String[]
    only_groups = false
    thin_rtol = Bench.BenchmarkAnalysis.DEFAULT_THIN_RTOL
    i = 1
    while i <= length(args)
        a = args[i]
        if !startswith(a, "--")
            push!(pos, a); i += 1
        elseif a == "--group" && i < length(args)
            push!(group_specs, Bench.BenchmarkAnalysis.parse_group_spec(args[i+1])); i += 2
        elseif a == "--groups" && i < length(args)
            push!(group_files, args[i+1]); i += 2
        elseif a == "--only-groups"
            only_groups = true; i += 1
        elseif a == "--thin-rtol" && i < length(args)
            thin_rtol = parse(Float64, args[i+1]); i += 2
        else
            error("unexpected argument '$a'\n$USAGE")
        end
    end
    1 <= length(pos) <= 2 || error(USAGE)
    for f in group_files
        append!(group_specs, Bench.BenchmarkAnalysis.load_group_file(f))
    end
    groups = isempty(group_specs) ? nothing : group_specs
    only_groups && groups === nothing && error("--only-groups needs --group or --groups\n$USAGE")
    return pos, groups, only_groups, thin_rtol
end

pos, groups, only_groups, thin_rtol = parse_args(ARGS)
results_dir = pos[1]
reference   = length(pos) >= 2 ? pos[2] : Bench.REFERENCE
run_analyses(results_dir; reference_label=reference,
             groups=groups, only_groups=only_groups, thin_rtol=thin_rtol)
