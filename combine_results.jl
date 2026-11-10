#!/usr/bin/env julia
# Merge several sweeps' results directories into one, then analyse them together.
#
# Every consumer in this harness takes exactly ONE results dir, so two sweeps that were
# designed to be comparable -- the same `[problems]` with different `[[algorithms]]`, or
# the same `[[algorithms]]` over different `[problems]` -- cannot be analysed together.
# This assembles them into a new directory that is indistinguishable from a single sweep's,
# so `run_analyses.jl` / `render_gap_videos.jl` read it without knowing the difference.
#
#   julia --project combine_results.jl "results/<run A>" "results/<run B>"
#   julia --project combine_results.jl results/part{1,2,3,4,5,6}_*   # N sources, not just 2
#   julia --project combine_results.jl "results/<combined>" "results/<run C>"   # adds C to it
#   julia --project combine_results.jl "results/<A>" "results/<B>" --out results/merged
#   julia --project combine_results.jl "results/<A>" "results/<B>" --no-analyses
#
# A COMBINED DIRECTORY ABSORBS. When one of the arguments was itself produced by this tool,
# the merge happens INSIDE it -- the other results are added to it, its existing samples
# keep the numbers they already have, and no new directory appears -- whatever position it
# was given in. With several combined directories among the arguments, the FIRST one is the
# target and the others' contents (and their own merge history) are folded into it. So
# re-running as each sweep finishes grows ONE directory instead of a chain of them:
#
#   julia --project combine_results.jl results/part1_* results/part2_*   # -> smoke_combined_...
#   julia --project combine_results.jl results/smoke_combined_* results/part3_*   # adds part3
#   julia --project combine_results.jl results/smoke_combined_* results/part4_*   # adds part4
#
# Pass --out to force a fresh directory instead and leave every source untouched. Re-adding
# a run that is already inside the target is refused, since it would double-count it.
#
# The merge is a union over the {problems} x {algorithms} cells. A cell present in more
# than one source -- the shared `gsrdp_exhaustive_state` reference arm, typically -- has its
# samples POOLED: the rows are concatenated and `sample_index` renumbered 1..N, with
# `runs/<cell>/sample<k>/` renamed and `trace_path` rewritten to match, so the cell reads
# downstream as one cell that was simply run more times.
#
# Sources other than an absorbing target are opened read-only and never modified. The
# combined directory carries a COMBINED_FROM.md naming every run that went into it, which
# sample came from which, and every caveat any merge raised (different host, different
# commit, repeated seeds, ...). It is rendered from combine_log.json, which holds one entry
# per merge, so a directory merged into repeatedly keeps its whole history.
#
# Options:
#   --out DIR          output directory; forces a FRESH one even when a source is combined
#                      (default: absorb into the first combined source, else
#                      <parent of the first source>/<name>_combined_<timestamp>)
#   --name NAME        experiment name for the combined dir (default: the sources' shared name)
#   --mode link|copy   how run/ and raw/ files land (default: link -- hardlink, falling
#                      back to a copy across filesystems; the trace matrices are GBs)
#   --dedupe-seeds     keep only the first measurement of each (cell_id, seed) pair --
#                      REQUIRED whenever the sources pinned the same `seeds` and share an arm
#   --reference LABEL  reference algorithm label for the merged config and the analysis
#   --force            downgrade the blocking compatibility checks to recorded caveats
#   --no-analyses      assemble the directory but do not run the analysis suite
#   --no-gap-videos    do not (re-)render gap videos during the analysis pass
#
# Blocking by default (use --force to override, and read COMBINED_FROM.md afterwards):
# a differing `[property].convergence_eps` / `restrict_to_initial`, a differing
# `[benchmark].timeout_s`, one algorithm label meaning two different things, a problem
# whose state/action/transition counts changed between the runs, and -- unless
# --dedupe-seeds is passed -- a (cell_id, seed) pair measured in more than one source.
#
# That last one blocks rather than warns. The analysis reduces by (algorithm, problem,
# seed) with no sample identity (`run_scalars`, src/analysis/reduce.jl), so two runs sharing
# a seed merge into ONE group, their traces sort together by wall-clock, and the reported
# solve time is read off the interleaved mixture -- a fabricated number, not a noisy one.
# It is exactly the shape a split sweep leaves when every part pins `seeds` and re-runs the
# reference arm, so the arm it corrupts is the reference every normalized figure divides by:
#
#   julia --project combine_results.jl results/<sweep>_part{1,2,3}_* --dedupe-seeds

import Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "src", "Bench.jl"))
using .Bench

const USAGE = "usage: julia --project combine_results.jl <results_dir_A> <results_dir_B> " *
              "[<results_dir_C> ...] [--out DIR] [--name NAME] [--mode link|copy] " *
              "[--dedupe-seeds] [--reference LABEL] [--force] [--no-analyses] [--no-gap-videos]"

function parse_args(args)
    isempty(args) && error(USAGE)
    dirs = String[]
    opts = Dict{Symbol,Any}(:out => nothing, :name => nothing, :mode => :link,
                            :dedupe_seeds => false, :reference => nothing, :force => false,
                            :analyses => true, :gap_videos => nothing)
    i = 1
    while i <= length(args)
        a = args[i]
        if !startswith(a, "--")
            push!(dirs, a); i += 1
        elseif a == "--out" && i < length(args)
            opts[:out] = args[i+1]; i += 2
        elseif a == "--name" && i < length(args)
            opts[:name] = args[i+1]; i += 2
        elseif a == "--mode" && i < length(args)
            m = Symbol(args[i+1])
            m in (:link, :copy) || error("--mode must be 'link' or 'copy', got '$(args[i+1])'")
            opts[:mode] = m; i += 2
        elseif a == "--reference" && i < length(args)
            opts[:reference] = args[i+1]; i += 2
        elseif a == "--dedupe-seeds"
            opts[:dedupe_seeds] = true; i += 1
        elseif a == "--force"
            opts[:force] = true; i += 1
        elseif a == "--no-analyses"
            opts[:analyses] = false; i += 1
        elseif a == "--no-gap-videos"
            opts[:gap_videos] = false; i += 1
        else
            error("unexpected argument '$a'\n$USAGE")
        end
    end
    length(dirs) >= 2 ||
        error("need at least two results directories to combine, got $(length(dirs))\n$USAGE")
    for d in dirs
        isdir(d) || error("no such results directory: $d")
    end
    return dirs, opts
end

dirs, opts = parse_args(ARGS)

out = Bench.combine_results(dirs;
        out=opts[:out], name=opts[:name], mode=opts[:mode],
        dedupe_seeds=opts[:dedupe_seeds], reference_label=opts[:reference],
        force=opts[:force], analyses=opts[:analyses], gap_videos=opts[:gap_videos],
        command="julia --project combine_results.jl " * join(ARGS, " "))

println()
println(out in abspath.(dirs) ? "added into -> " : "combined  -> ", out)
println("sources      ", joinpath(out, Bench.COMBINE_SOURCES_FILE))
opts[:analyses] || println("\nanalysis skipped (--no-analyses); run " *
                           "`julia --project run_analyses.jl \"$out\"` to produce figures")
