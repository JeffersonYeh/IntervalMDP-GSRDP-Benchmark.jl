#!/usr/bin/env julia
# Shrink the report-facing `trace.csv` files of a finished results directory in place,
# with the same relative-gap thinning the analysis loader applies (src/thinning.jl).
# A sampled arm records a row every few Bellman updates, so one trace can reach 6.6 GB;
# thinned at rtol = 1e-3 it keeps a few tens of thousands of rows and every analysis
# result is unchanged to within a relative 1e-3 in the gap.
#
#   julia --project compact_traces.jl "results/<dir>"                 # rtol = 1e-3
#   julia --project compact_traces.jl "results/<dir>" --rtol 1e-4
#   julia --project compact_traces.jl "results/<dir>" --dry-run       # report only
#
# For each runs/**/trace.csv: the thinned copy is written to `trace.csv.tmp`, the
# original is renamed to `trace.full.csv`, and the copy takes its place, so
# results.csv's `trace_path` needs no change. Originals are NEVER deleted — the script
# ends by printing their total size so you can remove them yourself:
#
#   find "results/<dir>/runs" -name trace.full.csv -delete
#
# Idempotent: a sample that already has `trace.full.csv` is skipped. Each run appends
# what it did to `<dir>/compaction.json`.
#
# Rows the reader cannot parse (the truncated final line a SIGKILL leaves) are copied
# through verbatim, so the analysis still counts that trace as truncated. The matrix
# traces (gap_trace.csv, value_trace_*.csv, update_trace.csv) are not touched: the
# videos need every iteration.

import Pkg
Pkg.activate(@__DIR__; io=devnull)

using JSON
using Dates

include(joinpath(@__DIR__, "src", "thinning.jl"))

const USAGE = "usage: julia --project compact_traces.jl <results_dir> [--rtol X] [--dry-run]"

function parse_args(args)
    dir = nothing; rtol = 1e-3; dry = false
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--rtol" && i < length(args)
            rtol = parse(Float64, args[i+1]); i += 2
        elseif a == "--dry-run"
            dry = true; i += 1
        elseif !startswith(a, "--") && dir === nothing
            dir = a; i += 1
        else
            error("unexpected argument '$a'\n$USAGE")
        end
    end
    dir === nothing && error(USAGE)
    isdir(dir) || error("not a directory: $dir")
    0 < rtol < 1 || error("--rtol must be in (0, 1), got $rtol")
    return dir, rtol, dry
end

"""
    compact_trace(src, dst, rtol) -> (rows_before, rows_after)

Stream `src`, writing the header and the thinned rows to `dst` (`dst === nothing`
counts without writing). `nothing` when `src` is not a report trace (no gap column).
"""
function compact_trace(src::AbstractString, dst, rtol::Real)
    open(src, "r") do io
        header = readline(io)
        ncol = count(==(','), header) + 1
        cols = trace_header_columns(header)
        jw = get(cols, "wallclock_s", 0); jb = get(cols, "bellman_updates", 0)
        jg = get(cols, "gap_max", 0);     ji = get(cols, "initial_gap_max", 0)
        (jw > 0 && jb > 0 && (jg > 0 || ji > 0)) || return nothing

        out = dst === nothing ? devnull : open(dst, "w")
        rows_before = rows_after = 0
        try
            println(out, header)
            emit(line) = (rows_after += 1; println(out, line); nothing)
            tb = TailBuffer{String}(GapThinner(rtol, 2))
            fields = SubString{String}[]
            for line in eachline(io)
                isempty(line) && continue
                rows_before += 1
                split_fields!(fields, line)
                ok = length(fields) == ncol &&
                     _field_float(fields, jw) !== nothing && _field_float(fields, jb) !== nothing
                if !ok
                    # Keep an unparseable row exactly where it was (see the header).
                    flush_tail!(emit, tb)
                    emit(line)
                    continue
                end
                g  = jg > 0 ? something(_field_float(fields, jg), NaN) : NaN
                gi = ji > 0 ? something(_field_float(fields, ji), NaN) : NaN
                offer!(emit, tb, line, (g, gi))
            end
            flush_tail!(emit, tb)
        finally
            out === devnull || close(out)
        end
        return rows_before, rows_after
    end
end

_gb(bytes) = round(bytes / 2^30; digits=2)

function main(args)
    dir, rtol, dry = parse_args(args)
    runs = joinpath(dir, "runs")
    isdir(runs) || error("no runs/ directory under $dir")
    traces = String[]
    for (root, _, files) in walkdir(runs)
        "trace.csv" in files && push!(traces, joinpath(root, "trace.csv"))
    end
    sort!(traces)
    println("$(length(traces)) trace.csv file(s) under $runs; rtol = $rtol", dry ? " (dry run)" : "")

    entries = Dict{String,Any}[]
    for path in traces
        sdir = dirname(path)
        full = joinpath(sdir, "trace.full.csv")
        rel = relpath(path, dir)
        if isfile(full)
            println("  skip  $rel (already compacted)")
            continue
        end
        before = filesize(path)
        t0 = time()
        tmp = path * ".tmp"
        r = try
            compact_trace(path, dry ? nothing : tmp, rtol)
        catch e
            isfile(tmp) && rm(tmp)
            @warn "failed to compact; left untouched" path exception=e
            continue
        end
        if r === nothing
            println("  skip  $rel (not a report trace)")
            continue
        end
        rows_before, rows_after = r
        after = dry ? missing : filesize(tmp)
        if !dry
            mv(path, full)
            mv(tmp, path)
        end
        dt = round(time() - t0; digits=1)
        println("  $(dry ? "would" : "done")  $rel: $rows_before → $rows_after rows, ",
                "$(_gb(before)) GB", dry ? "" : " → $(_gb(after)) GB", " ($(dt)s)")
        push!(entries, Dict{String,Any}("path" => rel, "rows_before" => rows_before,
                                        "rows_after" => rows_after, "bytes_before" => before,
                                        "bytes_after" => after))
    end

    if !dry && !isempty(entries)
        log = joinpath(dir, "compaction.json")
        runs_log = isfile(log) ? JSON.parsefile(log) : Any[]
        push!(runs_log, Dict{String,Any}("datetime" => string(now()), "rtol" => rtol,
                                         "files" => entries))
        open(io -> JSON.print(io, runs_log, 2), log, "w")
        println("logged to $log")
    end

    kept = String[]
    for (root, _, files) in walkdir(runs)
        "trace.full.csv" in files && push!(kept, joinpath(root, "trace.full.csv"))
    end
    if !isempty(kept)
        total = sum(filesize, kept)
        println("\n$(length(kept)) original(s) kept as trace.full.csv, $(_gb(total)) GB in total. ",
                "Once the re-analysis looks right, remove them with:\n",
                "  find \"$runs\" -name trace.full.csv -delete")
    end
end

main(ARGS)
