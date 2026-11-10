#!/usr/bin/env julia
# BFS depth on the abstraction problems with the out-of-domain sink taken out of the targets.
#
# An IntervalMDPAbstractions model ends in one sink state for "left the domain", which is in
# avoid and has a u > 0 edge from every cell, so mixing_stats' depth and depth_avoid are 1 for
# every non-terminal state. Here the BFS seeds are the non-terminal states with a u > 0 edge
# into (reach ∪ avoid) \ {sink}, resp. avoid \ {sink}; the graph is unchanged (the sink has no
# outgoing edges, so no path runs through it). Same definitions and quantile grid as
# tools/mixing_stats.jl otherwise:
#
#   depth        fewest u > 0 steps to (reach ∪ avoid) \ {sink}
#   depth_avoid  fewest u > 0 steps to avoid \ {sink}   (no data when avoid = {sink})
#
# over the non-terminal states that can reach that set. Writes <out>/<problem>.csv (q, depth,
# depth_avoid) and <out>/summary.csv (how many non-terminal states reach each set).
# A problem without such a sink (last state not in avoid, or with stored columns) is skipped.
#
# Usage (from the repository root):
#   julia --project tools/sink_free_depth.jl [--out DIR] [problems_dir] name ...
# Default out DIR is results/mixing/dists_nosink. Peak memory ≈ load_model of the .nc (the
# largest abstractions, ~30M transitions, stay under 3 GB); run one problem per process.

include(joinpath(@__DIR__, "mixing_stats.jl"))   # load_model, load_spec, maybe_graph, bfs, QLEVELS, ...

"Index of the abstraction sink: the last state, in avoid, with no stored columns; or nothing."
function abstraction_sink(m, avoid)
    m.n in avoid || return nothing
    ncols(m) == (m.n - 1) * m.A || return nothing
    return m.n
end

function sink_free_depth(dir, name, out)
    m = load_model(joinpath(dir, name, name * ".nc"))
    reach, avoid, _, _ = load_spec(joinpath(dir, name, name * ".json"))
    sink = abstraction_sink(m, avoid)
    sink === nothing && (println("  $name: no abstraction sink, skipped"); return nothing)
    n = m.n
    terminal = falses(n); terminal[reach] .= true; terminal[avoid] .= true
    maybe = findall(!, terminal)
    ptr, adj, _, _ = maybe_graph(m, terminal)
    rptr, radj = reverse_graph(n, ptr, adj)

    # Non-terminal states with a u > 0 edge into `targets` (the BFS seeds at depth 1).
    function frontier(targets)
        front = falses(n)
        for s in maybe, c in cols(m, s), j in m.ucol[c]:(m.ucol[c+1]-1)
            m.uval[j] > 0 && targets[m.urow[j]] && (front[s] = true)
        end
        return findall(front)
    end
    function depths(targets)
        any(targets) || return Int[]
        d = bfs(n, rptr, radj, frontier(targets), 1)
        return [d[s] for s in maybe if d[s] < typemax(Int)]
    end
    to_T = copy(terminal); to_T[sink] = false
    to_A = falses(n); to_A[avoid] .= true; to_A[sink] = false
    dT, dA = depths(to_T), depths(to_A)

    qcol(v) = isempty(v) ? fill(missing, length(QLEVELS)) :
              (sv = sort(v); [sv[clamp(ceil(Int, p * length(sv)), 1, length(sv))] for p in QLEVELS])
    CSV.write(joinpath(out, name * ".csv"),
              DataFrame(q = collect(QLEVELS), depth = qcol(Float64.(dT)), depth_avoid = qcol(Float64.(dA))))
    println("  $name: sink $sink, |S_nt| $(length(maybe)), reach (T∖sink) $(length(dT)), reach (avoid∖sink) $(length(dA))")
    return (problem = name, num_nt = length(maybe), n_depth = length(dT), n_depth_avoid = length(dA))
end

function main_nosink(args)
    out = joinpath(@__DIR__, "..", "results", "mixing", "dists_nosink")
    if !isempty(args) && args[1] == "--out"
        out = args[2]; args = args[3:end]
    end
    dir = joinpath(@__DIR__, "..", "problems")
    if !isempty(args) && isdir(args[1])
        dir = args[1]; args = args[2:end]
    end
    mkpath(out)
    rows = filter(!isnothing, [sink_free_depth(dir, a, out) for a in args])
    isempty(rows) && return
    # merge into summary.csv by problem, so one-problem-per-process runs accumulate
    path = joinpath(out, "summary.csv")
    new = DataFrame(rows)
    old = isfile(path) ? CSV.read(path, DataFrame) : similar(new, 0)
    CSV.write(path, sort(vcat(filter(r -> !(r.problem in new.problem), old), new), :problem))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_nosink(ARGS)
end
