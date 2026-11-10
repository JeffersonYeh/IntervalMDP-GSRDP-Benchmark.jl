#!/usr/bin/env julia
# Upper and lower bound at the initial state per sweep of exhaustive-state GSRDP, for the plateau of
# the product IMDPs (thesis, baseline C3 paragraph; report-plots experiments-baseline-bounds.jl).
#
# A replica of the exhaustive sweep (Jacobi: every update reads the previous sweep; the action is
# greedy on the upper bound and used for both bounds; pessimistic O-maximisation). With
# BT_DRIVE=both it reproduces the sweep counts of the exhaustive runs of configs/thesis/rivi.toml
# (73 and 53 sweeps on pimdp_0 and pimdp_1), and records per sweep:
#   U0, L0, gap0     the bounds and gap at the initial state
#   npos             states with a positive lower bound
#   new_by_lower     states whose lower bound turned positive because an interval lower bound l > 0
#                    forces mass onto a successor that already has L > 0
#   new_by_capacity  ... because the successors with L = 0 cannot take all of the mass
# Writes results/bound_traces/<problem>.csv and one summary row per problem to summary.csv.
#
# Usage (from the repository root; pimdp_0/1: ~2 min and 2.5 GB each):
#   julia --project=. -t 8 tools/bound_traces.jl pimdp_0 pimdp_1
#   BT_DRIVE=both julia --project=. -t 8 tools/bound_traces.jl pimdp_0 pimdp_1
#       drive = "both" (configs/thesis/rivi.toml): the lower bound takes its own maximising action
#       instead of the one greedy on the upper bound; writes results/bound_traces_driveboth/

include(joinpath(@__DIR__, "mixing_stats.jl"))   # load_model, load_spec, align_lower, cols
using Printf, Base.Threads

const BOTH = get(ENV, "BT_DRIVE", "follow") == "both"
const OUT = mkpath(joinpath(@__DIR__, "..", "results", BOTH ? "bound_traces_driveboth" : "bound_traces"))
const TOL = 1e-9          # stop once the initial-state gap is below this
const MAXSWEEPS = 1000

"Pessimistic O-maximisation of column `c` at values `v`: mass beyond the lower bounds goes to the lowest values first."
function omax(m, la, c, v, buf)
    r = m.ucol[c]:(m.ucol[c+1]-1)
    resize!(buf, length(r)); buf .= r
    sort!(buf; by = j -> v[m.urow[j]])
    budget = 1.0
    for j in r; budget -= la[j]; end
    val = 0.0
    for j in buf
        q = la[j] + min(m.uval[j] - la[j], max(budget, 0.0))
        budget -= q - la[j]; val += q * v[m.urow[j]]
    end
    return val
end

function sweep!(Un, Ln, U, L, m, la, term, act, bufs)
    chunks = collect(Iterators.partition(1:m.n, cld(m.n, length(bufs))))
    @threads for ci in eachindex(chunks)
        buf = bufs[ci]
        for s in chunks[ci]
            term[s] && continue
            best = -Inf; ba = 0
            for c in cols(m, s)
                x = omax(m, la, c, U, buf); x > best && (best = x; ba = c)
            end
            Un[s] = best
            if BOTH   # the lower bound maximises on its own; act = its action
                bl = -Inf
                for c in cols(m, s)
                    x = omax(m, la, c, L, buf); x > bl && (bl = x; ba = c)
                end
                Ln[s] = bl
            else
                Ln[s] = omax(m, la, ba, L, buf)
            end
            act[s] = ba
        end
    end
end

function bound_traces(name)
    dir = joinpath(@__DIR__, "..", "problems", name)
    m = load_model(joinpath(dir, name * ".nc"))
    reach, avoid, smode, amode = load_spec(joinpath(dir, name * ".json"))
    (smode, amode) == ("maximize", "pessimistic") || error("$name: only maximise/pessimistic")
    length(m.init) == 1 || error("$name: one initial state expected")
    la = align_lower(m)
    term = falses(m.n); term[reach] .= true; term[avoid] .= true
    s0 = Int(m.init[1])
    U = ones(m.n); U[avoid] .= 0; L = zeros(m.n); L[reach] .= 1
    Un, Ln = copy(U), copy(L); act = zeros(Int, m.n)
    bufs = [Int[] for _ in 1:nthreads()]
    firstpos = fill(typemax(Int), m.n)
    io = open(joinpath(OUT, name * ".csv"), "w")
    println(io, "sweep,U0,L0,gap0,npos,new_by_lower,new_by_capacity")
    @printf(io, "0,%.12g,%.12g,%.12g,%d,0,0\n", U[s0], L[s0], U[s0] - L[s0], count(>(0), L[.!term]))
    k = 0
    while k < MAXSWEEPS
        k += 1
        sweep!(Un, Ln, U, L, m, la, term, act, bufs)
        nl = 0; nc = 0
        for s in 1:m.n
            (term[s] || L[s] > 0 || Ln[s] <= 0) && continue
            c = act[s]; forced = false
            for j in m.ucol[c]:(m.ucol[c+1]-1)
                la[j] > 0 && L[m.urow[j]] > 0 && (forced = true; break)
            end
            forced ? (nl += 1) : (nc += 1)
            firstpos[s] = k
        end
        U .= Un; L .= Ln
        @printf(io, "%d,%.12g,%.12g,%.12g,%d,%d,%d\n", k, U[s0], L[s0], U[s0] - L[s0],
                count(>(0), L[.!term]), nl, nc)
        U[s0] - L[s0] < TOL && break
    end
    close(io)
    # summary: value at s0, when s0 joined the L > 0 region, and the spread of the joining sweeps
    lsum = [sum(la[m.ucol[c]:(m.ucol[c+1]-1)]) for c in act if c > 0]
    fp = sort(filter(<(typemax(Int)), firstpos[.!term]))
    q(v, p) = v[clamp(ceil(Int, p * length(v)), 1, length(v))]
    row = (name, k, (U[s0] + L[s0]) / 2, firstpos[s0], q(fp, 0.5), q(fp, 0.9), fp[end],
           count(==(typemax(Int)), firstpos[.!term]), length(fp), q(sort(lsum), 0.5))
    sp = joinpath(OUT, "summary.csv")
    rows = isfile(sp) ? filter(l -> !startswith(l, name * ","), readlines(sp)[2:end]) : String[]
    push!(rows, join(row, ","))
    open(sp, "w") do f
        println(f, "problem,sweeps,value_s0,s0_first_positive_L,join_median,join_p90,join_max,never_positive,ever_positive,lsum_median")
        foreach(r -> println(f, r), sort(rows))
    end
    @info "done" name k row
end

if abspath(PROGRAM_FILE) == @__FILE__
    foreach(bound_traces, ARGS)
end
