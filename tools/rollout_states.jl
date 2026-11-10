#!/usr/bin/env julia
# Which states BRTDP's rollouts sample, rollout by rollout: how many, what share of the model and of the
# remaining gap, and how evenly the visits fall over the states.
#
#   julia --project=. tools/rollout_states.jl <problem>...   (one seed, 20260922; seconds to minutes)
#
# Runs BRTDP as in configs/thesis/brtdp.toml (drive = "both", max-gap initial state, k = 1,
# dedup_scope = "rollout") and overwrites, in results/rollout_states_driveboth/:
#   rollouts.csv  one row per rollout:
#       problem, rollout, updates      the rollout's index, the Bellman updates made before it
#       gap0                           the largest initial-state gap when it was sampled
#       start                          its start state (linear index)
#       visits                         its length before dedup (states drawn, the start included)
#       distinct                       its length after dedup = the states it backs up (k = 1)
#       new                            its states that no earlier rollout visited
#       n_states                       all states of the model (terminals included)
#       n_nt, n_open                   the non-terminal states (gap > 0 at the start of the run), and
#                                      those still open (gap > state_converged_eps) when it was sampled
#       open_distinct                  its distinct states that were open when it was sampled
#       gap_sum, gap_updated           Σ (U − L) over the non-terminal states when it was sampled, and over
#                                      its distinct states: gap_updated / gap_sum is the share of the
#                                      remaining gap that the rollout updates (= distinct / n_nt if the
#                                      draw ignored the gap)
#       gap_rho                        the effective share (Σg)² / (n_nt Σg²) of that gap over the
#                                      non-terminal states, as gap_share in tools/gap_spread.jl
#   visits.csv    sparse, one row per (rollout, state) the rollout visited: problem, rollout, state, count
#   states.csv    one row per non-terminal state: problem, state, is_s0, visits, backups (= the rollouts
#                 that visited it), first_rollout (0 = never), U0, L0 (the initial bounds), U, L (final)
# A rollout is one sample() call's buffer; with k = 1 it is backed up one state per GSRDP iteration, in
# reverse, so a rollout is the natural "iteration" of BRTDP.
using Random, TOML, Printf
const BENCH = normpath(joinpath(@__DIR__, ".."))
include(joinpath(BENCH, "src", "Bench.jl"))
using .Bench
using IntervalMDP
const TS = IntervalMDP.TrajectorySampling

const EPS_CONV = 1e-6                 # = state_converged_eps below
mutable struct Run
    lin::Any                          # LinearIndices of the state space
    nt::BitVector                     # non-terminal: gap > 0 at the first sampled rollout
    S0::Vector{Int}
    rows::Vector{Any}                 # per-rollout tuples
    visits::Vector{Any}               # per-rollout Dict state => count
    updates::Int
    U0::Vector{Float64}; L0::Vector{Float64}   # the bounds before the first backup
end
const R = Ref{Run}()

function TS._sample_trajectory(ss::TS.TrajectorySampling, model, vf, spec)
    r = R[]
    U = vf.upper.current; L = vf.lower.current
    if isempty(r.nt)                  # the first rollout: nothing is backed up yet
        r.lin = LinearIndices(size(U)); r.nt = BitVector([U[i] - L[i] > 0 for i in eachindex(U)])
        r.U0 = Float64.(vec(U)); r.L0 = Float64.(vec(L))
    end
    open = BitVector([r.nt[i] && U[i] - L[i] > EPS_CONV for i in eachindex(U)])
    g = maximum(U[s] - L[s] for s in r.S0)
    gv = [r.nt[i] ? Float64(U[i] - L[i]) : 0.0 for i in eachindex(U)]
    gsum = sum(gv); g2 = sum(abs2, gv)
    grho = g2 == 0 ? 0.0 : gsum^2 / (count(r.nt) * g2)
    t = TS._sample_trajectory(ss, model, vf, spec, TS._rollout_scratch(ss, model))
    c = Dict{Int,Int}()
    for s in t
        i = s isa CartesianIndex ? r.lin[s] : Int(s)
        c[i] = get(c, i, 0) + 1
    end
    start = isempty(t) ? 0 : (t[1] isa CartesianIndex ? r.lin[t[1]] : Int(t[1]))
    push!(r.rows, (r.updates, g, start, length(t), length(c), count(open), count(i -> open[i], keys(c)),
                   gsum, sum((gv[i] for i in keys(c)); init=0.0), grho))
    push!(r.visits, c)
    return t
end

const BRTDP = Dict("sampling" => "trajectory_brtdp", "action_epsilon" => 0.02, "k" => 1,
    "transition_bound" => "lower", "state_temperature" => 1.0, "state_alpha_e" => 1.0,
    "state_exclude_converged" => true, "state_converged_eps" => EPS_CONV, "dedup" => true,
    "dedup_scope" => "rollout", "terminate" => ["max_steps"], "initial_state" => "max", "drive" => "both")
cfgpath = tempname() * ".toml"
open(cfgpath, "w") do io
    TOML.print(io, Dict("name" => "rollout_states",
        "property" => Dict("convergence_eps" => 1e-5, "restrict_to_initial" => true),
        "problems" => Dict("root" => joinpath(BENCH, "problems"), "names" => ARGS),
        "algorithms" => [Dict("name" => "gsrdp", "label" => "brtdp", "params" => BRTDP)]))
end
cfg = load_experiment_config(cfgpath)
const OUT = mkpath(joinpath(BENCH, "results", "rollout_states_driveboth"))
fr = open(joinpath(OUT, "rollouts.csv"), "w")
fv = open(joinpath(OUT, "visits.csv"), "w")
fs = open(joinpath(OUT, "states.csv"), "w")
println(fr, "problem,rollout,updates,gap0,start,visits,distinct,new,n_states,n_nt,n_open,open_distinct,gap_sum,gap_updated,gap_rho")
println(fv, "problem,rollout,state,count")
println(fs, "problem,state,is_s0,visits,backups,first_rollout,U0,L0,U,L")
for ps in cfg.problems
    bp = build_problem(ps.name, ps.params)
    ok, bp = Bench.override_specification(bp, cfg)
    S0 = Int.(Bench.problem_initial_states(bp))
    built = build_algorithm("gsrdp", bp, BRTDP)
    Random.seed!(20260922)
    R[] = Run(nothing, BitVector(), S0, Any[], Any[], 0, Float64[], Float64[])
    lastvf = Ref{Any}(nothing)
    cb = function (vf, bu, _=nothing)
        R[].updates = Int(bu); lastvf[] = vf
        nothing
    end
    t = @elapsed solve(bp.problem, built.alg; callback=cb)
    r = R[]; name = ps.instance_name
    nt = findall(r.nt); nnt = length(nt)
    first = zeros(Int, length(r.nt)); vis = zeros(Int, length(r.nt)); bak = zeros(Int, length(r.nt))
    for (k, (row, c)) in enumerate(zip(r.rows, r.visits))
        new = 0
        for (s, n) in c
            first[s] == 0 && (first[s] = k; new += 1)
            vis[s] += n; bak[s] += 1
            println(fv, join((name, k, s, n), ","))
        end
        u, g, st, v, d, no, od, gs, gu, gr = row
        println(fr, join((name, k, u, g, st, v, d, new, length(r.nt), nnt, no, od, gs, gu, gr), ","))
    end
    U = vec(lastvf[].upper.current); L = vec(lastvf[].lower.current)
    for s in nt
        println(fs, join((name, s, s in S0, vis[s], bak[s], first[s], r.U0[s], r.L0[s], U[s], L[s]), ","))
    end
    foreach(flush, (fr, fv, fs))
    @printf("%-40s updates=%d rollouts=%d n_nt=%d visited=%d  wall=%.1fs\n", name, r.updates,
            length(r.rows), nnt, count(>(0), vis[nt]), t)
end
foreach(close, (fr, fv, fs))
