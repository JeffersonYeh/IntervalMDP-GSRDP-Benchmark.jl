#!/usr/bin/env julia
# How evenly the gap is spread over the states during a solve, and how evenly BRTDP spreads its
# updates: the evidence for the car_parking paragraph of the thesis (BRTDP section; report-plots
# experiments-brtdp-carparking.jl), with the other C4 abstractions for contrast.
#
#   julia --project=. tools/gap_spread.jl <problem>...   (one seed, 20260922; seconds per problem)
#   DIAG_DRIVE=both [DIAG_ALGS=brtdp,exhaustive] julia --project=. tools/gap_spread.jl <problem>...
#       the arms of configs/thesis/{rivi,arivi-k1,brtdp}.toml instead (drive = "both", BRTDP with the max-gap initial
#       state); writes results/gap_spread_driveboth/trace.csv
#
# Runs BRTDP (tune6 brtdp_rw6_k1), exhaustive states and round robin (k = 1) as in
# tools/brtdp_diagnostics.jl and appends to results/gap_spread/trace.csv (delete it first to rerun a
# problem) log-spaced rows, the last one at the final update of the run, of
#   updates, gap0, U0, L0   largest gap over the initial states, and U, L at that state
#   gap_share               effective share of the non-terminal states that carry the gap,
#                           (Σ g)² / (n Σ g²) over the n non-terminal states (1 = spread evenly
#                           over all of them, 1/n = a single state)
#   visit_share             the same for BRTDP's cumulative updates per state (empty otherwise)
using Random, Printf, TOML, Statistics
const BENCH = normpath(joinpath(@__DIR__, ".."))
include(joinpath(BENCH, "src", "Bench.jl"))
using .Bench
using IntervalMDP
const TS = IntervalMDP.TrajectorySampling
const VISITS = Dict{Any,Int}()
function TS._sample_trajectory(ss::TS.TrajectorySampling, model, value_function, spec)
    t = TS._sample_trajectory(ss, model, value_function, spec, TS._rollout_scratch(ss, model))
    foreach(s -> VISITS[s] = get(VISITS, s, 0) + 1, t)
    return t
end

effshare(x, n) = (s2 = sum(abs2, x); s2 == 0 ? 0.0 : sum(x)^2 / (n * s2))

mutable struct Rec
    S0::Vector{Int}; nt::Vector{Int}; io::IO; alg::String; prob::String; next::Float64; upd::Int
    vf::Any; logged::Int                 # the latest value function, the updates of the last row
end
function (r::Rec)(vf, bu::Integer, _=nothing)
    r.upd = Int(bu); r.vf = vf
    r.upd < r.next && !isempty(r.nt) && return nothing
    r.next = max(r.next * 1.03, r.upd + 1)
    logrow(r)
end
function logrow(r::Rec)
    r.logged = r.upd
    U = r.vf.upper.current; L = r.vf.lower.current
    # non-terminal states: those with a gap at the first call (terminals start with U = L)
    isempty(r.nt) && append!(r.nt, findall(i -> U[i] - L[i] > 0, eachindex(U)))
    g, j = findmax([U[s] - L[s] for s in r.S0])
    s = r.S0[j]
    gs = [U[i] - L[i] for i in r.nt]
    vs = isempty(VISITS) ? "" : string(effshare(collect(values(VISITS)), length(r.nt)))
    println(r.io, join([r.alg, r.prob, r.upd, g, U[s], L[s], effshare(gs, length(r.nt)), vs], ","))
    nothing
end
(r::Rec)(vf, bu::Integer) = r(vf, bu, nothing)

const BRTDP = Dict("sampling" => "trajectory_brtdp", "action_epsilon" => 0.02, "k" => 1,
    "transition_bound" => "lower", "state_temperature" => 1.0, "state_alpha_e" => 1.0,
    "state_exclude_converged" => true, "state_converged_eps" => 1e-6, "dedup" => true,
    "dedup_scope" => "rollout", "terminate" => ["max_steps"])
const ALGS = ["brtdp" => BRTDP, "exhaustive" => Dict("sampling" => "exhaustive_state"),
              "rr1" => Dict("sampling" => "round_robin", "k" => 1)]
const BOTH = get(ENV, "DIAG_DRIVE", "follow") == "both"
if BOTH
    BRTDP["initial_state"] = "max"
    foreach(((_, p),) -> p["drive"] = "both", ALGS)
end
const ONLY = split(get(ENV, "DIAG_ALGS", "brtdp,exhaustive,rr1"), ",")   # a subset of the arms
cfgpath = tempname() * ".toml"
open(cfgpath, "w") do io
    TOML.print(io, Dict("name" => "gap_spread",
        "property" => Dict("convergence_eps" => 1e-5, "restrict_to_initial" => true),
        "problems" => Dict("root" => joinpath(BENCH, "problems"), "names" => ARGS),
        "algorithms" => [Dict("name" => "gsrdp", "label" => a, "params" => p) for (a, p) in ALGS if a in ONLY]))
end
cfg = load_experiment_config(cfgpath)
const OUT = mkpath(joinpath(BENCH, "results", BOTH ? "gap_spread_driveboth" : "gap_spread"))
out = joinpath(OUT, "trace.csv")
io = open(out, isfile(out) ? "a" : "w")
filesize(out) == 0 && println(io, "alg,problem,updates,gap0,U0,L0,gap_share,visit_share")
for ps in cfg.problems
    bp = build_problem(ps.name, ps.params)
    ok, bp = Bench.override_specification(bp, cfg)
    S0 = Int.(Bench.problem_initial_states(bp))
    for as in cfg.algorithms
        built = build_algorithm(as.name, bp, as.params)
        Random.seed!(20260922); empty!(VISITS)
        r = Rec(S0, Int[], io, as.label, ps.instance_name, 1.0, 0, nothing, -1)
        t = @elapsed solve(bp.problem, built.alg; callback=r)
        r.logged == r.upd || logrow(r)   # always the final state of the run, at its last update
        flush(io)
        @info "done" ps.instance_name as.label r.upd t
    end
end
close(io)
