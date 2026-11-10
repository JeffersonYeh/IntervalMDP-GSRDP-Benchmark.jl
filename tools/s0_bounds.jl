#!/usr/bin/env julia
# U and L at every initial state during a solve, for problems with several initial states: the curves
# of gap_spread.jl are read at the state with the largest gap, which changes during a BRTDP run, so
# they jump between states although each state's bounds are monotone (report-plots
# experiments-brtdp-carparking.jl, the S20 panel).
#
#   julia --project=. tools/s0_bounds.jl <problem>...   (one seed, 20260922; seconds per problem)
#
# Runs the drive = "both" arms of configs/thesis/{rivi,arivi-k1,brtdp}.toml and overwrites
# results/s0_bounds_driveboth/trace.csv with one row per initial state (alg, problem, updates, s0, U, L,
# holds_max_gap) at each logged update, the last at the final update of the run. BRTDP logs whenever a
# bound at an initial state changes, so its values hold exactly from one logged update to the next; the
# baselines log at log-spaced updates, like gap_spread.jl.
using Random, TOML
const BENCH = normpath(joinpath(@__DIR__, ".."))
include(joinpath(BENCH, "src", "Bench.jl"))
using .Bench
using IntervalMDP

mutable struct Rec
    S0::Vector{Int}; io::IO; alg::String; prob::String; next::Float64; upd::Int
    vf::Any; logged::Int                 # the latest value function, the updates of the last row
    on_change::Bool; last::Vector{Float64}   # log on any change at S0; U and L there at the last row
end
s0_bounds(r::Rec) = [x[s] for x in (r.vf.upper.current, r.vf.lower.current) for s in r.S0]
function (r::Rec)(vf, bu::Integer, _=nothing)
    r.upd = Int(bu); r.vf = vf
    if r.on_change
        s0_bounds(r) == r.last || logrow(r)
        return nothing
    end
    r.upd < r.next && return nothing
    r.next = max(r.next * 1.03, r.upd + 1)
    logrow(r)
end
function logrow(r::Rec)
    r.logged = r.upd; r.last = s0_bounds(r)
    U = r.vf.upper.current; L = r.vf.lower.current
    _, j = findmax([U[s] - L[s] for s in r.S0])
    for (i, s) in enumerate(r.S0)
        println(r.io, join([r.alg, r.prob, r.upd, s, U[s], L[s], i == j], ","))
    end
end

const BRTDP = Dict("sampling" => "trajectory_brtdp", "action_epsilon" => 0.02, "k" => 1,
    "transition_bound" => "lower", "state_temperature" => 1.0, "state_alpha_e" => 1.0,
    "state_exclude_converged" => true, "state_converged_eps" => 1e-6, "dedup" => true,
    "dedup_scope" => "rollout", "terminate" => ["max_steps"], "initial_state" => "max", "drive" => "both")
const ALGS = ["brtdp" => BRTDP, "exhaustive" => Dict("sampling" => "exhaustive_state", "drive" => "both"),
              "rr1" => Dict("sampling" => "round_robin", "k" => 1, "drive" => "both")]
cfgpath = tempname() * ".toml"
open(cfgpath, "w") do io
    TOML.print(io, Dict("name" => "s0_bounds",
        "property" => Dict("convergence_eps" => 1e-5, "restrict_to_initial" => true),
        "problems" => Dict("root" => joinpath(BENCH, "problems"), "names" => ARGS),
        "algorithms" => [Dict("name" => "gsrdp", "label" => a, "params" => p) for (a, p) in ALGS]))
end
cfg = load_experiment_config(cfgpath)
out = joinpath(mkpath(joinpath(BENCH, "results", "s0_bounds_driveboth")), "trace.csv")
open(out, "w") do io
    println(io, "alg,problem,updates,s0,U,L,holds_max_gap")
    for ps in cfg.problems
        bp = build_problem(ps.name, ps.params)
        ok, bp = Bench.override_specification(bp, cfg)
        S0 = Int.(Bench.problem_initial_states(bp))
        for as in cfg.algorithms
            built = build_algorithm(as.name, bp, as.params)
            Random.seed!(20260922)
            r = Rec(S0, io, as.label, ps.instance_name, 1.0, 0, nothing, -1, as.label == "brtdp", Float64[])
            solve(bp.problem, built.alg; callback=r)
            r.logged == r.upd || logrow(r)
            println(ps.instance_name, " ", as.label, " updates=", r.upd)
        end
    end
end
