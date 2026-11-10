# Diagnostic traces of BRTDP (tune6 brtdp_rw6_k1) against the exhaustive sweep and round robin (k = 1):
#   julia --project=. tools/brtdp_diagnostics.jl <problem>...   (one seed, 20260922; seconds per problem)
#   DIAG_DRIVE=both julia --project=. tools/brtdp_diagnostics.jl <problem>...
#       the arms of configs/thesis/{rivi,arivi-k1,brtdp}.toml instead: drive = "both" on all three, BRTDP with the
#       max-gap initial state (initial_state = "max"); writes results/brtdp_diagnostics_driveboth/
#   DIAG_DRIVE=both DIAG_TB=upper DIAG_MAX_S=1500 DIAG_ALGS=brtdp julia --project=. tools/brtdp_diagnostics.jl <problem>...
#       BRTDP with transition_bound = DIAG_TB (default lower), censored after DIAG_MAX_S seconds per
#       solve (default none, the last row is the state at the censoring point); with DIAG_TB set
#       writes results/brtdp_diagnostics_tb_<bound>/
# Appends (delete the files first to rerun a problem) to results/brtdp_diagnostics/trace.csv
# log-spaced rows, the last one at the final update of the run, of (updates, max initial-state gap,
# the initial state holding it, U and L there, mean U and L over the initial states, rollouts, rollout
# steps, distinct states visited) and one line per solve to summary.txt (rollouts, mean rollout length,
# visited states, rollout starts per initial state). Used for the thesis text on car_parking_S10/S20 and
# linear_stochastically_switched_S20/S40 (report-plots experiments-brtdp.jl, experiments_brtdp_selected).
using Random, Printf, TOML, CSV, DataFrames, Statistics
const BENCH = normpath(joinpath(@__DIR__, ".."))
include(joinpath(BENCH, "src", "Bench.jl"))
using .Bench
using IntervalMDP
const TS = IntervalMDP.TrajectorySampling
const RST = zeros(Int, 2)            # rollouts, states
const VISITED = Set{Any}()
const STARTS = Dict{Any,Int}()
function TS._sample_trajectory(ss::TS.TrajectorySampling, model, value_function, spec)
    t = TS._sample_trajectory(ss, model, value_function, spec, TS._rollout_scratch(ss, model))
    RST[1] += 1; RST[2] += length(t)
    union!(VISITED, t)
    isempty(t) || (STARTS[t[1]] = get(STARTS, t[1], 0) + 1)
    return t
end
mutable struct Rec
    S0::Vector{Int}; io::IO; alg::String; prob::String; next::Float64; upd::Int
    vf::Any; logged::Int                 # the latest value function, the updates of the last row
    below::Bool                          # the largest initial-state gap has dropped below 1
    deadline::Float64                    # time() after which the solve is censored
end
struct DiagTimeout <: Exception end
function (r::Rec)(vf, bu::Integer, _=nothing)
    r.upd = Int(bu); r.vf = vf
    time() > r.deadline && throw(DiagTimeout())
    # also a row at the exact update where the largest gap leaves 1, off the log-spaced schedule
    first = !r.below && maximum(vf.upper.current[s] - vf.lower.current[s] for s in r.S0) < 1
    first && (r.below = true)
    r.upd < r.next && !first && return nothing
    r.upd >= r.next && (r.next = max(r.next * 1.03, r.upd + 1))
    logrow(r)
end
function logrow(r::Rec)
    r.logged = r.upd
    U = r.vf.upper.current; L = r.vf.lower.current
    g, j = findmax([U[s] - L[s] for s in r.S0])
    s = r.S0[j]
    println(r.io, join([r.alg, r.prob, r.upd, g, s, U[s], L[s], mean(U[r.S0]), mean(L[r.S0]),
                        RST[1], RST[2], length(VISITED)], ","))
    nothing
end
(r::Rec)(vf, bu::Integer) = r(vf, bu, nothing)

const BOTH = get(ENV, "DIAG_DRIVE", "follow") == "both"
const TB = get(ENV, "DIAG_TB", "")                    # "", "lower" or "upper"
const MAX_S = parse(Float64, get(ENV, "DIAG_MAX_S", "Inf"))
const BRTDP = Dict("sampling" => "trajectory_brtdp", "action_epsilon" => 0.02, "k" => 1,
    "transition_bound" => isempty(TB) ? "lower" : TB, "state_temperature" => 1.0, "state_alpha_e" => 1.0,
    "state_exclude_converged" => true, "state_converged_eps" => 1e-6, "dedup" => true,
    "dedup_scope" => "rollout", "terminate" => ["max_steps"])
const ALGS = ["brtdp" => BRTDP, "exhaustive" => Dict("sampling" => "exhaustive_state"),
              "rr1" => Dict("sampling" => "round_robin", "k" => 1)]
if BOTH
    BRTDP["initial_state"] = "max"
    foreach(((_, p),) -> p["drive"] = "both", ALGS)
end
const ONLY = split(get(ENV, "DIAG_ALGS", "brtdp,exhaustive,rr1"), ",")   # a subset of the arms
problems = ARGS
cfgpath = tempname() * ".toml"
open(cfgpath, "w") do io
    TOML.print(io, Dict("name" => "diag",
        "property" => Dict("convergence_eps" => 1e-5, "restrict_to_initial" => true),
        "problems" => Dict("root" => joinpath(BENCH, "problems"), "names" => problems),
        "algorithms" => [Dict("name" => "gsrdp", "label" => a, "params" => p) for (a, p) in ALGS if a in ONLY]))
end
cfg = load_experiment_config(cfgpath)
const OUT = mkpath(haskey(ENV, "DIAG_OUT") ? ENV["DIAG_OUT"] :   # DIAG_OUT overrides the directory
                  joinpath(BENCH, "results", !isempty(TB) ? "brtdp_diagnostics_tb_$TB" :
                                             BOTH ? "brtdp_diagnostics_driveboth" : "brtdp_diagnostics"))
out = joinpath(OUT, "trace.csv")
io = open(out, isfile(out) ? "a" : "w")
filesize(out) == 0 && println(io, "alg,problem,updates,gap,s0,U,L,meanU,meanL,rollouts,states,visited")
summ = open(joinpath(OUT, "summary.txt"), "a")
for ps in cfg.problems
    bp = build_problem(ps.name, ps.params)
    ok, bp = Bench.override_specification(bp, cfg)
    S0 = Int.(Bench.problem_initial_states(bp))
    for as in cfg.algorithms
        built = build_algorithm(as.name, bp, as.params)
        Random.seed!(20260922); fill!(RST, 0); empty!(VISITED); empty!(STARTS)
        r = Rec(S0, io, as.label, ps.instance_name, 1.0, 0, nothing, -1, false, time() + MAX_S)
        censored = false
        t = @elapsed try
            solve(bp.problem, built.alg; callback=r)
        catch e
            e isa DiagTimeout || rethrow()
            censored = true
        end
        r.logged == r.upd || logrow(r)   # always the final state of the run, at its last update
        println(summ, ps.instance_name, " ", as.label, " updates=", r.upd, " wall=", round(t, digits=2), " censored=", censored,
                " rollouts=", RST[1], " meanlen=", round(RST[2] / max(RST[1], 1), digits=2),
                " visited=", length(VISITED), " starts=", sort(collect(values(STARTS))))
        flush(io); flush(summ)
        @info "done" ps.instance_name as.label r.upd t
    end
end
close(io); close(summ)
