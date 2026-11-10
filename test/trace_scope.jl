# Regression test for the executor's scoped gap (`_gap_scope` in src/trace.jl).
#
# When every trace that reads the per-state gap is an initial-state trace, the executor
# fills in only the initial states' entries of its gap buffer. These tests pin that it
# then records exactly what it records when an all-states trace forces the full gap, and
# that any all-states reader turns the scope off.
#
# Needs the solver stack, so it lives in its own file (like test/update_trace.jl):
#
#   julia --project=. test/trace_scope.jl

using Test
using Random
using IntervalMDP

include(joinpath(@__DIR__, "..", "src", "Bench.jl"))
using .Bench

# The 3-state reach MDP of test/update_trace.jl (every action from every state; action 3
# reaches the absorbing target 3 with probability 1, so the gap closes in ~60 iterations),
# with two initial states.
function _problem()
    p = IntervalMDP.IntervalAmbiguitySets(;
        lower = [0 0.5 0; 0.1 0.3 0; 0.2 0.1 1.0],
        upper = [0.5 0.7 0; 0.6 0.5 0; 0.7 0.3 1.0])
    mdp = IntervalMDP.IntervalMarkovDecisionProcess([p, p, p], [1, 2])
    spec = IntervalMDP.Specification(IntervalMDP.InfiniteTimeReachability([3], 1e-6),
                                     IntervalMDP.Pessimistic, IntervalMDP.Maximize)
    return mdp, IntervalMDP.VerificationProblem(mdp, spec)
end

# `timeout_s` caps each solve: should a solve ever fail to converge, it is censored
# instead of growing the trace without bound.
function _solve(kinds; track_initial = true, sampling = IntervalMDP.RoundRobinState(1))
    mdp, problem = _problem()
    Random.seed!(11)
    alg = IntervalMDP.GeneralizedSamplingbasedRobustDynamicProgramming(
        IntervalMDP.default_bellman_algorithm(mdp); sampling_strategy = sampling)
    ex = Bench.build_trace_executor(kinds, Int32[1, 2]; track_initial, timeout_s = 30.0)
    IntervalMDP.solve(problem, alg; callback = ex)
    return ex
end

_initial(ex) = only(filter(t -> t isa Bench.InitialTrace, collect(ex.traces)))

@testset "trace gap scope" begin
    @testset "initial-only executors fill in only the initial states" begin
        @test _solve(String[]).gap_scope == [1, 2]
        @test _solve(["initial_gap"]).gap_scope == [1, 2]
    end

    @testset "any all-states gap reader turns the scope off" begin
        @test _solve(["trace"]).gap_scope === nothing
        @test _solve(["gap"]).gap_scope === nothing
        # No gap reader at all: nothing to scope.
        @test _solve(["value"]; track_initial = false).gap_scope === nothing
    end

    @testset "the scoped executor records the same initial-state trace" begin
        scoped, full = _solve(String[]), _solve(["trace"])
        a, b = _initial(scoped), _initial(full)
        @test length(a.gap_max) > 2
        @test a.gap_max == b.gap_max
        @test a.gap_mean == b.gap_mean
        @test a.bellman_updates == b.bellman_updates
        @test Bench.final_gap(scoped) == (a.gap_max[end], a.gap_mean[end])
        @test Bench.final_iteration_count(scoped) == Bench.final_iteration_count(full)

        g1, g2 = _solve(["initial_gap"]), _solve(["initial_gap", "trace"])
        t1 = only(filter(t -> t isa Bench.InitialGapTrace, collect(g1.traces)))
        t2 = only(filter(t -> t isa Bench.InitialGapTrace, collect(g2.traces)))
        @test t1.gaps == t2.gaps
    end
end
