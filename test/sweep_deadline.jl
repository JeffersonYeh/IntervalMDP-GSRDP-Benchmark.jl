# Regression test: the sweep deadline caps EVERY solve in a cell, not just the first.
#
# The harness used to turn the time left before `SLURM_JOB_END_TIME - wall_reserve_s` into a
# duration once per cell and hand that same duration to each sample. A sample starting after
# earlier ones had finished then got the full budget again, outlived the job, and SLURM's kill
# took the whole cell with it -- finished samples and the partial curve alike. Each solve must
# instead be clamped to the time left when IT starts, so the in-flight one is censored with its
# partial trace and later ones are never started.
#
# A fake solver with a fixed per-solve duration drives the real `run_sampled_solve`, so the
# timing is deterministic. Needs IntervalMDP (for a real value function), not the full Bench
# stack:
#
#   julia --project=. test/sweep_deadline.jl

using Test
using IntervalMDP
using BenchmarkTools
using Random
using Statistics

include(joinpath(@__DIR__, "..", "src", "problem_provider.jl"))
include(joinpath(@__DIR__, "..", "src", "registry.jl"))
include(joinpath(@__DIR__, "..", "src", "thinning.jl"))
include(joinpath(@__DIR__, "..", "src", "trace.jl"))
include(joinpath(@__DIR__, "..", "src", "config.jl"))
include(joinpath(@__DIR__, "..", "src", "benchmark.jl"))

const PROBLEM = joinpath(@__DIR__, "..", "problems", "storm_tiny01")
const PROB = IntervalMDP.Data.read_intervalmdp_jl(joinpath(PROBLEM, "storm_tiny01.nc"),
                                                  joinpath(PROBLEM, "storm_tiny01.json"))

"A solve that takes `iters × dt` seconds, calling back once per iteration."
struct SlowAlg
    iters::Int
    dt::Float64
end

function IntervalMDP.solve(prob::typeof(PROB), a::SlowAlg; callback)
    vf = IntervalMDP.IntervalValueFunction(IntervalMDP.StateValueFunction(prob, IntervalMDP.Lower),
                                           IntervalMDP.StateValueFunction(prob, IntervalMDP.Upper))
    for i in 1:a.iters
        sleep(a.dt)
        callback(vf, i)
    end
    return :solved
end

@testset "sweep deadline" begin
    @testset "_clamp_to_deadline" begin
        @test _clamp_to_deadline(Inf, Inf) == Inf
        @test _clamp_to_deadline(5.0, Inf) == 5.0
        @test _clamp_to_deadline(Inf, time() - 1.0) == 0.0          # past: censor at once
        @test 9.0 < _clamp_to_deadline(Inf, time() + 10.0) <= 10.0
        @test _clamp_to_deadline(2.0, time() + 10.0) == 2.0
    end

    # Warm the solve + BenchmarkTools path so compilation does not eat into the budgets below.
    run_sampled_solve(PROB, SlowAlg(2, 0.01), [1], Int[], ["trace"], false)

    @testset "a later sample is censored at the deadline, not given a fresh budget" begin
        alg = SlowAlg(10, 0.1)                     # ~1.0 s per solve
        t0 = time()
        trial, sols, traces, walls, aborted =
            run_sampled_solve(PROB, alg, [1, 2, 3, 4], Int[], ["trace"], false;
                              timeout_s=Inf, sweep_deadline=t0 + 2.5)
        took = time() - t0

        # Samples 1-2 finish (~2.0 s); sample 3 starts with ~0.5 s left and is censored.
        # Given a fresh 2.5 s, as before the fix, it would have finished instead.
        @test length(sols) == 2
        @test length(traces) == 3                  # the censored sample's trace is kept
        @test aborted isa CellTimeout
        @test took < 2.5 + 0.5                     # stopped at the deadline, not after
        @test has_records(traces[3])               # ... and it holds a partial curve
        @test 0 < last(traces[3].traces[1].bellman_updates) < alg.iters
        @test length(walls) == 2                   # completed samples keep their timings
    end

    @testset "per-solve timeout_s still applies when it is the tighter bound" begin
        trial, sols, traces, walls, aborted =
            run_sampled_solve(PROB, SlowAlg(10, 0.1), [1, 2], Int[], ["trace"], false;
                              timeout_s=0.35, sweep_deadline=time() + 100.0)
        @test isempty(sols)
        @test aborted isa CellTimeout
        @test aborted.timeout_s == 0.35
    end

    @testset "no deadline, no timeout: every sample completes" begin
        trial, sols, traces, walls, aborted =
            run_sampled_solve(PROB, SlowAlg(3, 0.01), [1, 2], Int[], ["trace"], false)
        @test length(sols) == 2
        @test aborted === nothing
    end
end
