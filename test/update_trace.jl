# Regression tests for `UpdateTrace` — the per-iteration record of WHICH states the
# solver backed up.
#
# This is the only trace that depends on the solver handing the callback a third
# argument (IntervalMDP `src/gsrdp.jl`, `_invoke_callback`). Everything else in the trace
# stack works off the value function alone, so nothing else would notice if that hook
# regressed to the two-argument form — the counts would silently become all-zero rows and
# the videos would show an empty updates panel that reads as "the sampler never ran".
# Hence a test that pins the hook itself, not just the bookkeeping.
#
# Unlike runtests.jl these need the solver stack, so they live in their own file
# (matching test/initial_states.jl):
#
#   julia --project=. test/update_trace.jl

using Test
using Random
using IntervalMDP

include(joinpath(@__DIR__, "..", "src", "Bench.jl"))
using .Bench

# A 3-state reach-avoid MDP: state 3 absorbing and the reach target. Float64, not
# Rational: exact rationals make the solver iterate in Rational{Int64}, whose
# denominators overflow long before the ~60 iterations a 1e-6 tolerance takes.
function _tiny_problem()
    p = IntervalMDP.IntervalAmbiguitySets(;
        lower = [0 0.5 0; 0.1 0.3 0; 0.2 0.1 1.0],
        upper = [0.5 0.7 0; 0.6 0.5 0; 0.7 0.3 1.0])
    mdp = IntervalMDP.IntervalMarkovDecisionProcess([p, p, p], [1])
    spec = IntervalMDP.Specification(IntervalMDP.InfiniteTimeReachability([3], 1e-6),
                                     IntervalMDP.Pessimistic, IntervalMDP.Maximize)
    return mdp, IntervalMDP.VerificationProblem(mdp, spec)
end

function _solve_with(kinds, sampling; seed=7)
    mdp, problem = _tiny_problem()
    Random.seed!(seed)
    alg = IntervalMDP.GeneralizedSamplingbasedRobustDynamicProgramming(
        IntervalMDP.default_bellman_algorithm(mdp); sampling_strategy=sampling)
    ex = Bench.build_trace_executor(kinds, Int[])
    IntervalMDP.solve(problem, alg; callback=ex)
    return mdp, ex
end

_update_trace(ex) = only(filter(t -> t isa Bench.UpdateTrace, collect(ex.traces)))

@testset "UpdateTrace" begin

    @testset "\"updates\" is a registered trace kind" begin
        @test Bench.known_trace_kind("updates")
        @test "updates" in Bench.trace_kinds()
        @test Bench.build_trace("updates", Int[]) isa Bench.UpdateTrace
    end

    @testset "a full sweep backs up every state exactly once per iteration" begin
        mdp, ex = _solve_with(["updates"], IntervalMDP.ExhaustiveState())
        t = _update_trace(ex)
        @test length(t.counts) > 1
        # Row 1 is the pre-update callback: nothing has been sampled yet.
        @test all(iszero, t.counts[1])
        @test all(c -> c == ones(Int32, IntervalMDP.num_states(mdp)), t.counts[2:end])
    end

    @testset "a sampled variant backs up only k states, duplicates included" begin
        mdp, ex = _solve_with(["updates"], IntervalMDP.RandomSubsetState(2))
        t = _update_trace(ex)
        @test all(iszero, t.counts[1])
        # `RandomSubsetState(k)` draws k states iid, so a state can come up twice in one
        # iteration and really is relaxed twice. The total is always k; the per-state
        # count is not always 0 or 1, which is exactly what the updates panel shows.
        @test all(c -> sum(c) == 2, t.counts[2:end])
        @test any(c -> any(>(1), c), t.counts[2:end])
        @test any(c -> any(iszero, c), t.counts[2:end])
    end

    @testset "counts reconcile with the solver's own bellman_updates" begin
        # `_bellman_update_count` is `|state_seq| * (num_actions + 1)`, and `sum(counts)`
        # is `|state_seq|`. Tying the two together is what proves the trace is counting
        # the sequence the solver actually swept rather than something adjacent to it.
        for ss in (IntervalMDP.ExhaustiveState(), IntervalMDP.RandomSubsetState(2))
            mdp, ex = _solve_with(["updates"], ss)
            t = _update_trace(ex)
            per_state = IntervalMDP.num_actions(mdp) + 1
            @test t.bellman_updates[1] == 0
            @test diff(t.bellman_updates) ==
                  [sum(t.counts[i]) * per_state for i in 2:length(t.counts)]
        end
    end

    @testset "rows line up with the other heavy traces" begin
        # The videos overlay these files frame by frame, so every trace in one executor
        # must record the same number of iterations against the same update counts.
        _, ex = _solve_with(["trace", "gap", "value", "updates"],
                            IntervalMDP.RandomSubsetState(2))
        traces = collect(ex.traces)
        n = Bench.nrecorded(first(traces))
        @test n > 1
        @test all(Bench.nrecorded(t) == n for t in traces)
        @test all(t.bellman_updates == first(traces).bellman_updates for t in traces)
    end

    @testset "final_gap still comes from the gap traces, not the update trace" begin
        _, ex = _solve_with(["trace", "updates"], IntervalMDP.ExhaustiveState())
        @test Bench.final_gap(_update_trace(ex)) === (NaN, NaN)  # carries no value info
        gmax, gmean = Bench.final_gap(ex)                        # ...so the scan skips it
        @test !isnan(gmax) && !isnan(gmean)
    end

    @testset "CSV round-trip preserves the counts" begin
        _, ex = _solve_with(["updates"], IntervalMDP.RandomSubsetState(2))
        t = _update_trace(ex)
        dir = mktempdir()
        path = joinpath(dir, Bench.trace_filename(t))
        @test basename(path) == "update_trace.csv"
        Bench.write_trace_csv(t, path)

        lines = collect(eachline(path))
        @test lines[1] == "wallclock_s,bellman_updates,n[1],n[2],n[3]"
        @test length(lines) == length(t.counts) + 1
        for (i, c) in enumerate(t.counts)
            fields = split(lines[i + 1], ',')
            @test [parse(Int, f) for f in fields[3:end]] == Int.(c)
        end
    end

    @testset "a two-argument-only solver yields zero rows, not an error" begin
        # The executor keeps its two-argument form for callers that never pass a
        # sequence (RVI, and any solver predating the hook). An UpdateTrace then records
        # a zero row rather than failing the whole run.
        _, problem = _tiny_problem()
        ex = Bench.build_trace_executor(["updates"], Int[])
        vf = IntervalMDP.IntervalValueFunction(
            IntervalMDP.StateValueFunction(problem, IntervalMDP.Lower),
            IntervalMDP.StateValueFunction(problem, IntervalMDP.Upper))
        ex(vf, 0)
        t = _update_trace(ex)
        @test length(t.counts) == 1 && all(iszero, t.counts[1])
    end
end

# `[output].trace_thin_rtol` thins the scalar traces while they record (TraceExecutor in
# src/trace.jl). The two scalar traces must stay row-aligned — `write_report_trace_csv`
# pairs them by index — and everything the harness reads off the executor afterwards
# must be what an unthinned run reports.
@testset "scalar-trace thinning in the executor" begin
    function _solve_thinned(rtol)
        mdp, problem = _tiny_problem()
        alg = IntervalMDP.GeneralizedSamplingbasedRobustDynamicProgramming(
            IntervalMDP.default_bellman_algorithm(mdp);
            sampling_strategy=IntervalMDP.ExhaustiveState())
        ex = Bench.build_trace_executor(["trace", "updates"], [1]; track_initial=true,
                                        thin_rtol=rtol)
        IntervalMDP.solve(problem, alg; callback=ex)
        return ex
    end
    scalar(ex, T) = only(filter(t -> t isa T, collect(ex.traces)))

    full = _solve_thinned(0.0)
    # The gap shrinks 20% per iteration, so any rtol below that keeps every row (each is an
    # anchor or its corner); 0.6 spans several iterations and actually drops some.
    thin = _solve_thinned(0.6)
    tf, tt = scalar(full, Bench.Trace), scalar(thin, Bench.Trace)
    it = scalar(thin, Bench.InitialTrace)

    @test full.thin === nothing
    @test length(tt.gap_max) < length(tf.gap_max)
    # the two scalar traces stay aligned, row for row
    @test length(it.gap_max) == length(tt.gap_max)
    @test it.bellman_updates == tt.bellman_updates && it.wallclock == tt.wallclock
    # kept rows are a subsequence of the full trace, with its first and last rows
    @test issubset(tt.bellman_updates, tf.bellman_updates)
    @test first(tt.bellman_updates) == first(tf.bellman_updates)
    @test last(tt.bellman_updates) == last(tf.bellman_updates)
    @test tt.gap_max == tf.gap_max[indexin(tt.bellman_updates, tf.bellman_updates)]
    # what the harness reads off the executor is unchanged
    @test Bench.final_gap(thin) == Bench.final_gap(full)
    @test Bench.final_bellman_updates(thin) == Bench.final_bellman_updates(full)
    @test Bench.final_iteration_count(thin) == Bench.final_iteration_count(full)
    @test Bench.final_iteration_count(full) == length(tf.gap_max) - 1
    # the matrix trace is never thinned
    @test length(_update_trace(thin).counts) == length(tf.gap_max)

    # the report CSV carries only the kept rows
    path = joinpath(mktempdir(), "trace.csv")
    Bench.write_report_trace_csv(thin, path)
    @test countlines(path) == length(tt.gap_max) + 1
end
