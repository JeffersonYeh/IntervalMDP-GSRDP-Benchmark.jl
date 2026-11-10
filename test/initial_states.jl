# Regression test for the silent `restrict_to_initial` no-op.
#
# A problem whose model declares `AllStates()` instead of a real initial-state set makes
# `restrict_to_initial = true` do nothing at all: IntervalMDP's `apply_initial_restriction`
# (src/termination.jl:38-42) returns the unwrapped all-states criterion, so the solve converges
# over every state while the results claim otherwise, and the benchmark's `initial_gap_*` trace
# columns come out NaN on every row. Nothing in the output distinguishes that from a correct
# restricted run, which is why it needs a test rather than an eyeball.
#
# Unlike runtests.jl these need the solver stack (they read .nc models and build termination
# criteria), so they live in their own file:
#
#   julia --project=. test/initial_states.jl

using Test
using IntervalMDP

const PROBLEMS = joinpath(@__DIR__, "..", "problems")

"Problem directories holding a native .nc + .json pair (the PRISM models are converted to this too)."
function nc_problems()
    isdir(PROBLEMS) || return String[]
    return filter(readdir(PROBLEMS)) do name
        isfile(joinpath(PROBLEMS, name, string(name, ".nc")))
    end
end

load(name) = IntervalMDP.Data.read_intervalmdp_jl(
    joinpath(PROBLEMS, name, string(name, ".nc")),
    joinpath(PROBLEMS, name, string(name, ".json")),
)

"The GSRDP algorithm as `src/algorithms.jl` builds it, with the default full sweep."
gsrdp(prob) = GeneralizedSamplingbasedRobustDynamicProgramming(
    IntervalMDP.RobustBellman();
    sampling_strategy = IntervalMDP.ExhaustiveState(),
)

"Rebuild `prob`'s spec with `restrict_to_initial`, the way `override_specification` does."
function with_restriction(prob, rti::Bool)
    spec = IntervalMDP.specification(prob)
    prop = IntervalMDP.system_property(spec)
    prop_new = IntervalMDP.InfiniteTimeReachAvoid(
        IntervalMDP.reach(prop), IntervalMDP.avoid(prop),
        IntervalMDP.convergence_eps(prop), rti,
    )
    return IntervalMDP.Specification(prop_new, IntervalMDP.satisfaction_mode(spec),
                                     IntervalMDP.strategy_mode(spec))
end

@testset "initial states" begin
    names = nc_problems()
    @test !isempty(names)   # a green run over zero problems would prove nothing

    @testset "$name" for name in names
        prob = load(name)
        sys = IntervalMDP.system(prob)
        S0 = IntervalMDP.initial_states(sys)

        # The .nc round-trip turns an empty initial-state set into `AllStates()`
        # (IntervalMDP src/Data/intervalmdp.jl:36-39, 225-228), so this is the state a problem
        # generated without an INITIAL_REGION lands in — and it cannot be recovered afterwards.
        @test !(S0 isa IntervalMDP.AllStates)
        @test !isempty(S0)

        # The actual claim: with the flag on, the solver must terminate on the initial-state
        # gap, not the all-states gap.
        tc = IntervalMDP.termination_criteria(gsrdp(prob), with_restriction(prob, true), sys)
        @test tc isa IntervalMDP.InitialStateCriteria

        # ...and with it off, it must not.
        tc_off = IntervalMDP.termination_criteria(gsrdp(prob), with_restriction(prob, false), sys)
        @test !(tc_off isa IntervalMDP.InitialStateCriteria)
    end
end
