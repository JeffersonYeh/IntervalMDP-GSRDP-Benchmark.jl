# Tests for the sampling-strategy resolvers in `src/registry.jl` — the layer
# that turns flat TOML values into IntervalMDP strategy objects.
#
# Unlike the analysis tests in `runtests.jl`, these do pull in IntervalMDP: the
# resolvers construct real strategy objects, and asserting against anything less
# would not check the thing that actually breaks when the library changes. They
# stop short of `Bench.jl` itself, so the plotting stack stays out.

using Test
using IntervalMDP

include(joinpath(@__DIR__, "..", "src", "problem_provider.jl"))
include(joinpath(@__DIR__, "..", "src", "registry.jl"))
include(joinpath(@__DIR__, "..", "src", "thinning.jl"))
include(joinpath(@__DIR__, "..", "src", "trace.jl"))
include(joinpath(@__DIR__, "..", "src", "config.jl"))

const TSamp = IntervalMDP.TrajectorySampling

@testset "resolve_sampling: non-trajectory strategies still resolve" begin
    @test resolve_sampling(Dict{String,Any}()) isa IntervalMDP.ExhaustiveState
    @test resolve_sampling(Dict{String,Any}("sampling" => "random_state", "k" => 8)) isa
          IntervalMDP.RandomSubsetState
    @test resolve_sampling(Dict{String,Any}("sampling" => "random_state_action", "k" => 8)) isa
          IntervalMDP.RandomSubsetStateActions
    @test resolve_sampling(Dict{String,Any}("sampling" => "round_robin", "k" => 8)) isa
          IntervalMDP.RoundRobinState
    @test resolve_sampling(
        Dict{String,Any}("sampling" => "round_robin_state_action", "k" => 8),
    ) isa IntervalMDP.RoundRobinStateActions
    # `GapPriorityQueueSampling` is a deprecated *constructor*, not a type: it
    # returns one configuration of `PrioritizedSweep`.
    @test resolve_sampling(Dict{String,Any}("sampling" => "priority_gap", "k" => 4)) isa
          IntervalMDP.PriorityQueueSampling.PrioritizedSweep
    @test_throws ErrorException resolve_sampling(Dict{String,Any}("sampling" => "nope"))

    # Every `k`-taking strategy must say so rather than silently defaulting.
    for name in
        ["random_state", "random_state_action", "round_robin", "round_robin_state_action"]
        @test_throws ErrorException resolve_sampling(Dict{String,Any}("sampling" => name))
    end
end

# `drive` picks which GSRDP bound optimizes; only a non-default one reaches the label.
@testset "resolve_bound_update / drive label" begin
    P(pairs...) = Dict{String,Any}(pairs...)
    @test resolve_bound_update(P()) isa IntervalMDP.FollowDrive
    @test resolve_bound_update(P("drive" => "follow")) isa IntervalMDP.FollowDrive
    @test resolve_bound_update(P("drive" => "Both")) isa IntervalMDP.BothDrive
    @test_throws ErrorException resolve_bound_update(P("drive" => "upper"))
    @test_throws ErrorException resolve_bound_update(P("drive" => "lower"))
    @test_throws ErrorException resolve_bound_update(P("drive" => "uppermono"))

    base = P("sampling" => "round_robin", "k_frac" => 0.05)
    @test algo_label("gsrdp", base) == "gsrdp_round_robin_kf0.05"
    @test algo_label("gsrdp", merge(base, P("drive" => "follow"))) == "gsrdp_round_robin_kf0.05"
    @test algo_label("gsrdp", merge(base, P("drive" => "both"))) ==
          "gsrdp_round_robin_kf0.05_driveboth"
    @test algo_label("gsrdp", P("sampling" => "exhaustive_state", "drive" => "both")) ==
          "gsrdp_exhaustive_state_driveboth"
end

# `k_frac` is resolved per problem into the absolute `k` the samplers take.
@testset "resolve_k_frac: size-relative k" begin
    P(pairs...) = Dict{String,Any}(pairs...)

    # No k_frac: untouched (the same object).
    p = P("sampling" => "round_robin", "k" => 8)
    @test resolve_k_frac(p, 100, 4) === p

    # ceil, never below 1; k_frac removed, other keys kept.
    r = resolve_k_frac(P("sampling" => "round_robin", "k_frac" => 0.25), 101, 9)
    @test r["k"] == 26 && !haskey(r, "k_frac") && r["sampling"] == "round_robin"
    @test resolve_k_frac(P("sampling" => "random_state", "k_frac" => 0.05), 3, 1)["k"] == 1
    @test resolve_k_frac(P("sampling" => "random_state", "k_frac" => 0.5), 3, 1)["k"] == 2
    @test resolve_k_frac(P("sampling" => "random_state", "k_frac" => 1.0), 513, 36)["k"] == 513

    # State-action samplers take a share of the (state, action) pairs.
    @test resolve_k_frac(P("sampling" => "random_state_action", "k_frac" => 0.1), 100, 4)["k"] == 40
    @test resolve_k_frac(P("sampling" => "round_robin_state_action", "k_frac" => 0.5), 10, 3)["k"] == 15

    # The resolved params still resolve to the right strategy.
    @test resolve_sampling(resolve_k_frac(P("sampling" => "round_robin", "k_frac" => 0.25), 101, 9)) isa
          IntervalMDP.RoundRobinState

    # Trajectory sampling: k_frac sets the Gauss-Seidel batch size, a share of the states.
    traj = resolve_k_frac(P("sampling" => "trajectory", "gauss_seidel" => true, "k_frac" => 0.1), 101, 9)
    @test traj["k"] == 11 && !haskey(traj, "k_frac")
    ts = resolve_sampling(traj)
    @test ts.gauss_seidel && ts.k == 11
    @test algo_label("gsrdp", P("sampling" => "trajectory", "gauss_seidel" => true, "k_frac" => 0.1)) ==
          "gsrdp_trajectory_kf0.1"

    # Rejections: both k and k_frac, out of range, trajectory k_frac without batching.
    @test_throws ErrorException resolve_k_frac(P("sampling" => "round_robin", "k" => 4, "k_frac" => 0.5), 10, 1)
    @test_throws ErrorException resolve_k_frac(P("sampling" => "round_robin", "k_frac" => 0.0), 10, 1)
    @test_throws ErrorException resolve_k_frac(P("sampling" => "round_robin", "k_frac" => 1.5), 10, 1)
    @test_throws ErrorException resolve_k_frac(P("sampling" => "trajectory", "k_frac" => 0.5), 10, 1)

    # The label carries the fraction, not a per-problem k.
    @test algo_label("gsrdp", P("sampling" => "random_state", "k_frac" => 0.25)) == "gsrdp_random_state_kf0.25"
    @test algo_label("gsrdp", P("sampling" => "round_robin", "k_frac" => 0.05)) == "gsrdp_round_robin_kf0.05"
end

# The library renamed `AllStatesSweep`/`AllSampling` to `ExhaustiveState`/
# `ExhaustiveStateActions`; the config vocabulary accepts both spellings so
# banked configs keep resolving.
@testset "resolve_sampling: exhaustive spellings are interchangeable" begin
    for name in ["all_states_sweep", "sweep", "all_sweep", "exhaustive_state"]
        @test resolve_sampling(Dict{String,Any}("sampling" => name)) isa
              IntervalMDP.ExhaustiveState
    end
    for name in ["all", "all_sampling", "exhaustive_state_action", "exhaustive_state_actions"]
        @test resolve_sampling(Dict{String,Any}("sampling" => name)) isa
              IntervalMDP.ExhaustiveStateActions
    end
end

@testset "removed trajectory strategies give a migration error" begin
    for name in [
        "trajectory_transition",
        "trajectory_probability",
        "trajectory_gap",
        "trajectory_expected_gap",
        "trajectory_action_uncertainty",
        "trajectory_reach",
        "trajectory_reach_probability",
    ]
        err = try
            resolve_sampling(Dict{String,Any}("sampling" => name))
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        # The message must say what to do, not just that the name is unknown.
        @test occursin("was removed", err.msg)
        @test occursin("sampling='trajectory'", err.msg)
        @test occursin("NOT directly comparable", err.msg)
    end
end

@testset "sampling = \"trajectory\" with defaults" begin
    ss = resolve_sampling(Dict{String,Any}("sampling" => "trajectory"))
    @test ss isa TSamp.TrajectorySampling
    # Library defaults, including the non-zero epsilons that keep the rollout
    # from locking onto a cycle.
    @test ss.action_policy == TSamp.EpsilonGreedy(0.1)
    @test ss.state_policy == TSamp.EpsilonGreedy(0.1)
    @test ss.action_score isa TSamp.UpperBoundScore
    # Under epsilon_greedy only the mixture's exploit bound is read.
    @test ss.state_score isa TSamp.MixtureStateScore
    @test ss.state_score.exploit === IntervalMDP.Upper
    @test ss.transition.bound === IntervalMDP.Upper
    @test ss.transition.adversary === nothing
    @test ss.gauss_seidel == false
    @test ss.reverse == true
    @test ss.dedup == false
    @test ss.initial_state === :random
    @test length(ss.terminate) == 1 && ss.terminate[1] isa TSamp.MaxSteps
    @test !ss.exclude_visited && !ss.exclude_converged && ss.converged_eps === nothing
end

@testset "successor filters" begin
    st(; kw...) = resolve_sampling(
        Dict{String,Any}("sampling" => "trajectory", (String(k) => v for (k, v) in kw)...),
    )
    @test st(state_exclude_visited = true).exclude_visited
    c = st(state_exclude_converged = true)
    @test c.exclude_converged && c.converged_eps === nothing   # -> the property's eps
    @test st(state_exclude_converged = true, state_converged_eps = 1e-4).converged_eps == 1e-4
    # Policy-independent: also under boltzmann.
    b = st(state_policy = "boltzmann", state_exclude_visited = true, state_exclude_converged = true)
    @test b.exclude_visited && b.exclude_converged
    # eps only under state_exclude_converged = true.
    @test_throws ErrorException st(state_converged_eps = 1e-4)
    @test_throws ErrorException st(state_exclude_converged = false, state_converged_eps = 1e-4)
    @test_throws ArgumentError st(state_exclude_converged = true, state_converged_eps = -1.0)
end

@testset "dedup" begin
    st(; kw...) = resolve_sampling(
        Dict{String,Any}("sampling" => "trajectory", (String(k) => v for (k, v) in kw)...),
    )
    @test st(dedup = true).dedup
    @test st(gauss_seidel = true, k = 4, dedup = true).dedup
    @test !resolve_sampling(Dict{String,Any}("sampling" => "trajectory_brtdp", "k" => 8)).dedup
    @test resolve_sampling(
        Dict{String,Any}("sampling" => "trajectory_brtdp", "k" => 8, "dedup" => true),
    ).dedup
end

@testset "initial_state" begin
    st(; kw...) = resolve_sampling(
        Dict{String,Any}("sampling" => "trajectory", (String(k) => v for (k, v) in kw)...),
    )
    @test st(initial_state = "max").initial_state === :max
    @test st(initial_state = "Random").initial_state === :random
    @test resolve_sampling(
        Dict{String,Any}("sampling" => "trajectory_brtdp", "k" => 8, "initial_state" => "max"),
    ).initial_state === :max
    @test_throws ArgumentError st(initial_state = "min")
end

@testset "trajectory_brtdp preset" begin
    P(kv...) = Dict{String,Any}("sampling" => "trajectory_brtdp", kv...)
    @test haskey(TRAJECTORY_PRESETS, "trajectory_brtdp")

    ss = resolve_sampling(P("k" => 8))
    @test ss isa TSamp.TrajectorySampling
    # Draw ∝ p·(U − L): the power rule at T = 1 over the pure explore term.
    @test ss.state_policy == TSamp.Boltzmann(1.0)
    @test ss.state_score isa TSamp.LogStateScore
    m = ss.state_score.inner
    @test m isa TSamp.MixtureStateScore
    @test m.explore isa TSamp.GapExplore
    @test m.alpha_explore == TSamp.FixedAlpha(1.0)
    @test m.alpha_prob == TSamp.FixedAlpha(0.0)
    @test length(ss.terminate) == 2
    @test ss.terminate[1] isa TSamp.MaxSteps
    @test ss.terminate[2] == TSamp.ExpectedGapStop(10.0)
    @test ss.gauss_seidel && ss.reverse && ss.k == 8
    # Action side and filters stay at the defaults.
    @test ss.action_policy == TSamp.EpsilonGreedy(0.1)
    @test ss.action_score isa TSamp.UpperBoundScore
    @test !ss.exclude_visited && !ss.exclude_converged

    # Gauss-Seidel batching needs a batch size.
    @test_throws ErrorException resolve_sampling(P())
    kf = resolve_k_frac(P("k_frac" => 0.1), 101, 9)
    @test kf["k"] == 11 && kf["sampling"] == "trajectory_brtdp"
    @test resolve_sampling(kf).k == 11
    @test algo_label("gsrdp", P("k" => 8)) == "gsrdp_trajectory_brtdp_k8"

    # Explicit keys win over the preset.
    o = resolve_sampling(P("k" => 8, "expected_gap_tau" => 5.0, "state_temperature" => 2.0))
    @test o.terminate[2] == TSamp.ExpectedGapStop(5.0)
    @test o.state_policy == TSamp.Boltzmann(2.0)
    # Validation still applies to the merged keys.
    @test_throws ErrorException resolve_sampling(P("k" => 8, "state_tempreature" => 2.0))
end

@testset "sampling = \"trajectory\" resolves every knob" begin
    ss = resolve_sampling(
        Dict{String,Any}(
            "sampling" => "trajectory",
            "action_policy" => "boltzmann",
            "action_temperature_schedule" => "gap_based",
            "action_temperature_k" => 2.5,
            "action_temperature_min" => 0.3,
            "action_score" => "log_weighted",
            "action_beta" => 0.75,
            "transition_bound" => "lower",
            "transition_adversary" => "pessimistic",
            "state_exploit" => "lower",
            "state_policy" => "boltzmann",
            "state_temperature" => 2.0,
            "state_score" => "mixture",
            "state_explore" => "upper",
            "state_alpha_e_schedule" => "power",
            "state_alpha_e_k" => 2.0,
            "state_alpha_e_min" => 0.1,
            "state_alpha_e_max" => 0.8,
            "state_alpha_p" => 0.25,
            "terminate" => ["max_steps", "expected_gap"],
            "max_steps" => 17,
            "expected_gap_tau" => 5.0,
            "gauss_seidel" => true,
            "k" => 6,
            "reverse" => false,
        ),
    )

    @test ss.action_policy == TSamp.Boltzmann(TSamp.GapBasedTemperature(2.5, 0.3))
    @test ss.action_score isa TSamp.LogScore
    @test ss.action_score.inner isa TSamp.WeightedAverageScore
    @test ss.action_score.inner.beta == 0.75
    @test ss.transition.bound === IntervalMDP.Lower
    @test ss.transition.adversary === Pessimistic
    @test ss.state_policy == TSamp.Boltzmann(TSamp.FixedTemperature(2.0))
    # The plain softmax, i.e. not the default log_ wrapper.
    mix = ss.state_score
    @test mix isa TSamp.MixtureStateScore
    @test mix.exploit === IntervalMDP.Lower
    @test mix.explore isa TSamp.UpperBoundExplore
    @test mix.alpha_explore == TSamp.ClippedAlpha(TSamp.PowerAlpha(2.0), 0.1, 0.8)
    @test mix.alpha_prob == TSamp.FixedAlpha(0.25)
    @test length(ss.terminate) == 2
    @test ss.terminate[1] isa TSamp.MaxSteps && ss.terminate[1].n == 17
    @test ss.terminate[2] isa TSamp.ExpectedGapStop && ss.terminate[2].tau == 5.0
    @test ss.gauss_seidel == true
    @test ss.k == 6
    @test ss.reverse == false
end

@testset "action_score log_ prefix and score aliases" begin
    mk(score) = resolve_sampling(
        Dict{String,Any}("sampling" => "trajectory", "action_score" => score),
    ).action_score
    @test mk("upper") isa TSamp.UpperBoundScore
    @test mk("lower") isa TSamp.LowerBoundScore
    @test mk("weighted") isa TSamp.WeightedAverageScore
    @test mk("weighted_average") isa TSamp.WeightedAverageScore
    @test mk("log_upper") isa TSamp.LogScore
    @test mk("log_upper").inner isa TSamp.UpperBoundScore
    @test mk("log_lower").inner isa TSamp.LowerBoundScore
end

@testset "state selection parameters are hierarchical" begin
    st(; kw...) = resolve_sampling(
        Dict{String,Any}("sampling" => "trajectory", (String(k) => v for (k, v) in kw)...),
    )
    bz(; kw...) = st(; state_policy = "boltzmann", kw...)
    # The mixture under the default log_ wrapper.
    mixof(ss) = ss.state_score isa TSamp.LogStateScore ? ss.state_score.inner : ss.state_score

    # epsilon_greedy: only state_epsilon and state_exploit apply.
    @test st().state_policy == TSamp.EpsilonGreedy(0.1)
    @test st(state_epsilon = 0.3).state_policy == TSamp.EpsilonGreedy(0.3)
    @test st(state_exploit = "lower").state_score.exploit === IntervalMDP.Lower
    @test mixof(bz(state_exploit = "lower")).exploit === IntervalMDP.Lower

    # boltzmann defaults: T_s = 1 under the power rule (log_mixture), gap explore,
    # alpha_e = 0.5, alpha_p = 0.
    d = bz()
    @test d.state_policy == TSamp.Boltzmann(TSamp.FixedTemperature(1.0))
    @test d.state_score isa TSamp.LogStateScore
    @test d.state_score.inner isa TSamp.MixtureStateScore
    @test mixof(d).exploit === IntervalMDP.Upper
    @test mixof(d).explore isa TSamp.GapExplore
    @test mixof(d).alpha_explore == TSamp.FixedAlpha(0.5)
    @test mixof(d).alpha_prob == TSamp.FixedAlpha(0.0)
    @test bz(state_temperature = 0.05).state_policy ==
          TSamp.Boltzmann(TSamp.FixedTemperature(0.05))
    @test bz(state_score = "log_mixture").state_score isa TSamp.LogStateScore
    @test bz(state_score = "mixture").state_score isa TSamp.MixtureStateScore

    # alpha schedules, for both weights.
    for w in ("e", "p")
        field = w == "e" ? :alpha_explore : :alpha_prob
        # Keys are given as suffixes of `state_alpha_<w>`: "" is the weight itself.
        a(kv::Pair...) = getfield(
            mixof(bz(; (Symbol("state_alpha_$(w)$(k)") => v for (k, v) in kv)...)),
            field,
        )
        @test a("" => 0.7) == TSamp.FixedAlpha(0.7)
        @test a("_schedule" => "power") == TSamp.PowerAlpha(1.0)
        @test a("_schedule" => "power", "_k" => 3.0) == TSamp.PowerAlpha(3.0)
        @test a("_schedule" => "power", "_min" => 0.2) ==
              TSamp.ClippedAlpha(TSamp.PowerAlpha(1.0), 0.2, 1.0)
        @test a("_schedule" => "power", "_max" => 0.6) ==
              TSamp.ClippedAlpha(TSamp.PowerAlpha(1.0), 0.0, 0.6)
        # A key under the wrong schedule is an error.
        @test_throws ErrorException a("_k" => 2.0)
        @test_throws ErrorException a("_min" => 0.1)
        @test_throws ErrorException a("_max" => 0.9)
        @test_throws ErrorException a("_schedule" => "power", "" => 0.5)
        @test_throws ErrorException a("_schedule" => "linear")
        # Library validation still applies.
        @test_throws ArgumentError a("" => 1.5)
        @test_throws ArgumentError a("_schedule" => "power", "_k" => 0.0)
        @test_throws ArgumentError a("_schedule" => "power", "_min" => 0.7, "_max" => 0.3)
    end

    # state_explore picks f_explore.
    ex(name) = mixof(bz(state_explore = name)).explore
    @test ex("gap") isa TSamp.GapExplore
    @test ex("upper") isa TSamp.UpperBoundExplore
    @test ex("action_uncertainty") isa TSamp.ActionUncertaintyExplore
    @test ex("max_bellman_error") isa TSamp.MaxBellmanErrorExplore
    @test ex("vpi") isa TSamp.VPIExplore
    for (name, bound) in (
        "bellman_error" => IntervalMDP.Upper,
        "bellman_error_upper" => IntervalMDP.Upper,
        "bellman_error_lower" => IntervalMDP.Lower,
    )
        e = ex(name)
        @test e isa TSamp.BellmanErrorExplore
        @test e.bound === bound
    end
    @test_throws ErrorException ex("wat")

    # Each resolve builds its own term, so tracked backups never leak between
    # two algorithms that happen to share a config.
    @test ex("bellman_error").tracker !== ex("bellman_error").tracker

    # Boltzmann-only keys under epsilon_greedy, and vice versa, are errors.
    @test_throws ErrorException st(state_temperature = 0.5)
    @test_throws ErrorException st(state_score = "mixture")
    @test_throws ErrorException st(state_explore = "gap")
    @test_throws ErrorException st(state_alpha_e = 0.5)
    @test_throws ErrorException st(state_alpha_p_schedule = "power")
    @test_throws ErrorException bz(state_epsilon = 0.1)

    # The old scores and their keys are gone.
    @test_throws ErrorException bz(state_score = "gap_weighted")
    @test_throws ErrorException bz(state_score = "log_vpi")
    @test_throws ErrorException bz(state_bound = "upper")
    @test_throws ErrorException bz(state_temperature_schedule = "gap_decay")
    @test_throws ErrorException bz(state_policy = "softmax")
end

@testset "transition_adversary = spec defers to the specification" begin
    mk(a) = resolve_sampling(
        Dict{String,Any}("sampling" => "trajectory", "transition_adversary" => a),
    ).transition.adversary
    @test mk("spec") === nothing
    @test mk("optimistic") === Optimistic
    @test mk("pessimistic") === Pessimistic
end

@testset "temperature schedules" begin
    # No schedule key: a bare temperature is a fixed one, as before.
    fixed = resolve_sampling(
        Dict{String,Any}(
            "sampling" => "trajectory",
            "action_policy" => "boltzmann",
            "action_temperature" => 0.25,
        ),
    )
    @test fixed.action_policy == TSamp.Boltzmann(0.25)
    @test fixed.action_policy.T === TSamp.FixedTemperature(0.25)

    # An unknown schedule name is a config error, not a silent default.
    @test_throws ErrorException resolve_sampling(
        Dict{String,Any}(
            "sampling" => "trajectory",
            "action_policy" => "boltzmann",
            "action_temperature_schedule" => "linear",
        ),
    )
end

@testset "action selection parameters are hierarchical" begin
    act(; kw...) = resolve_sampling(
        Dict{String,Any}("sampling" => "trajectory", (String(k) => v for (k, v) in kw)...),
    ).action_policy

    # Defaults at every level.
    @test act() == TSamp.EpsilonGreedy(0.1)
    @test act(action_epsilon = 0.4) == TSamp.EpsilonGreedy(0.4)
    @test act(action_policy = "boltzmann") == TSamp.Boltzmann(TSamp.FixedTemperature(1.0))
    @test act(action_policy = "boltzmann", action_temperature_schedule = "gap_based") ==
          TSamp.Boltzmann(TSamp.GapBasedTemperature(1.0, 0.1))
    @test act(
        action_policy = "boltzmann",
        action_temperature_schedule = "gap_based",
        action_temperature_k = 3.0,
    ) == TSamp.Boltzmann(TSamp.GapBasedTemperature(3.0, 0.1))
    # t_min = 0 removes the floor.
    @test act(
        action_policy = "boltzmann",
        action_temperature_schedule = "gap_based",
        action_temperature_min = 0.0,
    ) == TSamp.Boltzmann(TSamp.GapBasedTemperature(1.0, 0.0))

    # A key under the wrong parent is an error, not silently ignored.
    @test_throws ErrorException act(action_temperature = 0.5)
    @test_throws ErrorException act(action_temperature_schedule = "fixed")
    @test_throws ErrorException act(action_temperature_k = 2.0)
    @test_throws ErrorException act(action_policy = "boltzmann", action_epsilon = 0.2)
    @test_throws ErrorException act(action_policy = "boltzmann", action_temperature_k = 2.0)
    @test_throws ErrorException act(action_policy = "boltzmann", action_temperature_min = 0.1)
    @test_throws ErrorException act(action_temperature_min = 0.1)
    @test_throws ErrorException act(
        action_policy = "boltzmann",
        action_temperature_schedule = "gap_based",
        action_temperature = 0.5,
    )

    # The annealing schedules are gone from the action side, as are their keys.
    @test_throws ErrorException act(
        action_policy = "boltzmann",
        action_temperature_schedule = "gap_decay",
    )

    # Library-level validation still applies through the resolver.
    @test_throws ArgumentError act(
        action_policy = "boltzmann",
        action_temperature_schedule = "gap_based",
        action_temperature_k = 0.0,
    )
    @test_throws ArgumentError act(
        action_policy = "boltzmann",
        action_temperature_schedule = "gap_based",
        action_temperature_min = -0.1,
    )
end

@testset "validation: typos and missing required params are errors" begin
    # A typo in a sweep config must fail rather than silently use a default.
    @test_throws ErrorException resolve_sampling(
        Dict{String,Any}("sampling" => "trajectory", "state_temprature" => 1.0),
    )
    @test_throws ErrorException resolve_sampling(
        Dict{String,Any}("sampling" => "trajectory", "action_policy" => "greedy"),
    )
    @test_throws ErrorException resolve_sampling(
        Dict{String,Any}("sampling" => "trajectory", "action_score" => "middle"),
    )
    @test_throws ErrorException resolve_sampling(
        Dict{String,Any}("sampling" => "trajectory", "state_score" => "wat"),
    )
    @test_throws ErrorException resolve_sampling(
        Dict{String,Any}(
            "sampling" => "trajectory",
            "state_policy" => "boltzmann",
            "state_score" => "wat",
        ),
    )
    @test_throws ErrorException resolve_sampling(
        Dict{String,Any}("sampling" => "trajectory", "transition_bound" => "middle"),
    )
    @test_throws ErrorException resolve_sampling(
        Dict{String,Any}("sampling" => "trajectory", "terminate" => ["never"]),
    )
    # gauss_seidel needs a batch size.
    @test_throws ErrorException resolve_sampling(
        Dict{String,Any}("sampling" => "trajectory", "gauss_seidel" => true),
    )
    # ...and a batch size needs gauss_seidel, rather than being silently ignored.
    @test_throws ErrorException resolve_sampling(
        Dict{String,Any}("sampling" => "trajectory", "k" => 4),
    )
    @test resolve_sampling(
        Dict{String,Any}("sampling" => "trajectory", "gauss_seidel" => true, "k" => 4),
    ).k == 4
    # Library-level validation still applies through the resolver.
    @test_throws ArgumentError resolve_sampling(
        Dict{String,Any}(
            "sampling" => "trajectory",
            "action_policy" => "boltzmann",
            "action_temperature" => 0.0,
        ),
    )
end

@testset "with_label keeps BuiltAlgorithm in sync with AlgorithmSpec" begin
    built = BuiltAlgorithm(nothing, "generated", "gsrdp", Dict{String,Any}())
    relabelled = with_label(built, "hand-written")
    @test relabelled.label == "hand-written"
    @test relabelled.kind == built.kind
    @test relabelled.alg === built.alg
    @test relabelled.descriptor === built.descriptor
end

@testset "validate_config rejects duplicate algorithm labels" begin
    problems = [ProblemSpec("car_parking", "car_parking_S10_I3", Dict{String,Any}())]
    register_problem!("car_parking", params -> nothing)
    register_algorithm!("gsrdp_test", (bp, params) -> nothing)

    # save_raw, save_trace, save_policy, track_initial, traces, reference_label,
    # results_root, image_formats, analysis, gap_videos, max_frames, framerate,
    # trace_thin_rtol
    output = OutputSettings(
        false, false, false, false, ["trace"], "a", "results", ["png"], false,
        false, 100, 10, 0.0,
    )

    dup = [
        AlgorithmSpec("gsrdp_test", "same", Dict{String,Any}()),
        AlgorithmSpec("gsrdp_test", "same", Dict{String,Any}()),
    ]
    err = try
        validate_config(problems, dup, output)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("same label", err.msg)
    @test occursin("label", err.msg)

    distinct = [
        AlgorithmSpec("gsrdp_test", "a", Dict{String,Any}()),
        AlgorithmSpec("gsrdp_test", "b", Dict{String,Any}()),
    ]
    @test validate_config(problems, distinct, output) === nothing

    # [output].trace_thin_rtol is a relative tolerance: [0, 1)
    bad = OutputSettings(
        false, false, false, false, ["trace"], "a", "results", ["png"], false,
        false, 100, 10, 1.0,
    )
    @test_throws ErrorException validate_config(problems, distinct, bad)
end
