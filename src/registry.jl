# Two registries + the resolvers that turn flat config strings/numbers into the
# concrete Julia objects `solve` needs. This indirection is the whole point of
# the harness: adding a problem source or an algorithm is a single
# `register_*!` call, after which it is selectable by name from any TOML config.

# ---------------------------------------------------------------------------
# Problem registry:  name => (params::Dict) -> BenchmarkProblem
# ---------------------------------------------------------------------------
const PROBLEM_REGISTRY = Dict{String,Function}()

"""
    register_problem!(name, builder)

Register a side-effect-free `builder(params::Dict) -> BenchmarkProblem` under `name`.
Builders MUST NOT write files or plot — persistence is centralized in results.jl —
so they are safe to call once per instance and reuse across the whole sweep.
"""
function register_problem!(name::AbstractString, builder)
    haskey(PROBLEM_REGISTRY, name) && @warn "overwriting problem builder '$name'"
    PROBLEM_REGISTRY[String(name)] = builder
    return name
end

build_problem(name::AbstractString, params::AbstractDict) = PROBLEM_REGISTRY[String(name)](params)

has_problem(name::AbstractString) = haskey(PROBLEM_REGISTRY, String(name))

# ---------------------------------------------------------------------------
# Algorithm registry:  name => (bp::BenchmarkProblem, params::Dict) -> BuiltAlgorithm
# ---------------------------------------------------------------------------
const ALGORITHM_REGISTRY = Dict{String,Function}()

"""
    BuiltAlgorithm

A ready-to-solve algorithm plus a display label. `kind` is a coarse tag (e.g.
`"gsrdp"`) used for grouping in results; `descriptor` captures the resolved
knobs (bellman / sampling / k / term) for the `results.csv` columns.
"""
struct BuiltAlgorithm
    alg::Any                       # <: IntervalMDP.ModelCheckingAlgorithm
    label::String
    kind::String
    descriptor::Dict{String,Any}
end

"""
    register_algorithm!(name, builder)

Register `builder(bp::BenchmarkProblem, params::Dict) -> BuiltAlgorithm` under `name`.
The builder receives the *constructed* problem because the default Bellman
operator depends on the model (`default_bellman_algorithm`).
"""
function register_algorithm!(name::AbstractString, builder)
    haskey(ALGORITHM_REGISTRY, name) && @warn "overwriting algorithm builder '$name'"
    ALGORITHM_REGISTRY[String(name)] = builder
    return name
end

build_algorithm(name::AbstractString, bp, params::AbstractDict) =
    ALGORITHM_REGISTRY[String(name)](bp, params)

"""
    with_label(built, label) -> BuiltAlgorithm

`built` with its display label replaced. A builder computes a label from its
params via `algo_label`, but the config may carry an explicit `label` key that
the builder never sees — and it is that one, `AlgorithmSpec.label`, that names
run directories and results rows. This keeps the two in agreement.
"""
with_label(built::BuiltAlgorithm, label::AbstractString) =
    BuiltAlgorithm(built.alg, String(label), built.kind, built.descriptor)

has_algorithm(name::AbstractString) = haskey(ALGORITHM_REGISTRY, String(name))

# ---------------------------------------------------------------------------
# Resolvers: config value -> IntervalMDP object
# (sampling strategies are NOT exported by IntervalMDP, so they are referenced via
#  the qualified `IntervalMDP.` path. GSRDP termination is no longer configurable
#  here — it is auto-derived from the specification's property.)
# ---------------------------------------------------------------------------

"Resolve the Bellman operator; `\"default\"` is problem-dependent."
function resolve_bellman(bp, params)
    b = lowercase(String(get(params, "bellman", "default")))
    if b == "default"
        return default_bellman_algorithm(system(bp.problem))
    elseif b in ("omax", "omaximization")
        return OMaximization()
    elseif b in ("vertex", "vertexenumeration")
        return VertexEnumeration()
    elseif b in ("lp", "mccormick", "lpmccormick", "lpmccormickrelaxation")
        return LPMcCormickRelaxation()
    else
        error("unknown bellman algorithm: '$b' (expected default|omax|vertex|lp)")
    end
end

# ---------------------------------------------------------------------------
# Trajectory sampling
# ---------------------------------------------------------------------------
# `IntervalMDP.TrajectorySampling.TrajectorySampling` is one strategy
# parameterized by a product of orthogonal choices (selection policy × score
# function × concrete transition × Gauss-Seidel batching) rather than a family
# of fixed strategies. That product is what `sampling = "trajectory"` exposes:
# every knob is an optional key in `[algorithms.params]`, defaulting to the
# library default. `TRAJECTORY_PRESETS` names useful points in that space.

const TS = IntervalMDP.TrajectorySampling

# Keys `resolve_sampling` shares with every other strategy, or that other
# resolvers read — allowed alongside the trajectory-specific ones below.
const TRAJECTORY_COMMON_KEYS = Set(["bellman", "sampling", "k", "term", "tol", "drive"])

const TRAJECTORY_KEYS = Set([
    "action_policy", "action_epsilon",
    "action_temperature_schedule", "action_temperature", "action_temperature_k",
    "action_temperature_min",
    "action_score", "action_beta",
    "transition_bound", "transition_adversary",
    "state_exploit", "state_policy", "state_epsilon", "state_temperature",
    "state_score", "state_explore",
    "state_alpha_e_schedule", "state_alpha_e", "state_alpha_e_k",
    "state_alpha_e_min", "state_alpha_e_max",
    "state_alpha_p_schedule", "state_alpha_p", "state_alpha_p_k",
    "state_alpha_p_min", "state_alpha_p_max",
    "state_exclude_visited", "state_exclude_converged", "state_converged_eps",
    "terminate", "max_steps", "expected_gap_tau",
    "gauss_seidel", "reverse", "dedup", "dedup_scope", "initial_state",
])

# The four strategies the rework removed. Named explicitly so an old config
# fails with a migration message instead of the generic "unknown sampling".
const REMOVED_TRAJECTORY_SAMPLINGS = Set([
    "trajectory_transition", "trajectory_probability",
    "trajectory_gap", "trajectory_expected_gap",
    "trajectory_action_uncertainty",
    "trajectory_reach", "trajectory_reach_probability",
])

# Named points in the trajectory parameter space: `sampling = "<preset>"` is
# `sampling = "trajectory"` with these keys filled in. Any explicit key still
# wins over the preset (`_expand_preset`).
const TRAJECTORY_PRESETS = Dict{String,Dict{String,Any}}(
    # BRTDP (McMahan et al., 2005) in the reworked parameter space: draw s' in
    # proportion to p(s')·(U − L)(s') (the power rule at T = 1 over the pure
    # explore term, α_e = 1, α_p = 0), stop a rollout on the expected-gap rule,
    # and hand it out goal-end first in Gauss-Seidel batches. The closest point
    # to slow-tail's `trajectory_brtdp`, but not equal to it: that one weighs by
    # max(p_U, p_L)·(U − L), stops on the max-adversary expected gap and dedups
    # batches (docs/experiment-config.md; `dedup = true` covers the last). Requires
    # `k` or `k_frac`.
    "trajectory_brtdp" => Dict{String,Any}(
        "state_policy" => "boltzmann",
        "state_score" => "log_mixture", "state_temperature" => 1.0,
        "state_explore" => "gap",
        "state_alpha_e_schedule" => "fixed", "state_alpha_e" => 1.0,
        "state_alpha_p_schedule" => "fixed", "state_alpha_p" => 0.0,
        "terminate" => ["max_steps", "expected_gap"], "expected_gap_tau" => 10.0,
        "gauss_seidel" => true, "reverse" => true,
    ),
)

"""
    _expand_preset(params) -> params

For `sampling` naming a `TRAJECTORY_PRESETS` entry, the preset's keys merged
under `params` (explicit keys win) with `sampling = "trajectory"`. Any other
`params` comes back untouched.
"""
function _expand_preset(params)
    s = lowercase(String(get(params, "sampling", "exhaustive_state")))
    haskey(TRAJECTORY_PRESETS, s) || return params
    p = merge(TRAJECTORY_PRESETS[s], Dict{String,Any}(String(k) => v for (k, v) in params))
    p["sampling"] = "trajectory"
    return p
end

_traj_str(params, key, default) = lowercase(String(get(params, key, default)))

# A key that only applies under another key's value is an error anywhere else,
# rather than silently ignored: a sweep that sets `action_epsilon` on a Boltzmann
# arm is varying nothing.
function _traj_require_parent(params, keys, parent, value)
    for key in keys
        haskey(params, key) && error("$key only applies under $parent = $value")
    end
end

# Action selection is hierarchical; each key applies only under its parent's value:
#
#   action_policy = "epsilon_greedy"         (default)
#       action_epsilon                       (default 0.1)
#   action_policy = "boltzmann"
#       action_temperature_schedule = "fixed"      (default)
#           action_temperature               T = t            (default 1.0)
#       action_temperature_schedule = "gap_based"
#           action_temperature_k             T = max(t_min, k·δ / (1 − δ)),
#                                            δ = max_{s ∈ S₀} Diff(s)
#                                            (default 1.0)
#           action_temperature_min           t_min, the exploration floor (default 0.1;
#                                            0 = no floor)
function _traj_action_policy(params)
    p = _traj_str(params, "action_policy", "epsilon_greedy")
    boltzmann_keys = (
        "action_temperature_schedule", "action_temperature",
        "action_temperature_k", "action_temperature_min",
    )
    if p in ("epsilon_greedy", "eps_greedy", "egreedy")
        _traj_require_parent(params, boltzmann_keys, "action_policy", "boltzmann")
        return TS.EpsilonGreedy(Float64(get(params, "action_epsilon", 0.1)))
    elseif p == "boltzmann"
        _traj_require_parent(params, ("action_epsilon",), "action_policy", "epsilon_greedy")
        return TS.Boltzmann(_traj_action_temperature(params))
    else
        error("unknown action_policy: '$p' (expected epsilon_greedy|boltzmann)")
    end
end

function _traj_action_temperature(params)
    sched = _traj_str(params, "action_temperature_schedule", "fixed")
    if sched == "fixed"
        _traj_require_parent(
            params, ("action_temperature_k", "action_temperature_min"),
            "action_temperature_schedule", "gap_based",
        )
        return TS.FixedTemperature(Float64(get(params, "action_temperature", 1.0)))
    elseif sched == "gap_based"
        _traj_require_parent(
            params, ("action_temperature",), "action_temperature_schedule", "fixed",
        )
        return TS.GapBasedTemperature(
            Float64(get(params, "action_temperature_k", 1.0)),
            Float64(get(params, "action_temperature_min", 0.1)),
        )
    else
        error("unknown action_temperature_schedule: '$sched' (expected fixed|gap_based)")
    end
end

# Successor selection is hierarchical as well:
#
#   state_exploit = "upper" | "lower"        f_exploit (default upper); read by both
#                                            policies
#   state_policy = "epsilon_greedy"          (default) argmax of p·f_exploit
#       state_epsilon                        (default 0.1)
#   state_policy = "boltzmann"
#       state_temperature                    T_s, fixed (default 1.0)
#       state_score = "log_mixture"          (default) the power rule P ∝ f_S^(1/T_s):
#                                            T_s = 1 draws exactly from f_S, and the
#                                            draw does not depend on the support size
#       state_score = "mixture"              softmax of f_S / T_s; f_S sums to 1, so the
#                                            same T_s is near-argmax on a small support
#                                            and near-uniform on a large one
#                                            f_S = (1 − α_e)·f'_exploit + α_e·f'_explore
#       state_explore                        f_explore (default gap)
#       state_alpha_e_schedule = "fixed"     (default)
#           state_alpha_e                    α_e (default 0.5)
#       state_alpha_e_schedule = "power"
#           state_alpha_e_k                  α_e = δ^k, δ = max_{s ∈ S₀} Diff(s) (default 1.0)
#           state_alpha_e_min, _max          optional clip of δ^k (defaults 0, 1)
#       state_alpha_p_*                      the same for α_p, which flattens p towards
#                                            uniform (default fixed, 0.0)
#
# Under epsilon_greedy only `state_exploit` is read out of the score.
#
# The successor filters apply under both policies (`_traj_state_filters`):
#
#   state_exclude_visited                    a rollout never revisits a state (default false)
#   state_exclude_converged                  skip successors with U − L < eps (default false)
#       state_converged_eps                  eps (default: the property's convergence_eps)
function _traj_state_selection(params)
    exploit = _traj_bound(params, "state_exploit", "upper")
    p = _traj_str(params, "state_policy", "epsilon_greedy")
    boltzmann_keys = (
        "state_temperature", "state_score", "state_explore",
        (k for k in TRAJECTORY_KEYS if startswith(k, "state_alpha_"))...,
    )
    if p in ("epsilon_greedy", "eps_greedy", "egreedy")
        _traj_require_parent(params, boltzmann_keys, "state_policy", "boltzmann")
        policy = TS.EpsilonGreedy(Float64(get(params, "state_epsilon", 0.1)))
        return policy, TS.MixtureStateScore(; exploit = exploit)
    elseif p == "boltzmann"
        _traj_require_parent(params, ("state_epsilon",), "state_policy", "epsilon_greedy")
        policy = TS.Boltzmann(TS.FixedTemperature(Float64(get(params, "state_temperature", 1.0))))
        return policy, _traj_state_mixture(params, exploit)
    else
        error("unknown state_policy: '$p' (expected epsilon_greedy|boltzmann)")
    end
end

function _traj_state_mixture(params, exploit)
    s = _traj_str(params, "state_score", "log_mixture")
    s in ("mixture", "log_mixture") ||
        error("unknown state_score: '$s' (expected mixture|log_mixture)")
    score = TS.MixtureStateScore(;
        exploit = exploit,
        explore = _traj_explore(params),
        alpha_explore = _traj_alpha(params, "e", 0.5),
        alpha_prob = _traj_alpha(params, "p", 0.0),
    )
    return s == "log_mixture" ? TS.LogStateScore(score) : score
end

function _traj_state_filters(params)
    exclude_converged = Bool(get(params, "state_exclude_converged", false))
    exclude_converged || _traj_require_parent(
        params, ("state_converged_eps",), "state_exclude_converged", "true",
    )
    eps = get(params, "state_converged_eps", nothing)
    return (;
        exclude_visited = Bool(get(params, "state_exclude_visited", false)),
        exclude_converged = exclude_converged,
        converged_eps = eps === nothing ? nothing : Float64(eps),
    )
end

# `which` is "e" (α_e) or "p" (α_p).
function _traj_alpha(params, which, default)
    key = "state_alpha_$which"
    sched = _traj_str(params, "$(key)_schedule", "fixed")
    if sched == "fixed"
        _traj_require_parent(
            params, ("$(key)_k", "$(key)_min", "$(key)_max"), "$(key)_schedule", "power",
        )
        return TS.FixedAlpha(Float64(get(params, key, default)))
    elseif sched == "power"
        _traj_require_parent(params, (key,), "$(key)_schedule", "fixed")
        alpha = TS.PowerAlpha(Float64(get(params, "$(key)_k", 1.0)))
        (haskey(params, "$(key)_min") || haskey(params, "$(key)_max")) || return alpha
        return TS.ClippedAlpha(
            alpha,
            Float64(get(params, "$(key)_min", 0.0)),
            Float64(get(params, "$(key)_max", 1.0)),
        )
    else
        error("unknown $(key)_schedule: '$sched' (expected fixed|power)")
    end
end

function _traj_bound(params, key, default)
    b = _traj_str(params, key, default)
    b == "upper" && return IntervalMDP.Upper
    b == "lower" && return IntervalMDP.Lower
    error("unknown $key: '$b' (expected upper|lower)")
end

function _traj_action_score(params)
    s = _traj_str(params, "action_score", "upper")
    base, islog = startswith(s, "log_") ? (s[5:end], true) : (s, false)
    score = if base == "upper"
        TS.UpperBoundScore()
    elseif base == "lower"
        TS.LowerBoundScore()
    elseif base in ("weighted", "weighted_average")
        TS.WeightedAverageScore(Float64(get(params, "action_beta", 0.5)))
    else
        error("unknown action_score: '$s' (expected upper|lower|weighted, optionally log_-prefixed)")
    end
    return islog ? TS.LogScore(score) : score
end

function _traj_adversary(params)
    a = _traj_str(params, "transition_adversary", "spec")
    a == "spec" && return nothing
    a == "optimistic" && return IntervalMDP.Optimistic
    a == "pessimistic" && return IntervalMDP.Pessimistic
    error("unknown transition_adversary: '$a' (expected spec|optimistic|pessimistic)")
end

# `f_explore`, the explore term of the state mixture. Built fresh per resolve:
# the Bellman-error terms carry per-solve state.
function _traj_explore(params)
    e = _traj_str(params, "state_explore", "gap")
    if e == "gap"
        return TS.GapExplore()
    elseif e in ("upper", "upper_bound")
        return TS.UpperBoundExplore()
    elseif e == "action_uncertainty"
        return TS.ActionUncertaintyExplore()
    elseif e in ("bellman_error", "bellman_error_upper")
        return TS.BellmanErrorExplore(; bound = IntervalMDP.Upper)
    elseif e == "bellman_error_lower"
        return TS.BellmanErrorExplore(; bound = IntervalMDP.Lower)
    elseif e == "max_bellman_error"
        return TS.MaxBellmanErrorExplore()
    elseif e == "vpi"
        return TS.VPIExplore()
    else
        error(
            "unknown state_explore: '$e' (expected gap|upper|action_uncertainty|" *
            "bellman_error_upper|bellman_error_lower|max_bellman_error|vpi)",
        )
    end
end

function _traj_terminate(params)
    names = get(params, "terminate", ["max_steps"])
    names isa AbstractString && (names = [names])

    rules = TS.TerminationRule[]
    for n in names
        n = lowercase(String(n))
        if n == "max_steps"
            ms = get(params, "max_steps", nothing)
            push!(rules, TS.MaxSteps(ms === nothing ? nothing : Int(ms)))
        elseif n in ("expected_gap", "expected_gap_stop")
            push!(rules, TS.ExpectedGapStop(Float64(get(params, "expected_gap_tau", 10.0))))
        else
            error("unknown terminate rule: '$n' (expected max_steps|expected_gap)")
        end
    end
    return rules
end

"""
    resolve_trajectory(params) -> TrajectorySampling

Build the configurable trajectory sampler from a flat `[algorithms.params]`
table. Every key is optional; see `docs/experiment-config.md` for the full
table. Unknown keys are an error rather than a silent default, so a typo in a
sweep config fails instead of quietly running the wrong thing.
"""
function resolve_trajectory(params)
    for key in keys(params)
        (String(key) in TRAJECTORY_COMMON_KEYS || String(key) in TRAJECTORY_KEYS) || error(
            "unknown trajectory sampling parameter '$key'. Valid: " *
            "$(sort(collect(TRAJECTORY_KEYS)))",
        )
    end

    gauss_seidel = Bool(get(params, "gauss_seidel", false))
    # `k` is the Gauss-Seidel batch size; without batching the whole trajectory is
    # one batch and a `k` would be silently ignored (as `k_frac` already rejects).
    gauss_seidel || !haskey(params, "k") || error(
        "'k' for sampling='trajectory' requires gauss_seidel = true: it sets the " *
        "Gauss-Seidel batch size, and without batching the whole trajectory is one batch",
    )
    k = gauss_seidel ? _require_k(params, "trajectory with gauss_seidel=true") : 1
    state_policy, state_score = _traj_state_selection(params)

    return TS.TrajectorySampling(;
        action_policy = _traj_action_policy(params),
        action_score = _traj_action_score(params),
        transition = TS.ConcreteTransition(;
            bound = _traj_bound(params, "transition_bound", "upper"),
            adversary = _traj_adversary(params),
        ),
        state_policy = state_policy,
        state_score = state_score,
        terminate = _traj_terminate(params),
        gauss_seidel = gauss_seidel,
        k = k,
        reverse = Bool(get(params, "reverse", true)),
        dedup = Bool(get(params, "dedup", false)),
        dedup_scope = Symbol(_traj_str(params, "dedup_scope", "batch")),
        initial_state = Symbol(_traj_str(params, "initial_state", "random")),
        _traj_state_filters(params)...,
    )
end

"Resolve a GSRDP sampling strategy; every family but the exhaustive sweeps reads `k` from params."
function resolve_sampling(params)
    s = lowercase(String(get(params, "sampling", "exhaustive_state")))
    # The `all_*` spellings predate the library's `Exhaustive*` rename and stay
    # accepted: banked configs and the recorded run metadata use them.
    if s in ("exhaustive_state", "all_states_sweep", "sweep", "all_sweep")
        return IntervalMDP.ExhaustiveState()
    elseif s in ("exhaustive_state_action", "exhaustive_state_actions", "all", "all_sampling")
        return IntervalMDP.ExhaustiveStateActions()
    elseif s in ("random_state", "random_subset_state")
        return IntervalMDP.RandomSubsetState(_require_k(params, s))
    elseif s in ("random_state_action", "random_subset_state_actions", "random_state_actions")
        return IntervalMDP.RandomSubsetStateActions(_require_k(params, s))
    elseif s in ("round_robin", "round_robin_state")
        return IntervalMDP.RoundRobinState(_require_k(params, s))
    elseif s in ("round_robin_state_action", "round_robin_state_actions")
        return IntervalMDP.RoundRobinStateActions(_require_k(params, s))
    elseif s == "trajectory"
        return resolve_trajectory(params)
    elseif haskey(TRAJECTORY_PRESETS, s)
        return resolve_trajectory(_expand_preset(params))
    elseif s in REMOVED_TRAJECTORY_SAMPLINGS
        error(
            "sampling='$s' was removed when trajectory sampling was reworked. It draws " *
            "the next state *proportionally* to a weight, which the new argmax / softmax " *
            "policies cannot express, so there is no configuration that reproduces it. " *
            "Use sampling='trajectory' with explicit params (see docs/experiment-config.md). " *
            "NOTE: results from any replacement are NOT directly comparable to banked " *
            "'$s' runs — re-run the baseline for any figure that spans the change.",
        )
    elseif s in ("value_ordered", "value_function_ordered", "value_function_ordered_sampling")
        # The struct has no zero-arg form: order states by the current gap,
        # widest first, and take the top `k`.
        return IntervalMDP.ValueFunctionOrderedSampling(
            IntervalMDP.gap,
            false,
            _require_k(params, s),
        )
    elseif s in ("priority_gap", "priority_queue_gap")
        return IntervalMDP.PriorityQueueSampling.GapPriorityQueueSampling(_require_k(params, s))
    elseif s in ("priority_upper", "priority_queue_upper", "priority_upper_bound")
        return IntervalMDP.PriorityQueueSampling.UpperBoundPriorityQueueSampling(_require_k(params, s))
    elseif s in ("priority_action_uncertainty", "priority_queue_action_uncertainty")
        return IntervalMDP.PriorityQueueSampling.ActionUncertaintyPriorityQueueSampling(
            _require_k(params, s),
        )
    else
        error("unknown sampling strategy: '$s'")
    end
end

"""
    resolve_bound_update(params) -> IntervalMDP.BoundUpdateMode

Which GSRDP bound runs the optimizing Bellman update, from `drive`: `"follow"` (the
default — one bound picks the action and the other follows it; the upper bound drives
under Maximize, the lower bound under Minimize) or `"both"` (each bound optimizes on its
own).
"""
function resolve_bound_update(params)
    d = lowercase(String(get(params, "drive", "follow")))
    d == "follow" && return IntervalMDP.FollowDrive()
    d == "both"   && return IntervalMDP.BothDrive()
    error("unknown drive: '$d' (expected follow|both)")
end

function _require_k(params, s)
    haskey(params, "k") || error("sampling='$s' requires an integer 'k' in [algorithms.params]")
    return Int(params["k"])
end

"""
    resolve_k_frac(params, nstates, nactions) -> params

Turn a size-relative `k_frac` into the absolute `k` the samplers take, for one problem:
`k = max(1, ceil(k_frac * n))`, where `n` is the sampler's domain — `nstates` for the
state samplers, `nstates * nactions` for the `*_state_action` ones. Returns `params`
untouched when there is no `k_frac`, else a copy with `k` set and `k_frac` removed.

An absolute `k` means very different things across problems of 3 and 330k states (a
full sweep on one, a sliver of the other); `k_frac` gives every problem the same slice.
The label keeps `k_frac` (`algo_label`), so it is identical across problems, while the
resolved per-problem `k` is what reaches `results.csv` (`sampling_k`).

For trajectory sampling `k` is the Gauss-Seidel batch size (states of the active
trajectory relaxed per iteration), so `n = nstates`, and `k_frac` requires
`gauss_seidel = true` — without batching there is no `k` for it to set.
"""
function resolve_k_frac(params, nstates::Integer, nactions::Integer)
    haskey(params, "k_frac") || return params
    s = lowercase(String(get(params, "sampling", "exhaustive_state")))
    haskey(params, "k") && error("give either 'k' or 'k_frac' in [algorithms.params], not both")
    expanded = _expand_preset(params)   # a preset carries its own gauss_seidel
    lowercase(String(get(expanded, "sampling", ""))) == "trajectory" &&
        !Bool(get(expanded, "gauss_seidel", false)) &&
        error("'k_frac' for sampling='$s' requires gauss_seidel = true: it sets " *
              "the batch size, which only exists in Gauss-Seidel mode")
    f = Float64(params["k_frac"])
    0 < f <= 1 || error("'k_frac' must be in (0, 1], got $f")
    n = occursin("state_action", s) ? nstates * nactions : nstates
    p = Dict{String,Any}(String(k) => v for (k, v) in params if String(k) != "k_frac")
    p["k"] = max(1, ceil(Int, f * n))
    return p
end

"""
`resolve_k_frac` for a built benchmark problem: the domain is its model's states/actions.

`k_chain_outdeg = d` (trajectory sampling, with `k_frac`) switches to `k = 1` on models whose
mean successor support -- transitions per state-action pair -- is below `d`: near-deterministic
chains, where single-state Gauss-Seidel batches propagate a rollout's values end to end, while
wider-support models keep the size-relative batch.
"""
function resolve_k_frac(params, bp)
    haskey(params, "k_frac") || return params
    sys = IntervalMDP.system(bp.problem)
    ns, na = IntervalMDP.num_states(sys), IntervalMDP.num_actions(sys)
    if haskey(params, "k_chain_outdeg")
        d = Float64(params["k_chain_outdeg"])
        p = Dict{String,Any}(String(k) => v for (k, v) in params if String(k) != "k_chain_outdeg")
        nt = num_transitions(sys)
        if !ismissing(nt) && nt / (ns * na) < d
            delete!(p, "k_frac")
            p["k"] = 1
            return p
        end
        return resolve_k_frac(p, ns, na)
    end
    return resolve_k_frac(params, ns, na)
end

"The `Property` of a BenchmarkProblem's specification."
system_property_of(bp) = IntervalMDP.system_property(specification(bp.problem))

"""
    override_specification(bp, cfg) -> (ok::Bool, bp)

Confirm `bp`'s specification carries an `InfiniteTimeReachAvoid` property — the only
property type this harness currently targets. Warns and returns `(false, bp)` for
any other property so the caller can skip the cell. On success applies any
`[property]` overrides from `cfg` (`convergence_eps` / `restrict_to_initial`) via
`apply_property_overrides`, returning the possibly-rebuilt problem.
"""
function override_specification(bp, cfg)
    system = IntervalMDP.system(bp.problem)
    spec = specification(bp.problem)
    prop = IntervalMDP.system_property(spec)

    if (prop isa IntervalMDP.InfiniteTimeReachAvoid) || (prop isa IntervalMDP.FiniteTimeReachAvoid)

        reach = IntervalMDP.reach(prop)
        avoid = IntervalMDP.avoid(prop)

        # A problem file with no reach states is vacuous — every value is 0 — so the cell
        # would "converge" instantly and report a meaningless speed-up. Skip it loudly.
        # (Generated by too coarse or misaligned a state grid; see tools/IntervalMDPAbstractions/spec.jl.)
        if isempty(reach)
            @warn "skipping: property has an empty reach set" problem=bp.name
            return (false, bp)
        end

        # An unset `[property]` override leaves the field `nothing`; fall back to the
        # problem's own stored value rather than passing `nothing` into a `R<:Real`/`Bool`
        # constructor, which is a MethodError.
        # (Lazily — `something` would evaluate the fallback eagerly, and `convergence_eps`
        # is undefined for a FiniteTimeReachAvoid property, which this branch also accepts.)
        eps = cfg.property.convergence_eps === nothing ?
              IntervalMDP.convergence_eps(prop) : cfg.property.convergence_eps
        rti = cfg.property.restrict_to_initial === nothing ?
              IntervalMDP.restrict_to_initial(prop) : cfg.property.restrict_to_initial

        # `restrict_to_initial` is only honoured when the *model* declares initial states:
        # `apply_initial_restriction` (IntervalMDP src/termination.jl:38-42) returns the
        # all-states criterion unwrapped for an `AllStates()` model, with no warning. Left to
        # itself that produces an all-states result labelled as initial-state-restricted --
        # and NaN `initial_gap_*` columns, since the trace has no states to index. Skip the
        # cell instead, the way an empty reach set is skipped above. Fix by regenerating the
        # problem with an initial region (tools/IntervalMDPAbstractions/spec.jl).
        if rti === true && IntervalMDP.initial_states(system) isa IntervalMDP.AllStates
            @warn "skipping: restrict_to_initial=true but the model declares no initial states \
                   (AllStates); regenerate the problem with an initial region" problem=bp.name
            return (false, bp)
        end

        prop_new = IntervalMDP.InfiniteTimeReachAvoid(reach, avoid, eps, rti)
        spec_new = IntervalMDP.Specification(prop_new, IntervalMDP.satisfaction_mode(spec), IntervalMDP.strategy_mode(spec))
        prob_new = IntervalMDP.ControlSynthesisProblem(system, spec_new)

        return (true, set_problem(bp, prob_new))
    
    else
        @warn "skipping: property is not ReachAvoid" problem=bp.name property=typeof(prop)
        return (false, bp)
    end
end

"Build a short, unique-ish label for results, e.g. `gsrdp_random_state_k250` or, for a
size-relative `k_frac`, `gsrdp_round_robin_kf0.25` (`..._driveboth` for a non-default `drive`)."
function algo_label(name, params)
    parts = String[String(name)]
    haskey(params, "sampling") && push!(parts, lowercase(String(params["sampling"])))
    haskey(params, "k")        && push!(parts, "k$(Int(params["k"]))")
    haskey(params, "k_frac")   && push!(parts, "kf$(Float64(params["k_frac"]))")
    if haskey(params, "bellman") && lowercase(String(params["bellman"])) != "default"
        push!(parts, lowercase(String(params["bellman"])))
    end
    # The default keeps no suffix, so a follow arm pools with runs that predate `drive`
    # (all Maximize, where follow is the old upper-driven update).
    if haskey(params, "drive") && lowercase(String(params["drive"])) != "follow"
        push!(parts, "drive$(lowercase(String(params["drive"])))")
    end
    return join(parts, "_")
end
