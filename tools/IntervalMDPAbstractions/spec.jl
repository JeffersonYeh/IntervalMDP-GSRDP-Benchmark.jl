# Shared spec post-processing for the problem generators.
#
# The upstream examples (IntervalMDPAbstractions.jl/examples/systems/*.jl) all build
# *finite*-horizon region properties, so `abstraction` hands back a
# `FiniteTimeReachAvoid`. The benchmark solves infinite-horizon reach-avoid, and the
# harness rewraps the property on load anyway (src/registry.jl, `override_specification`);
# rewrapping here instead makes the stored problem agree with what is actually solved,
# the way `package_delivery.jl` already builds its property directly.
#
# Each generator is wrapped in its own module by `run.jl`, so every one of them includes
# this file and gets its own copy — the module wrapper is what keeps the shared `main`,
# `DIR`, `NAME`, ... names from clashing.

"""
    infinite_time_reachavoid(spec, eps; restrict_to_initial=false) -> Specification

Rewrap the abstract reach-avoid `spec` as an `InfiniteTimeReachAvoid` with convergence
tolerance `eps`, keeping the satisfaction and strategy modes.

Errors on an empty reach set. That is not a corner case worth tolerating: a cell only
becomes a reach state when it is *fully contained* in the reach region (the `⊆` test in
`IntervalMDPAbstractions`' `convert_property`, pessimistic mode), so a grid that is too
coarse — or merely misaligned with the region — yields no reach states at all and a
vacuous property whose value is 0 everywhere. Written to disk that produced
`robot_2d_S10_I5`, whose `"reach": []` then failed to even load.
"""
function infinite_time_reachavoid(spec, eps; restrict_to_initial::Bool = false)
    prop = IntervalMDP.system_property(spec)
    if !(prop isa IntervalMDP.FiniteTimeReachAvoid ||
         prop isa IntervalMDP.InfiniteTimeReachAvoid)
        error("expected a reach-avoid property, got $(typeof(prop))")
    end

    reach = IntervalMDP.reach(prop)
    avoid = IntervalMDP.avoid(prop)

    isempty(reach) && error("""
                            empty reach set — the specification would be vacuous (value 0 everywhere).
                            A cell becomes a reach state only when it is fully contained in the reach
                            region, so the state grid has to be both fine enough to fit a cell inside
                            the region and aligned with it. Choose a different state split.
                            """)

    # Positional 4-arg form, as in the harness (`override_specification`): `reach`/`avoid`
    # come back as `CartesianIndex` vectors, which the `restrict_to_initial` keyword
    # constructor does not accept.
    return IntervalMDP.Specification(
        IntervalMDP.InfiniteTimeReachAvoid(reach, avoid, eps, restrict_to_initial),
        IntervalMDP.satisfaction_mode(spec),
        IntervalMDP.strategy_mode(spec),
    )
end

# The upstream example systems all declare `initial_region = EmptySet(2)`
# (examples/systems/{car_parking,robot_2d,van_der_pol,linear_stochastically_switched}.jl), so
# `abstraction` records an empty initial-state set (IntervalMDPAbstractions
# src/abstractions/abstraction.jl:131-139). `write_intervalmdp_jl_model` stores that as a
# zero-length array and the reader turns it back into `AllStates()`, which is unrecoverable.
# A model in that state silently disables `restrict_to_initial` in the solver's termination
# criterion (IntervalMDP src/termination.jl:38-42 returns the all-states criterion unwrapped)
# and makes the benchmark's `initial_gap_*` trace columns NaN. Substituting a real region is
# what puts S0 on the model.

"The time horizon handed to the upstream `*_sys` builders. Irrelevant to the stored problem —
`infinite_time_reachavoid` rewraps the property — but the builders require one."
const TIME_HORIZON = 10

"""
    abstract_with_initial_region(sys, spec, initial_region, state_abs, input_abs, target_model)

Abstract `sys` with `initial_region` substituted for its own (empty) initial set, so the
resulting IMDP carries a real S0. Returns `(imdp, abstract_spec)` just like `abstraction`.

Errors on an empty S0: a region that misses the grid entirely would reintroduce exactly the
silent failure this exists to prevent.
"""
function abstract_with_initial_region(sys, spec, initial_region, state_abs, input_abs,
                                      target_model)
    sys = System(IntervalMDPAbstractions.dynamics(sys), initial_region)
    imdp, abstract_spec =
        abstraction(AbstractionProblem(sys, spec), state_abs, input_abs, target_model)

    S0 = IntervalMDP.initial_states(imdp)
    isempty(S0) && error("""
                         empty initial-state set — `initial_region` $(initial_region) overlaps
                         no cell of the state grid with non-zero measure. Written to disk that
                         becomes `AllStates()`, which silently disables `restrict_to_initial`
                         and yields NaN `initial_gap_*`. Move or enlarge the region; note that
                         a region sitting exactly on cell boundaries can also pick up its
                         neighbours, so keep a margin.
                         """)

    @info "initial states" region=initial_region count=length(S0) cells=numregions(state_abs)
    return imdp, abstract_spec
end
