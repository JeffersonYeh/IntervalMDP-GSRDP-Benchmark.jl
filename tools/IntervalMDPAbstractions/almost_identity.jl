include(joinpath(@__DIR__, "../../../../..", "IntervalMDPAbstractions.jl", "examples", "systems", "almost_identity.jl"))
using IntervalMDPAbstractions
using IntervalMDP

include(joinpath(@__DIR__, "spec.jl"))

const DIR = joinpath(@__DIR__, "..","..", "problems")
const NAME = "almost_identity"
const SPARSE = true
const EPS = 0.0001

# n-D "almost identity" linear system (upstream `almost_identity_sys`): x+ = A x + w with
# A = 0.7 I - 0.1 (cyclic shift), w ~ N(0, 0.01 I), on [-1,1]^n. It has NO input, so the IMDP
# has a single action per state — a pure verification (interval Markov chain) problem, useful
# as a dimension-scaling family where GSRDP's action sampling plays no role.
#
# AUGMENTED property. Upstream is finite-horizon safety. Here: reach the centre box
# [-0.25,0.25]^n, leaving [-1,1]^n is the sink (avoid).

"""
    main(n, state_split_per_dim)

`state_split_per_dim` must be a multiple of 8 for the grid to align with the reach box.
"""
function main(n::Integer, state_split_per_dim::Integer, restrict_to_initial::Bool=false)
    # create path
    fullname = string(NAME, "_n$(n)_S$(state_split_per_dim)")
    fullpath = joinpath(DIR, fullname)

    # load and create problem — the upstream builder supplies only the dynamics here
    sys, _ = almost_identity_sys(n, TIME_HORIZON)
    reach_region = Hyperrectangle(; low=fill(-0.25, n), high=fill(0.25, n))
    prop = InfiniteTimeRegionReachAvoid(reach_region, EmptySet(n), EPS)
    spec = Specification(prop, Pessimistic, Maximize)

    # One cell in the positive corner.
    cell = 2 / state_split_per_dim
    initial_region = Hyperrectangle(; low=fill(1 - cell, n), high=fill(1.0, n))

    X = Hyperrectangle(; low=fill(-1.0, n), high=fill(1.0, n))
    state_abs = StateUniformGridSplit(X, ntuple(_ -> state_split_per_dim, n))
    input_abs = InputDiscrete([Singleton([0.0])])
    imdp, spec = abstract_with_initial_region(sys, spec, initial_region, state_abs, input_abs,
                                              SPARSE ? SparseIMDPTarget() : IMDPTarget())

    spec = infinite_time_reachavoid(spec, EPS; restrict_to_initial=restrict_to_initial)

    mkpath(fullpath)
    IntervalMDP.Data.write_intervalmdp_jl_model(joinpath(fullpath, string(fullname, ".nc")), imdp)
    IntervalMDP.Data.write_intervalmdp_jl_spec(joinpath(fullpath, string(fullname, ".json")), spec)

    println("Saved problem to ", fullpath)
end
