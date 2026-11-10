include(joinpath(@__DIR__, "../../../../..", "IntervalMDPAbstractions.jl", "examples", "systems", "bas_4d.jl"))
using IntervalMDPAbstractions
using IntervalMDP

include(joinpath(@__DIR__, "spec.jl"))

const DIR = joinpath(@__DIR__, "..","..", "problems")
const NAME = "bas_4d"
const SPARSE = true
const EPS = 0.0001

# 4-D building automation system (ARCH-COMP BAS CS1; upstream `building_automation_system_4d`):
# two zone temperatures (x1, x2) and two radiator temperatures (x3, x4), one heater input.
#
# AUGMENTED property. Upstream is finite-horizon safety of the region of interest, which as an
# unbounded property has no reach set (and infinite-time safety under Gaussian noise decays to 0).
# Here: reach the comfort band x1, x2 in [19.75, 20.25] (radiators unconstrained) while staying
# in the region of interest — leaving it is the sink, i.e. the avoid set.
const X = Hyperrectangle(; low=[18.75, 18.75, 29.5, 29.5], high=[21.25, 21.25, 36.5, 36.5])
const U = Hyperrectangle(; low=[17.0], high=[20.0])
const REACH = Hyperrectangle(; low=[19.75, 19.75, 29.5, 29.5], high=[20.25, 20.25, 36.5, 36.5])

# One cell at the coarsest split in use ((5,5,7,7): widths 0.5 x 0.5 x 1 x 1): both zones cold,
# both radiators cold.
const INITIAL_REGION = Hyperrectangle(; low=[18.75, 18.75, 29.5, 29.5], high=[19.25, 19.25, 30.5, 30.5])

"""
    main(state_split::NTuple{4,Int}, input_split::Integer)

x1/x2 splits must be multiples of 5 for the grid to align with the comfort band.
"""
function main(state_split::NTuple{4, Int}, input_split::Integer, restrict_to_initial::Bool=false)
    # create path
    fullname = string(NAME, "_S", join(state_split, "x"), "_I$(input_split)")
    fullpath = joinpath(DIR, fullname)

    # load and create problem — the upstream builder supplies only the dynamics here
    sys, _ = building_automation_system_4d(TIME_HORIZON)
    prop = InfiniteTimeRegionReachAvoid(REACH, EmptySet(4), EPS)
    spec = Specification(prop, Pessimistic, Maximize)

    state_abs = StateUniformGridSplit(X, state_split)
    input_abs = InputLinRange(U, (input_split,))
    imdp, spec = abstract_with_initial_region(sys, spec, INITIAL_REGION, state_abs, input_abs,
                                              SPARSE ? SparseIMDPTarget() : IMDPTarget())

    spec = infinite_time_reachavoid(spec, EPS; restrict_to_initial=restrict_to_initial)

    mkpath(fullpath)
    IntervalMDP.Data.write_intervalmdp_jl_model(joinpath(fullpath, string(fullname, ".nc")), imdp)
    IntervalMDP.Data.write_intervalmdp_jl_spec(joinpath(fullpath, string(fullname, ".json")), spec)

    println("Saved problem to ", fullpath)
end
