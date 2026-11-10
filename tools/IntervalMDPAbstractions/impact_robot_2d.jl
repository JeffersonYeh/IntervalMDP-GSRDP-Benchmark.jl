using IntervalMDPAbstractions
using IntervalMDP
using LazySets

include(joinpath(@__DIR__, "spec.jl"))

const DIR = joinpath(@__DIR__, "..","..", "problems")
const NAME = "impact_robot_2d_ra"
const SPARSE = true
const EPS = 0.0001

# IMPaCT's 2D robot reach-avoid example (IMPaCT examples/ex_2Drobot-RA-U/robot2D.cpp, identical
# to its ARCH-COMP entry examples/ARCH-COMP/2025/PR_minimal):
#   x+ = x + 2 u1 [cos u2, sin u2] + w,  w ~ N(0, I / 1.3333),  X = [-10,10]^2, U = [-1,1]^2,
#   reach [5,8]^2, avoid [-1.5,1.5]^2, infinite-horizon pessimistic reach-avoid.
# Not a duplicate of patrol_robot/robot_2d: those take steps of 10 u (not 2 u), with u2 over a
# full [-1,1] *polar* range, and reach [5,7]^2 / avoid [-2,2]^2.
#
# For a fixed input the dynamics are a translation, so they are written as affine with B = 2 I
# and one singleton input [u1 cos u2, u1 sin u2] per point of IMPaCT's input grid (the way
# upstream `InputRobot` handles robot_2d), which keeps the one-step image exact. IMPaCT's own
# grid is 40x40 states (eta 0.5) x 21x21 inputs (eta 0.1).

# One cell at the coarsest split in use (S20, cell width 1.0) in the bottom-left corner, on the
# far side of the avoid region from the reach region.
const INITIAL_REGION = Hyperrectangle(; low=[-10.0, -10.0], high=[-9.0, -9.0])

const X = Hyperrectangle(; low=[-10.0, -10.0], high=[10.0, 10.0])
const REACH = Hyperrectangle(; low=[5.0, 5.0], high=[8.0, 8.0])
const AVOID = Hyperrectangle(; low=[-1.5, -1.5], high=[1.5, 1.5])

"The input grid `LinRange(-1, 1, n)^2`, mapped to the translation it causes (before B = 2 I)."
function robot_inputs(n)
    r = LinRange(-1.0, 1.0, n)
    return vec([Singleton([u1 * cos(u2), u1 * sin(u2)]) for (u1, u2) in Iterators.product(r, r)])
end

"""
    main(state_split_per_dim, input_split_per_dim)

`state_split_per_dim` must be a multiple of 20 for the grid to align with the reach region.
"""
function main(state_split_per_dim::Integer, input_split_per_dim::Integer, restrict_to_initial::Bool=false)
    fullname = string(NAME, "_S$(state_split_per_dim)_I$(input_split_per_dim)")
    fullpath = joinpath(DIR, fullname)

    w_stddev = fill(sqrt(1 / 1.3333), 2)
    dyn = AffineAdditiveNoiseDynamics([1.0 0.0; 0.0 1.0], [2.0 0.0; 0.0 2.0],
                                      AdditiveDiagonalGaussianNoise(w_stddev))

    prop = InfiniteTimeRegionReachAvoid(REACH, AVOID, EPS)
    spec = Specification(prop, Pessimistic, Maximize)

    state_abs = StateUniformGridSplit(X, (state_split_per_dim, state_split_per_dim))
    input_abs = InputDiscrete(robot_inputs(input_split_per_dim))
    imdp, spec = abstract_with_initial_region(System(dyn), spec, INITIAL_REGION, state_abs, input_abs,
                                              SPARSE ? SparseIMDPTarget() : IMDPTarget())

    spec = infinite_time_reachavoid(spec, EPS; restrict_to_initial=restrict_to_initial)

    mkpath(fullpath)
    IntervalMDP.Data.write_intervalmdp_jl_model(joinpath(fullpath, string(fullname, ".nc")), imdp)
    IntervalMDP.Data.write_intervalmdp_jl_spec(joinpath(fullpath, string(fullname, ".json")), spec)

    println("Saved problem to ", fullpath)
end
