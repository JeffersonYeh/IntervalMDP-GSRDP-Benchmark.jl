using ArchCompStochasticModels
using IntervalMDPAbstractions
using IntervalMDP
using LazySets

include(joinpath(@__DIR__, "spec.jl"))

const DIR = joinpath(@__DIR__, "..","..", "problems")
const NAME = "patrol_robot"
const SPARSE = true
const EPS = 0.0001

# ARCH-COMP "reduced patrol robot" (ArchCompStochasticModels `reduced_patrol_robot_infinite_time_reachavoid`),
# the one benchmark whose published property already is infinite-time reach-avoid:
#   x+ = x + 10 u1 [cos u2, sin u2] + w,  w ~ N(0, 0.75 I),  X = [-10,10]^2, U = [-1,1]^2,
#   reach [5,7]^2, avoid [-2,2]^2.
# Abstraction mirrors `odimdp_rpr_it_ra` (IntervalMDPAbstractions.jl arch-comp/2025/patrol_robot.jl:104),
# but to a plain sparse IMDP rather than an odIMDP, as every other generator here does.
#
# Reach/avoid are taken from `reduced_patrol_robot_regions()`, NOT from the problem's property:
# `InfiniteTimeReachAvoidSpecification`'s fields are `(avoid_set, target_set)`, but
# `reduced_patrol_robot_infinite_time_reachavoid` constructs it as `(T, A)`, so its
# `target_set` is the avoid box [-2,2]^2 and vice versa (and the arch-comp script inherits it).

# One cell at the coarsest split in use (S20, cell width 1.0) in the bottom-left corner, on
# the far side of the avoid region from the reach region.
const INITIAL_REGION = Hyperrectangle(; low=[-9.0, -9.0], high=[-8.0, -8.0])

function main(state_split_per_dim::Integer, input_split_per_dim::Integer, restrict_to_initial::Bool=false)
    # create path
    fullname = string(NAME, "_S$(state_split_per_dim)_I$(input_split_per_dim)")
    fullpath = joinpath(DIR, fullname)

    # load and create problem
    arch_comp_problem = ArchCompStochasticModels.reduced_patrol_robot_infinite_time_reachavoid()
    arch_comp_system = arch_comp_problem.system
    reach_region, avoid_region = ArchCompStochasticModels.reduced_patrol_robot_regions()

    Tx = ArchCompStochasticModels.transition_kernel(arch_comp_system)
    @assert Tx.mean isa Smooth2
    noise = AdditiveDiagonalGaussianNoise(sqrt.(Tx.variance))

    X = arch_comp_system.state_space
    U = ArchCompStochasticModels.control_space(arch_comp_system)
    dyn = NonlinearAdditiveNoiseDynamics(Tx.mean.func, LazySets.dim(X), LazySets.dim(U), noise)

    prop = InfiniteTimeRegionReachAvoid(reach_region, avoid_region, EPS)
    spec = Specification(prop, Pessimistic, Maximize)

    state_abs = StateUniformGridSplit(X, (state_split_per_dim, state_split_per_dim))
    input_abs = InputLinRange(U, (input_split_per_dim, input_split_per_dim))
    imdp, spec = abstract_with_initial_region(System(dyn), spec, INITIAL_REGION, state_abs, input_abs,
                                              SPARSE ? SparseIMDPTarget() : IMDPTarget())

    # Already infinite-time; the rewrap still sets eps/restrict_to_initial and rejects an
    # empty reach set before anything is written.
    spec = infinite_time_reachavoid(spec, EPS; restrict_to_initial=restrict_to_initial)

    mkpath(fullpath)
    IntervalMDP.Data.write_intervalmdp_jl_model(joinpath(fullpath, string(fullname, ".nc")), imdp)
    IntervalMDP.Data.write_intervalmdp_jl_spec(joinpath(fullpath, string(fullname, ".json")), spec)

    println("Saved problem to ", fullpath)
end
