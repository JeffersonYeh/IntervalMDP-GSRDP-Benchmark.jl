include(joinpath(@__DIR__, "../../../../..", "IntervalMDPAbstractions.jl", "examples", "systems", "linear_stochastically_switched.jl"))
using IntervalMDPAbstractions
using IntervalMDP

include(joinpath(@__DIR__, "spec.jl"))

const DIR = joinpath(@__DIR__, "..","..", "problems")
const NAME = "linear_stochastically_switched"
const SPARSE = true
const EPS = 0.0001

# One cell at the coarsest split in use (S10, cell width 0.4), in the bottom-right quadrant —
# clear of both the reach ([-1,0]x[-1,1]) and avoid ([1,2]x[0,1]) regions. Deliberately placed
# *inside* a cell rather than on its boundaries: the exactly-aligned [1.2,1.6]x[-2.0,-1.6]
# picks up 2 cells instead of 1, because floating-point grid boundaries leave a ~1e-16 overlap
# with the neighbour and `iszeromeasure` is exact. Keep this margin when retuning.
const INITIAL_REGION = Hyperrectangle(; low=[1.25, -1.95], high=[1.55, -1.65])

# Mirrors `linear_stochastically_switched_direct` (IntervalMDPAbstractions.jl
# examples/systems/linear_stochastically_switched.jl:33-62). We rebuild its body here rather
# than calling it because the initial region has to go into the `System` *before* `abstraction`
# runs, and `*_direct` neither takes one nor returns `state_abs`.
const X = Hyperrectangle(; low=[-2.0, -2.0], high=[2.0, 2.0])

function main(state_split_per_dim::Integer, restrict_to_initial::Bool=false)
    # create path
    fullname = string(NAME, "_S$(state_split_per_dim)")
    fullpath = joinpath(DIR, fullname)

    # load and create problem
    state_split = (state_split_per_dim, state_split_per_dim)

    sys, spec = linear_stochastically_switched_sys(TIME_HORIZON)
    state_abs = StateUniformGridSplit(X, state_split)
    input_abs = InputDiscrete([Singleton([0.0])])
    imdp, spec = abstract_with_initial_region(sys, spec, INITIAL_REGION, state_abs, input_abs,
                                              SPARSE ? SparseIMDPTarget() : IMDPTarget())

    # Infinite-horizon is what the benchmark solves; the upstream example builds a
    # finite-horizon property. Rewrap before writing anything, so a rejected spec
    # (empty reach set) leaves no half-written problem directory behind.
    spec = infinite_time_reachavoid(spec, EPS; restrict_to_initial=restrict_to_initial)

    mkpath(fullpath)
    IntervalMDP.Data.write_intervalmdp_jl_model(joinpath(fullpath, string(fullname, ".nc")), imdp)
    IntervalMDP.Data.write_intervalmdp_jl_spec(joinpath(fullpath, string(fullname, ".json")), spec)

    println("Saved problem to ", fullpath)
end
