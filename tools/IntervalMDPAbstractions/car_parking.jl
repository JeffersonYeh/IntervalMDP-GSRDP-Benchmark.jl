include(joinpath(@__DIR__, "../../../../..", "IntervalMDPAbstractions.jl", "examples", "systems", "car_parking.jl"))
using IntervalMDPAbstractions
using IntervalMDP

include(joinpath(@__DIR__, "spec.jl"))

const DIR = joinpath(@__DIR__, "..","..", "problems")
const NAME = "car_parking"
const SPARSE = true
const EPS = 0.0001

# One cell at the coarsest split in use (S10, cell width 2.0), in the bottom-left corner —
# clear of both the reach ([4,10]x[-4,0]) and avoid ([4,10]x[0,4]) regions. Sized deliberately
# small: restricting convergence to S0 only buys anything when S0 is a small fraction of the
# state space. See `abstract_with_initial_region` in spec.jl for why this has to exist at all.
const INITIAL_REGION = Hyperrectangle(; low=[-10.0, -10.0], high=[-8.0, -8.0])

# Mirrors `car_parking_direct` (IntervalMDPAbstractions.jl examples/systems/car_parking.jl:62-90).
# We rebuild its body here rather than calling it because the initial region has to go into the
# `System` *before* `abstraction` runs, and `*_direct` neither takes one nor returns `state_abs`.
const X = Hyperrectangle(; low=[-10.0, -10.0], high=[10.0, 10.0])
const U = Hyperrectangle(; low=[-1.0, -1.0], high=[1.0, 1.0])

function main(state_split_per_dim::Integer, input_split_per_dim::Integer, restrict_to_initial::Bool=false)
    # create path
    fullname = string(NAME, "_S$(state_split_per_dim)_I$(input_split_per_dim)")
    fullpath = joinpath(DIR, fullname)

    # load and create problem
    state_split = (state_split_per_dim, state_split_per_dim)
    input_split = (input_split_per_dim, input_split_per_dim)

    sys, spec = car_parking_sys(TIME_HORIZON)
    state_abs = StateUniformGridSplit(X, state_split)
    input_abs = InputGridSplit(U, input_split)   # `range_vs_grid=:grid`, the upstream default
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
