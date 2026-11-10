using IntervalMDPAbstractions
using IntervalMDP
using LazySets

include(joinpath(@__DIR__, "spec.jl"))

const DIR = joinpath(@__DIR__, "..","..", "problems")
const NAME = "room_temp"
const SPARSE = true
const EPS = 0.0001

# IMPaCT's multi-room temperature examples (IMPaCT examples/ex_3Droom-S/roomTemp3D.cpp and
# ex_5Droom-S/roomTemp5D.cpp): n rooms, rooms 1 and 3 with a heater (inputs u1, u2 in [0,1]),
#   x_i+ = (a - γ u_i) x_i + η Σ_{j ∈ N(i)} x_j + γ T_h u_i + β T_e + w_i     (heated rooms)
#   x_i+ =  a x_i          + η Σ_{j ∈ N(i)} x_j               + β T_e + w_i     (other rooms)
# with a = 1 - 2η - β, β = 0.022, γ = 0.05, T_h = 50, T_e = -1, X = [19,21]^n, and
#   3D: η = 0.20, neighbours all-to-all,   w ~ N(0, I/50)
#   5D: η = 0.30, neighbours on a ring,    w ~ N(0, I/100).
# The u·x term makes them bilinear, hence `NonlinearAdditiveNoiseDynamics`; with singleton
# inputs they are affine in x, so its first-order Taylor model is exact.
#
# AUGMENTED property. IMPaCT checks safety of X (finite-horizon in 3D; infinite-horizon safety
# under Gaussian noise decays to 0). Here: warm both heated rooms to >= 20.25 (3D) / >= 20.2 (5D),
# the other rooms free, while no room leaves X (leaving X is the sink, i.e. the avoid set).
# Tighter targets were tried on 3D S8 and are near-vacuous under the pessimistic abstraction:
# heated rooms in the top band [20.5,21] *and* room 2 >= 19.5 gives V(S0) = 0.003 (max 0.19 over
# all states); heated rooms in [20.5,21] alone at most 0.29. An all-rooms band does not work
# either: an unheated room between two rooms held at h1, h2 settles at (η(h1 + h2) - β)/(2η + β), which is below 20 for
# every h1, h2 inside X (3D: 19.85 at h = 21), so one band around 20 for every room is
# unreachable from below.
#
# The initial cell is cool but viable. Starting every room at 19 is not: the unheated room moves
# first, before any heating reaches it, and drops out of X on the first step whatever the input.

const DIMS = Dict(
    # η, neighbours of each room, noise variance, reach low corner, initial cell
    3 => (η = 0.20, nbrs = [[2, 3], [1, 3], [1, 2]], var = 1 / 50,
          reach_low = [20.25, 19.0, 20.25], init = ([19.5, 19.5, 19.5], [19.75, 19.75, 19.75])),
    5 => (η = 0.30, nbrs = [[2, 5], [1, 3], [2, 4], [3, 5], [4, 1]], var = 1 / 100,
          reach_low = [20.2, 19.0, 20.2, 19.0, 19.0],
          init = ([19.4, 19.4, 19.4, 19.4, 19.4], [19.8, 19.8, 19.8, 19.8, 19.8])),
)
const HEATED = (1, 3)   # rooms with heaters u1, u2
const β, γ, T_h, T_e = 0.022, 0.05, 50.0, -1.0

function room_dynamics(n)
    p = DIMS[n]
    a = 1 - 2 * p.η - β
    f(x, u) = [begin
                   h = findfirst(==(i), HEATED)
                   drift = p.η * sum(x[j] for j in p.nbrs[i]) + β * T_e
                   h === nothing ? a * x[i] + drift :
                   (a - γ * u[h]) * x[i] + drift + γ * T_h * u[h]
               end for i in 1:n]
    return NonlinearAdditiveNoiseDynamics(f, n, 2, AdditiveDiagonalGaussianNoise(fill(sqrt(p.var), n)))
end

"""
    main(n, state_split_per_dim, input_split_per_dim)

`n` ∈ (3, 5). The reach band has to fall on grid lines: 3D needs a multiple of 8 cells per
dimension (width 0.25), 5D a multiple of 5 (width 0.4).
"""
function main(n::Integer, state_split_per_dim::Integer, input_split_per_dim::Integer,
              restrict_to_initial::Bool=false)
    fullname = string(NAME, "_$(n)d_S$(state_split_per_dim)_I$(input_split_per_dim)")
    fullpath = joinpath(DIR, fullname)
    p = DIMS[n]

    X = Hyperrectangle(; low=fill(19.0, n), high=fill(21.0, n))
    U = Hyperrectangle(; low=[0.0, 0.0], high=[1.0, 1.0])
    reach = Hyperrectangle(; low=p.reach_low, high=fill(21.0, n))
    initial_region = Hyperrectangle(; low=p.init[1], high=p.init[2])

    prop = InfiniteTimeRegionReachAvoid(reach, EmptySet(n), EPS)
    spec = Specification(prop, Pessimistic, Maximize)

    state_abs = StateUniformGridSplit(X, ntuple(_ -> state_split_per_dim, n))
    input_abs = InputLinRange(U, (input_split_per_dim, input_split_per_dim))
    imdp, spec = abstract_with_initial_region(System(room_dynamics(n)), spec, initial_region,
                                              state_abs, input_abs,
                                              SPARSE ? SparseIMDPTarget() : IMDPTarget())

    spec = infinite_time_reachavoid(spec, EPS; restrict_to_initial=restrict_to_initial)

    mkpath(fullpath)
    IntervalMDP.Data.write_intervalmdp_jl_model(joinpath(fullpath, string(fullname, ".nc")), imdp)
    IntervalMDP.Data.write_intervalmdp_jl_spec(joinpath(fullpath, string(fullname, ".json")), spec)

    println("Saved problem to ", fullpath)
end
