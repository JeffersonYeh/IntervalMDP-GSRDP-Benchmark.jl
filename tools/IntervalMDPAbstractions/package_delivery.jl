include(joinpath(@__DIR__, "../../..", "midterm", "problem.jl"))
using ArchCompStochasticModels
using IntervalMDPAbstractions
using IntervalMDP
using LazySets


const DIR = joinpath(@__DIR__, "..","..", "problems")
const NAME = "package_delivery"
const SPARSE = true
const EPS = 0.0001

function main(state_split_per_dim::Integer, input_split_per_dim::Integer, restrict_to_initial::Bool=false)
    # create path
    fullname = string(NAME, "_S$(state_split_per_dim)_I$(input_split_per_dim)")
    fullpath = joinpath(DIR, fullname)

    mkpath(fullpath)

    # load and create problem
    state_abs, input_abs, sets = get_abstractions(state_split_per_dim, input_split_per_dim)

    imdp = product_delivery_2_imdp(state_split_per_dim, input_split_per_dim)
    spec = product_delivery_2_spec(restrict_to_initial, state_abs, sets)

    IntervalMDP.Data.write_intervalmdp_jl_model(joinpath(fullpath, string(fullname, ".nc")), imdp)
    IntervalMDP.Data.write_intervalmdp_jl_spec(joinpath(fullpath, string(fullname, ".json")), spec)

    println("Saved problem to ", fullpath)
end


# Package Delivery Construction
function get_abstractions(state_split_per_dim, input_split_per_dim)
    arch_comp_problem = ArchCompStochasticModels.package_delivery_synthesis()
    arch_comp_system = arch_comp_problem.system
   
    # 4. Define State & Input Space Abstraction
    X = arch_comp_system.state_space
    U = ArchCompStochasticModels.control_space(arch_comp_system)
    Usub = Hyperrectangle(; low=[-1.0, -1.0], high=[1.0, 1.0])

    xdims = ArchCompStochasticModels.num_continuous_dims(arch_comp_system)
    udims = LazySets.dim(U)

    state_abs = StateUniformGridSplit(
        X,
        ntuple(_ -> state_split_per_dim, xdims)
    )
    input_abs = InputLinRange(
        Usub, 
        ntuple(_ -> input_split_per_dim, udims)
    )


    arch_comp_spec = arch_comp_problem.specification
    arch_comp_prop = arch_comp_spec.underlying_prop
    sets = arch_comp_prop.sets

    state_abs, input_abs, sets
end

"""
    initial_region_states(state_abstraction, sets) -> Vector{Int32}

The abstract states overlapping the `p1` region, i.e. the model's S0.

Recorded unconditionally, *not* gated on `restrict_to_initial`. Initial states are a property
of the model; whether convergence is restricted to them is a property of the specification,
and the harness overrides that flag from the TOML anyway (benchmark src/registry.jl). Keeping
them separate means `restrict_to_initial` can be flipped in a config without regenerating any
`.nc`. Gating them together is what left every other problem with an empty S0, which writes to
disk as `AllStates()` and then silently disables the restriction (IntervalMDP
src/termination.jl:38-42) and yields NaN `initial_gap_*` in the traces.
"""
function initial_region_states(state_abstraction, sets)
    S0 = compute_region_states(state_abstraction, sets["p1"])
    isempty(S0) && error("""
                         empty initial-state set — the `p1` region overlaps no cell of the
                         state grid. Written to disk that becomes `AllStates()`, which silently
                         disables `restrict_to_initial` and yields NaN `initial_gap_*`.
                         """)
    return S0
end

function compute_region_states(state_abstraction, region)
    # compute reach set for region

    num_states = numregions(state_abstraction)
    states = Int32[]
    grid = diameter(state_abstraction.state_space) / state_abstraction.splits[1]

    for (i, source_region) in enumerate(IntervalMDPAbstractions.regions(state_abstraction)) # for every abstract state

        if strict_intersects(source_region, region, grid)
            push!(states, Int32(i))
        end
    end

    return states
end

function strict_intersects(H1, H2, grid)
    l1 = round.(Int, low(H1) ./ grid)
    h1 = round.(Int, high(H1) ./ grid)
    l2 = round.(Int, low(H2) ./ grid)
    h2 = round.(Int, high(H2) ./ grid)

    return all(max.(l1, l2) .< min.(h1, h2))
end

function product_delivery_2_imdp(state_split_per_dim, input_split_per_dim)

    # 1. Load Problem
    arch_comp_problem = ArchCompStochasticModels.package_delivery_synthesis()
    arch_comp_system = arch_comp_problem.system
    arch_comp_spec = arch_comp_problem.specification

    @assert arch_comp_spec isa ArchCompStochasticModels.ControllerSynthesisSpecification
    arch_comp_prop = arch_comp_spec.underlying_prop
    @assert arch_comp_prop isa ArchCompStochasticModels.scLTLSpecification


    # 2. Dynamics Abstraction
    Tx = ArchCompStochasticModels.transition_kernel(arch_comp_system)
    stddev = sqrt.(Tx.variance)
    noise = AdditiveDiagonalGaussianNoise(stddev)

    @assert Tx.mean isa Linear2
    additive_dynamics = AffineAdditiveNoiseDynamics(Tx.mean.A, Tx.mean.B, noise)
    system = System(additive_dynamics)

    if SPARSE
        target_model = SparseIMDPTarget()
    else
        target_model = IMDPTarget()
    end

    # 3. State & Input Space Abstraction
    state_abs, input_abs, sets = get_abstractions(state_split_per_dim, input_split_per_dim)

    # 4. Abstract System — initial states come from the p1 region. This is the only place
    #    S0 is recorded; the spec's restrict_to_initial flag and the benchmark's trace
    #    callback both read it back off the model.
    S0 = initial_region_states(state_abs, sets)
    imdp = abstraction_imdp(system, state_abs, input_abs, target_model; initial_states = S0)

    return imdp
end

function product_delivery_2_spec(restrict_to_initial, state_abs, sets)

    reach = compute_region_states(state_abs, sets["p3"])
    avoid = compute_region_states(state_abs, sets["p2"])
    push!(avoid, numregions(state_abs)+1) # add sink state to avoid set

    # `restrict_to_initial` is only a flag on the property; the initial *states* live on the
    # model (see `product_delivery_2_imdp`). The harness overrides this flag from the TOML
    # anyway, so it is the model's S0 that actually decides whether restriction takes effect.
    prop = InfiniteTimeReachAvoid(
        reach, avoid, EPS;
        restrict_to_initial = restrict_to_initial,
    )
    spec = Specification(prop, Pessimistic, Maximize)

    return spec
end