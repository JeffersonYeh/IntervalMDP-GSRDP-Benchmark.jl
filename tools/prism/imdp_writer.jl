# Shared back end of the explicit-model converters (prism_to_imdp.jl, ../storm/drn_to_imdp.jl):
# per-state choice lists -> padded CSC interval matrices -> IntervalMDP.jl .nc + .json with an
# infinite-time reach-avoid specification, round-tripped before returning.
#
# `choices[s]` is the vector of choices of state `s` (1-based), each a vector of
# `(dest, lower, upper)` tuples with 1-based `dest`. The caller activates the benchmark project
# and loads IntervalMDP, IntervalMDP.Data and SparseArrays.

"Build lower/upper CSC matrices with every state padded to `A` actions."
function build_matrices(choices, n)
    A = maximum(length, choices)
    colptr = Int32[1]
    rowval = Int32[]
    lower = Float64[]
    upper = Float64[]
    for s in 1:n
        cs = choices[s]
        isempty(cs) && error("state $(s - 1) has no choices (PRISM deadlock fixing disabled?)")
        for (c, entries) in enumerate(cs)
            isempty(entries) && error("state $(s - 1) choice $(c - 1) has no transitions")
            sort!(entries; by = first)
            allunique(first.(entries)) || error("state $(s - 1) choice $(c - 1): duplicate successor")
            lsum = sum(e -> e[2], entries)
            usum = sum(e -> e[3], entries)
            (lsum <= 1 + 1e-9 && usum >= 1 - 1e-9) ||
                error("state $(s - 1) choice $(c - 1): infeasible interval (Σl=$lsum, Σu=$usum)")
        end
        for a in 1:A
            for (d, l, u) in cs[min(a, length(cs))]
                push!(rowval, d); push!(lower, l); push!(upper, u)
            end
            push!(colptr, Int32(length(rowval) + 1))
        end
    end
    P̲ = SparseMatrixCSC(n, n * A, colptr, rowval, lower)
    P̅ = SparseMatrixCSC(n, n * A, copy(colptr), copy(rowval), upper)
    padded = count(cs -> length(cs) < A, choices)
    return P̲, P̅, A, padded
end

"Parse the `--strategy` / `--satisfaction` option values shared by the converters."
function parse_modes(strategy, satisfaction)
    s = strategy == "max" ? Maximize :
        strategy == "min" ? Minimize : error("--strategy must be max or min")
    m = satisfaction == "pessimistic" ? Pessimistic :
        satisfaction == "optimistic" ? Optimistic :
        error("--satisfaction must be pessimistic or optimistic")
    return s, m
end

"""
    write_reachavoid_problem(out, name, choices, n, init, reach, avoid, strategy, satisfaction, eps)

Write `<out>/<name>.nc` + `<out>/<name>.json` (infinite-time reach-avoid, `restrict_to_initial =
false` — the harness sets it per config) and check the files round-trip. `init`/`reach`/`avoid`
are sorted 1-based state indices.
"""
function write_reachavoid_problem(out, name, choices, n, init, reach, avoid,
                                  strategy, satisfaction, eps)
    isempty(init) && error("$name: no initial states")
    isempty(reach) && error("$name: no reach states")
    isempty(avoid) && error("$name: no avoid states — IntervalMDP.Data.read_intervalmdp_jl " *
                            "cannot read back an empty \"avoid\" array")
    isempty(intersect(reach, avoid)) || error("$name: reach and avoid overlap")

    P̲, P̅, A, padded = build_matrices(choices, n)
    mdp = IntervalMarkovDecisionProcess(IntervalAmbiguitySets(; lower = P̲, upper = P̅), A, init)
    prop = InfiniteTimeReachAvoid(reach, avoid, eps; restrict_to_initial = false)
    problem = ControlSynthesisProblem(mdp, Specification(prop, satisfaction, strategy))

    mkpath(out)
    model_path = joinpath(out, name * ".nc")
    spec_path = joinpath(out, name * ".json")
    Data.write_intervalmdp_jl_model(model_path, problem)
    Data.write_intervalmdp_jl_spec(spec_path, problem)

    # Round-trip: the files must reproduce what was built.
    rt = Data.read_intervalmdp_jl(model_path, spec_path; control_synthesis = true)
    sys = IntervalMDP.system(rt)
    rprop = IntervalMDP.system_property(IntervalMDP.specification(rt))
    idx(v) = sort!([Int32(first(Tuple(c))) for c in v])
    @assert num_states(sys) == n
    @assert num_actions(sys) == A
    @assert sort!(Int32.(collect(IntervalMDP.initial_states(sys)))) == init
    @assert idx(IntervalMDP.reach(rprop)) == reach
    @assert idx(IntervalMDP.avoid(rprop)) == avoid

    println("$name: $n states × $A actions ($padded states padded), ",
            "$(nnz(P̲)) transitions, init=$(length(init)) reach=$(length(reach)) ",
            "avoid=$(length(avoid)) -> $model_path")
end
