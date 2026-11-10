# Per-problem structural facts, written once per experiment to `problems.csv`.
#
# Unlike `results.csv` (one row per problem × algorithm × sample), `problems.csv`
# has exactly one row per problem instance and records the *shape* of the model —
# size and connectivity — independent of any algorithm:
#   num_states        : |S|
#   num_actions       : |A|
#   num_maybe_states  : states that are neither reach nor avoid (the states value
#                       iteration actually iterates on; reach/avoid are absorbing)
#   num_transitions   : number of edges, i.e. (source-action, target) pairs with a
#                       non-zero upper transition probability, summed over marginals
#   initial_states    : S0 as a space-separated index list (empty when the model
#                       declares `AllStates()`), so the analysis can mark the initial
#                       set without reopening the model — see analysis/gap_video.jl.
#                       Space-separated rather than comma-separated to stay one CSV field.
# A field is `missing` when the underlying accessor is unavailable for that model.

"Run `f`, returning its `Int` result or `missing` if it throws."
_try_int(f) = try Int(f()) catch; missing end

"Count matrix entries strictly greater than zero, without densifying sparse input."
_count_positive(M::SparseArrays.AbstractSparseMatrix) = count(>(0), SparseArrays.nonzeros(M))
_count_positive(M::AbstractMatrix) = count(>(0), M)

"""
    num_transitions(sys) -> Int | missing

Total number of edges: across every transition marginal, the count of
(source-action, target) pairs whose upper bound (`lower + gap`) is non-zero.
Falls back to the embedded Markov process for product systems; `missing` if the
model exposes no marginals.
"""
function num_transitions(sys)
    ms = try
        marginals(sys)
    catch
        try marginals(IntervalMDP.markov_process(sys)) catch; return missing end
    end
    total = 0
    for marginal in ms
        sets = IntervalMDP.ambiguity_sets(marginal)
        # upper = lower + gap; an edge exists wherever upper > 0.
        total += _count_positive(sets.lower .+ sets.gap)
    end
    return total
end

"""
    num_maybe_states(bp, num_states) -> Int | missing

Number of "maybe" states: those that are neither in the property's reach set nor
its avoid set (the absorbing target/sink states). `missing` if the state count is
unknown; reach/avoid default to empty when the property does not define them.
"""
function num_maybe_states(bp, ns)
    ns === missing && return missing
    try
        prop = system_property_of(bp)
        # reach/avoid are indices into the state space — `Int` vectors for plain
        # IMDPs, `CartesianIndex` vectors for factored/product models. Collect the
        # raw elements (no numeric conversion) so both kinds dedup correctly.
        target = Set{Any}()
        try union!(target, IntervalMDP.reach(prop)) catch end
        try union!(target, IntervalMDP.avoid(prop)) catch end
        return ns - length(target)
    catch
        return missing
    end
end

"""
    problem_stats(bp::BenchmarkProblem) -> NamedTuple

One `problems.csv` row of structural facts for a built problem.
"""
function problem_stats(bp::BenchmarkProblem)
    sys = system(bp.problem)
    ns  = _try_int(() -> num_states(sys))
    return (
        problem          = bp.name,
        source           = String(bp.source),
        num_states       = ns,
        num_actions      = _try_int(() -> num_actions(sys)),
        num_maybe_states = num_maybe_states(bp, ns),
        num_transitions  = num_transitions(sys),
        initial_states   = join(problem_initial_states(bp), ' '),
        status           = "ok",
    )
end

"""
    problem_stats_error(name, status) -> NamedTuple

A `problems.csv` row with no structural facts. `status` separates the two reasons a
row can be empty — `"build_error"` (the builder threw) from `"not_built"` (the sweep
was censored or killed before reaching this instance) — which are indistinguishable
from the `missing`s alone but mean very different things when reading a partial run.
"""
function problem_stats_error(name, status::AbstractString)
    return (
        problem          = String(name),
        source           = missing,
        num_states       = missing,
        num_actions      = missing,
        num_maybe_states = missing,
        num_transitions  = missing,
        initial_states   = missing,
        status           = String(status),
    )
end

"""
    write_problems_csv(path, specs, problem_cache)

Write one `problems.csv` row per unique problem instance (in `specs` order,
de-duplicated by `instance_name`), reading the already-built `BenchmarkProblem`
(or build Exception) from `problem_cache`.

Called on every cache miss during the sweep, not just at the end: `num_transitions` is
the x-axis of the scaling fit, and a file written only after the loop is lost to any
kill. Because every configured instance always gets a row — `not_built` until it is
built — a mid-sweep file is self-describing.

Written atomically (temp file + rename) so a kill mid-write cannot leave a truncated
CSV that the analysis would then mis-parse.
"""
function write_problems_csv(path::AbstractString, specs, problem_cache)
    seen = Set{String}()
    rows = NamedTuple[]
    for spec in specs
        spec.instance_name in seen && continue
        push!(seen, spec.instance_name)
        bpx = get(problem_cache, spec.instance_name, nothing)
        if bpx isa BenchmarkProblem
            push!(rows, problem_stats(bpx))
        else
            push!(rows, problem_stats_error(spec.instance_name,
                                            bpx === nothing ? "not_built" : "build_error"))
        end
    end
    isempty(rows) || _write_atomic(io -> CSV.write(io, DataFrame(rows)), path)
    return path
end
