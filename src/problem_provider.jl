# Uniform, source-agnostic representation of one constructed benchmark instance.
#
# A `BenchmarkProblem` is the *materialized* result of (an often expensive)
# construction step: the IntervalMDP problem is already built and ready to hand
# to `solve`. Builders registered in `PROBLEM_REGISTRY` (see registry.jl) return
# one of these; everything downstream (run matrix, timing, results) only ever
# sees this type, which is what makes new problem sources one-file additions.

"""
    BenchmarkProblem

Fields:
- `name`           : unique instance id, e.g. `"grid_world_24x24"`.
- `problem`        : a `ControlSynthesisProblem` (or `VerificationProblem`) ready for `solve`.
- `source`         : provenance tag, e.g. `:grid_world`, `:builtin`, `:prism`.
- `metadata`       : cheap-to-compute facts (`num_states`, `num_actions`, dims, source file…)
                     surfaced into `results.csv`.
- `builder_params` : the exact params the builder was called with (provenance).
"""
struct BenchmarkProblem
    name::String
    problem::Any
    source::Symbol
    metadata::Dict{String,Any}
    builder_params::Dict{String,Any}
end

"""
    set_problem(bp::BenchmarkProblem, problem) -> BenchmarkProblem

Return a copy of `bp` with the `problem` field replaced. `BenchmarkProblem` is
immutable, so this constructs a new instance rather than mutating in place; all
other fields (`name`, `source`, `metadata`, `builder_params`) are preserved.
"""
set_problem(bp::BenchmarkProblem, problem) =
    BenchmarkProblem(bp.name, problem, bp.source, bp.metadata, bp.builder_params)

"Number of states of the underlying system, or `missing` if unavailable."
problem_num_states(bp::BenchmarkProblem) = get(bp.metadata, "num_states", missing)

"Number of actions of the underlying system, or `missing` if unavailable."
problem_num_actions(bp::BenchmarkProblem) = get(bp.metadata, "num_actions", missing)

"""
    safe_size_metadata(system) -> Dict

Best-effort `num_states` / `num_actions` extraction; never throws (some system
types may not support both accessors).
"""
function safe_size_metadata(sys)
    md = Dict{String,Any}()
    try
        md["num_states"] = Int(num_states(sys))
    catch
    end
    try
        md["num_actions"] = Int(num_actions(sys))
    catch
    end
    return md
end
