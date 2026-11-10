# The experiment is the cartesian product {problems} × {algorithms} — one RunUnit
# (= one cell) per pair. Each cell is solved `samples` times as the samples of one
# benchmark (see benchmark.jl); the per-sample seeds are resolved once for the whole
# sweep by `resolve_sample_seeds` and shared across all cells (paired comparison).
# The cell id is deterministic and collision-free; it is also the per-cell output
# directory name.

struct RunUnit
    problem::ProblemSpec
    algorithm::AlgorithmSpec
    id::String
end

"Sanitize a label into a filesystem-safe id fragment."
_safe(s) = replace(String(s), r"[^A-Za-z0-9_.+-]" => "-")

run_unit_id(p::ProblemSpec, a::AlgorithmSpec) = string(_safe(p.instance_name), "__", _safe(a.label))

"Draw `n` random per-sample seeds from OS entropy (a fresh set on each invocation)."
draw_run_seeds(n::Integer) = rand(Random.RandomDevice(), 1:typemax(Int32), Int(n))

"""
    resolve_sample_seeds(cfg) -> Vector{Int}

The per-sample seeds for the whole sweep, chosen once and shared across every cell
(paired comparison): the explicit `cfg.benchmark.seeds` if given, otherwise
`cfg.benchmark.samples` seeds drawn randomly from OS entropy.
"""
resolve_sample_seeds(cfg::ExperimentConfig) =
    isempty(cfg.benchmark.seeds) ? draw_run_seeds(cfg.benchmark.samples) : cfg.benchmark.seeds

"""
    build_run_matrix(cfg) -> Vector{RunUnit}

Full cartesian product of {problems} × {algorithms}, one RunUnit per pair.
(GSRDP × finite-horizon) incompatibilities are NOT removed here — they can only be
detected after the problem is constructed, so the driver skips them at run time and
records `status=\"skipped:finite_horizon\"`.)
"""
function build_run_matrix(cfg::ExperimentConfig)::Vector{RunUnit}
    units = RunUnit[]
    for p in cfg.problems, a in cfg.algorithms
        push!(units, RunUnit(p, a, run_unit_id(p, a)))
    end
    return units
end

"True if the problem's specification has a fixed-horizon (finite-time) property,
which GSRDP rejects."
function is_finite_horizon(bp)
    try
        return IntervalMDP.isfinitetime(system_property_of(bp))
    catch
        return false
    end
end
