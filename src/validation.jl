# Solution-based validation / sanity checking. Each trial returns a
# `ControlSynthesisSolution` whose `value_function` is the lower bound and whose
# `residual` is the elementwise gap (= |upper - lower|); the upper bound is therefore
# `value_function .+ residual`. Two checks run against these, OUTSIDE the timed region:
#
#   1. Per trial — `maximum(gap) < convergence_eps` (did this solve converge).
#   2. Per problem, across ALL trials (any algorithm / sample) — every trial's value
#      interval [lower, upper] must overlap every other trial's interval, elementwise
#      per state. Soundly-bounding trials of the same problem cannot disagree, so a
#      violation flags a correctness bug in some algorithm/sampling variant.
#
# When the property's `restrict_to_initial` is set, both checks consider only the
# model's initial states (the solver only converges those); otherwise all states.

# Cross-trial overlap slack, in units of `convergence_eps`: two trials that each
# bracket the true value within their own gap (< eps) can have lower/upper bounds
# disagree by up to the sum of those gaps (~2 eps) in the worst converged case.
const BOUNDS_TOL_FACTOR = 2.0

"""
    TrialBounds

Per-trial value bounds harvested from a `ControlSynthesisSolution`, plus the metadata
needed to attribute a validation failure back to a specific (problem, algorithm,
sample) trial. `lower`/`upper` are full state vectors; `scope` holds the state indices
the checks apply to (empty ⇒ all states). All trials of one `instance_name` solve the
same `ControlSynthesisProblem`, so their vectors align by index.
"""
struct TrialBounds
    instance_name::String
    cell_id::String
    alg_label::String
    sample_index::Int
    seed::Union{Int,Missing}
    lower::Vector{Float64}
    upper::Vector{Float64}
    scope::Vector{Int32}
    eps::Float64
    gap_max::Float64
    converged::Bool
end

# A view restricted to `scope`, or the whole vector when scope is empty (all states).
_scoped(v::AbstractVector, scope::AbstractVector{<:Integer}) =
    isempty(scope) ? v : @view v[scope]

"""
    extract_trial_bounds(sol, bp, S0, restrict_to_initial; cell_id, alg_label, sample_index, seed)
        -> Union{TrialBounds,Nothing}

Build a `TrialBounds` from one trial's solution. `lower = value_function(sol)`,
`gap = residual(sol)`, `upper = lower .+ gap`. Scope is `S0` when `restrict_to_initial`
is set and the model has initial states, else all states. Returns `nothing` (logged) if
the solution does not expose `value_function`/`residual`, so a bad trial never aborts
the sweep. `eps` defaults to the problem's `convergence_eps`.
"""
function extract_trial_bounds(sol, bp, S0::AbstractVector{<:Integer},
                              restrict_to_initial::Bool;
                              cell_id::AbstractString, alg_label::AbstractString,
                              sample_index::Integer, seed=missing,
                              eps=problem_convergence_eps(bp))
    lower = try vec(Float64.(value_function(sol))) catch; nothing end
    gap   = try vec(Float64.(residual(sol)))       catch; nothing end
    if lower === nothing || gap === nothing
        @warn "skipping trial in validation: solution lacks value_function/residual" cell=cell_id sample=sample_index
        return nothing
    end
    upper = lower .+ gap
    scope = (restrict_to_initial && !isempty(S0)) ? Int32.(collect(S0)) : Int32[]
    e = eps === missing ? NaN : Float64(eps)
    gmax = maximum(_scoped(gap, scope))
    conv = isnan(e) ? false : (gmax < e)
    return TrialBounds(String(bp.name), String(cell_id), String(alg_label),
                       Int(sample_index), seed === missing ? missing : Int(seed),
                       lower, upper, scope, e, gmax, conv)
end

# --- validation records ----------------------------------------------------

"Per-trial validation outcome (one row in validation.csv)."
struct TrialValidation
    instance_name::String
    cell_id::String
    alg_label::String
    sample_index::Int
    seed::Union{Int,Missing}
    scope::String                 # "initial" | "all"
    n_scope_states::Int
    eps::Float64
    gap_max::Float64
    converged::Bool
    bounds_violation::Float64     # max amount this trial's interval fails to overlap the rest
    bounds_consistent::Bool
end

"Per-problem rollup (one entry in validation_summary.json)."
struct ProblemValidation
    instance_name::String
    n_trials::Int
    n_converged::Int
    max_violation::Float64        # max over states of (Lmax - Umin), tolerance not subtracted
    consistent::Bool
end

"""
    validate_bounds(by_instance; tol_eps_factor=BOUNDS_TOL_FACTOR)
        -> (trials::Vector{TrialValidation}, problems::Vector{ProblemValidation})

For each instance, the pairwise overlap conditions (`lowerᵢ ≤ upperⱼ` and
`upperᵢ ≥ lowerⱼ` for all i,j over the scope) are equivalent to the per-state extrema
condition `Lmax(s) ≤ Umin(s) + tol`, where `Lmax = maxₜ lowerₜ`, `Umin = minₜ upperₜ`
and `tol = tol_eps_factor * eps`. Each trial's `bounds_violation` is the worst amount
its own interval misses the common intersection,
`maxₛ max(lowerᵢ(s) − Umin(s), Lmax(s) − upperᵢ(s))`.
"""
function validate_bounds(by_instance::AbstractDict{String,<:AbstractVector{TrialBounds}};
                         tol_eps_factor::Real=BOUNDS_TOL_FACTOR)
    trials = TrialValidation[]
    problems = ProblemValidation[]

    for instance in sort(collect(keys(by_instance)))
        tbs = by_instance[instance]
        isempty(tbs) && continue

        scope = tbs[1].scope
        @assert all(t -> t.scope == scope, tbs) "inconsistent scope across trials of $instance"
        @assert all(t -> length(t.lower) == length(tbs[1].lower), tbs) "mismatched state-space length for $instance"

        # Per-state extrema over the scope. `_scoped` returns the full vector when scope
        # is empty (all-states case); reduce trial-wise.
        lowers = [collect(_scoped(t.lower, scope)) for t in tbs]
        uppers = [collect(_scoped(t.upper, scope)) for t in tbs]
        Lmax = reduce((a, b) -> max.(a, b), lowers)
        Umin = reduce((a, b) -> min.(a, b), uppers)

        eps = maximum(t.eps for t in tbs)            # tolerance off the (shared) eps
        tol = (isnan(eps) ? 0.0 : tol_eps_factor * eps)
        max_violation = maximum(Lmax .- Umin)        # >0 means some state's intervals miss each other
        prob_consistent = max_violation <= tol

        for (i, t) in enumerate(tbs)
            viol = maximum(max.(lowers[i] .- Umin, Lmax .- uppers[i]))
            push!(trials, TrialValidation(
                instance, t.cell_id, t.alg_label, t.sample_index, t.seed,
                isempty(scope) ? "all" : "initial",
                isempty(scope) ? length(t.lower) : length(scope),
                t.eps, t.gap_max, t.converged, viol, viol <= tol))
        end

        push!(problems, ProblemValidation(instance, length(tbs),
                                          count(t -> t.converged, tbs),
                                          max_violation, prob_consistent))
    end

    return trials, problems
end

# --- report writing --------------------------------------------------------

"""
    write_validation_report(dir, trials, problems) -> (csv_path, json_path)

Writes per-trial `validation.csv` and per-problem `validation_summary.json` to `dir`,
`@warn`s each non-converged / inconsistent entry, and logs a final rollup.
"""
function write_validation_report(dir::AbstractString,
                                 trials::AbstractVector{TrialValidation},
                                 problems::AbstractVector{ProblemValidation})
    csv_path = joinpath(dir, "validation.csv")
    json_path = joinpath(dir, "validation_summary.json")

    rows = [(instance_name = t.instance_name, cell_id = t.cell_id, alg_label = t.alg_label,
             sample_index = t.sample_index, seed = t.seed, scope = t.scope,
             n_scope_states = t.n_scope_states, eps = t.eps, gap_max = t.gap_max,
             converged = t.converged, bounds_violation = t.bounds_violation,
             bounds_consistent = t.bounds_consistent) for t in trials]
    CSV.write(csv_path, DataFrame(rows))

    summary = [Dict("instance_name" => p.instance_name, "n_trials" => p.n_trials,
                    "n_converged" => p.n_converged, "max_violation" => p.max_violation,
                    "consistent" => p.consistent) for p in problems]
    open(io -> JSON.print(io, summary, 2), json_path, "w")

    for t in trials
        t.converged || @warn "trial did not converge" instance=t.instance_name cell=t.cell_id sample=t.sample_index gap_max=t.gap_max eps=t.eps
        t.bounds_consistent || @warn "trial bounds inconsistent with other trials" instance=t.instance_name cell=t.cell_id sample=t.sample_index violation=t.bounds_violation
    end
    n_unconverged = count(t -> !t.converged, trials)
    n_inconsistent = count(p -> !p.consistent, problems)
    @info "validation complete" trials=length(trials) problems=length(problems) unconverged_trials=n_unconverged inconsistent_problems=n_inconsistent report=csv_path

    return csv_path, json_path
end
