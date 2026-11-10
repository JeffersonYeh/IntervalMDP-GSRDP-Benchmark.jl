# Solution quality + correctness. A faster-but-wrong algorithm is useless, so
# every cell records how many iterations it took, the final gap, whether it
# converged, and — optionally — how far its value function is from a configured
# trusted reference. All derived from the single untimed instrumented solve;
# none of this runs inside the timed region.

struct QualityResult
    num_iterations::Union{Int,Missing}
    bellman_updates::Union{Int,Missing}
    final_gap_max::Union{Float64,Missing}
    final_gap_mean::Union{Float64,Missing}
    converged::Union{Bool,Missing}
    primary_value::Union{Vector{Float64},Missing}   # primary bound, for ref comparison
end

empty_quality() = QualityResult(missing, missing, missing, missing, missing, missing)

"The model's initial-state indices (S0), or `Int32[]` if it has none / declares `AllStates`."
function problem_initial_states(bp)::Vector{Int32}
    try
        init = IntervalMDP.initial_states(system(bp.problem))
        return init isa IntervalMDP.AllStates ? Int32[] : Int32.(collect(init))
    catch
        return Int32[]
    end
end

"Specification convergence tolerance, or `missing`."
function problem_convergence_eps(bp)
    try
        return Float64(convergence_eps(system_property_of(bp)))
    catch
        return missing
    end
end

"""
    extract_quality(executor, sol, bp, S0, restrict_to_initial) -> QualityResult

`sol` is the returned solution; `executor` the populated `TraceExecutor` from the
instrumented solve. The final U-L gap is read from the solution's `residual` (the
elementwise gap), restricted to the initial states `S0` when `restrict_to_initial`,
and `converged` is derived from it (`gap_max < eps`) — matching the cross-trial
validation report. Falls back to the executor's per-iteration trace gap if the
solution does not expose `residual`.

`num_iterations` and `bellman_updates` are DIFFERENT quantities and come from
different places. The solution carries only the iteration counter; the Bellman-update
count reaches us solely through the trace callback, so it is read from the executor
(see `final_bellman_updates`). A sampled strategy does a small batch of updates per
iteration and a full sweep does `num_states × (num_actions + 1)`, so the two differ by
orders of magnitude and are not interchangeable.
"""
function extract_quality(executor::TraceExecutor, sol, bp,
                         S0::AbstractVector{<:Integer}=Int32[],
                         restrict_to_initial::Bool=false)
    iters = try Int(num_iterations(sol)) catch; missing end
    bupd = final_bellman_updates(executor)
    V = try vec(Float64.(value_function(sol))) catch; missing end
    eps = problem_convergence_eps(bp)

    gap = try vec(Float64.(residual(sol))) catch; nothing end
    if gap === nothing
        gmax, gmean = final_gap(executor)               # fallback: per-iteration trace gap
    else
        g = (restrict_to_initial && !isempty(S0)) ? @view(gap[S0]) : gap
        gmax, gmean = maximum(g), mean(g)
    end
    conv = (eps === missing || isnan(gmax)) ? missing : (gmax < eps)
    return QualityResult(iters, bupd, gmax, gmean, conv, V)
end

"""
    censored_quality(executor) -> QualityResult

Quality for a run stopped by the per-solve deadline. There is no solution to read a
`residual` from, so the gap comes from the partial trace's last recorded iteration and
`converged` is `false` by construction — a censored run demonstrably did not reach its
tolerance. `primary_value` is `missing`, so such a run never feeds the reference
comparison or the cross-trial bound check.

Both counts likewise come from the partial trace. They must land in the fields that
match their units: `final_iteration_count` in `num_iterations` and
`final_bellman_updates` in `bellman_updates`, so that a censored row is directly
comparable with a completed one column by column.
"""
function censored_quality(executor::TraceExecutor)
    gmax, gmean = final_gap(executor)
    return QualityResult(final_iteration_count(executor),
                         final_bellman_updates(executor),
                         isnan(gmax) ? missing : gmax,
                         isnan(gmean) ? missing : gmean,
                         false, missing)
end

struct ReferenceAgreement
    ref_max_diff::Union{Float64,Missing}
    ref_mean_diff::Union{Float64,Missing}
    ref_max_diff_initial::Union{Float64,Missing}
    agrees::Union{Bool,Missing}
end

empty_agreement() = ReferenceAgreement(missing, missing, missing, missing)

"""
    value_agreement(V_alg, V_ref, initial_states, eps; tol_factor=10.0)

Correctness scaffolding. The reference-agreement check has been removed pending a
new correctness method; this currently returns `empty_agreement()` regardless of
inputs. The signature and `ReferenceAgreement` struct are kept so the report
columns and call sites stay in place for the replacement.
"""
function value_agreement(V_alg, V_ref, initial_states::AbstractVector{<:Integer}, eps;
                         tol_factor::Real=10.0)
    return empty_agreement()
end
