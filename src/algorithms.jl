# Algorithm builders. Each receives the constructed problem (the Bellman default
# depends on the model) plus the config params, and returns a ready-to-solve
# `BuiltAlgorithm`. The GSRDP sampling strategy lives ON the constructor; its
# termination criterion is no longer an algorithm option — it is auto-derived from
# the specification's property (a gap criterion from `convergence_eps`, optionally
# restricted to the model's initial states when the property sets `restrict_to_initial`).
# Adding a new algorithm is a single `register_algorithm!` here — then select it by
# name from any TOML.

function _descriptor(params)
    d = Dict{String,Any}()
    d["bellman"] = lowercase(String(get(params, "bellman", "default")))
    d["sampling"] = lowercase(String(get(params, "sampling", "exhaustive_state")))
    haskey(params, "k") && (d["k"] = Int(params["k"]))
    # Only these three reach `results.csv` (see `assemble_row` in results.jl),
    # so there is no point widening this: the trajectory sampler's dozen-odd
    # knobs are recorded per algorithm in the run's `resolved_config.toml`,
    # keyed by the label, and `validate_config` guarantees labels are unique.
    return d
end

# Generalized Sampling-based Robust Dynamic Programming — the thesis algorithm.
register_algorithm!("gsrdp", function (bp, params)
    # A size-relative `k_frac` becomes this problem's absolute `k` here, the first point
    # the model is known. The label is built from the unresolved params, so it matches
    # the config-time label and stays the same across problems; the descriptor gets the
    # resolved `k`, so `results.csv` records the k each problem actually ran with.
    rp = resolve_k_frac(params, bp)
    alg = GeneralizedSamplingbasedRobustDynamicProgramming(
        resolve_bellman(bp, rp);
        sampling_strategy=resolve_sampling(rp),  # termination auto-derived from the spec
        bound_update=resolve_bound_update(rp),
    )
    return BuiltAlgorithm(alg, algo_label("gsrdp", params), "gsrdp", _descriptor(rp))
end)
