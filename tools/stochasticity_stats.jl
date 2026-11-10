#!/usr/bin/env julia
# How random is each problem's transition structure? No-solve, per-(s,a) and per-state
# stochasticity metrics, read straight from the interval bounds (companion to
# tools/mixing_stats.jl; same model reader, quantile grid and output conventions).
#
# Structural out-degree (`outdeg` in mixing_stats: successors with u > 0) overstates how
# random a transition is when most successors carry tiny mass -- an abstraction cell can have
# 40 successors of which two hold 90% of the probability. These metrics weigh the successors:
#
#   per (s, a), over the non-terminal sources s and their stored actions a:
#     outdeg    successors with u > 0 (as in mixing_stats)
#     neff      effective number of successors, exp(H(p̂)), H the Shannon entropy (nats) of the
#               normalised midpoint distribution p̂(t) ∝ (l(t) + u(t)) / 2 -- the number of
#               equally likely successors with the same entropy (1 = deterministic)
#     neff_ratio  neff / outdeg in [1/outdeg, 1], the share of the support that carries mass
#               (1 = all successors equally likely, or a single successor)
#     pmax      max_t p̂(t), the midpoint mass of the most likely successor (1 = deterministic)
#     lmax      max_t l(t), the largest guaranteed lower-bound mass: one fixed successor keeps it,
#               i.e. under every adversary (≥ 0.9: near-deterministic whatever the adversary)
#   per non-terminal state s:
#     outdeg_state  distinct successors with u > 0 over all actions of s
#
# Writes <out>/<problem>.csv (q and the inverse ECDF of each metric on the QLEVELS grid, like
# results/mixing/dists/) and merges a row per problem into <out>/summary.csv:
#   n_sa, n_nt, outdeg_mean, outdeg_state_mean, neff_mean, neff_median, neff_ratio_median,
#   pmax_median,
#   frac_near_det (share of (s,a) with pmax ≥ 0.9), frac_forced_det (lmax ≥ 0.9),
#   frac_single (outdeg = 1)
#
# Usage (from the repository root):
#   julia --project tools/stochasticity_stats.jl [--out DIR] [problems_dir] name ...
# Default out DIR is results/mixing/dists_stoch. Peak memory ≈ load_model of the .nc (the
# largest models, ~30M transitions, stay under ~3 GB); run one problem per process.

include(joinpath(@__DIR__, "mixing_stats.jl"))   # load_model, load_spec, align_lower, cols, QLEVELS, ...

const NEAR_DET = 0.9

function stochasticity(dir, name, out)
    m = load_model(joinpath(dir, name, name * ".nc"))
    reach, avoid, _, _ = load_spec(joinpath(dir, name, name * ".json"))
    n = m.n
    terminal = falses(n); terminal[reach] .= true; terminal[avoid] .= true
    maybe = findall(!, terminal)
    la = align_lower(m)

    outdeg = Int[]; neff = Float64[]; neff_ratio = Float64[]; pmax = Float64[]; lmax = Float64[]
    outdeg_state = Int[]
    mark = zeros(Int, n)
    for s in maybe
        distinct = 0
        for c in cols(m, s)
            rng = m.ucol[c]:(m.ucol[c+1]-1)
            d = 0; tot = 0.0; lm = 0.0
            for j in rng
                u = m.uval[j]; u > 0 || continue
                d += 1; tot += (la[j] + u) / 2; lm = max(lm, la[j])
                t = m.urow[j]
                mark[t] != s && (mark[t] = s; distinct += 1)
            end
            d == 0 && continue
            H = 0.0; pm = 0.0
            for j in rng
                u = m.uval[j]; u > 0 || continue
                p = (la[j] + u) / 2 / tot
                p > 0 && (H -= p * log(p))
                pm = max(pm, p)
            end
            push!(outdeg, d); push!(neff, exp(H)); push!(neff_ratio, exp(H) / d)
            push!(pmax, pm); push!(lmax, lm)
        end
        push!(outdeg_state, distinct)
    end

    qcol(v) = isempty(v) ? fill(missing, length(QLEVELS)) :
              (sv = sort(v); [sv[clamp(ceil(Int, p * length(sv)), 1, length(sv))] for p in QLEVELS])
    CSV.write(joinpath(out, name * ".csv"),
              DataFrame(q = collect(QLEVELS), outdeg = qcol(Float64.(outdeg)), neff = qcol(neff),
                        neff_ratio = qcol(neff_ratio),
                        pmax = qcol(pmax), lmax = qcol(lmax),
                        outdeg_state = qcol(Float64.(outdeg_state))))
    m_or(v) = isempty(v) ? missing : mean(v)
    row = (problem = name, n_sa = length(outdeg), n_nt = length(maybe),
           outdeg_mean = m_or(outdeg), outdeg_state_mean = m_or(outdeg_state),
           neff_mean = m_or(neff), neff_median = isempty(neff) ? missing : median(neff),
           neff_ratio_median = isempty(neff_ratio) ? missing : median(neff_ratio),
           pmax_median = isempty(pmax) ? missing : median(pmax),
           frac_near_det = m_or(pmax .>= NEAR_DET), frac_forced_det = m_or(lmax .>= NEAR_DET),
           frac_single = m_or(outdeg .== 1))
    @printf("  %-36s |S_nt| %7d  (s,a) %8d  outdeg %7.2f  neff %6.2f  pmax~ %.3f  near-det %.2f  forced %.2f\n",
            name, row.n_nt, row.n_sa, coalesce(row.outdeg_mean, NaN), coalesce(row.neff_mean, NaN),
            coalesce(row.pmax_median, NaN), coalesce(row.frac_near_det, NaN),
            coalesce(row.frac_forced_det, NaN))
    return row
end

function main_stoch(args)
    out = joinpath(@__DIR__, "..", "results", "mixing", "dists_stoch")
    if !isempty(args) && args[1] == "--out"
        out = args[2]; args = args[3:end]
    end
    dir = joinpath(@__DIR__, "..", "problems")
    if !isempty(args) && isdir(args[1])
        dir = args[1]; args = args[2:end]
    end
    mkpath(out)
    rows = [stochasticity(dir, a, out) for a in args]
    isempty(rows) && return
    # merge into summary.csv by problem, so one-problem-per-process runs accumulate
    path = joinpath(out, "summary.csv")
    new = DataFrame(rows)
    old = isfile(path) ? CSV.read(path, DataFrame) : similar(new, 0)
    CSV.write(path, sort(vcat(filter(r -> !(r.problem in new.problem), old), new; cols = :union), :problem))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_stoch(ARGS)
end
