#!/usr/bin/env julia
# Spectral density of the midpoint chain: every eigenvalue magnitude |λ| of P on S_nt, not
# just the spectral radius that mixing_stats --spectral brackets.
#
# P is the same proxy as for the absorption time (mixing_stats.proxy_matrices): each interval
# replaced by its normalised midpoint, actions averaged uniformly, restricted to the
# non-terminal states. P is block-triangular over the SCCs of its support, so its spectrum is
# the union of the diagonal blocks' spectra:
#   - a singleton SCC contributes its self-loop P(s,s) exactly (0 without one);
#   - a larger SCC B contributes the eigenvalues of the dense block P[B,B] (LAPACK geevx,
#     O(|B|³)).
# A dense eigenvalue within its own LAPACK error bound eps·‖P[B,B]‖ / rcond (rcond: reciprocal
# condition number of that eigenvalue) cannot be told apart from 0 and is set to 0. Defective
# zero eigenvalues (nilpotent Jordan blocks, e.g. in the near-deterministic coin chains) come
# out of the solver as rings of radius ~eps^(1/k) (1e-8, 1e-4, ...) with tiny rcond; a fixed
# tolerance cannot remove them without also removing the abstractions' genuine, well-
# conditioned eigenvalues down to 1e-10. All magnitudes are clamped to ≤ 1 (P is
# substochastic). There is one value per non-terminal state.
#
# Writes <out>/<problem>.csv (q and the inverse ECDF of lambda_abs on the QLEVELS grid, like
# results/mixing/dists/) and merges a row per problem into <out>/summary.csv:
#   n_nt, n_scc, max_scc, rho (max |λ|), lambda_median, frac_zero (|λ| = 0),
#   frac_slow (|λ| ≥ SLOW), status, time_s
# A problem whose largest SCC exceeds --max-block is skipped (status "skipped") with no
# quantile file; the dense block and its two eigenvector matrices need ~3 · 8·max_scc² bytes
# plus LAPACK workspace.
#
# Usage (from the repository root):
#   julia --project tools/spectral_density.jl [--out DIR] [--max-block N] [problems_dir] name ...
# Default out DIR is results/mixing/dists_spectral, default --max-block 10000 (~4 GB peak).
# Cluster-only at the default: the problems whose largest SCC is bigger (see the skipped rows
# of summary.csv); rerun them there with --max-block Inf. Run one problem per process.

include(joinpath(@__DIR__, "mixing_stats.jl"))   # load_model, load_spec, align_lower, proxy_matrices, support_sccs, QLEVELS, ...

const SLOW = 0.9

function spectral_density(dir, name, out; max_block)
    t0 = time()
    m = load_model(joinpath(dir, name, name * ".nc"))
    reach, avoid, _, _ = load_spec(joinpath(dir, name, name * ".json"))
    n = m.n
    terminal = falses(n); terminal[reach] .= true; terminal[avoid] .= true
    maybe = findall(!, terminal)
    _, P = proxy_matrices(m, terminal, align_lower(m))
    comp = support_sccs(P, .!terminal)
    blocks = Dict{Int,Vector{Int}}()
    for s in maybe
        push!(get!(blocks, comp[s], Int[]), s)
    end
    max_scc = maximum(length, values(blocks); init = 0)
    row = (problem = name, n_nt = length(maybe), n_scc = length(blocks), max_scc = max_scc)
    if max_scc > max_block
        println("  $name: |S_nt| $(length(maybe)), largest SCC $max_scc > --max-block $max_block, skipped")
        return (; row..., rho = missing, lambda_median = missing, frac_zero = missing,
                frac_slow = missing, status = "skipped", time_s = round(time() - t0; digits = 2))
    end

    lam = Float64[]
    for B in values(blocks)
        if length(B) == 1
            push!(lam, P[B[1], B[1]])
        else
            length(B) >= 2000 && (println("    $name: dense block $(length(B)) ...");
                                  flush(stdout))
            r = LAPACK.geevx!('B', 'V', 'V', 'E', Matrix(P[B, B]))
            wr, wi, abnrm, rconde = r[2], r[3], r[9], r[10]
            for i in eachindex(wr)
                v = hypot(wr[i], wi[i])
                push!(lam, v <= eps() * abnrm / rconde[i] ? 0.0 : min(v, 1.0))
            end
        end
    end

    qcol(v) = isempty(v) ? fill(missing, length(QLEVELS)) :
              (sv = sort(v); [sv[clamp(ceil(Int, p * length(sv)), 1, length(sv))] for p in QLEVELS])
    CSV.write(joinpath(out, name * ".csv"), DataFrame(q = collect(QLEVELS), lambda_abs = qcol(lam)))
    e = isempty(lam)
    res = (; row..., rho = e ? missing : maximum(lam), lambda_median = e ? missing : median(lam),
           frac_zero = e ? missing : mean(lam .== 0), frac_slow = e ? missing : mean(lam .>= SLOW),
           status = "ok", time_s = round(time() - t0; digits = 2))
    @printf("  %-36s |S_nt| %7d  SCCs %7d  largest %6d  ρ %.6f  |λ|~ %.3g  zero %.2f  slow %.3f  %.1fs\n",
            name, res.n_nt, res.n_scc, max_scc, coalesce(res.rho, NaN),
            coalesce(res.lambda_median, NaN), coalesce(res.frac_zero, NaN),
            coalesce(res.frac_slow, NaN), res.time_s)
    return res
end

function main_spectral(args)
    out = joinpath(@__DIR__, "..", "results", "mixing", "dists_spectral")
    max_block = 10_000.0
    while !isempty(args) && args[1] in ("--out", "--max-block")
        args[1] == "--out" ? (out = args[2]) : (max_block = parse(Float64, args[2]))
        args = args[3:end]
    end
    dir = joinpath(@__DIR__, "..", "problems")
    if !isempty(args) && isdir(args[1])
        dir = args[1]; args = args[2:end]
    end
    mkpath(out)
    rows = [spectral_density(dir, a, out; max_block) for a in args]
    isempty(rows) && return
    # merge into summary.csv by problem, so one-problem-per-process runs accumulate
    path = joinpath(out, "summary.csv")
    new = DataFrame(rows)
    old = isfile(path) ? CSV.read(path, DataFrame) : similar(new, 0)
    CSV.write(path, sort(vcat(filter(r -> !(r.problem in new.problem), old), new; cols = :union), :problem))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_spectral(ARGS)
end
