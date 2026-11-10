#!/usr/bin/env julia
# How far the initial states are from the reach set and from the avoid set, measured apart:
# per initial state s ∈ S0 ∖ (reach ∪ avoid), no solve of the IMDP, same graph and midpoint
# chain as tools/mixing_stats.jl (report-plots/depth.md, absorption-time.md §9).
#
#   depth            fewest u > 0 steps to reach ∪ avoid          (mixing_stats `depth`)
#   depth_reach      fewest u > 0 steps to reach
#   depth_avoid      fewest u > 0 steps to avoid
#   depth_avoid_nosink  ... to avoid ∖ {abstraction sink} (missing without such a sink)
#   h_reach, h_avoid absorption probabilities into reach / avoid under the midpoint chain P
#   tau              expected steps to reach ∪ avoid, = h_reach τ_reach + h_avoid τ_avoid
#                    (missing when h_reach + h_avoid < 1: some mass is never absorbed)
#   tau_reach        E[N | absorbed in reach], N the absorption step (missing when h_reach = 0)
#   tau_avoid        E[N | absorbed in avoid]
#
# Conditional times: with X ∈ {reach, avoid}, D_X the non-terminal states with a u > 0 path
# to X (exactly those with h_X > 0) and r_X(s) the one-step probability into X,
#   (I − P_DD) h_X = r_X,   (I − P_DD) g_X = h_X,   τ_X = g_X / h_X,
# since g_X(s) = Σ_k Pr_s(N > k, absorbed in X) = E_s[N · 1{absorbed in X}]. Restricting to
# D_X keeps I − P_DD nonsingular even when other states never reach a terminal.
# Infinite depths and undefined times are written as missing.
#
# Writes one row per (problem, initial state) to <out>/s0_horizon.csv; re-running a subset
# replaces those problems' rows. Peak memory ≈ mixing_stats without the set structure: run one
# problem per process (pimdp_* are the largest, ~34M midpoint entries, iterative solves).
#
# Usage (from the repository root):
#   julia --project tools/s0_horizon.jl [--out DIR] [problems_dir] name ...
# Default out DIR is results/mixing.

include(joinpath(@__DIR__, "sink_free_depth.jl"))   # mixing_stats helpers + abstraction_sink

"Non-terminal states with a u > 0 edge into `targets` (the BFS seeds at depth 1)."
function frontier(m, maybe, targets)
    front = falses(m.n)
    for s in maybe, c in cols(m, s), j in m.ucol[c]:(m.ucol[c+1]-1)
        m.uval[j] > 0 && targets[m.urow[j]] && (front[s] = true)
    end
    return findall(front)
end

"One-step probability into `targets` under the midpoint chain (normalised midpoint, actions averaged)."
function step_into(m, maybe, targets, la)
    r = zeros(m.n)
    for s in maybe, c in cols(m, s)
        tot = 0.0
        for j in m.ucol[c]:(m.ucol[c+1]-1); tot += (la[j] + m.uval[j]) / 2; end
        for j in m.ucol[c]:(m.ucol[c+1]-1)
            m.uval[j] > 0 && targets[m.urow[j]] && (r[s] += (la[j] + m.uval[j]) / 2 / tot / m.A)
        end
    end
    return r
end

"""
Solve (I − Pm) x = b for b ≥ 0 and ρ(Pm) < 1, as absorption_time does: LU, then QR (UMFPACK's
LU is silently wrong on the nearly singular coin chains), each accepted only if finite,
non-negative and with residual ≤ 1e-9 · max(1, max x); otherwise (or above `maxnnz_direct`)
the monotone iteration x ← b + Pm x from 0. Returns (x, exact).
"""
function absorbing_solve(Pm, b; maxnnz_direct = 2_000_000, maxiter = 100_000, maxtime = 600.0)
    n = length(b)
    n == 0 && return Float64[], true
    if nnz(Pm) <= maxnnz_direct
        A = I_sparse(n) - Pm
        good(x) = all(isfinite, x) && minimum(x) >= -1e-9 * max(1.0, maximum(x)) &&
                  maximum(abs.(A * x .- b)) <= 1e-9 * max(1.0, maximum(x))
        for f in (lu, qr)
            x = try f(A) \ b catch; fill(NaN, n) end
            good(x) && return max.(x, 0.0), true
        end
    end
    x = zeros(n); xn = similar(x); t0 = time()
    for _ in 1:maxiter
        mul!(xn, Pm, x); xn .+= b
        maximum(i -> xn[i] > 0 ? abs(xn[i] - x[i]) / xn[i] : 0.0, 1:n) < 1e-10 && return xn, true
        x, xn = xn, x
        time() - t0 > maxtime && break
    end
    return x, false
end

function s0_horizon(dir, name)
    t0 = time()
    m = load_model(joinpath(dir, name, name * ".nc"))
    reach, avoid, _, _ = load_spec(joinpath(dir, name, name * ".json"))
    n = m.n
    terminal = falses(n); terminal[reach] .= true; terminal[avoid] .= true
    maybe = findall(!, terminal)
    init = [Int(s) for s in m.init if !terminal[s]]
    length(init) < length(m.init) && println("  $name: $(length(m.init) - length(init)) initial state(s) terminal, skipped")
    la = align_lower(m)
    ptr, adj, _, _ = maybe_graph(m, terminal)
    rptr, radj = reverse_graph(n, ptr, adj)

    in_reach = falses(n); in_reach[reach] .= true
    in_avoid = falses(n); in_avoid[avoid] .= true
    sink = abstraction_sink(m, avoid)
    depth_to(targets) = any(targets) ? bfs(n, rptr, radj, frontier(m, maybe, targets), 1) :
                                       fill(typemax(Int), n)
    fin(d, s) = d[s] < typemax(Int) ? d[s] : missing
    dT, dR, dA = depth_to(terminal), depth_to(in_reach), depth_to(in_avoid)
    dAns = if sink === nothing
        nothing
    else
        t = copy(in_avoid); t[sink] = false; depth_to(t)
    end

    _, P = proxy_matrices(m, terminal, la)
    exact = true
    cond = Dict{Symbol,Any}()
    for (key, targets, d) in ((:reach, in_reach, dR), (:avoid, in_avoid, dA))
        D = [s for s in maybe if d[s] < typemax(Int)]
        pos = Dict(s => i for (i, s) in enumerate(D))
        PD = SparseMatrixCSC{Float64,Int}(P[D, D])
        h, e1 = absorbing_solve(PD, step_into(m, D, targets, la)[D])
        g, e2 = absorbing_solve(PD, h)
        exact &= e1 && e2
        cond[key] = (pos, h, g)
    end
    function cval(key, s)
        pos, h, g = cond[key]
        i = get(pos, s, 0)
        i == 0 || h[i] <= 0 ? (0.0, missing) : (h[i], g[i] / h[i])
    end

    rows = map(init) do s
        hR, tR = cval(:reach, s); hA, tA = cval(:avoid, s)
        tau = hR + hA >= 1 - 1e-9 ? (hR > 0 ? hR * tR : 0.0) + (hA > 0 ? hA * tA : 0.0) : missing
        (problem = name, state = s, depth = fin(dT, s), depth_reach = fin(dR, s),
         depth_avoid = fin(dA, s),
         depth_avoid_nosink = dAns === nothing ? missing : fin(dAns, s),
         h_reach = hR, h_avoid = hA, tau = tau, tau_reach = tR, tau_avoid = tA, exact = exact)
    end
    @printf("  %-36s |S0∖T| %3d  d_reach %s  d_avoid %s  h_reach %.3g  τ_reach %s  τ_avoid %s  %s  %.1fs\n",
            name, length(init), join(unique(r.depth_reach for r in rows), "/"),
            join(unique(r.depth_avoid for r in rows), "/"),
            isempty(rows) ? NaN : mean(r.h_reach for r in rows),
            join(unique(round.(skipmissing(r.tau_reach for r in rows); sigdigits = 3)), "/"),
            join(unique(round.(skipmissing(r.tau_avoid for r in rows); sigdigits = 3)), "/"),
            exact ? "" : "(INEXACT)", time() - t0)
    return rows
end

function main_s0(args)
    out = joinpath(@__DIR__, "..", "results", "mixing")
    if !isempty(args) && args[1] == "--out"
        out = args[2]; args = args[3:end]
    end
    dir = joinpath(@__DIR__, "..", "problems")
    if !isempty(args) && isdir(args[1])
        dir = args[1]; args = args[2:end]
    end
    mkpath(out)
    new = DataFrame(reduce(vcat, [s0_horizon(dir, a) for a in args]; init = NamedTuple[]))
    nrow(new) == 0 && return
    path = joinpath(out, "s0_horizon.csv")
    old = isfile(path) ? CSV.read(path, DataFrame) : similar(new, 0)
    CSV.write(path, sort(vcat(filter(r -> !(r.problem in new.problem), old), new; cols = :union),
                         [:problem, :state]))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_s0(ARGS)
end
