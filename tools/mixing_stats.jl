#!/usr/bin/env julia
# No-solve "mixing" metrics for the benchmark problems: how far value information has to travel
# and how fast maybe-state mass drains into the terminal (reach ∪ avoid) states, read straight
# from the model's interval bounds -- no value iteration, no policy.
#
# Metric families (columns of mixing_stats.csv; per-state distributions in dists/<problem>.csv):
#   propagation  backward BFS depth to the terminals over u > 0 edges (a lower bound on the sweeps
#                before a state's bounds can move), forward reach/depth from S0
#   sets         density, frontier (one-step in-neighbourhood) and co-successor clusters of the
#                reach / avoid / S0 sets, over all u > 0 edges and over each column's 90%-mass
#                core (hubs like an abstraction's sink excluded); depth to reach / avoid apart
#   cycles       SCCs of the maybe-state graph, condensation-DAG depth, sure/possible MECs
#                (acyclic => finite convergence; cycles => geometric)
#   drain        guaranteed one-step exit e(s,a) = max(Σ_T l, 1 − Σ_maybe u) (best/worst action),
#                forced self-loop min_a l(s|s,a)
#   adversary    interval width u − l, degenerate fraction, budget 1 − Σ l, out-degree
#   spectral     (--spectral) a bracket on ρ(Q) of the maybe→maybe block Q of ANY policy/adversary
#                chain, hence also of the optimal one that sets VI's asymptotic rate:
#                  rho_lo      ρ(L), L(s,t) = min_a l(t|s,a) ≤ Q entrywise  (certified lower bound)
#                  rho_hi_nK   (max_s P_max[stay in own SCC K steps])^(1/K) ≥ ρ(Q) (certified
#                              upper bound; K sweeps of a finite-horizon max/max recursion, not
#                              a solve; exactly 1 on a possible MEC, 0 on an acyclic model)
#                  rho_mid     ρ of the action-averaged normalised midpoint chain (a point
#                              estimate of the optimal chain's ρ, NOT a bound on it; the
#                              _s0_lo/_hi columns bracket ρ(midpoint) itself, certified)
#                plus abs_time_mid_*: expected steps to a terminal under that same midpoint
#                chain, (I − P)⁻¹ 1 (one sparse linear solve). On the abstractions every state
#                is one u > 0 step from the sink, so BFS depth is 1 everywhere and this is the
#                propagation measure that still discriminates.
#                `_s0` columns restrict to the maybe states reachable from S0, which is all that
#                matters under restrict_to_initial = true.
# GSRDP updates are Jacobi within an iteration (it reads `previous`), so sweep ORDER does not
# enter any of this.
#
# Reads the .nc/.json directly via tools/end_components.jl (no IntervalMDP model). Memory peaks
# at a few × the .nc size: pimdp_*, package_delivery_S50_I5 and robot_2d_S30_I15 are
# cluster-only (tools/mixing_stats.sbatch); locally use --max-mb.
#
# Usage (from the repository root):
#   julia --project tools/mixing_stats.jl [--spectral] [--horizon K] [--max-mb X] [--verbose]
#                                         [--out DIR] [problems_dir] [name ...]
# Default out DIR is results/mixing. Re-running a subset replaces those problems' rows in an
# existing mixing_stats.csv and keeps the rest, so local and cluster passes can be merged.

using CSV, DataFrames, LinearAlgebra, SparseArrays, Statistics
include(joinpath(@__DIR__, "end_components.jl"))   # Model, load_model, load_spec, tarjan, mec, ...

const EPS_CONV = 1e-5          # baseline convergence_eps, for the predicted-sweeps columns
const QLEVELS = 0:0.005:1      # quantile grid of the per-state distribution files
const VERBOSE = Ref(false)     # --verbose: per-stage elapsed time and peak RSS

stage(name, t0) = VERBOSE[] && @printf("    %-22s %7.1fs  peak RSS %6.0f MB\n", name, time() - t0,
                                        Sys.maxrss() / 2^20)

"Problem family, as in problems/baseline-problems.md §2."
function family(name)
    any(p -> startswith(name, p), ("storm_", "brp_", "crowds_")) && return "Storm"
    startswith(name, "robot_delta") && return "PRISM"
    startswith(name, "coin") && return "PRISM coin"
    any(p -> startswith(name, p), ("package_delivery", "bas_", "patrol_robot")) && return "ARCH-COMP"
    any(p -> startswith(name, p), ("room_temp", "impact_")) && return "IMPaCT"
    any(p -> startswith(name, p), ("multiObj", "pimdp")) && return "Repro package"
    return "IMDPAbstractions"
end

# --- model views ------------------------------------------------------------------------

# Columns (state-action pairs) of state s. Abstraction models store no columns for their
# trailing sink state (stateptr has |S| entries, not |S| + 1), so clamp to what exists.
ncols(m) = length(m.ucol) - 1
cols(m, s) = ((s-1)*m.A+1):min(s*m.A, ncols(m))

"Lower bound of every upper-support entry (0 where l is not stored). Errors if l > 0 off u's support."
function align_lower(m)
    la = zeros(Float64, length(m.uval))
    for c in 1:ncols(m)
        j, jend = m.ucol[c], m.ucol[c+1] - 1
        for i in m.lcol[c]:(m.lcol[c+1]-1)
            r, v = m.lrow[i], m.lval[i]
            v > 0 || continue
            while j <= jend && m.urow[j] != r; j += 1; end   # rows sorted in both, so this is a merge
            j <= jend || error("lower bound outside upper support (column $c, row $r)")
            la[j] = v
        end
    end
    return la
end

"Deduplicated maybe→maybe adjacency over u > 0 edges (any action), as CSR, plus per-state flags."
function maybe_graph(m, terminal)
    n, A = m.n, m.A
    ptr = ones(Int, n + 1); adj = Int32[]
    selfloop = falses(n); to_T = falses(n)
    mark = zeros(Int, n)
    for s in 1:n
        if !terminal[s]
            for c in cols(m, s), j in m.ucol[c]:(m.ucol[c+1]-1)
                t = m.urow[j]
                m.uval[j] > 0 || continue
                if terminal[t]
                    to_T[s] = true
                elseif mark[t] != s
                    mark[t] = s; push!(adj, t)
                    t == s && (selfloop[s] = true)
                end
            end
        end
        ptr[s+1] = length(adj) + 1
    end
    return ptr, adj, selfloop, to_T
end

"Reverse of a CSR adjacency."
function reverse_graph(n, ptr, adj)
    rptr = zeros(Int, n + 1)
    for t in adj; rptr[t+1] += 1; end
    rptr[1] = 1; cumsum!(rptr, rptr)
    fill_ = copy(rptr); radj = Vector{Int32}(undef, length(adj))
    for s in 1:n, e in ptr[s]:ptr[s+1]-1
        t = adj[e]; radj[fill_[t]] = s; fill_[t] += 1
    end
    return rptr, radj
end

"BFS distances (typemax = unreached) from `seeds` (distance d0) along the CSR edges."
function bfs(n, ptr, adj, seeds, d0)
    dist = fill(typemax(Int), n); q = Int[]
    for s in seeds
        dist[s] == typemax(Int) || continue
        dist[s] = d0; push!(q, s)
    end
    head = 1
    while head <= length(q)
        s = q[head]; head += 1
        for e in ptr[s]:ptr[s+1]-1
            t = adj[e]
            if dist[t] == typemax(Int)
                dist[t] = dist[s] + 1; push!(q, t)
            end
        end
    end
    return dist
end

"Longest chain of SCCs (counted in SCCs) in the condensation. Tarjan numbers sinks first."
function condensation_depth(n, ptr, adj, comp)
    ncomp = maximum(comp; init = 0)
    longest = ones(Int, ncomp)
    order = sort!(filter(s -> comp[s] > 0, collect(1:n)); by = s -> comp[s])
    for s in order, e in ptr[s]:ptr[s+1]-1
        t = adj[e]; k, kt = comp[s], comp[t]
        (kt > 0 && kt != k) && (longest[k] = max(longest[k], longest[kt] + 1))
    end
    return maximum(longest; init = 0)
end

# --- spectral bracket -------------------------------------------------------------------

"""
    block_rho(M, comp, masks) -> (brackets, converged, iters)

Certified bracket (lo, hi) on ρ(M) = max over SCC blocks, from Collatz–Wielandt ratios of a
power iteration on the block-diagonal part of M (intra-SCC entries only), shifted to (B + I)/2
so periodic blocks converge. Each SCC is normalised separately; states with comp 0 (terminal)
are ignored. One bracket per entry of `masks`, each a collection of SCC ids to maximise over
(all maybe SCCs, or only the S0-reachable ones for the `_s0` columns). `x0` is an optional
positive start vector over all states.
"""
function block_rho(M::SparseMatrixCSC, comp, masks; maxiter = 20_000, maxtime = 600.0,
                   x0 = nothing)
    n = size(M, 1); ncomp = maximum(comp; init = 0)
    # keep intra-SCC entries only (spectrum of a block-triangular matrix = union of its blocks')
    I, J, V = findnz(M)
    keep = [comp[I[k]] == comp[J[k]] for k in eachindex(I)]
    Bt = sparse(J[keep], I[keep], V[keep], n, n)          # transposed: columns of Bt = rows of B
    # any positive start is valid; a near-Perron one (the absorption times) converges far faster
    x = x0 === nothing ? ones(n) : max.(x0, 1e-300); y = similar(x)
    lo = fill(Inf, ncomp); hi = fill(-Inf, ncomp); mx = zeros(ncomp)
    # every iterate gives a valid bracket, so keep the tightest seen (the last one can drift
    # under roundoff when ρ is within ~1e-7 of 1, e.g. coin2_K1024)
    best_lo = fill(-Inf, ncomp); best_hi = fill(Inf, ncomp)
    t0 = time(); it = 0
    brackets = [(0.0, 0.0) for _ in masks]; conv = false
    while it < maxiter && time() - t0 < maxtime
        it += 1
        # y = (B x + x)/2 with B = Bt'
        @inbounds for s in 1:n
            acc = x[s]
            for e in nzrange(Bt, s); acc += nonzeros(Bt)[e] * x[rowvals(Bt)[e]]; end
            y[s] = acc / 2
        end
        fill!(lo, Inf); fill!(hi, -Inf); fill!(mx, 0.0)
        @inbounds for s in 1:n
            k = comp[s]; k == 0 && continue
            r = y[s] / x[s]
            lo[k] = min(lo[k], r); hi[k] = max(hi[k], r); mx[k] = max(mx[k], y[s])
        end
        @inbounds for s in 1:n; comp[s] > 0 && (x[s] = y[s] / mx[comp[s]]); end
        @inbounds for k in 1:ncomp
            best_lo[k] = max(best_lo[k], lo[k]); best_hi[k] = min(best_hi[k], hi[k])
        end
        conv = true
        for (i, mask) in enumerate(masks)
            l = maximum(2best_lo[k] - 1 for k in mask; init = 0.0)
            h = maximum(2best_hi[k] - 1 for k in mask; init = 0.0)
            brackets[i] = (max(l, 0.0), max(h, 0.0))
            # tight relative to 1 − ρ, which is what the plots and predicted sweeps care about
            conv &= h - l <= 1e-3 * (1 - h) + 1e-12
        end
        conv && break
    end
    return brackets, conv, it
end

"Tarjan SCCs of the support of a maybe×maybe sparse matrix (states outside `active` get 0)."
function support_sccs(M::SparseMatrixCSC, active)
    Mt = sparse(M')                                        # columns of Mt = rows of M
    n = size(M, 1)
    ptr = Mt.colptr; adj = rowvals(Mt)
    return tarjan(n, ptr, adj, active)
end

"""
Maybe×maybe matrices over state indices (terminal rows/cols empty):
  L[s,t] = min_a l(t|s,a)                      (0 unless l > 0 for EVERY action)
  P[s,t] = mean_a p(t|s,a), p = (l+u)/2 normalised to sum 1 over all successors
"""
function proxy_matrices(m, terminal, la)
    n, A = m.n, m.A
    LI, LJ, LV = Int32[], Int32[], Float64[]
    PI, PJ, PV = Int32[], Int32[], Float64[]
    lmin = zeros(n); lcnt = zeros(Int, n); pacc = zeros(n); touched = Int[]; mark = zeros(Int, n)
    for s in 1:n
        terminal[s] && continue
        empty!(touched)
        for c in cols(m, s)
            tot = 0.0
            for j in m.ucol[c]:(m.ucol[c+1]-1); tot += (la[j] + m.uval[j]) / 2; end
            for j in m.ucol[c]:(m.ucol[c+1]-1)
                t = m.urow[j]
                (terminal[t] || m.uval[j] <= 0) && continue
                if mark[t] != s
                    mark[t] = s; push!(touched, t); lmin[t] = Inf; lcnt[t] = 0; pacc[t] = 0.0
                end
                if la[j] > 0
                    lmin[t] = min(lmin[t], la[j]); lcnt[t] += 1
                end
                pacc[t] += (la[j] + m.uval[j]) / 2 / tot / A
            end
        end
        for t in touched
            if lcnt[t] == A
                push!(LI, s); push!(LJ, t); push!(LV, lmin[t])
            end
            push!(PI, s); push!(PJ, t); push!(PV, pacc[t])
        end
    end
    return sparse(LI, LJ, LV, n, n), sparse(PI, PJ, PV, n, n)
end

"O-maximised Σ_t p(t) w(t) [comp(t) = k] over the interval ambiguity set of column c."
function omax_stay(m, la, lsum_c, w, comp, k, c, buf)
    r = m.ucol[c]:(m.ucol[c+1]-1)
    wv(j) = (t = m.urow[j]; comp[t] == k ? w[t] : 0.0)
    v = 0.0
    @inbounds for j in r; v += la[j] * wv(j); end
    budget = 1 - lsum_c
    resize!(buf, length(r)); copyto!(buf, r)
    sort!(buf; by = wv, rev = true)
    @inbounds for j in buf
        budget <= 0 && break
        wt = wv(j)
        wt <= 0 && break
        add = min(m.uval[j] - la[j], budget)
        v += add * wt; budget -= add
    end
    return v
end

"""
    stay_bounds(m, terminal, la, comp, horizons, sets) -> Dict((K, set index) => rho_hi)

Per-SCC stay recursion. w_0 = 1 on maybe states; w_{k+1}(s) = max_a max_{p ∈ Γ(s,a)} Σ_t p(t)
w_k(t) [comp(t) = comp(s)], by O-maximisation (lower bounds first, remaining budget to the
successors with the largest w). Successors outside s's own SCC count as an exit, so w_K(s) is
the largest probability any policy/adversary pair stays inside SCC(s) for K steps. For every
induced Q, ‖Q_C^K‖_∞ ≤ max_{s∈C} w_K(s), and ρ(Q) = max_C ρ(Q_C), so
max_{s∈set} w_K(s)^(1/K) ≥ ρ(Q) over the SCCs meeting `set`. An acyclic model gets 0.
"""
function stay_bounds(m, terminal, la, comp, horizons, sets)
    n = m.n
    w = [terminal[s] ? 0.0 : 1.0 for s in 1:n]; wn = copy(w)
    lsum = zeros(ncols(m))
    for c in 1:ncols(m), j in m.ucol[c]:(m.ucol[c+1]-1); lsum[c] += la[j]; end
    buf = Int[]
    out = Dict{Tuple{Int,Int},Float64}()
    K = maximum(horizons)
    for k in 1:K
        @inbounds for s in 1:n
            terminal[s] && continue
            best = 0.0
            for c in cols(m, s)
                best = max(best, omax_stay(m, la, lsum[c], w, comp, comp[s], c, buf))
            end
            wn[s] = min(best, 1.0)
        end
        w, wn = wn, w
        if k in horizons
            for (i, set) in enumerate(sets)
                mw = maximum((w[s] for s in set); init = 0.0)
                out[(k, i)] = mw^(1 / k)
            end
        end
    end
    return out
end

"""
    absorption_time(P, maybe) -> (t, exact)

Expected steps to reach a terminal state under the proxy chain P (maybe×maybe block),
t = (I − P)⁻¹ 1 on the maybe states. Every solve is checked by its residual: sparse LU first,
then sparse QR (UMFPACK's LU hits ~1e-17 pivots on the coins, where nearly every row of P sums
to 1 within the maybe set, and returns garbage without failing), then `maxiter` sweeps of the
monotone fixed point t ← 1 + P t from 0 (a lower bound, `exact = false` unless it settles).
Direct solves only up to `maxnnz_direct` stored entries: on pimdp_* (≈34M entries over 21–32k
states) LU/QR fill-in heads for a dense 21k² factor and blew a 7 GB budget.
Only called when every maybe state has a path to a terminal, so I − P is nonsingular.
"""
function absorption_time(P::SparseMatrixCSC, maybe; maxnnz_direct = 2_000_000,
                         maxiter = 100_000, maxtime = 600.0)
    Pm = SparseMatrixCSC{Float64,Int}(P[maybe, maybe])
    b = ones(length(maybe))
    if nnz(Pm) <= maxnnz_direct
        A = I_sparse(length(maybe)) - Pm
        good(t) = all(isfinite, t) && minimum(t) >= 1 - 1e-6 &&
                  maximum(abs.(A * t .- b)) <= 1e-6 * max(1.0, maximum(t))
        t = try lu(A) \ b catch; fill(NaN, length(b)) end
        good(t) && return t, true
        t = try qr(A) \ b catch; fill(NaN, length(b)) end
        good(t) && return t, true
        A = nothing
    end
    t = zeros(length(maybe)); tn = similar(t); t0 = time()
    for _ in 1:maxiter
        mul!(tn, Pm, t); tn .+= 1
        maximum(abs.(tn .- t) ./ tn) < 1e-9 && return tn, true
        t, tn = tn, t
        time() - t0 > maxtime && break
    end
    return t, false
end
I_sparse(n) = sparse(1.0I, n, n)

predicted_sweeps(ρ) = ρ <= 0 ? 1.0 : ρ >= 1 ? Inf : max(1.0, log(EPS_CONV) / log(ρ))

# --- per-problem driver -----------------------------------------------------------------

q(v, p) = isempty(v) ? missing : quantile(v, p)
mean_or_missing(v) = isempty(v) ? missing : mean(v)

# --- target / initial set structure -----------------------------------------------------

const HUB_FRAC = 0.5    # a set state with ≥ this share of the maybe states as predecessors is a hub

"Union-find root with path halving."
function uf_find(parent, x)
    @inbounds while parent[x] != x
        parent[x] = parent[parent[x]]; x = parent[x]
    end
    return x
end
uf_union!(parent, a, b) = (ra = uf_find(parent, a); rb = uf_find(parent, b); ra != rb && (parent[rb] = ra); nothing)

"Component sizes of the `members` under a union-find `parent`."
function uf_sizes(parent, members)
    counts = Dict{Int,Int}()
    for t in members; r = uf_find(parent, t); counts[r] = get(counts, r, 0) + 1; end
    return collect(values(counts))
end

const CORE_MASS = 0.9   # a column's core = fewest successors holding this share of its midpoint mass
const HUB_MIN_MAYBE = 10

"""
    mass_core(m, la) -> BitVector over upper entries

Marks, per column (s,a), the smallest set of successors whose midpoint mass (l+u)/2 reaches
CORE_MASS of the column total (largest first). On the abstractions u > 0 almost everywhere
(and l > 0 wherever u > 0), with tails of ~1e-6; the core keeps the successors that carry the
probability, which is what makes the set clusters below mean something there.
"""
function mass_core(m, la)
    core = falses(length(m.uval)); buf = Int[]
    for c in 1:ncols(m)
        r = m.ucol[c]:(m.ucol[c+1]-1)
        resize!(buf, length(r)); copyto!(buf, r)
        filter!(j -> m.uval[j] > 0, buf)
        tot = sum((la[j] + m.uval[j]) / 2 for j in buf; init = 0.0)
        sort!(buf; by = j -> la[j] + m.uval[j], rev = true)
        acc = 0.0
        for j in buf
            core[j] = true; acc += (la[j] + m.uval[j]) / 2
            acc >= CORE_MASS * tot && break
        end
    end
    return core
end

"""
    set_structure(m, terminal, core, X, nmaybe, key) -> (row entries, frontier)

Density and clustering of a state set X (reach, avoid or S0), from the dynamics alone. Two
edge sets: all u > 0 edges, and the `_core` edges from `mass_core`.
  frontier F(X) = maybe states with an edge into X   → `_frontier_frac[_core]` (÷ |M|) and
                  `_frontier_ratio[_core]` (÷ |X|, surface-to-volume: small = compact)
  hubs          = t ∈ X with core predecessors ≥ HUB_FRAC·|M| (only when |M| ≥ HUB_MIN_MAYBE),
                  e.g. a sink every state drifts to; excluded from clustering
  clusters      = components of X \\ hubs, t₁ ~ t₂ when an edge joins them or one column
                  (s,a) reaches both (co-successors)
Returns the row entries keyed `<key>_…` and the u-frontier as a BitVector (the BFS seeds for
the depth to X).
"""
function set_structure(m, terminal, core, X, nmaybe, key)
    n = m.n
    inX = falses(n); inX[X] .= true
    members = findall(inX)
    pred = zeros(Int, n); mark = zeros(Int, n)
    front = falses(n); front_c = falses(n)
    for s in 1:n
        terminal[s] && continue
        for c in cols(m, s), j in m.ucol[c]:(m.ucol[c+1]-1)
            t = m.urow[j]
            (inX[t] && m.uval[j] > 0) || continue
            front[s] = true
            if core[j]
                front_c[s] = true
                mark[t] != s && (mark[t] = s; pred[t] += 1)
            end
        end
    end
    hub = falses(n)
    if nmaybe >= HUB_MIN_MAYBE
        for t in members; pred[t] >= HUB_FRAC * nmaybe && (hub[t] = true); end
    end
    kept = [t for t in members if !hub[t]]
    parent = collect(1:n); parent_c = collect(1:n)
    # Every stored column counts here, terminal ones included: the abstractions keep the cell
    # dynamics of reach/avoid cells, which is the only thing linking an obstacle's interior
    # cells (no maybe state's core reaches them). Storm/PRISM terminals are self-loops: no effect.
    for s in 1:n
        own = inX[s] && !hub[s]           # direct edges from a member
        for c in cols(m, s)
            first = 0; first_c = 0
            for j in m.ucol[c]:(m.ucol[c+1]-1)
                t = m.urow[j]
                (inX[t] && !hub[t] && m.uval[j] > 0) || continue
                first == 0 ? (first = t) : uf_union!(parent, first, t)
                own && uf_union!(parent, s, t)
                if core[j]
                    first_c == 0 ? (first_c = t) : uf_union!(parent_c, first_c, t)
                    own && uf_union!(parent_c, s, t)
                end
            end
        end
    end
    sz = uf_sizes(parent, kept); sz_c = uf_sizes(parent_c, kept)
    nf, nfc = count(front), count(front_c)
    k(x) = Symbol(key, "_", x)
    row = Dict{Symbol,Any}(
        k("n") => length(members), k("frac") => length(members) / n,
        k("frontier_frac") => nmaybe == 0 ? missing : nf / nmaybe,
        k("frontier_frac_core") => nmaybe == 0 ? missing : nfc / nmaybe,
        k("frontier_ratio") => isempty(members) ? missing : nf / length(members),
        k("frontier_ratio_core") => isempty(members) ? missing : nfc / length(members),
        k("hubs") => count(hub),
        k("clusters") => length(sz), k("cluster_max") => maximum(sz; init = 0),
        k("cluster_mean") => mean_or_missing(sz),
        k("cluster_max_frac") => isempty(kept) ? missing : maximum(sz) / length(kept),
        k("clusters_core") => length(sz_c), k("cluster_max_core") => maximum(sz_c; init = 0),
        k("cluster_mean_core") => mean_or_missing(sz_c),
        k("cluster_max_frac_core") => isempty(kept) ? missing : maximum(sz_c) / length(kept),
    )
    return row, front
end

function problem_stats(dir, name; spectral::Bool, horizon::Int, qdir)
    t0 = time()
    nc = joinpath(dir, name, name * ".nc"); js = joinpath(dir, name, name * ".json")
    m = load_model(nc)
    reach, avoid, _, _ = load_spec(js)
    n, A = m.n, m.A
    terminal = falses(n); terminal[reach] .= true; terminal[avoid] .= true
    maybe = findall(!, terminal); nmaybe = length(maybe)
    la = align_lower(m)
    stage("load", t0)

    # propagation
    ptr, adj, selfloop, to_T = maybe_graph(m, terminal)
    rptr, radj = reverse_graph(n, ptr, adj)
    depth = bfs(n, rptr, radj, findall(to_T), 1)          # 1 = one step from a terminal
    dm = [depth[s] for s in maybe if depth[s] < typemax(Int)]
    init_maybe = [Int(s) for s in m.init if !terminal[s]]
    fdist = bfs(n, ptr, adj, init_maybe, 0)
    fwd = [s for s in maybe if fdist[s] < typemax(Int)]
    depth_s0 = maximum((depth[s] for s in init_maybe if depth[s] < typemax(Int)); init = 0)

    stage("propagation", t0)
    # target / initial set structure, and the depth to reach and to avoid separately
    setrow = Dict{Symbol,Any}()
    setdepth = Dict{String,Vector{Int}}()
    core = mass_core(m, la)
    for (key, X) in (("reach", reach), ("avoid", avoid), ("s0", m.init))
        r, front = set_structure(m, terminal, core, X, nmaybe, key)
        merge!(setrow, r)
        key == "s0" && continue
        dX = bfs(n, rptr, radj, findall(front), 1)
        fin = [dX[s] for s in maybe if dX[s] < typemax(Int)]
        setdepth[key] = fin
        setrow[Symbol("depth_", key, "_s0")] =
            maximum((dX[s] for s in init_maybe if dX[s] < typemax(Int)); init = 0)
        setrow[Symbol("depth_", key, "_max")] = maximum(fin; init = 0)
        setrow[Symbol("n_no_path_", key)] = nmaybe - length(fin)
    end
    stage("set structure", t0)
    # cycles
    comp = tarjan(n, ptr, adj, .!terminal)
    csize = zeros(Int, maximum(comp; init = 0))
    for s in maybe; csize[comp[s]] += 1; end
    nontriv(k, s) = csize[k] > 1 || selfloop[s]
    cyclic = [s for s in maybe if nontriv(comp[s], s)]
    nontriv_ids = unique(comp[s] for s in cyclic)
    mecs = Dict(nt => summarize(mec(m, terminal, nt), reachable(m, terminal)) for nt in (:sure, :possible))

    stage("cycles + MECs", t0)
    # drain + adversary freedom
    best_exit = zeros(nmaybe); worst_exit = zeros(nmaybe); self_lmin = zeros(nmaybe)
    widths = Float64[]; budgets = Float64[]; outdeg = Int[]
    ntrans = 0
    for (i, s) in enumerate(maybe)
        be, we, sl = 0.0, 1.0, 1.0
        for c in cols(m, s)
            lT = 0.0; uM = 0.0; ls = 0.0; lsum = 0.0; d = 0
            for j in m.ucol[c]:(m.ucol[c+1]-1)
                t = m.urow[j]; u = m.uval[j]; l = la[j]
                u > 0 || continue
                d += 1; lsum += l
                push!(widths, u - l)
                terminal[t] ? (lT += l) : (uM += u)
                t == s && (ls = l)
            end
            e = clamp(max(lT, 1 - uM), 0.0, 1.0)
            be = max(be, e); we = min(we, e); sl = min(sl, ls)
            push!(budgets, 1 - lsum); push!(outdeg, d); ntrans += d
        end
        best_exit[i] = be; worst_exit[i] = we; self_lmin[i] = sl
    end
    # count terminal-state columns too, to match problems.csv num_transitions
    for s in findall(terminal), c in cols(m, s), j in m.ucol[c]:(m.ucol[c+1]-1)
        m.uval[j] > 0 && (ntrans += 1)
    end

    row = Dict{Symbol,Any}(
        :problem => name, :family => family(name), :status => "ok",
        :num_states => n, :num_actions => A, :num_maybe => nmaybe, :num_s0 => length(m.init),
        :num_transitions => ntrans, :nc_mb => round(filesize(nc) / 2^20; digits = 2),
        :depth_max => isempty(dm) ? missing : maximum(dm),
        :depth_mean => mean_or_missing(dm), :depth_p90 => q(dm, 0.9), :depth_s0 => depth_s0,
        :n_no_path_T => nmaybe - length(dm),
        :fwd_reach_frac => nmaybe == 0 ? missing : length(fwd) / nmaybe,
        :fwd_depth_max => maximum((fdist[s] for s in fwd); init = 0),
        :scc_nontrivial => length(nontriv_ids),
        :scc_largest => maximum((csize[k] for k in nontriv_ids); init = 0),
        :cyclic_frac => nmaybe == 0 ? missing : length(cyclic) / nmaybe,
        :cond_depth => condensation_depth(n, ptr, adj, comp),
        :mec_sure_n => mecs[:sure].n, :mec_sure_states => mecs[:sure].states,
        :mec_possible_n => mecs[:possible].n, :mec_possible_states => mecs[:possible].states,
        :exit_best_median => q(best_exit, 0.5), :exit_best_p10 => q(best_exit, 0.1),
        :exit_best_min => isempty(best_exit) ? missing : minimum(best_exit),
        :frac_no_guaranteed_exit => mean_or_missing(best_exit .<= 0),
        :exit_worst_median => q(worst_exit, 0.5),
        :selfloop_lmin_max => isempty(self_lmin) ? missing : maximum(self_lmin),
        :selfloop_lmin_mean => mean_or_missing(self_lmin),
        :frac_selfloop_forced => mean_or_missing(self_lmin .> 0),
        :width_median => q(widths, 0.5), :width_mean => mean_or_missing(widths),
        :frac_degenerate => mean_or_missing(widths .<= TOL),
        :budget_median => q(budgets, 0.5), :budget_mean => mean_or_missing(budgets),
        :outdeg_mean => mean_or_missing(outdeg), :outdeg_max => maximum(outdeg; init = 0),
    )
    merge!(row, setrow)

    abs_t = Float64[]
    stage("drain + adversary", t0)
    if spectral
        L, P = proxy_matrices(m, terminal, la)
        stage("proxy matrices", t0)
        # Midpoint absorption time first: t = (I − P)⁻¹ 1 satisfies P t = t − 1, so it is close
        # to P's Perron vector on a slowly mixing model and seeds the power iteration below.
        x0 = nothing
        if row[:n_no_path_T] == 0 && nmaybe > 0
            t, exact = absorption_time(P, maybe)
            pos = Dict(s => i for (i, s) in enumerate(maybe))
            row[:abs_time_mid_s0] = mean_or_missing([t[pos[s]] for s in init_maybe])
            row[:abs_time_mid_max] = maximum(t); row[:abs_time_mid_median] = median(t)
            row[:abs_time_mid_exact] = exact
            abs_t = t
            x0 = ones(n); x0[maybe] .= t
            stage("absorption time", t0)
        end
        # L: SCCs of its own (sparser) support; P: same support as the u > 0 graph
        for (label, M, cmp, start) in (("lo", L, support_sccs(L, .!terminal), nothing),
                                       ("mid", P, comp, x0))
            ks_all = unique(cmp[s] for s in maybe)
            ks_s0 = unique(cmp[s] for s in fwd)
            (b_all, b_s0), conv, it = block_rho(M, cmp, [ks_all, ks_s0]; x0 = start)
            if label == "lo"
                row[:rho_lo] = b_all[1]; row[:rho_lo_s0] = b_s0[1]      # certified: lower end
                row[:rho_lo_bracket_hi] = b_all[2]
            else
                # point estimate: geometric middle of the certified bracket in gap (1 − ρ) space
                gmid(b) = 1 - sqrt(max(1 - b[2], 0.0) * max(1 - b[1], 0.0))
                row[:rho_mid] = gmid(b_all); row[:rho_mid_s0] = gmid(b_s0)
                row[:rho_mid_s0_lo] = b_s0[1]; row[:rho_mid_s0_hi] = b_s0[2]
            end
            row[Symbol("rho_", label, "_converged")] = conv
            row[Symbol("rho_", label, "_iters")] = it
        end
        stage("power iterations", t0)
        hz = sort(unique([1, 10, horizon] ∩ (1:horizon)))
        hi = stay_bounds(m, terminal, la, comp, hz, [maybe, fwd])
        for K in hz
            row[Symbol("rho_hi_n", K)] = hi[(K, 1)]
        end
        row[:rho_hi] = hi[(horizon, 1)]; row[:rho_hi_s0] = hi[(horizon, 2)]
        row[:horizon] = horizon
        row[:sweeps_min] = predicted_sweeps(row[:rho_lo_s0])
        row[:sweeps_max] = predicted_sweeps(row[:rho_hi_s0])
    end
    spectral && stage("stay bounds", t0)
    row[:time_s] = round(time() - t0; digits = 2)

    # per-state (and per-transition, for width) marginal distributions, as quantiles
    # inverse ECDF (no interpolation), so a 2-state model reads as two steps, not a ramp
    function qcol(v)
        isempty(v) && return fill(missing, length(QLEVELS))
        sv = sort(v); n = length(sv)
        return [sv[clamp(ceil(Int, p * n), 1, n)] for p in QLEVELS]
    end
    CSV.write(joinpath(qdir, name * ".csv"), DataFrame(
        q = collect(QLEVELS),
        depth = qcol(Float64.(dm)),
        exit_best = qcol(best_exit), exit_worst = qcol(worst_exit),
        selfloop_lmin = qcol(self_lmin), width = qcol(widths), outdeg = qcol(Float64.(outdeg)),
        abs_time_mid = qcol(abs_t),
        depth_reach = qcol(Float64.(setdepth["reach"])), depth_avoid = qcol(Float64.(setdepth["avoid"]))))
    return row
end

# --- main -------------------------------------------------------------------------------

# CSV column order (rows are Dicts); anything not listed goes last, alphabetically
const COLUMN_ORDER = [:problem, :family, :status, :num_states, :num_actions, :num_maybe, :num_s0,
    :num_transitions, :nc_mb, :depth_max, :depth_mean, :depth_p90, :depth_s0, :n_no_path_T,
    :fwd_reach_frac, :fwd_depth_max,
    :reach_n, :reach_frac, :reach_frontier_frac, :reach_frontier_frac_core,
    :reach_frontier_ratio, :reach_frontier_ratio_core, :reach_hubs, :reach_clusters,
    :reach_cluster_max, :reach_cluster_mean, :reach_cluster_max_frac, :reach_clusters_core,
    :reach_cluster_max_core, :reach_cluster_mean_core, :reach_cluster_max_frac_core, :avoid_n,
    :avoid_frac, :avoid_frontier_frac, :avoid_frontier_frac_core, :avoid_frontier_ratio,
    :avoid_frontier_ratio_core, :avoid_hubs, :avoid_clusters, :avoid_cluster_max,
    :avoid_cluster_mean, :avoid_cluster_max_frac, :avoid_clusters_core,
    :avoid_cluster_max_core, :avoid_cluster_mean_core, :avoid_cluster_max_frac_core, :s0_n,
    :s0_frac, :s0_frontier_frac, :s0_frontier_frac_core, :s0_frontier_ratio,
    :s0_frontier_ratio_core, :s0_hubs, :s0_clusters, :s0_cluster_max, :s0_cluster_mean,
    :s0_cluster_max_frac, :s0_clusters_core, :s0_cluster_max_core, :s0_cluster_mean_core,
    :s0_cluster_max_frac_core, :depth_reach_s0, :depth_reach_max, :n_no_path_reach,
    :depth_avoid_s0, :depth_avoid_max, :n_no_path_avoid,
    :scc_nontrivial, :scc_largest, :cyclic_frac, :cond_depth,
    :mec_sure_n, :mec_sure_states, :mec_possible_n, :mec_possible_states, :exit_best_median,
    :exit_best_p10, :exit_best_min, :frac_no_guaranteed_exit, :exit_worst_median,
    :selfloop_lmin_max, :selfloop_lmin_mean, :frac_selfloop_forced, :width_median, :width_mean,
    :frac_degenerate, :budget_median, :budget_mean, :outdeg_mean, :outdeg_max,
    :rho_lo, :rho_lo_s0, :rho_lo_bracket_hi, :rho_lo_converged, :rho_lo_iters,
    :rho_mid, :rho_mid_s0, :rho_mid_s0_lo, :rho_mid_s0_hi, :rho_mid_converged, :rho_mid_iters,
    :rho_hi_n1, :rho_hi_n10, :rho_hi, :rho_hi_s0, :horizon, :sweeps_min, :sweeps_max,
    :abs_time_mid_s0, :abs_time_mid_median, :abs_time_mid_max, :abs_time_mid_exact, :time_s]

function order_columns(df)
    have = propertynames(df)
    first = [c for c in COLUMN_ORDER if c in have]
    return select(df, first, sort(setdiff(have, first)))
end

function parse_args(args)
    opts = Dict{Symbol,Any}(:spectral => false, :horizon => 100, :max_mb => Inf,
                            :out => joinpath("results", "mixing"))
    pos = String[]
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--spectral"; opts[:spectral] = true
        elseif a == "--horizon"; opts[:horizon] = parse(Int, args[i+=1])
        elseif a == "--max-mb"; opts[:max_mb] = parse(Float64, args[i+=1])
        elseif a == "--out"; opts[:out] = args[i+=1]
        elseif a == "--verbose"; VERBOSE[] = true
        else push!(pos, a)
        end
        i += 1
    end
    return opts, pos
end

function main(args)
    opts, pos = parse_args(args)
    dir = isempty(pos) ? "problems" : pos[1]
    names = length(pos) > 1 ? pos[2:end] :
            sort(filter(d -> isfile(joinpath(dir, d, d * ".nc")), readdir(dir)))
    out = opts[:out]; qdir = joinpath(out, "dists"); mkpath(qdir)
    csv = joinpath(out, "mixing_stats.csv")
    old = isfile(csv) ? CSV.read(csv, DataFrame) : DataFrame()
    rows = Dict{Symbol,Any}[]
    for name in names
        nc = joinpath(dir, name, name * ".nc")
        mb = filesize(nc) / 2^20
        row = if mb > opts[:max_mb]
            @printf("%-36s skipped (%.0f MB > --max-mb)\n", name, mb)
            Dict{Symbol,Any}(:problem => name, :family => family(name), :status => "skipped_size",
                             :nc_mb => round(mb; digits = 2))
        else
            try
                r = problem_stats(dir, name; spectral = opts[:spectral], horizon = opts[:horizon], qdir)
                @printf("%-36s %7d maybe  depth %s  cyclic %.2f  exit_med %.3g  %s(%.1fs)\n", name,
                        r[:num_maybe], string(r[:depth_max]), coalesce(r[:cyclic_frac], NaN),
                        coalesce(r[:exit_best_median], NaN),
                        opts[:spectral] ? @sprintf("gap 1−ρ ∈ [%.3g, %.3g]  ", 1 - r[:rho_hi_s0], 1 - r[:rho_lo_s0]) : "",
                        r[:time_s])
                r
            catch e
                @printf("%-36s ERROR %s\n", name, sprint(showerror, e))
                Dict{Symbol,Any}(:problem => name, :family => family(name), :status => "error")
            end
        end
        push!(rows, row)
        # rewrite after every problem so a kill keeps what is done; merge with earlier passes
        new = vcat(DataFrame.(rows)...; cols = :union)
        keep = isempty(old) ? old : old[.!in.(old.problem, Ref(Set(new.problem))), :]
        all_ = vcat(keep, new; cols = :union)
        sort!(all_, :problem); all_ = order_columns(all_)
        tmp = csv * ".tmp"; CSV.write(tmp, all_); mv(tmp, csv; force = true)
        flush(stdout); GC.gc()
    end
    println("wrote ", csv)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
