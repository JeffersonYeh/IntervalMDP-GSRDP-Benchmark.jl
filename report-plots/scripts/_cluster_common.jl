# Shared helpers of dataset-cluster.jl (not run on its own: `_`-prefixed scripts are skipped by
# make_all.jl): the candidate model metrics, their values per problem, the problems left out of
# the clustering, the metric selection, and the PCA and clustering helpers. Expects
# _dataset_common.jl to be included first.

const CLUSTER_VAR = 0.75     # share of the variance the retained components explain
const CLUSTER_K = 2:10       # k shown in the decision plots
const CLUSTER_KRANGE = 4:8   # k read (k ≤ 3 merges all 21 abstractions and product IMDPs)

# The candidate metrics: column, group, figure label (text), symbol (LaTeX math), definition,
# and the per-problem view it summarises. "log" in the definition = log10 (zeros floored, see
# cluster_metrics).
const PCA_SPEC_ALL = [
    (:log_nt, "size", "non-terminal states", "|S_{\\mathrm{nt}}|",
     "states in neither the reach nor the avoid set (log)", "scatter/size"),
    (:log_actions, "size", "actions", "|A|", "actions (log)", "scatter/size"),
    (:log_trans, "size", "transitions", "|T|",
     "stored transitions, \\(\\overline{p}_{s,a}(t) > 0\\) (log)", "histogram/hist"),
    (:log_density, "size", "density", "|T| / |S|^2|A|",
     "stored over possible transitions (log)", "histogram/hist"),
    (:log_reach_frac, "sets", "reach share", "|S_{\\mathrm{reach}}| / |S|",
     "share of states in the reach set (log)", "scatter/sets"),
    (:log_avoid_frac, "sets", "avoid share", "|S_{\\mathrm{avoid}}| / |S|",
     "share of states in the avoid set (log, floor \\(10^{-6}\\))", "scatter/sets"),
    (:log_s0_frac, "sets", "initial share", "|S_0| / |S|",
     "share of initial states (log)", "scatter/sets"),
    (:log_neff, "stochasticity", "effective successors", "\\bar{N}_{\\mathrm{eff}}",
     "mean effective successors \\(e^{H(\\hat{p}_{s,a})}\\) over \\((s,a)\\) (log)", "scatter/stoch"),
    (:neff_ratio_med, "stochasticity", "effective share of support", "N_{\\mathrm{eff}} / \\mathrm{deg}",
     "median over \\((s,a)\\) of \\(N_{\\mathrm{eff}}\\) over the support size", "box/neff\\_ratio"),
    (:log_pmax_med, "stochasticity", "top-successor mass", "\\hat{p}_{\\max}",
     "median over \\((s,a)\\) of the largest normalised midpoint probability (log)", "box/pmax"),
    (:log_lmax_med, "stochasticity", "largest lower bound", "\\underline{p}_{\\max}",
     "median over \\((s,a)\\) of the largest lower bound (log)", "box/lmax"),
    (:frac_forced_det, "stochasticity", "forced-deterministic share", "\\Pr[\\underline{p}_{\\max} \\geq 0.9]",
     "share of \\((s,a)\\) whose lower bounds force \\(\\geq 0.9\\) onto one successor", "scatter/stoch"),
    (:log_width_med, "stochasticity", "interval width", "\\overline{p} - \\underline{p}",
     "median interval width over the transitions (log)", "box/width"),
    (:log_tau_s0, "horizon", "absorption time", "\\tau(S_0)",
     "midpoint absorption time from \\(S_0\\), steps (log)", "scatter/stoch"),
    (:log_tau_reach_s0, "horizon", "absorption time in reach", "\\tau_{\\mathrm{reach}}(S_0)",
     "midpoint absorption time from \\(S_0\\), given absorption in reach (log)", "s0"),
    (:log_tau_avoid_s0, "horizon", "absorption time in avoid", "\\tau_{\\mathrm{avoid}}(S_0)",
     "the same, given absorption in avoid (log; 0 without an avoid set)", "s0"),
    (:log_depth_reach_s0, "horizon", "depth to reach", "d_{\\mathrm{reach}}(S_0)",
     "fewest steps from \\(S_0\\) to the reach set (log)", "s0"),
    (:log_depth_max, "horizon", "maximum depth", "\\max_s d(s)",
     "largest depth to reach \\(\\cup\\) avoid over \\(S_{\\mathrm{nt}}\\) (log)", "box/depth"),
    (:tau_iqr, "horizon", "spread of absorption times", "\\mathrm{IQR}\\,\\log\\,\\tau(s)",
     "interquartile range of \\(\\log_{10} \\tau(s)\\) over \\(S_{\\mathrm{nt}}\\)", "box/abs\\_time"),
    (:log_exit_med, "horizon", "guaranteed exit probability", "e(s)",
     "median over \\(S_{\\mathrm{nt}}\\) of the best action's guaranteed exit probability (log)", "box/exit\\_best"),
    (:log_h_reach_s0, "horizon", "reach probability", "h_{\\mathrm{reach}}(S_0)",
     "midpoint probability of absorption in reach from \\(S_0\\) (log)", "s0"),
    (:log_lambda_med, "spectrum", "median eigenvalue", "|\\lambda|_{50}",
     "median eigenvalue magnitude of the midpoint chain on \\(S_{\\mathrm{nt}}\\) (log)", "box/lambda\\_abs"),
    (:log_lambda_q90, "spectrum", "90th-percentile eigenvalue", "|\\lambda|_{90}",
     "90th percentile of the same (log)", "box/lambda\\_abs"),
]
cluster_inputs = [joinpath(d, p * ".csv") for d in (DISTS_DIR, STOCH_DIR, SPECTRAL_DIR) for p in problems.problem]
const CLUSTER_INPUTS_OK = all(isfile, [S0_CSV; cluster_inputs])
if CLUSTER_INPUTS_OK
    mstats_c = CSV.read(joinpath(MIXING_DIR, "mixing_stats.csv"), DataFrame)
    ssum_c = CSV.read(joinpath(STOCH_DIR, "summary.csv"), DataFrame)
    s0h = CSV.read(S0_CSV, DataFrame)
    mix_q = Dict((d, p) => CSV.read(joinpath(d, p * ".csv"), DataFrame)
              for d in (DISTS_DIR, STOCH_DIR, SPECTRAL_DIR), p in problems.problem)
end
# value at quantile `level` of problem p's distribution of `col` in directory d
mix_qv(d, p, col, level) = (t = mix_q[(d, p)]; Float64(t[argmin(abs.(t.q .- level)), col]))
s0mean(p, col) = (v = collect(skipmissing(s0h[s0h.problem .== p, col])); isempty(v) ? missing : mean(v))
row_of(df, p) = df[only(findall(==(p), df.problem)), :]
lg(x; floor=1e-12) = log10(max(x, floor))

# The metrics of one problem, the PCA_SPEC_ALL columns.
function cluster_metrics(p)
    m, st = row_of(mstats_c, p), row_of(ssum_c, p)
    ns = m.num_states
    (; log_nt=log10(m.num_maybe), log_actions=log10(m.num_actions), log_trans=log10(m.num_transitions),
     log_density=log10(m.num_transitions / (ns^2 * m.num_actions)), log_reach_frac=lg(m.reach_frac),
     log_avoid_frac=lg(m.avoid_frac; floor=1e-6), log_s0_frac=lg(m.s0_frac),
     log_neff=log10(st.neff_mean), neff_ratio_med=mix_qv(STOCH_DIR, p, :neff_ratio, 0.5),
     log_pmax_med=lg(mix_qv(STOCH_DIR, p, :pmax, 0.5)), log_lmax_med=lg(mix_qv(STOCH_DIR, p, :lmax, 0.5)),
     frac_forced_det=st.frac_forced_det, log_width_med=lg(mix_qv(DISTS_DIR, p, :width, 0.5)),
     log_tau_s0=lg(m.abs_time_mid_s0), log_tau_reach_s0=lg(s0mean(p, :tau_reach)),
     log_tau_avoid_s0=lg(coalesce(s0mean(p, :tau_avoid), 0.0)),
     log_depth_reach_s0=log10(s0mean(p, :depth_reach)), log_depth_max=log10(mix_qv(DISTS_DIR, p, :depth, 1.0)),
     tau_iqr=lg(mix_qv(DISTS_DIR, p, :abs_time_mid, 0.75)) - lg(mix_qv(DISTS_DIR, p, :abs_time_mid, 0.25)),
     log_exit_med=lg(mix_qv(DISTS_DIR, p, :exit_best, 0.5)), log_h_reach_s0=lg(s0mean(p, :h_reach); floor=1e-8),
     log_lambda_med=lg(mix_qv(SPECTRAL_DIR, p, :lambda_abs, 0.5)),
     log_lambda_q90=lg(mix_qv(SPECTRAL_DIR, p, :lambda_abs, 0.9)))
end

# Problems left out of the clustering, picked by hand among the extremes (DBSCAN noise, outlier
# scores): at either end of the dataset, sampling has nothing to work with. Their metrics would
# also stretch the scores and the components. dataset-cluster.jl runs everything (PCA, Ward,
# figures) on the remaining problems; the left-out ones are listed in
# cache/dataset_cluster_excluded.csv, with the extreme (group) each belongs to.
const CLUSTER_EXCLUDE_GROUPS = [
    "near-deterministic" => ("forced-deterministic share 1, about one effective successor",
                             ["brp_16_2", "multiObj_robotIMDP"]),
    "degenerate" => ("guaranteed exit 1, absorption time 1, depth 1: exact in one sweep",
                     ["storm_tiny01", "storm_tiny03", "storm_tiny05"]),
]
const CLUSTER_EXCLUDE = [p => reason for (_, (reason, ps)) in CLUSTER_EXCLUDE_GROUPS for p in ps]

# --- PCA --------------------------------------------------------------------------------
normal_score(v) = (r = StatsBase.tiedrank(v);
                   Distributions.quantile.(Distributions.Normal(), (r .- 0.5) ./ length(v)))
standardise(X) = (X .- mean(X; dims=1)) ./ std(X; dims=1)
pca_of(X) = (F = svd(X); (; Z=F.U .* F.S', V=F.V, ev=F.S .^ 2 ./ sum(F.S .^ 2), eig=F.S .^ 2 ./ (size(X, 1) - 1)))

# --- Ward clustering on the retained components -------------------------------------------
# Agglomerative clustering on the distance matrix D (Lance–Williams). Merges are (a, b, height)
# with negative = leaf (hclust convention); Ward's height is the distance form sqrt of the
# Lance–Williams update on squared distances.
function hclust(D, method::Symbol)
    n = size(D, 1); M = method == :ward ? D .^ 2 : copy(D)
    active = collect(1:n); id = [-i for i in 1:n]; sz = ones(Int, n)
    merges = Tuple{Int,Int,Float64}[]
    while length(active) > 1
        best, bi, bj = Inf, 0, 0
        for a in eachindex(active), b in a+1:length(active)
            i, j = active[a], active[b]
            M[i, j] < best && ((best, bi, bj) = (M[i, j], i, j))
        end
        push!(merges, (id[bi], id[bj], method == :ward ? sqrt(best) : best))
        for k in active
            (k == bi || k == bj) && continue
            M[bi, k] = M[k, bi] =
                method == :ward ? ((sz[bi] + sz[k]) * M[bi, k] + (sz[bj] + sz[k]) * M[bj, k] -
                                   sz[k] * M[bi, bj]) / (sz[bi] + sz[bj] + sz[k]) :
                method == :average ? (sz[bi] * M[bi, k] + sz[bj] * M[bj, k]) / (sz[bi] + sz[bj]) :
                max(M[bi, k], M[bj, k])
        end
        sz[bi] += sz[bj]; id[bi] = length(merges)
        deleteat!(active, findfirst(==(bj), active))
    end
    return merges
end
leaves(merges, m) = m < 0 ? [-m] : vcat(leaves(merges, merges[m][1]), leaves(merges, merges[m][2]))
# labels 1..k of the k clusters left after undoing the k - 1 highest merges
function cut(merges, k)
    roots = [length(merges)]
    while length(roots) < k
        r = argmax(m -> m > 0 ? merges[m][3] : -Inf, roots)
        splice!(roots, findfirst(==(r), roots), [merges[r][1], merges[r][2]])
    end
    lab = zeros(Int, length(merges) + 1)
    for (c, r) in enumerate(roots), l in leaves(merges, r)
        lab[l] = c
    end
    return lab
end
distances(X) = [norm(X[i, :] .- X[j, :]) for i in axes(X, 1), j in axes(X, 1)]
# per-point silhouette (b - a) / max(a, b); a singleton scores 0
function silhouettes(D, lab)
    map(axes(D, 1)) do i
        own = [j for j in axes(D, 1) if lab[j] == lab[i] && j != i]
        isempty(own) && return 0.0
        a = mean(D[i, own])
        b = minimum(mean(D[i, lab .== c]) for c in unique(lab) if c != lab[i])
        (b - a) / max(a, b)
    end
end
wss(X, lab) = sum(sum(abs2, X[lab .== c, :] .- mean(X[lab .== c, :]; dims=1)) for c in unique(lab))
# Refinement of a Ward cut, keeping the cluster numbers. Ward's merges are greedy and cannot be
# undone, so a problem joined early to a subtree may end up nearer another cluster. (1) k-means
# (Lloyd) iterations started from the means of the clusters `lab`, until no problem changes
# cluster; (2) while a problem has a negative silhouette, the most negative one moves to the
# cluster with the smallest mean distance.
function refine_kmeans(X, lab)
    lab = copy(lab)
    for _ in 1:100
        cs = sort(unique(lab))
        C = Dict(c => vec(mean(X[lab .== c, :]; dims=1)) for c in cs)
        new = [argmin(c -> sum(abs2, X[i, :] .- C[c]), cs) for i in axes(X, 1)]
        new == lab && break
        lab = new
    end
    D = distances(X)
    for _ in axes(X, 1)
        sl = silhouettes(D, lab)
        i = argmin(sl)
        sl[i] >= 0 && break
        others = [c for c in unique(lab) if c != lab[i]]
        lab[i] = argmin(c -> mean(D[i, (lab .== c) .& (axes(X, 1) .!= i)]), others)
    end
    return lab
end
# --- Metric selection -------------------------------------------------------------------------
# The selection of the metrics `cols` of `df`: on the
# PCA of their normal scores (rank) or z-scores, components to CLUSTER_VAR, (1) keep the metrics
# with communality h² ≥ h2min on the retained components, (2) drop near-duplicates (|ρ| > rho,
# Pearson on the transformed values). Bottom-up (the thesis's order) goes from the lowest h² up
# and drops a metric when a still-kept one with at least its h² correlates with it; that one can
# be dropped later, leaving the first without a stand-in. `topdown` keeps from the highest h²
# down and drops a metric only for one already kept. `dup[i]`: the metric i is dropped for.
function select_metrics(df, cols; rank::Bool, h2min::Real, rho::Real=0.85, topdown::Bool=false)
    m = length(cols)
    X = standardise(reduce(hcat, [rank ? normal_score(Float64.(df[!, c])) : Float64.(df[!, c]) for c in cols]))
    p = pca_of(X)
    np = findfirst(>=(CLUSTER_VAR), cumsum(p.ev))
    h2 = [sum(p.eig[j] * p.V[i, j]^2 for j in 1:np) for i in 1:m]
    rh = cor(X)
    keep = h2 .>= h2min
    dup = zeros(Int, m)   # the kept metric a dropped near-duplicate stands in for (0: none)
    if topdown
        rel, keep = keep, falses(m)
        for i in sortperm(h2; rev=true)
            rel[i] || continue
            cand = [j for j in 1:m if keep[j] && abs(rh[i, j]) > rho]
            isempty(cand) ? (keep[i] = true) : (dup[i] = argmax(j -> abs(rh[i, j]), cand))
        end
    else
        for i in sortperm(h2)
            cand = [j for j in 1:m if j != i && keep[j] && abs(rh[i, j]) > rho && h2[j] >= h2[i]]
            keep[i] && !isempty(cand) && (keep[i] = false; dup[i] = argmax(j -> abs(rh[i, j]), cand))
        end
    end
    return (; h2, rho=rh, relevant=h2 .>= h2min, keep, dup, npc=np, ev=p.ev)
end
