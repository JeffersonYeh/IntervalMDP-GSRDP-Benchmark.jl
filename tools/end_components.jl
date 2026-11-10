#!/usr/bin/env julia
# Maximal end-component (MEC) check for the benchmark problems: run it when adding problems.
#
# An end component is a set of non-terminal states (reach and avoid states are terminal) where
# play can stay forever. GSRDP picks actions greedily on the upper bound, so inside one a staying
# action can tie with an exit: the lower bound then stalls, or the upper bound settles on the
# wrong fixed point. storm_tiny04 never terminates for exactly this reason.
#
# Two notions of "action a can stay in C", from the interval bounds l(s'|s,a), u(s'|s,a):
#   sure:     every successor with u > 0 is in C             -- stays whatever the adversary does
#             (a real trap: under Maximize the upper bound is pinned at 1 on C)
#   possible: every successor with l > 0 is in C, Σ_C u ≥ 1  -- the adversary *can* stay
#             (only a risk: harmless when leaving is always strictly better, e.g. pimdp_*)
# Columns: #MEC / states in MECs / largest MEC / MEC states reachable from S0. MECs of at most
# 10 states are listed (1-based state indices).
#
# Reads the .nc/.json directly (NCDatasets), no IntervalMDP model; pimdp_1 (36M transitions)
# peaks ~2.4 GB, all 42 problems take a few minutes.
#
# Usage (from the repository root):
#   julia --project tools/end_components.jl [problems_dir] [name ...]

using NCDatasets, JSON, Printf

const TOL = 1e-9

struct Model
    n::Int
    A::Int
    lcol::Vector{Int}; lrow::Vector{Int32}; lval::Vector{Float64}
    ucol::Vector{Int}; urow::Vector{Int32}; uval::Vector{Float64}
    init::Vector{Int32}
end

function load_model(path)
    NCDataset(path) do ds
        n = Int(ds.attrib["num_states"])
        sp = Int.(ds["stateptr"][:])
        A = sp[2] - sp[1]
        all(==(A), diff(sp)) || error("non-uniform action count")
        Model(n, A,
              Int.(ds["lower_colptr"][:]), Int32.(ds["lower_rowval"][:]), Float64.(ds["lower_nzval"][:]),
              Int.(ds["upper_colptr"][:]), Int32.(ds["upper_rowval"][:]), Float64.(ds["upper_nzval"][:]),
              Int32.(ds["initial_states"][:]))
    end
end

function load_spec(path)
    d = JSON.parsefile(path)
    idx(v) = Int32[x[1] for x in v]
    p = d["property"]
    return idx(p["reach"]), idx(p["avoid"]), d["strategy_mode"], d["satisfaction_mode"]
end

state_of(m, c) = (c - 1) ÷ m.A + 1

"Can column `c` (of state s, component id k) stay inside component k?"
function stays(m, comp, c, k, notion::Symbol)
    if notion === :sure
        for j in m.ucol[c]:(m.ucol[c+1]-1)
            m.uval[j] > 0 && comp[m.urow[j]] != k && return false
        end
        return true
    else
        for j in m.lcol[c]:(m.lcol[c+1]-1)
            m.lval[j] > 0 && comp[m.lrow[j]] != k && return false
        end
        s = 0.0
        for j in m.ucol[c]:(m.ucol[c+1]-1)
            comp[m.urow[j]] == k && (s += m.uval[j])
        end
        return s >= 1 - TOL
    end
end

"Iterative Tarjan over CSR (ptr, adj) restricted to `active` nodes. Returns component ids (0 = inactive)."
function tarjan(n, ptr, adj, active)
    index = zeros(Int, n); low = zeros(Int, n); onstack = falses(n)
    comp = zeros(Int, n)
    stack = Int[]; cs = Tuple{Int,Int}[]   # call stack: (node, next edge position)
    idx = 0; ncomp = 0
    for root in 1:n
        (active[root] && index[root] == 0) || continue
        push!(cs, (root, ptr[root]))
        idx += 1; index[root] = low[root] = idx; push!(stack, root); onstack[root] = true
        while !isempty(cs)
            v, e = cs[end]
            if e < ptr[v+1]
                cs[end] = (v, e + 1)
                w = adj[e]
                if index[w] == 0
                    idx += 1; index[w] = low[w] = idx; push!(stack, w); onstack[w] = true
                    push!(cs, (w, ptr[w]))
                elseif onstack[w]
                    low[v] = min(low[v], index[w])
                end
            else
                pop!(cs)
                if !isempty(cs)
                    u = cs[end][1]; low[u] = min(low[u], low[v])
                end
                if low[v] == index[v]
                    ncomp += 1
                    while true
                        w = pop!(stack); onstack[w] = false; comp[w] = ncomp
                        w == v && break
                    end
                end
            end
        end
    end
    return comp
end

function mec(m, terminal, notion)
    n, A = m.n, m.A
    comp = [terminal[s] ? 0 : 1 for s in 1:n]
    allowed = [!terminal[state_of(m, c)] for c in 1:n*A]
    first_pass = true
    while true
        removed = false
        for c in 1:n*A
            allowed[c] || continue
            s = state_of(m, c)
            if comp[s] == 0 || !stays(m, comp, c, comp[s], notion)
                allowed[c] = false; removed = true
            end
        end
        for s in 1:n
            comp[s] == 0 && continue
            any(allowed[(s-1)*A+a] for a in 1:A) || (comp[s] = 0)
        end
        (!removed && !first_pass) && break
        first_pass = false
        # edges: allowed actions, u > 0 successors in the same component
        ptr = zeros(Int, n + 1); ptr[1] = 1
        adj = Int[]
        for s in 1:n
            if comp[s] > 0
                for a in 1:A
                    c = (s-1)*A + a
                    allowed[c] || continue
                    for j in m.ucol[c]:(m.ucol[c+1]-1)
                        t = m.urow[j]
                        m.uval[j] > 0 && comp[t] == comp[s] && push!(adj, t)
                    end
                end
            end
            ptr[s+1] = length(adj) + 1
        end
        comp = tarjan(n, ptr, adj, comp .> 0)
    end
    return comp
end

"States reachable from S0 over u > 0 edges without passing through terminal states."
function reachable(m, terminal)
    seen = falses(m.n); q = Int[]
    for s in m.init; seen[s] = true; push!(q, s); end
    while !isempty(q)
        s = pop!(q)
        terminal[s] && continue
        for c in (s-1)*m.A+1:s*m.A, j in m.ucol[c]:(m.ucol[c+1]-1)
            t = m.urow[j]
            if m.uval[j] > 0 && !seen[t]
                seen[t] = true; push!(q, t)
            end
        end
    end
    return seen
end

function summarize(comp, reach0)
    ids = unique(filter(>(0), comp))
    sizes = [count(==(k), comp) for k in ids]
    nstates = sum(sizes; init = 0)
    nreach = count(s -> comp[s] > 0 && reach0[s], eachindex(comp))
    return (n = length(ids), states = nstates, largest = maximum(sizes; init = 0),
            singles = count(==(1), sizes), from_s0 = nreach)
end

function main(args)
    dir = isempty(args) ? "problems" : args[1]
    names = length(args) > 1 ? args[2:end] :
            sort(filter(d -> isdir(joinpath(dir, d)), readdir(dir)))
    @printf("%-36s %8s %8s %3s  %-26s  %-26s\n", "problem", "states", "nonterm", "A",
            "sure: #MEC/states/max/s0", "possible: #MEC/states/max/s0")
    for name in names
        nc = joinpath(dir, name, name * ".nc"); js = joinpath(dir, name, name * ".json")
        isfile(nc) && isfile(js) || continue
        m = load_model(nc)
        reach, avoid, strat, sat = load_spec(js)
        terminal = falses(m.n); terminal[reach] .= true; terminal[avoid] .= true
        r0 = reachable(m, terminal)
        comps = Dict(notion => mec(m, terminal, notion) for notion in (:sure, :possible))
        res = Dict(notion => summarize(comps[notion], r0) for notion in (:sure, :possible))
        f(r) = @sprintf("%d / %d / %d / %d", r.n, r.states, r.largest, r.from_s0)
        @printf("%-36s %8d %8d %3d  %-26s  %-26s  %s/%s\n", name, m.n, count(!, terminal), m.A,
                f(res[:sure]), f(res[:possible]), strat, sat)
        for notion in (:sure, :possible)
            comp = comps[notion]
            for k in unique(filter(>(0), comp))
                members = findall(==(k), comp)
                length(members) <= 10 && println("    $notion MEC: ", members)
            end
        end
        flush(stdout)
        m = nothing; GC.gc()
    end
end

# guarded so tools/mixing_stats.jl can include() the helpers above
if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
