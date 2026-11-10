# Convert IntervalMDP.jl reproducibility-package models into infinite-horizon reach-avoid
# benchmark problems.
#
#   tools/repro_package/fetch_lfs.sh /tmp/repro             # the .nc files are git-LFS objects
#   julia --project=. tools/repro_package/import_repro.jl \
#       /tmp/repro multiObj_robotIMDP pimdp_0 ...
#
# The published specs are finite-horizon (H=200) *reachability* of one state. What we store:
#   reach  — the published reach state(s), unchanged;
#   avoid  — every other absorbing state (P̲(s→s) = 1 under all actions): a trap that can never
#            reach the goal, e.g. the failure/obstacle sink of multiObj_robotIMDP (state 206).
#            Listing it changes no value, it only makes the terminal set explicit;
#   S0     — the files carry an "initial_states" of *all* states and no geometry, so there is no
#            published initial state. We take the non-terminal state farthest from the reach set
#            by shortest path over edges with P̅ > 0 (the hardest start; ties → lowest index),
#            restricted to states that can reach the goal at all, so V(S0) is not trivially 0.
using IntervalMDP, IntervalMDP.Data
using NCDatasets, SparseArrays, JSON

const DIR = joinpath(@__DIR__, "..", "..", "problems")
const EPS = 0.0001

function read_raw(path)
    NCDataset(path) do ds
        n = Int32(ds.attrib["num_states"])
        mat(p) = SparseMatrixCSC(n, length(ds["$(p)_colptr"]) - 1,
                                 convert.(Int32, ds["$(p)_colptr"][:]),
                                 convert.(Int32, ds["$(p)_rowval"][:]),
                                 ds["$(p)_nzval"][:])
        stateptr = convert.(Int32, ds["stateptr"][:])
        num_actions = diff(stateptr)
        all(==(num_actions[1]), num_actions) || throw(DimensionMismatch("non-uniform action count"))
        return mat("lower"), mat("upper"), Int(num_actions[1])
    end
end

"States whose lower self-transition is 1 under every action."
function absorbing_states(P̲, na)
    n = size(P̲, 1)
    return [s for s in 1:n if all(P̲[s, (s - 1) * na + a] >= 1 - 1e-12 for a in 1:na)]
end

"Shortest-path distance (in steps) from each state to `targets` over edges with P̅ > 0."
function distance_to(P̅, na, targets)
    n = size(P̅, 1)
    # reverse adjacency: preds[t] = sources with an edge into t
    preds = [Int[] for _ in 1:n]
    for c in 1:size(P̅, 2), k in nzrange(P̅, c)
        s = (c - 1) ÷ na + 1
        t = P̅.rowval[k]
        s != t && P̅.nzval[k] > 0 && push!(preds[t], s)
    end
    dist = fill(typemax(Int), n)
    queue = collect(targets)
    dist[queue] .= 0
    head = 1
    while head <= length(queue)
        t = queue[head]; head += 1
        for s in preds[t]
            if dist[s] == typemax(Int)
                dist[s] = dist[t] + 1
                push!(queue, s)
            end
        end
    end
    return dist
end

function convert_model(src_dir, name)
    P̲, P̅, na = read_raw(joinpath(src_dir, name * ".nc"))
    n = size(P̅, 1)

    spec_json = JSON.parsefile(joinpath(src_dir, name * ".json"))
    prop = spec_json["property"]
    prop["type"] == "reachability" || error("$name: expected a reachability property")
    reach = Int32.(vcat(prop["reach"]...))

    terminal = absorbing_states(P̲, na)
    avoid = Int32.(setdiff(terminal, reach))

    dist = distance_to(P̅, na, reach)
    candidates = [s for s in 1:n if s ∉ terminal && dist[s] < typemax(Int)]
    isempty(candidates) && error("$name: no non-terminal state can reach the goal")
    s0 = candidates[argmax(dist[candidates])]   # argmax returns the first (lowest index) maximum

    @info "converted" name n actions=na nnz=nnz(P̅) reach avoid s0 dist=dist[s0] unreachable=count(==(typemax(Int)), dist)

    imdp = IntervalMarkovDecisionProcess(IntervalAmbiguitySets(; lower=P̲, upper=P̅), na, Int32[s0])
    spec = Specification(InfiniteTimeReachAvoid(reach, avoid, EPS; restrict_to_initial=false),
                         spec_json["satisfaction_mode"] == "pessimistic" ? Pessimistic : Optimistic,
                         spec_json["strategy_mode"] == "maximize" ? Maximize : Minimize)

    out = joinpath(DIR, name)
    mkpath(out)
    write_intervalmdp_jl_model(joinpath(out, name * ".nc"), imdp)
    write_intervalmdp_jl_spec(joinpath(out, name * ".json"), spec)
    println("Saved problem to ", out)
end

if abspath(PROGRAM_FILE) == @__FILE__
    src_dir = ARGS[1]
    for name in ARGS[2:end]
        convert_model(src_dir, name)
        GC.gc()
    end
end
