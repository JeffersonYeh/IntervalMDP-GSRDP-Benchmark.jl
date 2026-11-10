#!/usr/bin/env julia
# Convert a Storm explicit DRN file of an interval MDP (`@value_type: double-interval`) into an
# IntervalMDP.jl .nc (model) + .json (spec) pair with an infinite-time reach-avoid specification.
#
# Usage:
#   julia --project=. tools/storm/drn_to_imdp.jl \
#     <file.drn> --out <dir> [--name <name>] [--reach-label target] [--avoid-label <item>[,<item>...]] \
#     [--strategy max|min] [--satisfaction pessimistic|optimistic] [--eps 1e-4]
#
# Initial states come from the "init" label; the reach set from `--reach-label` (non-empty) and
# the avoid set from `--avoid-label`, a comma-separated union of items: a label name,
# `state:<id>` (a 0-based DRN state id), or `@absorbing`. `@absorbing` is every absorbing
# non-reach state (all choices a probability-1 self-loop): those traps have value 0 whether or
# not they are avoided, so the value is unchanged. It exists because
# IntervalMDP.Data.read_intervalmdp_jl cannot read back an empty "avoid" array (it becomes a
# `Vector{Any}` the property constructor rejects), so a reach-only model still needs one.
# States with fewer choices are padded exactly as in the PRISM converter
# (tools/prism/imdp_writer.jl).
#
# The DRN subset handled is what Storm writes for (interval) MDPs/DTMCs:
#   state <id> [<rewards>] [<label> ...]      rewards in brackets, before or after the labels
#   	action <name> [<rewards>]
#   		<dest> : [<l>, <u>]   or   <dest> : <p>
# Parametric values, `@placeholders` and state valuations are not supported.

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)

using IntervalMDP
import IntervalMDP.Data
using SparseArrays

include(joinpath(@__DIR__, "..", "prism", "imdp_writer.jl"))

function parse_args(args)
    isempty(args) && error("usage: drn_to_imdp.jl <file.drn> --out <dir> [--name <name>] " *
                           "[--reach-label target] [--avoid-label <item>[,<item>...]] [--strategy max|min] " *
                           "[--satisfaction pessimistic|optimistic] [--eps 1e-4]")
    opts = Dict{String,String}("strategy" => "max", "satisfaction" => "pessimistic",
                               "eps" => "1e-4", "reach-label" => "target")
    path = nothing
    i = 1
    while i <= length(args)
        a = args[i]
        if startswith(a, "--") && i < length(args)
            opts[a[3:end]] = args[i + 1]; i += 2
        elseif path === nothing
            path = a; i += 1
        else
            error("unexpected argument '$a'")
        end
    end
    path === nothing && error("missing <file.drn> path")
    haskey(opts, "out") || error("--out <dir> is required")
    get!(opts, "name", first(splitext(basename(path))))
    return path, opts
end

"Drop bracketed reward blocks (`[1]`, `[1, 2]`) from a state or action line."
strip_rewards(line) = replace(line, r"\[[^\]]*\]" => " ")

"Parse an interval `[l, u]` or a point probability `p`."
function parse_prob(str)
    str = strip(str)
    if startswith(str, "[")
        l, u = parse.(Float64, split(str[2:end-1], ","))
        return l, u
    end
    p = parse(Float64, str)
    return p, p
end

"""
Parse a DRN file. Returns `(n, choices, labels)` with `choices` as in `imdp_writer.jl` and
`labels` = `label => sorted 1-based state indices`.
"""
function read_drn(path)
    n_declared = nothing
    choices = Vector{Vector{Vector{Tuple{Int32,Float64,Float64}}}}()
    labels = Dict{String,Vector{Int32}}()
    value_type = nothing
    in_model = false
    header = nothing   # the `@` header whose value is on the next line
    for raw in eachline(path)
        line = strip(raw)
        (isempty(line) || startswith(line, "//")) && continue
        if startswith(line, "@")
            header = line
            if line == "@model"
                in_model = true
            elseif startswith(line, "@value_type:")
                value_type = strip(split(line, ":"; limit = 2)[2])
            end
            continue
        end
        if !in_model
            header == "@nr_states" && (n_declared = parse(Int, line))
            continue
        end
        w = split(line)
        if w[1] == "state"
            s = parse(Int, w[2]) + 1
            s == length(choices) + 1 || error("states out of order at '$line'")
            push!(choices, Vector{Vector{Tuple{Int32,Float64,Float64}}}())
            for lab in split(strip_rewards(join(w[3:end], " ")))
                push!(get!(labels, lab, Int32[]), Int32(s))
            end
        elseif w[1] == "action"
            isempty(choices) && error("action before any state: '$line'")
            push!(last(choices), Tuple{Int32,Float64,Float64}[])
        else
            dest, prob = split(line, ":"; limit = 2)
            l, u = parse_prob(prob)
            (isempty(choices) || isempty(last(choices))) &&
                error("transition before any action: '$line'")
            push!(last(last(choices)), (Int32(parse(Int, strip(dest)) + 1), l, u))
        end
    end
    value_type in ("double-interval", "double", nothing) ||
        error("unsupported @value_type $(value_type)")
    n = length(choices)
    n_declared === nothing || n_declared == n ||
        throw(DimensionMismatch("@nr_states declares $n_declared states, parsed $n"))
    foreach(sort!, values(labels))
    return n, choices, labels
end

"Absorbing states: every choice is a single self-loop (whose interval must then contain 1)."
absorbing_states(choices) =
    Int32[s for (s, cs) in enumerate(choices) if all(c -> length(c) == 1 && c[1][1] == s, cs)]

function main(args)
    path, opts = parse_args(args)
    name = opts["name"]
    strategy, satisfaction = parse_modes(opts["strategy"], opts["satisfaction"])

    n, choices, labels = read_drn(path)
    lab(l) = haskey(labels, l) ? labels[l] :
             error("$name: label \"$l\" not in $(path) (labels: $(join(keys(labels), ", ")))")

    init = lab("init")
    reach = lab(opts["reach-label"])
    function avoid_item(item)
        item == "@absorbing" && return absorbing_states(choices)
        if startswith(item, "state:")
            s = parse(Int, item[7:end]) + 1
            1 <= s <= n || error("$name: $item is not a state of $(path)")
            return Int32[s]
        end
        return lab(item)
    end
    avoid = !haskey(opts, "avoid-label") ? Int32[] :
            setdiff(sort!(unique!(reduce(vcat, avoid_item.(strip.(split(opts["avoid-label"], ",")))))), reach)

    write_reachavoid_problem(opts["out"], name, choices, n, init, reach, avoid,
                             strategy, satisfaction, parse(Float64, opts["eps"]))
end

main(ARGS)
