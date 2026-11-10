#!/usr/bin/env julia
# Convert a PRISM explicit export (.sta/.tra/.lab) of an IMDP into an IntervalMDP.jl
# .nc (model) + .json (spec) pair with an infinite-time reach-avoid specification.
#
# Usage:
#   julia --project=. tools/prism/prism_to_imdp.jl \
#     <base> --out <dir> [--name <name>] [--strategy max|min] \
#     [--satisfaction pessimistic|optimistic] [--eps 1e-4]
#
# `<base>` is the path without extension. The .lab file must carry the labels "init"
# (PRISM adds it), "reach" (non-empty) and optionally "avoid"; the model files declare
# the latter two (see simple/robot.prism, consensus/coin*.prism).
#
# Why not IntervalMDP.Data.read_prism_file: both it and the .nc format require every
# state to have the same number of actions, while PRISM models have a varying number of
# choices per state. States with fewer choices are padded by repeating their last choice;
# a duplicated action never changes a max/min over actions, so values and optimal
# policies are preserved.

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."); io = devnull)

using IntervalMDP
import IntervalMDP.Data
using SparseArrays

include(joinpath(@__DIR__, "imdp_writer.jl"))

function parse_args(args)
    isempty(args) && error("usage: prism_to_imdp.jl <base> --out <dir> [--name <name>] " *
                           "[--strategy max|min] [--satisfaction pessimistic|optimistic] [--eps 1e-4]")
    opts = Dict{String,String}("strategy" => "max", "satisfaction" => "pessimistic",
                               "eps" => "1e-4")
    base = nothing
    i = 1
    while i <= length(args)
        a = args[i]
        if startswith(a, "--") && i < length(args)
            opts[a[3:end]] = args[i + 1]; i += 2
        elseif base === nothing
            base = a; i += 1
        else
            error("unexpected argument '$a'")
        end
    end
    base === nothing && error("missing <base> path")
    haskey(opts, "out") || error("--out <dir> is required")
    get!(opts, "name", basename(base))
    return base, opts
end

"Number of states: the .sta file is a header line plus one line per state."
num_states_from_sta(path) = open(countlines, path) - 1

"""
Parse a PRISM explicit .tra file of an IMDP (`src choice dest [l,u] [action]`, 0-based).
Returns `choices[s]` = vector of choices for state `s` (1-based), each a vector of
`(dest, lower, upper)` tuples with 1-based `dest`.
"""
function read_tra(path, n)
    choices = [Vector{Vector{Tuple{Int32,Float64,Float64}}}() for _ in 1:n]
    open(path) do io
        header = split(readline(io))
        parse(Int, header[1]) == n ||
            throw(DimensionMismatch(".tra declares $(header[1]) states, .sta has $n"))
        for line in eachline(io)
            isempty(line) && continue
            w = split(line)
            s = parse(Int, w[1]) + 1
            c = parse(Int, w[2]) + 1
            d = parse(Int32, w[3]) + 1
            lu = w[4]
            if startswith(lu, "[")
                l, u = parse.(Float64, split(lu[2:end-1], ","))
            else  # a point probability
                l = u = parse(Float64, lu)
            end
            cs = choices[s]
            while length(cs) < c
                push!(cs, Tuple{Int32,Float64,Float64}[])
            end
            push!(cs[c], (d, l, u))
        end
    end
    return choices
end

"Parse a PRISM .lab file into `label name => sorted 1-based state indices`."
function read_lab(path)
    labels = Dict{String,Vector{Int32}}()
    open(path) do io
        names = Dict{Int,String}()
        for word in split(readline(io))
            k, v = split(word, "="; limit = 2)
            names[parse(Int, k)] = strip(v, '"')
            labels[strip(v, '"')] = Int32[]
        end
        for line in eachline(io)
            isempty(line) && continue
            w = split(line)
            s = parse(Int32, chop(w[1])) + 1   # "12:" -> 13
            for k in w[2:end]
                push!(labels[names[parse(Int, k)]], s)
            end
        end
    end
    foreach(sort!, values(labels))
    return labels
end

function main(args)
    base, opts = parse_args(args)
    name = opts["name"]
    strategy, satisfaction = parse_modes(opts["strategy"], opts["satisfaction"])

    n = num_states_from_sta(base * ".sta")
    choices = read_tra(base * ".tra", n)
    labels = read_lab(base * ".lab")

    init = get(labels, "init", Int32[])
    reach = get(labels, "reach", Int32[])
    avoid = get(labels, "avoid", Int32[])
    isempty(init) && error("$name: no \"init\" states in .lab")
    isempty(reach) && error("$name: no \"reach\" states in .lab (add label \"reach\" to the model)")

    write_reachavoid_problem(opts["out"], name, choices, n, init, reach, avoid,
                             strategy, satisfaction, parse(Float64, opts["eps"]))
end

main(ARGS)
