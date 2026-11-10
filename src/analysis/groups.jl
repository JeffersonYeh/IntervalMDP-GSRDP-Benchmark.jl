# Algorithm groups — named subsets of the arms for which `run_analyses` redraws the
# cross-algorithm figures (cactus, normalized_gap_curves, scatter_vs_reference,
# vs_transitions) into `plots/groups/<name>/`. A sweep with tens of arms overlays them all
# on one axis and cycles the 10-colour palette several times over; a group is one readable
# slice of it — an OFAT axis plus its centre, say. Groups may overlap.
#
# A group is `name => patterns`, each pattern an exact label or a glob (`*`, `?`, anchored
# at both ends). Specs come from `--group NAME=PAT,PAT,...` or a TOML file of
# `name = ["PAT", ...]` (see run_analyses.jl).

const _GROUP_NAME_RE = r"^[A-Za-z0-9_.-]+$"

function _check_group_name(name::AbstractString)
    occursin(_GROUP_NAME_RE, name) ||
        error("group name '$name' must match [A-Za-z0-9_.-]+ (it becomes a directory name)")
    return String(name)
end

"""
    parse_group_spec(s) -> Pair{String,Vector{String}}

Parse one `--group` value, `NAME=PAT[,PAT...]`. Empty patterns are dropped; a spec with
no `=`, an invalid name, or no patterns at all is an error.
"""
function parse_group_spec(s::AbstractString)
    i = findfirst('=', s)
    i === nothing && error("--group expects NAME=PAT[,PAT...], got '$s'")
    name = _check_group_name(strip(s[1:prevind(s, i)]))
    pats = String[strip(p) for p in split(s[nextind(s, i):end], ',') if !isempty(strip(p))]
    isempty(pats) && error("--group '$name' lists no algorithms")
    return name => pats
end

"""
    load_group_file(path) -> Vector{Pair{String,Vector{String}}}

Read a TOML file of `name = ["PAT", ...]` entries, in file order.
"""
function load_group_file(path::AbstractString)
    isfile(path) || error("no such groups file: $path")
    # TOML.parsefile returns a Dict, which loses file order; the order groups are listed
    # in is the order a reader expects the directories/manifest entries to follow.
    order = String[]
    for line in eachline(path)
        m = match(r"^\s*([A-Za-z0-9_.-]+)\s*=", line)
        m === nothing || push!(order, m.captures[1])
    end
    d = TOML.parsefile(path)
    out = Pair{String,Vector{String}}[]
    for name in unique(vcat(filter(in(keys(d)), order), sort(collect(keys(d)))))
        v = d[name]
        (v isa AbstractVector && all(x -> x isa AbstractString, v)) ||
            error("groups file $path: '$name' must be an array of strings")
        pats = String[strip(x) for x in v if !isempty(strip(x))]
        isempty(pats) && error("groups file $path: '$name' lists no algorithms")
        push!(out, _check_group_name(name) => pats)
    end
    return out
end

# Anchored regex for a glob with `*` and `?`; everything else matches literally.
function _glob_regex(p::AbstractString)
    buf = IOBuffer()
    print(buf, '^')
    for c in p
        if c == '*'
            print(buf, ".*")
        elseif c == '?'
            print(buf, '.')
        elseif c in "\\^\$.|+()[]{}"
            print(buf, '\\', c)
        else
            print(buf, c)
        end
    end
    print(buf, '$')
    return Regex(String(take!(buf)))
end

"""
    resolve_groups(specs, algs, reference) -> Vector{NamedTuple}

Match each group's patterns against the algorithm labels `algs`. Returns one
`(name, patterns, algorithms, unmatched)` per group that selected at least one
non-reference label, in spec order; `algorithms` is sorted and always includes
`reference` (every grouped figure is read against it). A pattern matching nothing, and a
group left with nothing but the reference, are warned about — never fatal. A repeated
group name is an error.
"""
function resolve_groups(specs, algs, reference::AbstractString)
    labels = sort(unique(String.(algs)))
    out = NamedTuple{(:name, :patterns, :algorithms, :unmatched),
                     Tuple{String,Vector{String},Vector{String},Vector{String}}}[]
    seen = Set{String}()
    for (name, pats) in specs
        name in seen && error("group '$name' is defined more than once")
        push!(seen, name)
        members = Set{String}(); unmatched = String[]
        for p in pats
            re = _glob_regex(p)
            hits = filter(l -> occursin(re, l), labels)
            isempty(hits) ? push!(unmatched, p) : union!(members, hits)
        end
        isempty(unmatched) || @warn "group pattern(s) matched no algorithm" group=name unmatched
        if isempty(setdiff(members, (reference,)))
            @warn "group selects no algorithm besides the reference; skipped" group=name
            continue
        end
        push!(members, reference)
        push!(out, (name=String(name), patterns=String.(pats),
                    algorithms=sort(collect(members)), unmatched=unmatched))
    end
    return out
end
