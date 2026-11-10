# Stitch several sweeps' results directories into one.
#
# Every consumer in this harness -- `run_analyses`, `render_gap_videos`, the whole of
# `src/analysis/` -- takes exactly ONE results dir and reads `results.csv` / `problems.csv`
# / `meta.json` / `resolved_config.toml` plus trace paths relative to it. So two sweeps
# that were designed to be comparable (one design split across several configs to fit a
# cluster's wall-clock limit, or the arms of configs/thesis/) cannot be analysed together.
#
# `combine_results` takes N such directories and writes a new one that is indistinguishable
# from a single sweep, so every existing consumer just works. The merge is a union over the
# `{problems} x {algorithms}` cells:
#
#   same [problems], different [[algorithms]]  ->  more algorithm columns
#   same [[algorithms]], different [problems]  ->  more problem rows
#
# A cell present in more than one source (the shared reference arm the split configs re-run
# in every part) has its SAMPLES POOLED: its rows are concatenated and `sample_index` is
# renumbered 1..N across the sources, with `runs/<cell>/sample<k>/` renamed to match, so the
# cell reads downstream as one cell that was simply run more times.
#
# Sources are opened read-only and never modified. `COMBINED_FROM.md` at the root of the
# output records which run every merged sample came from, and every caveat the merge raised.
#
# This file deliberately leans on nothing from the solver stack -- only the tables and
# JSON/TOML records on disk -- so it can be included and tested without IntervalMDP.

const COMBINE_SOURCES_FILE = "COMBINED_FROM.md"

_combine_now() = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")

_as_dict(x) = x isa AbstractDict ? Dict{String,Any}(String(k) => v for (k, v) in x) :
                                   Dict{String,Any}()

# --- one source directory ---------------------------------------------------

struct CombineSource
    dir::String                     # absolute path
    tag::String                     # basename, used to label rows in COMBINED_FROM.md
    config::Dict{String,Any}        # resolved_config.toml (empty when absent)
    meta::Dict{String,Any}          # meta.json (empty when absent)
    results::DataFrame              # results.csv, every column read as String
    problems::DataFrame             # problems.csv (empty when absent)
    validation::DataFrame           # validation.csv (empty when absent)
end

# Read the tables as Strings rather than letting CSV infer types: this tool only ever
# reshuffles rows, and a round-trip through Float64 would silently reformat numbers the
# source sweep wrote. Empty fields stay `missing` and write back out empty.
_read_table(path) = isfile(path) ? CSV.read(path, DataFrame; types=String) : DataFrame()

_read_json_dict(path) = isfile(path) ? _as_dict(JSON.parsefile(path)) : Dict{String,Any}()
_read_toml_dict(path) = isfile(path) ? _as_dict(TOML.parsefile(path)) : Dict{String,Any}()

function load_combine_source(dir::AbstractString)
    d = abspath(String(dir))
    isdir(d) || error("no such results directory: $d")
    results = _read_table(joinpath(d, "results.csv"))
    isempty(results) && error("'$d' has no readable results.csv; nothing to combine")
    return CombineSource(d, basename(rstrip(d, '/')),
                         _read_toml_dict(joinpath(d, "resolved_config.toml")),
                         _read_json_dict(joinpath(d, "meta.json")),
                         results,
                         _read_table(joinpath(d, "problems.csv")),
                         _read_table(joinpath(d, "validation.csv")))
end

_experiment_name(s::CombineSource) =
    String(get(s.meta, "experiment", get(s.config, "name", s.tag)))

_cfg_section(s::CombineSource, name::AbstractString) = _as_dict(get(s.config, name, Dict()))

"`[[algorithms]]` / `[[problems]]` out of a resolved config, keyed by label / instance name."
function _cfg_entries(s::CombineSource, section::AbstractString, key::AbstractString)
    out = Dict{String,Dict{String,Any}}()
    for e in get(s.config, section, [])
        e isa AbstractDict && haskey(e, key) || continue
        out[String(e[key])] = _as_dict(e)
    end
    return out
end

"""
A directory this tool produced, rather than a sweep. Its `meta.json` carries `combined`
and a `combine_log.json` sits next to it recording how it was assembled.

A merged directory ABSORBS: when one is among the sources, the merge happens inside it
rather than into a fresh directory (see `combine_results`), so re-running the tool as more
sweeps finish grows one directory instead of spawning a chain of them.
"""
_is_combined(s::CombineSource) = get(s.meta, "combined", false) === true

const COMBINE_LOG_FILE = "combine_log.json"

"Every source run ever merged into `s` -- itself, or the runs behind it when it is merged."
_underlying_runs(s::CombineSource) =
    _is_combined(s) ? String[String(d) for d in get(s.meta, "combined_from", [])] : String[s.dir]

"""
    _source_summary(s) -> Dict

Everything `COMBINED_FROM.md` says about one source, captured at merge time. It is stored
in `combine_log.json` because a later in-place merge must still be able to describe the
runs that went in earlier, long after those directories may have moved or been deleted.
"""
function _source_summary(s::CombineSource)
    cols = names(s.results)
    col(c) = c in cols ? sort(unique(String.(skipmissing(s.results[!, c])))) : String[]
    d = Dict{String,Any}(
        "dir" => s.dir, "tag" => s.tag, "experiment" => _experiment_name(s),
        "combined" => _is_combined(s),
        "rows" => nrow(s.results), "cells" => length(col("cell_id")),
        "problems" => col("instance_name"), "algorithms" => col("alg_label"),
        "seeds" => [string(x) for x in get(_cfg_section(s, "benchmark"), "seeds", [])],
        "provenance" => _read_json_dict(joinpath(s.dir, "provenance.json")),
    )
    for k in ("datetime", "datetime_end", "hostname", "cpu", "ncpu", "julia_threads",
              "blas_threads", "julia_version", "intervalmdp_version",
              "n_ok", "n_error", "n_skipped", "n_timeout")
        d[k] = string(get(s.meta, k, "?"))
    end
    return d
end

# --- file transfer ----------------------------------------------------------

"""
    _transfer(src, dst, mode) -> Symbol

Place `src` at `dst` as `:link` (hardlink) or `:copy`. A sweep never rewrites its outputs
once it has finished, so sharing an inode is safe and makes the multi-gigabyte
`gap_trace.csv` / `value_trace_*.csv` files free to merge. Hardlinks cannot cross a
filesystem boundary -- results synced back from the cluster often live on another mount --
so `:link` falls back to a real copy rather than failing.
"""
function _transfer(src::AbstractString, dst::AbstractString, mode::Symbol)
    # An in-place merge has the target as its own first source, so src and dst are
    # routinely the same file. Without this the `rm` below would delete it and the
    # hardlink would then have nothing to link from.
    abspath(src) == abspath(dst) && return :kept
    mkpath(dirname(dst))
    ispath(dst) && rm(dst; force=true)
    if mode === :link
        try
            hardlink(src, dst)
            return :link
        catch e
            e isa Base.IOError || rethrow()
        end
    end
    cp(src, dst; force=true, follow_symlinks=true)
    return :copy
end

"Transfer every file under `src_dir` into `dst_dir`, preserving the relative layout."
function _transfer_tree(src_dir::AbstractString, dst_dir::AbstractString, mode::Symbol)
    n = 0
    isdir(src_dir) || return n
    for (root, _, files) in walkdir(src_dir)
        rel = relpath(root, src_dir)
        for f in files
            _transfer(joinpath(root, f),
                      joinpath(dst_dir, rel == "." ? f : joinpath(rel, f)), mode)
            n += 1
        end
    end
    return n
end

# --- merge plan -------------------------------------------------------------
# Computed before anything is written, so a compatibility failure costs nothing.

struct SampleMove
    source::Int          # index into the sources vector
    cell_id::String
    old_index::Int
    new_index::Int
    seed::String
end

_str_or_empty(x) = ismissing(x) ? "" : String(x)
_int_or_nothing(x) = ismissing(x) ? nothing : tryparse(Int, String(x))

"""
    _shared_name(names) -> String

One experiment name for a set of sources. Sweeps split across parts are named for their
part (`grid_search_1_part1` ... `grid_search_1_part6`), and joining those with `+` makes a
directory name nobody can read or type. Take the longest common prefix instead, trimmed back
to a `_`/`-` boundary so it cannot end mid-word (`grid_search_1_part` -> `grid_search_1`).

Falls back to joining when the names share no useful prefix, capped so the result stays a
usable directory name.
"""
function _shared_name(names::AbstractVector{<:AbstractString})
    length(names) == 1 && return String(first(names))
    pre = String(first(names))
    for n in names[2:end]
        k = 0
        while k < min(ncodeunits(pre), ncodeunits(n)) && codeunit(pre, k + 1) == codeunit(n, k + 1)
            k += 1
        end
        pre = pre[1:k]
    end
    cut = findlast(c -> c in ('_', '-'), pre)
    cut === nothing || (pre = pre[1:cut])
    pre = rstrip(pre, ['_', '-', '+', ' '])
    length(pre) >= 3 && return String(pre)
    joined = join(names, "+")
    return length(joined) <= 60 ? joined : String(first(names)) * "+$(length(names) - 1)more"
end

"Compact a sorted index list for the provenance table: `[1,2,3,5]` -> `\"1-3,5\"`."
function _compact_indices(v::AbstractVector{Int})
    isempty(v) && return "-"
    s = sort(v)
    parts = String[]
    i = 1
    while i <= length(s)
        j = i
        while j < length(s) && s[j + 1] == s[j] + 1
            j += 1
        end
        push!(parts, i == j ? string(s[i]) : string(s[i], "-", s[j]))
        i = j + 1
    end
    return join(parts, ",")
end

# --- compatibility ----------------------------------------------------------

mutable struct CombineReport
    warnings::Vector{String}
    force::Bool
end

"Raise a blocking incompatibility, or (under `force`) downgrade it to a recorded caveat."
function _blocking!(rep::CombineReport, msg::AbstractString)
    if rep.force
        push!(rep.warnings, "FORCED PAST: " * msg)
        @warn "combine_results: forced past a blocking incompatibility" detail=msg
    else
        error("combine_results: $msg\n(pass --force to combine anyway; the mismatch is " *
              "then recorded in $COMBINE_SOURCES_FILE)")
    end
    return nothing
end

function _caveat!(rep::CombineReport, msg::AbstractString)
    push!(rep.warnings, msg)
    @warn "combine_results: $msg"
    return nothing
end

"Describe one key's value in each source, for the mismatch messages."
_per_source(srcs, vals) = join(("$(s.tag)=$(repr(v))" for (s, v) in zip(srcs, vals)), ", ")

function _check_compatibility!(rep::CombineReport, srcs::Vector{CombineSource})
    # results.csv column order is the contract every downstream reader indexes by
    # (`assemble_row`, src/results.jl). Two sweeps built by different harness versions can
    # disagree, and concatenating them would shift columns.
    head = names(srcs[1].results)
    for s in srcs[2:end]
        names(s.results) == head || _blocking!(rep,
            "'$(s.tag)' results.csv has different columns than '$(srcs[1].tag)' " *
            "(missing $(setdiff(head, names(s.results))), " *
            "extra $(setdiff(names(s.results), head)))")
    end

    # Different tolerances are not comparable measurements, and the analysis reads
    # `restrict_to_initial` back per-directory (analysis/adapter.jl).
    for key in ("convergence_eps", "restrict_to_initial")
        vals = Any[get(_cfg_section(s, "property"), key, nothing) for s in srcs]
        length(unique(vals)) == 1 || _blocking!(rep,
            "[property].$key differs across sources ($(_per_source(srcs, vals)))")
    end

    # The analysis uses timeout_s as the censoring budget; mixing budgets mislabels which
    # points are censored.
    # An absent key is a config predating it, which ran uncapped -- the same thing the
    # default 0.0 means -- so normalise rather than treating "absent" as its own value.
    touts = Any[Float64(get(_cfg_section(s, "benchmark"), "timeout_s", 0.0)) for s in srcs]
    length(unique(touts)) == 1 || _blocking!(rep,
        "[benchmark].timeout_s differs across sources ($(_per_source(srcs, touts))); " *
        "the analysis would censor the merged runs against a single budget")

    # A label names the cell directory, the results row and the reference arm, so the same
    # label carrying different params merges two DIFFERENT algorithms into one cell -- the
    # exact collision `validate_config` forbids within a single config (src/config.jl).
    seen_alg = Dict{String,Tuple{String,Dict{String,Any}}}()
    for s in srcs, (label, entry) in _cfg_entries(s, "algorithms", "label")
        if haskey(seen_alg, label)
            tag, prev = seen_alg[label]
            if _as_dict(get(prev, "params", Dict())) != _as_dict(get(entry, "params", Dict())) ||
               get(prev, "name", "") != get(entry, "name", "")
                _blocking!(rep, "algorithm label '$label' means different things in " *
                                "'$tag' and '$(s.tag)'; the two would merge into a single " *
                                "cell. Give one of them a distinct `label = \"...\"`.")
            end
        else
            seen_alg[label] = (s.tag, entry)
        end
    end

    # A problem whose structural facts changed between sweeps is a different problem.
    structural = ("num_states", "num_actions", "num_transitions", "initial_states")
    seen_prob = Dict{String,Tuple{String,Dict{String,Any}}}()
    for s in srcs
        (isempty(s.problems) || !("problem" in names(s.problems))) && continue
        cols = names(s.problems)
        for row in eachrow(s.problems)
            ismissing(row.problem) && continue
            key = String(row.problem)
            facts = Dict{String,Any}(c => (c in cols ? row[c] : missing) for c in structural)
            if haskey(seen_prob, key)
                tag, prev = seen_prob[key]
                for c in structural
                    a, b = get(prev, c, missing), facts[c]
                    (ismissing(a) || ismissing(b) || a == b) && continue
                    _blocking!(rep, "problem '$key' has $c=$a in '$tag' but $b in " *
                                    "'$(s.tag)'; the problem file changed between the runs")
                end
            else
                seen_prob[key] = (s.tag, facts)
            end
        end
    end

    # Timings only compare within one machine. This is the single most important thing the
    # sources file has to say about a merged directory, so it is never silent.
    for key in ("hostname", "cpu", "julia_threads", "blas_threads")
        vals = Any[get(s.meta, key, nothing) for s in srcs]
        length(unique(vals)) == 1 || _caveat!(rep,
            "sources ran with different $key ($(_per_source(srcs, vals))); wall-clock " *
            "timings are only directly comparable within one machine")
    end
    for key in ("git_commit", "intervalmdp_version", "intervalmdp_source")
        vals = Any[get(s.meta, key, nothing) for s in srcs]
        length(unique(vals)) == 1 || _caveat!(rep,
            "sources were built from different $key ($(_per_source(srcs, vals)))")
    end
    dirty = [s.tag for s in srcs if get(s.meta, "git_dirty", false) === true]
    isempty(dirty) || _caveat!(rep,
        "ran from a dirty working tree: $(join(dirty, ", ")) (the patches are carried over " *
        "into this directory as <source>.<repo>.diff)")

    tracesets = Any[sort(String[String(t) for t in get(_cfg_section(s, "output"), "traces", [])])
                    for s in srcs]
    length(unique(tracesets)) == 1 || _caveat!(rep,
        "sources recorded different [output].traces ($(_per_source(srcs, tracesets))); " *
        "cells from the leaner runs will be missing the corresponding video panels")
    return nothing
end

# --- planning ---------------------------------------------------------------

"""
    _plan_samples(srcs, dedupe_seeds, rep; in_place) -> (moves, keep, counts)

Walk every source's `results.csv` in order and decide each row's fate. `moves` lists the
rows carrying a `sample_index`, renumbered `1..N` per cell across the sources; `keep[i]` is
the row mask for source `i` (false only for seed duplicates dropped by `dedupe_seeds`);
`counts` maps `cell_id` to its pooled sample count.

Rows WITHOUT a sample index -- `error:build`, `skipped:finite_horizon` -- own no run
directory and pass through untouched, so a cell that failed in one source and succeeded in
another keeps both records.

Under `in_place`, source 1 IS the output directory: its samples keep the numbers they
already have (their run directories and `trace_path`s are already correct, and renumbering
them would mean moving thousands of files that nothing asked to move), and the later
sources count on from there.
"""
function _plan_samples(srcs::Vector{CombineSource}, dedupe_seeds::Bool, rep::CombineReport;
                       in_place::Bool=false)
    counts = Dict{String,Int}()
    seen_seed = Set{Tuple{String,String}}()
    dupes = Dict{String,Vector{String}}()
    moves = SampleMove[]
    keep = [trues(nrow(s.results)) for s in srcs]

    for (i, s) in enumerate(srcs)
        cols = names(s.results)
        ("cell_id" in cols && "sample_index" in cols) ||
            error("'$(s.tag)' results.csv is missing cell_id/sample_index; not a results directory")
        has_seed = "seed" in cols
        pinned = in_place && i == 1        # already in the output directory: do not move it
        for (r, row) in enumerate(eachrow(s.results))
            idx = _int_or_nothing(row.sample_index)
            idx === nothing && continue          # error / skipped row: nothing to renumber
            cell = _str_or_empty(row.cell_id)
            seed = has_seed ? _str_or_empty(row.seed) : ""
            if !isempty(seed)
                key = (cell, seed)
                if key in seen_seed
                    push!(get!(dupes, cell, String[]), seed)
                    # A pinned row is already on disk in the output directory; dropping it
                    # would mean deleting data the caller did not ask to delete.
                    (dedupe_seeds && !pinned) && (keep[i][r] = false; continue)
                else
                    push!(seen_seed, key)
                end
            end
            # `max` rather than a running count, so a pinned cell whose numbering has a gap
            # still hands the next source an index that cannot collide with it.
            counts[cell] = pinned ? max(get(counts, cell, 0), idx) : get(counts, cell, 0) + 1
            push!(moves, SampleMove(i, cell, idx, pinned ? idx : counts[cell], seed))
        end
    end

    if !isempty(dupes)
        n = sum(length, values(dupes))
        cells = sort(collect(keys(dupes)))
        examples = join(cells[1:min(end, 3)], ", ") * (length(cells) > 3 ? ", ..." : "")
        head = "$n sample(s) across $(length(dupes)) cell(s) repeat a (cell, seed) pair " *
               "already measured in an earlier source (e.g. $examples)"
        if dedupe_seeds
            _caveat!(rep, head * " -- dropped (--dedupe-seeds)")
        else
            # BLOCKING, not a warning: the analysis reduces by `(algorithm, problem, seed)`
            # with no sample identity (`run_scalars`, analysis/reduce.jl), and `sample_idx`
            # cannot disambiguate because `load_canonical` sets it to the row index WITHIN a
            # trace file. Two runs sharing a seed therefore land in one group, their traces
            # are sorted together by wall-clock, and `first_crossing` reports a solve time
            # read off the interleaved mixture -- a fabricated number, not a noisy one. It
            # lands on exactly the arms a split sweep re-runs in every part, which is the
            # reference every normalized figure divides by.
            _blocking!(rep, head * ". Keeping them would not merely narrow the error bars: " *
                "the analysis groups by (algorithm, problem, seed) with no sample identity, " *
                "so the pooled traces interleave and the reported solve time is read off " *
                "that mixture. Re-run with --dedupe-seeds to keep the first measurement of " *
                "each pair")
        end
    end
    return moves, keep, counts
end

# --- merged records ---------------------------------------------------------

# One value when every source agrees, `"mixed"` otherwise -- so a merged meta.json never
# claims a single host or commit for a directory that has several.
_agree_or_mixed(vals) = length(unique(vals)) == 1 ? first(vals) : "mixed"

function _merged_meta(srcs::Vector{CombineSource}, name::AbstractString,
                      counts::Dict{String,Int}, n_rows::Int)
    cells = String[]
    for s in srcs, c in get(s.meta, "cells", [])
        String(c) in cells || push!(cells, String(c))
    end
    # A cell that only shows up in results.csv (a sweep whose meta.json predates the
    # expected-cell list) still belongs in the expected set, or the completeness banner
    # would call the merged directory incomplete.
    for c in sort(collect(keys(counts)))
        c in cells || push!(cells, c)
    end

    sum_key(k) = sum(Int(get(s.meta, k, 0)) for s in srcs)
    stamps(k) = String[String(v) for v in (get(s.meta, k, nothing) for s in srcs) if v !== nothing]
    starts, ends = stamps("datetime"), stamps("datetime_end")

    d = Dict{String,Any}(
        "experiment" => name,
        "combined" => true,
        # The underlying SWEEP directories, flattened: a merged source contributes the runs
        # behind it, not itself. `_underlying_runs` reads this back to refuse re-adding a
        # run that is already in here.
        "combined_from" => unique(String[r for s in srcs for r in _underlying_runs(s)]),
        "combined_at" => _combine_now(),
        "cells" => cells,
        "n_cells" => length(cells),
        "n_cells_run" => sum_key("n_cells_run"),
        "n_ok" => sum_key("n_ok"),
        "n_error" => sum_key("n_error"),
        "n_skipped" => sum_key("n_skipped"),
        "n_timeout" => sum_key("n_timeout"),
        "n_rows" => n_rows,
        "samples" => isempty(counts) ? 0 : maximum(values(counts)),
        "stopped_early" => any(get(s.meta, "stopped_early", false) === true for s in srcs),
        "stop_reason" => join(filter(!isempty,
            String[String(get(s.meta, "stop_reason", "")) for s in srcs]), "; "),
    )
    # `datetime_end` is how `load_completeness` decides a sweep reached its own end, so it
    # must be present: the merged directory is complete by construction.
    isempty(starts) || (d["datetime"] = minimum(starts))
    d["datetime_end"] = isempty(ends) ? _combine_now() : maximum(ends)

    for key in ("julia_version", "intervalmdp_version", "intervalmdp_source",
                "benchmarktools_version", "hostname", "cpu", "ncpu", "julia_threads",
                "blas_threads", "git_commit", "git_dirty")
        vals = Any[get(s.meta, key, nothing) for s in srcs]
        all(isnothing, vals) && continue
        d[key] = _agree_or_mixed(vals)
    end
    # Per-source detail, so a `"mixed"` above is never a dead end. A merged source
    # contributes the entries it already carries rather than nesting one level deeper on
    # every re-merge; the full record lives in `combine_log.json` either way.
    entries = Dict{String,Any}[]
    for s in srcs
        if _is_combined(s) && get(s.meta, "sources", nothing) isa AbstractVector
            append!(entries, Dict{String,Any}[_as_dict(e) for e in s.meta["sources"]])
        else
            push!(entries, Dict{String,Any}("dir" => s.dir, "tag" => s.tag,
                                            "experiment" => _experiment_name(s),
                                            "meta" => s.meta))
        end
    end
    d["sources"] = entries
    return d
end

"First source that sets `key`, else `default`."
function _first_set(sections, key, default)
    for d in sections
        haskey(d, key) && return d[key]
    end
    return default
end

_any_true(sections, key) = any(get(d, key, false) === true for d in sections)

function _union_strings(sections, key)
    acc = String[]
    for d in sections, v in get(d, key, String[])
        String(v) in acc || push!(acc, String(v))
    end
    return acc
end

"""
    _merged_config(srcs, name, counts, reference_label) -> Dict

Synthesize the merged `resolved_config.toml`: union the `[[problems]]` and
`[[algorithms]]` lists in first-seen order and take the widest of every output setting.

Built by hand rather than round-tripping through `load_experiment_config`, which cannot
read this file: it parses `problems` as a TOML *table* (`root`/`names`) while
`write_resolved_config` emits it as an array of tables, so the loader would silently fall
back to "every subdirectory under problems/". `run_analyses` reads `timeout_s`,
`restrict_to_initial`, `image_formats`, `save_policy` and the `gap_video*` keys back out of
this file, so it has to be complete.
"""
function _merged_config(srcs::Vector{CombineSource}, name::AbstractString,
                        counts::Dict{String,Int}, reference_label)
    problems, algorithms = Dict{String,Any}[], Dict{String,Any}[]
    seen_p, seen_a = Set{String}(), Set{String}()
    for s in srcs
        for e in get(s.config, "problems", [])
            e isa AbstractDict && haskey(e, "instance_name") || continue
            key = String(e["instance_name"])
            key in seen_p && continue
            push!(seen_p, key); push!(problems, _as_dict(e))
        end
        for e in get(s.config, "algorithms", [])
            e isa AbstractDict && haskey(e, "label") || continue
            key = String(e["label"])
            key in seen_a && continue
            push!(seen_a, key); push!(algorithms, _as_dict(e))
        end
    end

    bench = [_cfg_section(s, "benchmark") for s in srcs]
    out   = [_cfg_section(s, "output") for s in srcs]
    prop  = [_cfg_section(s, "property") for s in srcs]

    seeds = Int[]
    for d in bench, v in get(d, "seeds", Int[])
        Int(v) in seeds || push!(seeds, Int(v))
    end

    # The reference arm has to exist among the merged labels, or every analysis that plots
    # against a baseline silently falls back to the first algorithm.
    labels = String[String(a["label"]) for a in algorithms]
    ref = reference_label === nothing ?
          String(_first_set(out, "reference_label", BenchmarkAnalysis.REFERENCE)) :
          String(reference_label)
    isempty(labels) || ref in labels || (ref = first(labels))

    d = Dict{String,Any}(
        "name" => name,
        "benchmark" => Dict{String,Any}(
            "samples" => isempty(counts) ? Int(_first_set(bench, "samples", 0)) :
                                           maximum(values(counts)),
            "warmup" => _any_true(bench, "warmup"),
            "seeds" => seeds,
            "timeout_s" => Float64(_first_set(bench, "timeout_s", 0.0)),
            "wall_budget_s" => Float64(_first_set(bench, "wall_budget_s", 0.0)),
            "wall_reserve_s" => Float64(_first_set(bench, "wall_reserve_s", 300.0)),
        ),
        "output" => Dict{String,Any}(
            "save_raw" => _any_true(out, "save_raw"),
            "save_trace" => _any_true(out, "save_trace"),
            "save_policy" => _any_true(out, "save_policy"),
            "track_initial" => _any_true(out, "track_initial"),
            "traces" => _union_strings(out, "traces"),
            "reference_label" => ref,
            "results_root" => String(_first_set(out, "results_root", "results")),
            "image_formats" => _union_strings(out, "image_formats"),
            "analysis" => _any_true(out, "analysis"),
            "gap_videos" => _any_true(out, "gap_videos"),
            "gap_video_max_frames" => Int(_first_set(out, "gap_video_max_frames", 600)),
            "gap_video_framerate" => Int(_first_set(out, "gap_video_framerate", 15)),
        ),
        "problems" => problems,
        "algorithms" => algorithms,
    )
    p = Dict{String,Any}()
    for key in ("convergence_eps", "restrict_to_initial")
        i = findfirst(x -> haskey(x, key), prop)
        i === nothing || (p[key] = prop[i][key])
    end
    isempty(p) || (d["property"] = p)
    return d
end

# --- raw trials -------------------------------------------------------------

"""
    _merge_raw!(out, srcs, moves, mode, rep)

Carry `raw/<cell_id>.json` over. A cell drawn from one source transfers as-is; a pooled
cell has its `BenchmarkTools.Trial`s concatenated so the saved trial holds all the samples
its results rows do. Nothing under `src/analysis/` reads `raw/` -- it is a provenance
record -- so a failure here degrades to "keep the first source's trial" instead of failing
the merge.
"""
function _merge_raw!(out::AbstractString, srcs::Vector{CombineSource},
                     moves::Vector{SampleMove}, mode::Symbol, rep::CombineReport)
    by_cell = Dict{String,Vector{Int}}()
    for m in moves
        v = get!(by_cell, m.cell_id, Int[])
        m.source in v || push!(v, m.source)
    end
    for cell in sort(collect(keys(by_cell)))
        paths = [joinpath(srcs[i].dir, "raw", cell * ".json") for i in by_cell[cell]]
        filter!(isfile, paths)
        isempty(paths) && continue
        dst = joinpath(out, "raw", cell * ".json")
        if length(paths) == 1
            _transfer(paths[1], dst, mode)
            continue
        end
        try
            trials = [first(BenchmarkTools.load(p)) for p in paths]
            merged = trials[1]
            for t in trials[2:end]
                append!(merged.times, t.times)
                append!(merged.gctimes, t.gctimes)
            end
            BenchmarkTools.save(dst, merged)
        catch e
            _transfer(paths[1], dst, mode)
            _caveat!(rep, "could not concatenate the raw BenchmarkTools trials for cell " *
                          "'$cell' ($(sprint(showerror, e))); kept only one source's copy. " *
                          "Nothing in the analysis reads raw/, so the figures are unaffected.")
        end
    end
    return nothing
end

# --- provenance -------------------------------------------------------------

function _merge_provenance!(out::AbstractString, srcs::Vector{CombineSource}, mode::Symbol)
    per_source = Dict{String,Any}()
    for s in srcs
        prov = _read_json_dict(joinpath(s.dir, "provenance.json"))
        isempty(prov) && continue
        # A merged source's provenance is ALREADY a {"sources" => ...} map whose diffs are
        # named and placed. Splice its entries in rather than nesting them one level
        # deeper (and re-prefixing their filenames) on every re-merge.
        if haskey(prov, "sources") && prov["sources"] isa AbstractDict
            for (tag, inner) in prov["sources"]
                per_source[String(tag)] = inner
            end
            continue
        end
        # Each repo's patch is carried over under a source-qualified name, so two sources
        # with different dirty trees cannot overwrite each other's diff.
        for (_, info) in prov
            info isa AbstractDict && haskey(info, "diff") || continue
            src_diff = joinpath(s.dir, String(info["diff"]))
            isfile(src_diff) || continue
            renamed = string(s.tag, ".", String(info["diff"]))
            _transfer(src_diff, joinpath(out, renamed), mode)
            info["diff"] = renamed
        end
        per_source[s.tag] = prov
    end
    merged = Dict{String,Any}("sources" => per_source)
    # Promote the shared repo state to the top level only when every source agrees, so a
    # reader sees one commit exactly when there is one.
    for repo in ("thesis", "intervalmdp")
        infos = Any[get(p, repo, nothing) for p in values(per_source)]
        (isempty(infos) || any(isnothing, infos) || length(infos) != length(per_source)) && continue
        length(unique(get(i, "commit", "") for i in infos)) == 1 && (merged[repo] = first(infos))
    end
    open(io -> JSON.print(io, merged, 2), joinpath(out, "provenance.json"), "w")
    return nothing
end

# Videos the sources already rendered are carried over under their new sample number, so
# the analysis pass -- which keeps an existing file unless `--overwrite` -- does not
# re-render what the sources already paid for.
#
# Only ONE per cell, and specifically the one `_select_one_per_cell` would draw: the lowest
# sample number present. A pooled cell has a video from each source, and carrying both over
# would leave the merged directory with two videos of a cell where a real sweep has one.
function _merge_gap_videos!(out::AbstractString, srcs::Vector{CombineSource},
                            moves::Vector{SampleMove}, mode::Symbol)
    best = Dict{String,SampleMove}()
    for m in moves
        src = joinpath(srcs[m.source].dir, "plots", "gap_videos",
                       "$(m.cell_id)__sample$(m.old_index).mp4")
        isfile(src) || continue
        cur = get(best, m.cell_id, nothing)
        (cur === nothing || m.new_index < cur.new_index) && (best[m.cell_id] = m)
    end
    for m in values(best)
        _transfer(joinpath(srcs[m.source].dir, "plots", "gap_videos",
                           "$(m.cell_id)__sample$(m.old_index).mp4"),
                  joinpath(out, "plots", "gap_videos",
                           "$(m.cell_id)__sample$(m.new_index).mp4"), mode)
    end
    return nothing
end

# Per-problem rollup: trials sum, the worst violation wins, consistency is a conjunction.
function _merge_validation_summary(out::AbstractString, srcs::Vector{CombineSource})
    acc = Dict{String,Dict{String,Any}}()
    order = String[]
    for s in srcs
        path = joinpath(s.dir, "validation_summary.json")
        isfile(path) || continue
        for e in JSON.parsefile(path)
            e isa AbstractDict && haskey(e, "instance_name") || continue
            key = String(e["instance_name"])
            if haskey(acc, key)
                a = acc[key]
                a["n_trials"] += Int(get(e, "n_trials", 0))
                a["n_converged"] += Int(get(e, "n_converged", 0))
                a["max_violation"] = max(a["max_violation"], Float64(get(e, "max_violation", -Inf)))
                a["consistent"] = a["consistent"] && get(e, "consistent", false) === true
            else
                push!(order, key)
                acc[key] = Dict{String,Any}(
                    "instance_name" => key,
                    "n_trials" => Int(get(e, "n_trials", 0)),
                    "n_converged" => Int(get(e, "n_converged", 0)),
                    "max_violation" => Float64(get(e, "max_violation", -Inf)),
                    "consistent" => get(e, "consistent", false) === true)
            end
        end
    end
    isempty(order) && return nothing
    open(io -> JSON.print(io, [acc[k] for k in order], 2),
         joinpath(out, "validation_summary.json"), "w")
    return nothing
end

# --- the merge log ----------------------------------------------------------
# `COMBINED_FROM.md` is rendered from `combine_log.json`, not written directly, because a
# merged directory can be merged into AGAIN: the second pass must still describe the runs
# the first pass folded in, long after those directories may have moved or been deleted.
# The log is append-only -- one entry per merge operation -- and the prose is regenerated
# from the whole of it each time.

"One entry in the log: what this invocation added, and where each added sample landed."
function _log_entry(new_srcs::Vector{CombineSource}, tagged_moves, caveats::Vector{String};
                    command, mode, dedupe_seeds, in_place, target)
    groups = Dict{Tuple{String,String},Vector{SampleMove}}()
    for (tag, m) in tagged_moves
        push!(get!(groups, (m.cell_id, tag), SampleMove[]), m)
    end
    return Dict{String,Any}(
        "at" => _combine_now(),
        "command" => command,
        "mode" => String(mode),
        "dedupe_seeds" => dedupe_seeds,
        "in_place" => in_place,
        "target" => target,
        "sources" => [_source_summary(s) for s in new_srcs],
        "moves" => [Dict{String,Any}(
                        "cell_id" => k[1], "source" => k[2],
                        "from" => _compact_indices([m.old_index for m in g]),
                        "to" => _compact_indices([m.new_index for m in g]))
                    for (k, g) in sort(collect(groups); by=first)],
        "caveats" => caveats,
    )
end

"""
    _read_combine_log(s) -> Vector

A merged directory's log. When the file is absent -- a directory merged before the log
existed -- one entry is reconstructed from `meta.json`'s own source record, so merging into
it does not silently drop the history of how it was built. The per-sample moves are not
recoverable that way; the runs that went in are, which is the part that matters.
"""
function _read_combine_log(s::CombineSource)
    path = joinpath(s.dir, COMBINE_LOG_FILE)
    isfile(path) && return Vector{Any}(JSON.parsefile(path))
    _is_combined(s) || return Any[]
    sources = Dict{String,Any}[]
    for e in get(s.meta, "sources", [])
        e = _as_dict(e)
        m = _as_dict(get(e, "meta", Dict()))
        # rows/cells/problems/algorithms/seeds are left OUT, not empty: they cannot be
        # recovered from meta.json, and an empty list would read as "this run had none".
        d = Dict{String,Any}("dir" => String(get(e, "dir", "?")),
                             "tag" => String(get(e, "tag", "?")),
                             "experiment" => String(get(e, "experiment", "?")),
                             "combined" => false, "provenance" => Dict{String,Any}())
        for k in ("datetime", "datetime_end", "hostname", "cpu", "ncpu", "julia_threads",
                  "blas_threads", "julia_version", "intervalmdp_version",
                  "n_ok", "n_error", "n_skipped", "n_timeout")
            d[k] = string(get(m, k, "?"))
        end
        push!(sources, d)
    end
    isempty(sources) && return Any[]
    return Any[Dict{String,Any}(
        "at" => string(get(s.meta, "combined_at", get(s.meta, "datetime_end", "?"))),
        "command" => "", "mode" => "?", "dedupe_seeds" => false, "in_place" => false,
        "target" => nothing, "sources" => sources, "moves" => [],
        "caveats" => ["the per-sample provenance of this merge predates `$COMBINE_LOG_FILE` " *
                      "and could not be reconstructed; the runs it drew on are listed above"])]
end

# --- COMBINED_FROM.md -------------------------------------------------------

function _render_sources_file(out::AbstractString, log::Vector{Any}, name::AbstractString,
                              counts::Dict{String,Int}, n_samples::Int)
    runs = String[]
    for e in log, src in get(e, "sources", [])
        String(get(src, "dir", "")) in runs || push!(runs, String(get(src, "dir", "")))
    end
    caveats = String[String(c) for e in log for c in get(e, "caveats", [])]
    # A cell is pooled when its samples came from more than one run, whether that happened
    # in one merge or across several.
    by_cell = Dict{String,Set{String}}()
    for e in log, m in get(e, "moves", [])
        push!(get!(by_cell, String(get(m, "cell_id", "")), Set{String}()),
              String(get(m, "source", "")))
    end
    pooled = sort(String[c for (c, ss) in by_cell if length(ss) > 1])

    open(joinpath(out, COMBINE_SOURCES_FILE), "w") do io
        println(io, "# $name — combined results")
        println(io)
        println(io, "This directory was **not produced by a single sweep**. `combine_results.jl`")
        println(io, "assembled it from the $(length(runs)) runs below, laid out so that every consumer of a")
        println(io, "normal results directory (`run_analyses.jl`, `report-plots/`, the")
        println(io, "`src/analysis/` suite) reads it without knowing the difference.")
        println(io)
        println(io, "| | |")
        println(io, "|---|---|")
        println(io, "| Source runs | $(length(runs)) |")
        println(io, "| Merges | $(length(log)) |")
        println(io, "| Cells | $(length(counts)) |")
        println(io, "| Samples | $n_samples |")
        println(io, "| Last merged | $(isempty(log) ? "?" : get(log[end], "at", "?")) |")
        println(io)
        length(log) > 1 && println(io, "It was merged into $(length(log)) times; each pass is a \
                                        section below, in order.\n")

        for (n, e) in enumerate(log)
            _render_merge(io, e, n, length(log))
        end

        # An entry with sources but no moves is a merge whose per-sample provenance was
        # never recorded, so the list below is a lower bound rather than the answer.
        partial = any(e -> !isempty(get(e, "sources", [])) && isempty(get(e, "moves", [])), log)
        if partial
            println(io, "**Pooled cells** cannot be listed in full: one merge above predates ",
                        "`$COMBINE_LOG_FILE`")
            println(io, "and its per-sample provenance was not recorded. The per-cell sample counts")
            println(io, "in `results.csv` are authoritative.", isempty(pooled) ? "" :
                        " Known pooled from the merges that were\nrecorded: " * join(pooled, ", ") * ".")
        elseif isempty(pooled)
            println(io, "No cell drew samples from more than one run: this is a pure union, and")
            println(io, "every sample kept its own run.")
        else
            println(io, "**Pooled cells** ($(length(pooled))) — samples from more than one run, ")
            println(io, "renumbered into one sequence: ", join(pooled, ", "))
        end
        println(io)

        println(io, "## Caveats")
        println(io)
        if isempty(caveats)
            println(io, "None. Every merge found the sources agreeing on tolerance, timeout, machine,")
            println(io, "code version and recorded traces, with no `(cell, seed)` pair measured twice.")
        else
            println(io, "Raised across every merge into this directory. Read these before quoting any")
            println(io, "number from it.")
            println(io)
            for c in caveats
                println(io, "- ", c)
            end
        end
        println(io)
        println(io, "---")
        println(io)
        println(io, "`manifest.txt`, `METHODS_meanstd.md` and `plots/` here are written by the analysis")
        println(io, "suite over the merged data and say nothing about the merge; this file (rendered")
        println(io, "from `$COMBINE_LOG_FILE`) is the only record of it.")
    end
    return joinpath(out, COMBINE_SOURCES_FILE)
end

function _render_merge(io, e, n::Int, total::Int)
    srcs = get(e, "sources", [])
    inherited = String(get(e, "inherited_from", ""))
    heading = total == 1 ? "## Sources" :
              !isempty(inherited) ?
              "## Merge $n — $(get(e, "at", "?")) (history of `$inherited`, folded in here)" :
              get(e, "in_place", false) === true ?
              "## Merge $n — $(get(e, "at", "?")) (added into this directory)" :
              "## Merge $n — $(get(e, "at", "?"))"
    println(io, heading)
    println(io)
    if total > 1
        cmd = String(get(e, "command", ""))
        isempty(cmd) || println(io, "`$cmd`\n")
    elseif !isempty(String(get(e, "command", "")))
        println(io, "`$(get(e, "command", ""))` · mode `$(get(e, "mode", "?"))` · ",
                    "seed de-duplication $(get(e, "dedupe_seeds", false) === true ? "on" : "off")")
        println(io)
    end

    for (i, s) in enumerate(srcs)
        g(k) = string(get(s, k, "?"))
        println(io, "### $n.$i `$(g("tag"))`$(get(s, "combined", false) === true ? " (itself a merged directory)" : "")")
        println(io)
        println(io, "- Path: `$(g("dir"))`")
        println(io, "- Experiment: `$(g("experiment"))`")
        println(io, "- Ran: $(g("datetime")) → $(g("datetime_end"))")
        println(io, "- Host: $(g("hostname")) — $(g("cpu")), $(g("ncpu")) cpus, ",
                    "$(g("julia_threads")) julia / $(g("blas_threads")) blas threads")
        println(io, "- Julia $(g("julia_version")), IntervalMDP $(g("intervalmdp_version"))")
        prov = get(s, "provenance", Dict())
        for repo in sort(collect(keys(prov)))
            info = prov[repo]
            info isa AbstractDict && haskey(info, "commit") || continue
            dirty = get(info, "dirty", false) === true
            println(io, "- $repo: `$(get(info, "commit", "?"))` on `$(get(info, "branch", "?"))`",
                        dirty ? " — dirty, patch carried over as `$(get(info, "diff", "?"))`" : "")
        end
        # A reconstructed entry (one whose merge predates the log) cannot supply these.
        # Absent and empty both mean unrecorded here: a real run always has at least one
        # row and one problem, so an empty list is never a fact about the run.
        rec(k) = (v = get(s, k, nothing); v !== nothing && v != "?" && !isempty(v))
        rec("rows") && println(io, "- Rows: $(g("rows")) · cells: $(g("cells"))")
        println(io, "- Cells ok/error/skipped/timeout: $(g("n_ok"))/$(g("n_error"))/",
                    "$(g("n_skipped"))/$(g("n_timeout"))")
        if rec("problems")
            probs, algs = get(s, "problems", []), get(s, "algorithms", [])
            println(io, "- Seeds: ", let sd = get(s, "seeds", [])
                        isempty(sd) ? "drawn per run" : join(sd, ", ") end)
            println(io, "- Problems ($(length(probs))): $(join(probs, ", "))")
            println(io, "- Algorithms ($(length(algs))): $(join(algs, ", "))")
        else
            println(io, "- Seeds / problems / algorithms: not recorded — this merge predates ",
                        "`$COMBINE_LOG_FILE` and was reconstructed from `meta.json`")
        end
        println(io)
    end

    moves = get(e, "moves", [])
    if !isempty(moves)
        println(io, total == 1 ? "### Sample provenance" : "### Samples this merge added")
        println(io)
        println(io, "Where each `sample_index` in `results.csv` came from. A cell that appeared in")
        println(io, "more than one source had its samples **pooled** and renumbered, so it reads")
        println(io, "downstream as one cell that was simply run more times.")
        println(io)
        println(io, "| cell_id | source | samples there | → here |")
        println(io, "|---|---|---|---|")
        for m in moves
            println(io, "| `$(get(m, "cell_id", "?"))` | `$(get(m, "source", "?"))` | ",
                        "$(get(m, "from", "?")) | $(get(m, "to", "?")) |")
        end
        println(io)
    end
    return nothing
end

# --- entry point ------------------------------------------------------------

"Write via a sibling temp file and rename, so an in-place merge cannot leave a half-written record."
function _write_atomic_file(f, path::AbstractString)
    tmp = path * ".tmp"
    open(f, tmp, "w")
    mv(tmp, path; force=true)
    return path
end

"""
    combine_results(dirs; kwargs...) -> out_dir

Merge two or more results directories into one that behaves exactly like a single sweep's,
and (by default) run the analysis suite over it.

The merge is a union over `{problems} x {algorithms}` cells. A cell present in several
sources has its samples pooled and `sample_index` renumbered `1..N`, with
`runs/<cell>/sample<k>/` renamed to match and `trace_path` rewritten accordingly. Rows
without a sample index (`error:build`, `skipped:finite_horizon`) pass through untouched.

**A merged directory absorbs.** When one of `dirs` was itself produced by this tool, the
merge happens INSIDE it — the other results are added to it and its existing samples keep
the numbers they already have — rather than into a fresh directory. With several merged
directories among the sources, the FIRST one is the target. So re-running as more sweeps
finish grows one directory instead of spawning a chain of them, and `combine_log.json`
accumulates one entry per pass. Passing `out` always makes a fresh directory instead.

Keyword arguments:

  - `out`             — output directory; forces a fresh one even when a source is merged
                        (default: absorb into the first merged source, else
                        `<parent of dirs[1]>/<name>_combined_<timestamp>`)
  - `name`            — experiment name (default: the target's, else the shared source name)
  - `mode`            — `:link` (hardlink, falling back to a copy across filesystems) or `:copy`
  - `dedupe_seeds`    — keep only the first measurement of each `(cell_id, seed)` pair
  - `reference_label` — the baseline arm for the merged config and the analysis
  - `force`           — downgrade the blocking compatibility checks to recorded caveats
  - `analyses`        — run `run_analyses` on the result afterwards (default `true`)
  - `gap_videos`      — override `[output].gap_videos` for that analysis pass
  - `command`         — the invocation to record in `COMBINED_FROM.md`

Sources other than an absorbing target are opened read-only and never modified.
"""
function combine_results(dirs::AbstractVector{<:AbstractString};
                         out::Union{Nothing,AbstractString}=nothing,
                         name::Union{Nothing,AbstractString}=nothing,
                         mode::Symbol=:link,
                         dedupe_seeds::Bool=false,
                         reference_label::Union{Nothing,AbstractString}=nothing,
                         force::Bool=false,
                         analyses::Bool=true,
                         gap_videos::Union{Nothing,Bool}=nothing,
                         command::AbstractString="")
    length(dirs) >= 2 || error("combine_results needs at least two results directories")
    mode in (:link, :copy) || error("unknown file mode :$mode (expected :link or :copy)")
    seen = Set{String}()
    for d in dirs
        a = abspath(String(d))
        a in seen && error("results directory given twice: $a")
        push!(seen, a)
    end

    srcs = CombineSource[load_combine_source(d) for d in dirs]

    # A merged directory absorbs: it becomes the target and the first source, so its rows
    # and run directories stay exactly where they are and the rest are added to it.
    target_i = out === nothing ? findfirst(_is_combined, srcs) : nothing
    in_place = target_i !== nothing
    if in_place && target_i != 1
        srcs = CombineSource[srcs[target_i]; deleteat!(copy(srcs), target_i)]
        @info "merging into the existing combined directory" target=srcs[1].dir
    end

    rep = CombineReport(String[], force)
    _check_compatibility!(rep, srcs)

    if in_place
        # Re-adding a run already folded into the target would double-count it: the same
        # measurements would arrive a second time under fresh sample numbers.
        already = Set(_underlying_runs(srcs[1]))
        for s in srcs[2:end], run in _underlying_runs(s)
            run in already && _blocking!(rep,
                "'$run' is already part of '$(srcs[1].tag)'; merging it again would " *
                "double-count its samples")
        end
    end

    all_names = unique(String[_experiment_name(s) for s in srcs])
    exp_name = name !== nothing ? String(name) :
               in_place ? _experiment_name(srcs[1]) :
               _shared_name(all_names)

    out_dir = if in_place
        srcs[1].dir
    elseif out === nothing
        joinpath(dirname(srcs[1].dir),
                 string(exp_name, "_combined_",
                        Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")))
    else
        abspath(String(out))
    end
    if !in_place && ispath(out_dir)
        (isdir(out_dir) && isempty(readdir(out_dir))) ||
            error("output directory exists and is not empty: $out_dir")
    end

    moves, keep, counts = _plan_samples(srcs, dedupe_seeds, rep; in_place=in_place)

    mkpath(joinpath(out_dir, "runs"))
    mkpath(joinpath(out_dir, "raw"))
    mkpath(joinpath(out_dir, "plots"))

    # --- run directories, renumbered ---------------------------------------
    # Keyed by (source, cell, old index) so the results.csv rewrite and the video rename
    # below agree on where each sample landed. Done FIRST, and purely additively: an
    # in-place merge that dies here leaves the target's own data untouched and only some
    # orphan sample directories that nothing references.
    remap = Dict{Tuple{Int,String,Int},Int}()
    nfiles = 0
    for m in moves
        remap[(m.source, m.cell_id, m.old_index)] = m.new_index
        (in_place && m.source == 1) && continue      # already in the output directory
        nfiles += _transfer_tree(
            joinpath(srcs[m.source].dir, "runs", m.cell_id, "sample$(m.old_index)"),
            joinpath(out_dir, "runs", m.cell_id, "sample$(m.new_index)"), mode)
    end

    # --- results.csv --------------------------------------------------------
    frames = DataFrame[]
    for (i, s) in enumerate(srcs)
        df = copy(s.results[keep[i], :])
        isempty(df) && continue
        has_trace = "trace_path" in names(df)
        for r in 1:nrow(df)
            idx = _int_or_nothing(df[r, :sample_index])
            idx === nothing && continue
            cell = _str_or_empty(df[r, :cell_id])
            new = get(remap, (i, cell, idx), nothing)
            (new === nothing || new == idx) && continue      # pinned: already correct
            df[r, :sample_index] = string(new)
            # trace_path is stored relative to the results dir, so it has to follow the
            # sample into its new directory. The basename is kept as recorded rather than
            # assumed, so a future trace filename needs no change here.
            if has_trace && !ismissing(df[r, :trace_path])
                df[r, :trace_path] = join(["runs", cell, "sample$new",
                                           basename(String(df[r, :trace_path]))], "/")
            end
        end
        push!(frames, df)
    end
    merged = isempty(frames) ? DataFrame() : vcat(frames...; cols=:orderequal)
    _write_atomic_file(io -> CSV.write(io, merged), joinpath(out_dir, "results.csv"))

    # --- problems.csv (union by problem name) -------------------------------
    pframes = DataFrame[s.problems for s in srcs if !isempty(s.problems)]
    if !isempty(pframes)
        pmerged = vcat(pframes...; cols=:union)
        "problem" in names(pmerged) && (pmerged = unique(pmerged, :problem))
        _write_atomic_file(io -> CSV.write(io, pmerged), joinpath(out_dir, "problems.csv"))
    end

    # --- validation.csv + summary -------------------------------------------
    vframes = DataFrame[]
    for (i, s) in enumerate(srcs)
        isempty(s.validation) && continue
        df = copy(s.validation)
        if all(c -> c in names(df), ("cell_id", "sample_index"))
            for r in 1:nrow(df)
                idx = _int_or_nothing(df[r, :sample_index])
                idx === nothing && continue
                new = get(remap, (i, _str_or_empty(df[r, :cell_id]), idx), nothing)
                new === nothing || (df[r, :sample_index] = string(new))
            end
        end
        push!(vframes, df)
    end
    isempty(vframes) || _write_atomic_file(io -> CSV.write(io, vcat(vframes...; cols=:union)),
                                           joinpath(out_dir, "validation.csv"))
    _merge_validation_summary(out_dir, srcs)

    # --- provenance & records -----------------------------------------------
    _merge_raw!(out_dir, srcs, moves, mode, rep)
    _merge_provenance!(out_dir, srcs, mode)
    _merge_gap_videos!(out_dir, srcs, moves, mode)

    cfg = _merged_config(srcs, exp_name, counts, reference_label)
    _write_atomic_file(io -> TOML.print(io, cfg; sorted=true),
                       joinpath(out_dir, "resolved_config.toml"))

    meta = _merged_meta(srcs, exp_name, counts, nrow(merged))
    _write_atomic_file(io -> JSON.print(io, meta, 2), joinpath(out_dir, "meta.json"))

    _write_atomic_file(joinpath(out_dir, "progress.json")) do io
        JSON.print(io, Dict{String,Any}(
            "cells_done" => length(counts), "n_cells" => meta["n_cells"],
            "n_ok" => meta["n_ok"], "n_error" => meta["n_error"],
            "n_skipped" => meta["n_skipped"], "n_timeout" => meta["n_timeout"],
            "last_cell_id" => isempty(moves) ? "" : last(moves).cell_id,
            "datetime_last" => _combine_now()), 2)
    end

    # --- the merge log, and the prose rendered from it ----------------------
    added = in_place ? srcs[2:end] : srcs
    tagged = [(srcs[m.source].tag, m) for m in moves if !(in_place && m.source == 1)]
    log = in_place ? _read_combine_log(srcs[1]) : Any[]
    # An added source that is itself merged brings its own log with it, so the runs behind
    # it stay named here instead of collapsing into one opaque directory that may later be
    # deleted. Its entries happened earlier, so they go in before this one.
    for s in added
        _is_combined(s) || continue
        for e in _read_combine_log(s)
            entry = _as_dict(e)
            entry["inherited_from"] = s.tag
            push!(log, entry)
        end
    end
    push!(log, _log_entry(added, tagged, rep.warnings; command=command, mode=mode,
                          dedupe_seeds=dedupe_seeds, in_place=in_place,
                          target=in_place ? srcs[1].dir : nothing))
    _write_atomic_file(io -> JSON.print(io, log, 2), joinpath(out_dir, COMBINE_LOG_FILE))
    _render_sources_file(out_dir, log, exp_name, counts, length(moves))

    @info(in_place ? "added to combined results" : "combined results",
          sources=length(added), cells=length(counts), samples=length(moves),
          rows=nrow(merged), files=nfiles, out=out_dir)
    isempty(rep.warnings) ||
        @warn "$(length(rep.warnings)) caveat(s) recorded in $COMBINE_SOURCES_FILE" out=out_dir

    if analyses
        try
            BenchmarkAnalysis.run_analyses(out_dir;
                reference_label=String(cfg["output"]["reference_label"]),
                gap_videos=gap_videos)
        catch e
            # The assembled directory is the valuable output; a plotting failure must not
            # cost it. `run_analyses.jl <dir>` can always be retried by hand.
            @warn "analysis suite failed; the combined directory is complete and can be \
                   re-analysed with `julia --project run_analyses.jl <dir>`" out=out_dir exception=(e, catch_backtrace())
        end
    end
    return out_dir
end

combine_results(dirs::AbstractString...; kwargs...) = combine_results(collect(dirs); kwargs...)

"""
    rerender_sources_file(dir) -> path

Rewrite `COMBINED_FROM.md` from the `combine_log.json` already in `dir`, without merging
anything. The prose is derived, not authoritative — the log is — so this brings an older
directory's write-up up to date with the current renderer, or picks up a hand-edited log.
The cell and sample counts in the header come from the directory's own `results.csv`.
"""
function rerender_sources_file(dir::AbstractString)
    d = abspath(String(dir))
    log = _read_combine_log(load_combine_source(d))
    isempty(log) && error("'$d' has no $COMBINE_LOG_FILE and no combined meta.json to " *
                          "reconstruct one from; it was not built by combine_results")
    res = _read_table(joinpath(d, "results.csv"))
    counts = Dict{String,Int}()
    n_samples = 0
    if all(c -> c in names(res), ("cell_id", "sample_index"))
        for row in eachrow(res)
            _int_or_nothing(row.sample_index) === nothing && continue
            cell = _str_or_empty(row.cell_id)
            counts[cell] = get(counts, cell, 0) + 1
            n_samples += 1
        end
    end
    name = String(get(_read_json_dict(joinpath(d, "meta.json")), "experiment", basename(d)))
    return _render_sources_file(d, log, name, counts, n_samples)
end
