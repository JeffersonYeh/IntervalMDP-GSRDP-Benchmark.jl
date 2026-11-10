# Result storage: one tidy row per cell appended incrementally to results.csv
# (crash-safe + resumable), plus run-level provenance (meta.json, resolved
# config, git provenance (commit + diff per repo, provenance.json)). The CSV is
# the index; full convergence traces and
# raw BenchmarkTools.Trial objects live alongside it.

_now_string() = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")

"Create the timestamped results directory `results_root/<name>_<timestamp>/` and subdirs."
function make_results_dir(cfg::ExperimentConfig, base_dir::AbstractString)
    root = isabspath(cfg.output.results_root) ? cfg.output.results_root :
           joinpath(base_dir, cfg.output.results_root)
    stamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    dir = joinpath(root, string(cfg.name, "_", stamp))
    mkpath(joinpath(dir, "runs"))
    mkpath(joinpath(dir, "raw"))
    # Only pre-create plots/ when the report will actually run; leaving an empty one
    # behind makes `analysis = false` look like a failed analysis rather than a skipped
    # one. `run_analyses.jl` mkpath's it itself when invoked later on the directory.
    cfg.output.analysis && mkpath(joinpath(dir, "plots"))
    return dir
end

# --- timing extraction -----------------------------------------------------

"Empty timing fields (for skipped/error rows, or when there is no trial)."
empty_timing() = (time_min_s=missing, time_median_s=missing, time_mean_s=missing,
                  time_max_s=missing, time_std_s=missing, n_samples=0,
                  allocs=missing, memory_bytes=missing, gctime_min_s=missing)

"""
    sample_timing(trial, k) -> NamedTuple

Timing fields for sample `k` of a `BenchmarkTools.Trial`. Each sample is its own
independent data point (a distinct seed), so min=median=mean=max = that sample's
time and std=0; the across-sample spread is recovered by aggregating the per-sample
rows in the report. `allocs`/`memory_bytes` are the trial-level minima (BenchmarkTools
keeps only those), shared across the cell's rows.
"""
function sample_timing(trial, k::Integer)
    t = trial.times[k] / 1e9
    return (time_min_s = t, time_median_s = t, time_mean_s = t, time_max_s = t,
            time_std_s = 0.0, n_samples = 1,
            allocs = trial.allocs, memory_bytes = trial.memory,
            gctime_min_s = trial.gctimes[k] / 1e9)
end

"""
    wall_timing(seconds) -> NamedTuple

`sample_timing`'s shape from a wall-clock stamp taken around the solve, for samples
whose `BenchmarkTools.Trial` was lost — a `CellTimeout` unwinds out of the trial
runner and takes the completed samples' times with it. Allocation/GC fields are
`missing` because only the trial carried them; the timing columns stay comparable
because the stamp brackets the same region to within a `time_ns()` call.
"""
wall_timing(seconds::Real) =
    (time_min_s = Float64(seconds), time_median_s = Float64(seconds),
     time_mean_s = Float64(seconds), time_max_s = Float64(seconds),
     time_std_s = 0.0, n_samples = 1,
     allocs = missing, memory_bytes = missing, gctime_min_s = missing)

# --- one results row (fixed column order across all appends) ----------------

function assemble_row(unit::RunUnit, bp, built, timing::NamedTuple, q::QualityResult,
                      agr::ReferenceAgreement; status::AbstractString,
                      sample_index=missing, seed=missing,
                      error_msg=missing, error_type=missing, trace_path=missing,
                      wall_total_s=missing)
    head = (
        cell_id = unit.id,
        problem = unit.problem.name,
        instance_name = unit.problem.instance_name,
        source = bp === nothing ? missing : String(bp.source),
        num_states = bp === nothing ? missing : get(bp.metadata, "num_states", missing),
        num_actions = bp === nothing ? missing : get(bp.metadata, "num_actions", missing),
        algorithm = unit.algorithm.name,
        alg_label = unit.algorithm.label,
        alg_kind = built === nothing ? missing : built.kind,
        bellman = built === nothing ? missing : get(built.descriptor, "bellman", missing),
        sampling = built === nothing ? missing : get(built.descriptor, "sampling", missing),
        sampling_k = built === nothing ? missing : get(built.descriptor, "k", missing),
        convergence_eps = bp === nothing ? missing : problem_convergence_eps(bp),
        sample_index = sample_index,
        seed = seed,
    )
    quality = (
        num_iterations = q.num_iterations,
        bellman_updates = q.bellman_updates,
        final_gap_max = q.final_gap_max,
        final_gap_mean = q.final_gap_mean,
        converged = q.converged,
        trace_path = trace_path,
    )
    reference = (
        ref_max_diff = agr.ref_max_diff,
        ref_mean_diff = agr.ref_mean_diff,
        ref_max_diff_initial = agr.ref_max_diff_initial,
        agrees = agr.agrees,
    )
    tail = (
        status = String(status),
        error = error_msg,
        error_type = error_type,
        wall_total_s = wall_total_s,
        timestamp = _now_string(),
    )
    return merge(head, timing, quality, reference, tail)
end

"Append one row to results.csv, writing the header only on the first call."
function append_row!(csv_path::AbstractString, row::NamedTuple, header_written::Ref{Bool})
    CSV.write(csv_path, DataFrame([row]); append=header_written[], writeheader=!header_written[])
    header_written[] = true
    return nothing
end

# --- liveness records -------------------------------------------------------
# A killed sweep leaves no row for the cell it died in, and the post-loop writes
# (problems.csv, validation.csv, the final meta.json) never run. These two files are
# what let the analysis say "31 of 40 cells are missing" instead of quietly reporting
# on whatever survived.

"""
    _write_atomic(f, path)

Write via `path * \".tmp\"` then `mv` onto `path`. The rename is atomic within a
directory on POSIX, so a kill mid-write can never leave a half-parsed JSON file that
then breaks the loader — the whole point of these records is to be readable after a kill.
"""
function _write_atomic(f, path::AbstractString)
    tmp = path * ".tmp"
    open(f, tmp, "w")
    mv(tmp, path; force=true)
    return path
end

"Write the per-cell progress record (rewritten after every cell)."
write_progress(path::AbstractString, state::NamedTuple) =
    _write_atomic(path) do io
        JSON.print(io, merge(Dict(String(k) => v for (k, v) in pairs(state)),
                             Dict("datetime_last" => _now_string())), 2)
    end

"Write the abort marker. Best-effort, from an `atexit` hook — keep it to one small write."
write_aborted(path::AbstractString, state::NamedTuple, reason::AbstractString) =
    _write_atomic(path) do io
        JSON.print(io, merge(Dict(String(k) => v for (k, v) in pairs(state)),
                             Dict("reason" => reason, "datetime" => _now_string())), 2)
    end

# --- provenance ------------------------------------------------------------

function _git(args...; dir)
    try
        return strip(read(Cmd(`git -C $dir $(collect(args))`), String))
    catch
        return ""
    end
end

"Like `_git`, but returns stdout even when git exits nonzero (e.g. `diff --no-index`)."
function _git_capture(args...; dir)
    try
        return read(ignorestatus(Cmd(`git -C $dir $(collect(args))`)), String)
    catch
        return ""
    end
end

_pkg_field(deps, name, field) = begin
    out = "unknown"
    for (_, v) in deps
        if v.name == name
            out = string(getfield(v, field))
            break
        end
    end
    out
end

"Write run-level environment metadata to meta.json."
function write_meta(path::AbstractString, cfg::ExperimentConfig, n_cells::Int;
                    git_dir::AbstractString, extra::AbstractDict=Dict())
    deps = Pkg.dependencies()
    d = Dict{String,Any}(
        "experiment" => cfg.name,
        "datetime" => _now_string(),
        "julia_version" => string(VERSION),
        "intervalmdp_version" => _pkg_field(deps, "IntervalMDP", :version),
        "intervalmdp_source" => _pkg_field(deps, "IntervalMDP", :source),
        "benchmarktools_version" => _pkg_field(deps, "BenchmarkTools", :version),
        "hostname" => gethostname(),
        "cpu" => (try string(Sys.cpu_info()[1].model) catch; "unknown" end),
        "ncpu" => Sys.CPU_THREADS,
        "julia_threads" => Threads.nthreads(),
        "blas_threads" => (try LinearAlgebra.BLAS.get_num_threads() catch; -1 end),
        "n_cells" => n_cells,
        "git_commit" => _git("rev-parse", "HEAD"; dir=git_dir),
        "git_dirty" => !isempty(_git("status", "--porcelain"; dir=git_dir)),
    )
    for (k, v) in extra
        d[String(k)] = v
    end
    open(io -> JSON.print(io, d, 2), path, "w")
    return path
end

"Snapshot one git repo's state into `dir`: commit/branch/dirty in the returned
Dict, plus a `<name>.diff` patch file (tracked changes + untracked files) when
the tree is dirty. Returns `nothing` if `git_dir` is missing or not a git repo."
function _snapshot_repo(dir::AbstractString, name::AbstractString, git_dir)
    (git_dir === nothing || isempty(git_dir) || !isdir(git_dir)) && return nothing
    # Resolve the repo top-level and run every git call from there, so tracked
    # (`diff HEAD`, repo-root-relative) and untracked (`ls-files`/`--no-index`,
    # cwd-relative) paths share one root and the combined patch applies as a unit.
    root = _git("rev-parse", "--show-toplevel"; dir=git_dir)
    isempty(root) && return nothing              # not a git repo / git unavailable
    commit = _git("rev-parse", "HEAD"; dir=root)
    branch = _git("rev-parse", "--abbrev-ref", "HEAD"; dir=root)
    dirty  = !isempty(_git("status", "--porcelain"; dir=root))
    info = Dict{String,Any}("path" => root, "commit" => commit,
                            "branch" => branch, "dirty" => dirty)
    if dirty
        # `_git_capture` (not `_git`) so the diff keeps its trailing newline — `_git`
        # strips it and `git apply` then reports "corrupt patch". `--binary` makes
        # changes to binary files reapplicable rather than a "Binary files differ" stub.
        parts = String[]
        tracked = _git_capture("diff", "--binary", "HEAD"; dir=root)  # staged + unstaged tracked changes
        isempty(tracked) || push!(parts, tracked)
        untracked = split(_git("ls-files", "--others", "--exclude-standard"; dir=root),
                          '\n'; keepempty=false)
        for f in untracked                                  # each new file as an add-patch
            patch = _git_capture("diff", "--binary", "--no-index", "--", "/dev/null", f; dir=root)
            isempty(patch) || push!(parts, patch)
        end
        if !isempty(parts)
            diff_name = string(name, ".diff")
            write(joinpath(dir, diff_name), join(parts))
            info["diff"] = diff_name
        end
    end
    return info
end

"""
    snapshot_manifest(project_dir, dir) -> path

Record git provenance for reproducibility into `dir/provenance.json`: the commit
HEAD and branch for this repo (via `project_dir`) and for the dev-checked-out
IntervalMDP. For any repo with a dirty working tree, also writes a `git diff`
patch (`<repo>.diff` in `dir`) — tracked changes plus untracked files — that can
be reapplied with `git apply` on top of its commit to reconstruct the source.
"""
function snapshot_manifest(project_dir::AbstractString, dir::AbstractString)
    imdp_src = _pkg_field(Pkg.dependencies(), "IntervalMDP", :source)
    repos = ("thesis" => project_dir,
             "intervalmdp" => (imdp_src == "unknown" ? nothing : imdp_src))
    prov = Dict{String,Any}()
    for (name, gd) in repos
        info = _snapshot_repo(dir, name, gd)
        info === nothing || (prov[name] = info)
    end
    path = joinpath(dir, "provenance.json")
    open(io -> JSON.print(io, prov, 2), path, "w")
    return path
end
