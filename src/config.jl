# Declarative experiment definition: parse a TOML file into in-memory specs,
# expanding dataset globs and validating every referenced builder name against
# the registries BEFORE any expensive construction happens. The fully-resolved
# config is dumped back to the results dir for reproducibility.

struct ProblemSpec
    name::String              # registered problem-builder name
    instance_name::String     # disambiguates dataset members / param variants
    params::Dict{String,Any}
end

struct AlgorithmSpec
    name::String              # registered algorithm-builder name
    label::String             # disambiguates same-builder variants
    params::Dict{String,Any}
end

struct BenchmarkSettings
    samples::Int            # number of independent runs when `seeds` is unset (each a fresh random seed)
    warmup::Bool
    seeds::Vector{Int}      # explicit per-sample seeds; empty => draw `samples` random seeds
    # --- censoring / wall-clock budgets (0 disables) ---
    # Per-SOLVE deadline: the trace callback throws `CellTimeout` once a timed sample
    # exceeds this, producing a `status="timeout"` row plus the partial trace as a
    # censored convergence curve. Without it, a solve runs until convergence and a
    # cluster wall-clock kill leaves NO record at all. The untimed warm-up gets a much
    # looser multiple of this (see `_WARMUP_TIMEOUT_FACTOR`) because its cost is JIT,
    # not algorithmic.
    timeout_s::Float64
    # Total sweep budget. The loop stops cleanly BETWEEN cells once it is exhausted, so
    # the post-loop writes (problems.csv, validation.csv, the final meta.json) still run.
    # 0 => derive from SLURM's `SLURM_JOB_END_TIME` when set, else unlimited.
    wall_budget_s::Float64
    wall_reserve_s::Float64 # slack subtracted from the sweep deadline for those writes
end

struct OutputSettings
    save_raw::Bool
    save_trace::Bool
    # The solved policy, one small CSV per completed sample (analysis/policy_plot.jl draws
    # it). One row per state rather than per iteration, so unlike the trace matrices it
    # costs kilobytes, not gigabytes -- hence on by default where `gap_videos` is not.
    save_policy::Bool
    track_initial::Bool
    traces::Vector{String}      # which trace kinds to record (see TRACE_BUILDERS); default ["trace"]
    reference_label::String     # alg_label of the baseline variant the analysis compares against
    results_root::String
    image_formats::Vector{String}   # figure file types to emit, e.g. ["png", "svg"]
    analysis::Bool              # run the report/plot suite after the sweep (see below)
    # Grid-world gap videos (one mp4 per run per sample; see analysis/gap_video.jl).
    # Off by default: a single `gap_trace.csv` can be several GB and a sweep writes one
    # per sample, so this is only ever worth paying for when it is asked for.
    gap_videos::Bool
    gap_video_max_frames::Int   # iterations to render, counted from the first
    gap_video_framerate::Int    # frames per second
    # Relative-gap thinning of the scalar traces while they record (see TraceExecutor in
    # trace.jl and thinning.jl). 0 = off: every callback is a trace.csv row, which for a
    # sampled arm means tens of millions of rows and gigabytes held in memory per solve.
    trace_thin_rtol::Float64
end

# valid figure output types (lowercase, no leading dot)
const VALID_IMAGE_FORMATS = ("png", "svg", "pdf", "eps")

# accept either a single string ("png") or a list (["png", "svg"]); strip a leading dot
_parse_image_formats(x) = String[lowercase(strip(replace(String(s), r"^\." => ""))) for s in (x isa AbstractString ? [x] : collect(x))]

# Overrides for the reach-avoid property baked into each pre-built problem. A
# `nothing` field leaves the problem's own value untouched; a set field rebuilds
# the specification with the new value before solving (see `override_specification`).
struct PropertySettings
    convergence_eps::Union{Float64,Nothing}      # value-iteration tolerance
    restrict_to_initial::Union{Bool,Nothing}     # converge only on the model's initial states
end

struct ExperimentConfig
    name::String
    problems::Vector{ProblemSpec}
    algorithms::Vector{AlgorithmSpec}
    benchmark::BenchmarkSettings
    output::OutputSettings
    property::PropertySettings
    raw::Dict{String,Any}
end

# --- tiny helpers ----------------------------------------------------------

_basename_noext(p) = first(splitext(basename(String(p))))

_toml_dict(x) = x isa AbstractDict ? Dict{String,Any}(String(k) => v for (k, v) in x) : Dict{String,Any}()

# Expand `~`, `$VAR` and `${VAR}` in the path-valued config keys, so a config can say
# "/scratch/$USER/results" instead of hardcoding a username (cluster runs). An undefined
# variable is left as the literal `$VAR` rather than expanding to "", so the path fails
# loudly downstream instead of silently pointing somewhere else.
function _expand_path(p::AbstractString)
    s = replace(String(p), r"\$\{(\w+)\}|\$(\w+)" => m -> get(ENV, match(r"\w+", m).match, m))
    return expanduser(s)
end

# --- problem discovery ------------------------------------------------------
# Problems are pre-built files on disk, one subdirectory per problem (named
# after the problem). The loader detects the format from the files inside and
# targets the matching file-reader builder (`intervalmdp_file` / `prism`).

"""
    discover_problems(root, names) -> Vector{ProblemSpec}

Scan `root` for problem subdirectories. `names === nothing`/empty selects every
subdirectory (sorted); otherwise only the named ones (erroring on a missing name).
"""
function discover_problems(root::AbstractString, names)
    isdir(root) || error("problems root not found: $root")
    subdirs = names === nothing || isempty(names) ?
              sort!(filter(d -> isdir(joinpath(root, d)), readdir(root))) :
              String.(collect(names))
    problems = ProblemSpec[]
    for n in subdirs
        dir = joinpath(root, n)
        isdir(dir) || error("problem '$n' not found under '$root'")
        push!(problems, _problem_spec_from_dir(dir))
    end
    return problems
end

"""
    _problem_spec_from_dir(dir) -> ProblemSpec

Dispatch on the files in a problem subdirectory:
- exactly one `*.nc` (+ same-basename `.json` spec) -> `intervalmdp_file`
- a basename carrying the full PRISM set (`.sta`+`.tra`+`.lab`+`.pctl`) -> `prism`
The instance name is the subdirectory's basename.
"""
function _problem_spec_from_dir(dir::AbstractString)
    name  = basename(rstrip(String(dir), '/'))
    files = readdir(dir)

    ncs = filter(f -> endswith(f, ".nc"), files)
    if !isempty(ncs)
        length(ncs) == 1 ||
            error("problem '$name': expected one .nc model, found $(length(ncs)) ($(join(ncs, ", ")))")
        model = joinpath(dir, ncs[1])
        spec  = first(splitext(model)) * ".json"
        isfile(spec) ||
            error("problem '$name': model '$(basename(model))' has no same-basename .json spec")
        return ProblemSpec("intervalmdp_file", name,
                           Dict{String,Any}("model" => model, "spec" => spec))
    end

    # PRISM explicit format: a basename with the full .sta/.tra/.lab/.pctl set.
    for b in unique(_basename_noext.(files))
        has = ext -> isfile(joinpath(dir, b * ext))
        if has(".sta") && has(".tra") && has(".lab") && has(".pctl")
            return ProblemSpec("prism", name, Dict{String,Any}("file" => joinpath(dir, b)))
        end
    end

    error("problem '$name': no loadable model found in '$dir' " *
          "(expected a .nc+.json pair or a .sta/.tra/.lab/.pctl PRISM set)")
end

# --- the loader ------------------------------------------------------------

"""
    load_experiment_config(path; base_dir) -> ExperimentConfig

Parse a TOML experiment definition. Dataset globs (a `glob` key in a problem's
params) expand to one `ProblemSpec` per matching file (each with a `file` param).
Paths are resolved relative to `base_dir` (default: the config file's directory).
"""
function load_experiment_config(path::AbstractString; base_dir::AbstractString=dirname(abspath(path)))
    cfg = TOML.parsefile(path)

    name = String(get(cfg, "name", _basename_noext(path)))

    b = _toml_dict(get(cfg, "benchmark", Dict()))
    benchmark = BenchmarkSettings(
        Int(get(b, "samples", 5)),
        Bool(get(b, "warmup", true)),
        Int.(collect(get(b, "seeds", Int[]))),
        Float64(get(b, "timeout_s", 0.0)),
        Float64(get(b, "wall_budget_s", 0.0)),
        Float64(get(b, "wall_reserve_s", 300.0)),
    )

    o = _toml_dict(get(cfg, "output", Dict()))
    output = OutputSettings(
        Bool(get(o, "save_raw", true)),
        Bool(get(o, "save_trace", true)),
        Bool(get(o, "save_policy", true)),
        Bool(get(o, "track_initial", true)),
        String[lowercase(strip(String(t))) for t in get(o, "traces", ["trace"])],
        String(get(o, "reference_label", "gsrdp_exhaustive_state")),
        _expand_path(String(get(o, "results_root", "results"))),
        _parse_image_formats(get(o, "image_formats", ["png", "svg"])),
        # Gates only the in-harness report (figures, METHODS_meanstd.md, manifest.txt).
        # `run_analyses.jl` deliberately ignores this, so a sweep run with `analysis =
        # false` (e.g. on a cluster) can still be analysed later from its results dir.
        Bool(get(o, "analysis", true)),
        Bool(get(o, "gap_videos", false)),
        Int(get(o, "gap_video_max_frames", 600)),
        Int(get(o, "gap_video_framerate", 15)),
        Float64(get(o, "trace_thin_rtol", 0.0)),
    )

    pr = _toml_dict(get(cfg, "property", Dict()))
    property = PropertySettings(
        haskey(pr, "convergence_eps") ? Float64(pr["convergence_eps"]) : nothing,
        haskey(pr, "restrict_to_initial") ? Bool(pr["restrict_to_initial"]) : nothing,
    )

    pcfg  = _toml_dict(get(cfg, "problems", Dict()))
    proot = _expand_path(String(get(pcfg, "root", "problems")))
    proot = isabspath(proot) ? proot : joinpath(base_dir, proot)
    names = haskey(pcfg, "names") ? String.(collect(pcfg["names"])) : nothing
    problems = discover_problems(proot, names)

    algorithms = AlgorithmSpec[]
    for a in get(cfg, "algorithms", [])
        builder = String(a["name"])
        params = _toml_dict(get(a, "params", Dict()))
        label = haskey(a, "label") ? String(a["label"]) : algo_label(builder, params)
        push!(algorithms, AlgorithmSpec(builder, label, params))
    end

    validate_config(problems, algorithms, output; problems_root=proot, benchmark=benchmark)
    return ExperimentConfig(name, problems, algorithms, benchmark, output, property, cfg)
end

"Fail fast on unknown builder names or a reference algorithm not in the run set."
function validate_config(problems, algorithms, output; problems_root="problems", benchmark=nothing)
    isempty(problems)   && error("no problems found under '$problems_root' " *
                                 "(expected subdirectories with a .nc+.json pair or PRISM files)")
    isempty(algorithms) && error("config defines no [[algorithms]]")
    for p in problems
        has_problem(p.name) || error("unknown problem builder '$(p.name)'. Registered: $(sort(collect(keys(PROBLEM_REGISTRY))))")
    end
    for a in algorithms
        has_algorithm(a.name) || error("unknown algorithm builder '$(a.name)'. Registered: $(sort(collect(keys(ALGORITHM_REGISTRY))))")
    end
    # Labels name run directories (`runmatrix.jl`), results rows and the
    # reference algorithm, so a collision silently merges two cells into one.
    # `algo_label` only reads `sampling`/`k`/`bellman`, and the trajectory
    # sampler has a dozen more knobs, so two entries differing only in, say,
    # `state_temperature` generate the same label. Make that an error the
    # config can fix with an explicit `label`.
    seen = Dict{String,Int}()
    for a in algorithms
        seen[a.label] = get(seen, a.label, 0) + 1
    end
    for (label, n) in seen
        n == 1 || error("$n [[algorithms]] entries resolve to the same label '$label'; " *
                        "run directories and results rows would collide. Give each one a " *
                        "distinct `label = \"...\"` key.")
    end
    for t in output.traces
        known_trace_kind(t) || error("unknown trace kind '$t' in [output].traces. Valid: $(trace_kinds())")
    end
    isempty(output.image_formats) && error("[output].image_formats is empty; give at least one of $(VALID_IMAGE_FORMATS)")
    for f in output.image_formats
        f in VALID_IMAGE_FORMATS || error("unknown image format '$f' in [output].image_formats. Valid: $(VALID_IMAGE_FORMATS)")
    end
    output.gap_video_max_frames > 0 ||
        error("[output].gap_video_max_frames must be positive, got $(output.gap_video_max_frames)")
    output.gap_video_framerate > 0 ||
        error("[output].gap_video_framerate must be positive, got $(output.gap_video_framerate)")
    0 <= output.trace_thin_rtol < 1 ||
        error("[output].trace_thin_rtol must be in [0, 1), got $(output.trace_thin_rtol)")
    # Each video panel is rendered from its own heavy trace, so a missing trace kind
    # costs a panel rather than the whole video — except when ALL of them are missing,
    # which leaves nothing to draw. Named individually so the warning says what is lost.
    if output.gap_videos
        missing_panels = String[]
        "gap"   in output.traces || push!(missing_panels, "gap (needs traces = \"gap\")")
        "value" in output.traces ||
            push!(missing_panels, "lower/upper bounds (needs traces = \"value\")")
        "updates" in output.traces ||
            push!(missing_panels, "per-iteration update counts (needs traces = \"updates\")")
        if length(missing_panels) == 3
            @warn "[output].gap_videos = true but [output].traces records none of \"gap\", \
                   \"value\", \"updates\"; no heavy trace will be written and no video can \
                   be rendered" traces=output.traces
        elseif !isempty(missing_panels)
            @warn "[output].gap_videos = true; these video panels will be missing" panels=missing_panels traces=output.traces
        end
    end
    if !any(a -> a.label == output.reference_label, algorithms)
        @warn "reference_label '$(output.reference_label)' is not among the configured algorithm labels; the analysis will fall back to the first algorithm as reference."
    end
    if benchmark !== nothing
        benchmark.timeout_s      >= 0 || error("[benchmark].timeout_s must be >= 0 (0 disables the per-solve deadline)")
        benchmark.wall_budget_s  >= 0 || error("[benchmark].wall_budget_s must be >= 0 (0 => derive from SLURM, else unlimited)")
        benchmark.wall_reserve_s >= 0 || error("[benchmark].wall_reserve_s must be >= 0")
        # A cell needs `samples` solves plus (optionally) an untimed warm-up, each bounded
        # by `timeout_s` — warn when even a single cell cannot fit the declared budget.
        if benchmark.timeout_s > 0 && benchmark.wall_budget_s > 0
            # The warm-up's allowance is a multiple of timeout_s (see _WARMUP_TIMEOUT_FACTOR).
            per_cell = (10 * benchmark.warmup + benchmark.samples) * benchmark.timeout_s
            per_cell > benchmark.wall_budget_s &&
                @warn "one cell can take up to $(per_cell)s ((warmup + samples) x timeout_s) but [benchmark].wall_budget_s is $(benchmark.wall_budget_s)s; cells will be censored by the sweep deadline"
        end
    end
    return nothing
end

"""
    write_resolved_config(cfg, path)

Dump the fully-expanded config (every dataset member listed explicitly, every
default filled) back to TOML for an exactly-reproducible record.
"""
function write_resolved_config(cfg::ExperimentConfig, path::AbstractString)
    d = Dict(
        "name" => cfg.name,
        "benchmark" => Dict(
            "samples" => cfg.benchmark.samples,
            "warmup" => cfg.benchmark.warmup,
            "seeds" => cfg.benchmark.seeds,
            # The analysis reads `timeout_s` back from here to use as the censoring
            # budget instead of the max-observed fallback (see analysis/adapter.jl).
            "timeout_s" => cfg.benchmark.timeout_s,
            "wall_budget_s" => cfg.benchmark.wall_budget_s,
            "wall_reserve_s" => cfg.benchmark.wall_reserve_s,
        ),
        "output" => Dict(
            "save_raw" => cfg.output.save_raw, "save_trace" => cfg.output.save_trace,
            # Read back by `analysis_per_problem_policy`, so a standalone re-analysis can
            # tell "this sweep never recorded policies" from "it did and these cells failed".
            "save_policy" => cfg.output.save_policy,
            "track_initial" => cfg.output.track_initial,
            "traces" => cfg.output.traces,
            "reference_label" => cfg.output.reference_label,
            "results_root" => cfg.output.results_root,
            "image_formats" => cfg.output.image_formats,
            "analysis" => cfg.output.analysis,
            # Read back by `run_analyses` (analysis/BenchmarkAnalysis.jl), so a later
            # standalone re-analysis renders the same videos the sweep asked for.
            "gap_videos" => cfg.output.gap_videos,
            "gap_video_max_frames" => cfg.output.gap_video_max_frames,
            "gap_video_framerate" => cfg.output.gap_video_framerate,
            "trace_thin_rtol" => cfg.output.trace_thin_rtol,
        ),
        "problems" => [Dict("name" => p.name, "instance_name" => p.instance_name,
                            "params" => p.params) for p in cfg.problems],
        "algorithms" => [Dict("name" => a.name, "label" => a.label,
                              "params" => a.params) for a in cfg.algorithms],
    )
    # Only record property overrides that were actually set (an unset field leaves
    # the per-problem built-in value, which has no single config-level value).
    prop = Dict{String,Any}()
    cfg.property.convergence_eps     === nothing || (prop["convergence_eps"]     = cfg.property.convergence_eps)
    cfg.property.restrict_to_initial === nothing || (prop["restrict_to_initial"] = cfg.property.restrict_to_initial)
    isempty(prop) || (d["property"] = prop)
    open(io -> TOML.print(io, d; sorted=true), path, "w")
    return path
end
