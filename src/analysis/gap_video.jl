# Grid-world state-space videos: the value bounds, their gap, and where the sampler
# looked, all animated over the state space.
#
# The heavy trace CSVs (written by ../trace.jl) are wide matrices — one row per Bellman
# iteration, one column per state:
#
#   value_trace_lower.csv  `v[k]`    lower bound L        (`ValueTrace`)
#   value_trace_upper.csv  `v[k]`    upper bound U        (`ValueTrace`)
#   gap_trace.csv          `gap[k]`  U − L                (`GapTrace`)
#   update_trace.csv       `n[k]`    backups this iteration (`UpdateTrace`)
#
# Every scalar figure in analyses.jl reduces such a matrix to a curve; this file does the
# opposite and keeps the *spatial* structure, which is the only view that shows WHERE a
# sampled variant is still uncertain while the full sweep has already flattened the
# space — and, with the updates panel, whether the sampler ever looked there.
#
# The bounds answer a question the gap cannot: a stubborn gap because L is stuck low is a
# different problem from one because U is stuck high.
#
# Panels are independent. Whichever of the four files a sweep recorded get drawn; the
# rest are silently omitted, so every results dir written before the `"value"` and
# `"updates"` trace kinds were enabled still renders its single gap panel.
#
# Geometry. The abstraction problems are all built by `StateUniformGridSplit(X, (n, n))`
# plus one appended absorbing state for "left the partitioned region"
# (IntervalMDPAbstractions `src/abstractions/abstraction.jl`, "The resulting IMDP has
# `numregions(state_abstraction) + 1` states"). `LazySets.split` lays the regions out
# with `Iterators.product`, so the FIRST axis varies fastest and abstract state `s` is
# grid cell `(i, j)` with `s = (j - 1) * n + i` — plain Julia column-major order.
# Verified against `car_parking_S20_I3`: its reach region [4,10]×[-4,0] on X = [-10,10]²
# with n = 20 predicts states 135…140, 155…160, … and the stored spec lists exactly
# those. The trailing absorbing state has no place on the grid and is NOT drawn.
#
# Overlays. Each frame outlines the property's reach and avoid sets (from the stored spec
# JSON) and the model's initial states S0 (from problems.csv, else the model's netCDF —
# S0 lives in the model, not the spec). S0 is what `[property].restrict_to_initial`
# converges on, so it is the set a reader most needs to be able to find.
#
# Not every problem is a grid. The PRISM instances (`coin*`, `robot_delta0.1`) have no
# spatial structure at all, and reshaping them into a square would put unrelated states
# next to each other and read as noise. Those are skipped with a recorded reason rather
# than drawn — see `grid_geometry`.
#
# Cost. Videos are OFF by default (`[output].gap_videos`, or `gap_videos=true` on
# `run_analyses`). A single `gap_trace.csv` reaches 6.1 GB / 218k rows on a real sweep,
# and a results dir can hold well over a hundred of them, so nothing here ever reads a
# whole file: `_read_gap_head` streams and stops at `max_frames` rows. Only ONE sample
# per (problem, algorithm) cell is rendered — the samples of a cell differ only by random
# seed, so the rest would be near-duplicates at N× the cost (`_select_one_per_cell`).

# The problem families built by `StateUniformGridSplit`, i.e. the ones a grid picture is
# meaningful for, keyed by the prefix of the instance name. Membership is what makes a
# trace eligible at all (see `grid_geometry`); the value is the physical extent of the
# state space so the axes can carry real coordinates instead of cell indices, and may be
# `nothing` for a grid family whose bounds are not known here.
#
# The extents mirror the `const X = Hyperrectangle(...)` in each generator under
# `tools/IntervalMDPAbstractions/`; package_delivery's comes from
# `ArchCompStochasticModels.package_delivery_synthesis` (X ∈ [-6,6]²).
#
# ADDING A FAMILY: a new abstraction problem needs its prefix here, or its videos are
# skipped with "not a known uniform-grid family". That is deliberate — an opt-in list
# cannot silently draw a non-grid model on a grid.
const _STATE_SPACE_EXTENT = Dict{String,Union{Nothing,NTuple{4,Float64}}}(
    "robot_2d"                       => (-10.0, 10.0, -10.0, 10.0),
    "car_parking"                    => (-10.0, 10.0, -10.0, 10.0),
    "van_der_pol"                    => (-4.0, 4.0, -4.0, 4.0),
    "linear_stochastically_switched" => (-2.0, 2.0, -2.0, 2.0),
    "package_delivery"               => (-6.0, 6.0, -6.0, 6.0),
)

"""
    GridGeometry

How a flat gap vector maps onto a 2-D picture.

- `shape`       : `(n1, n2)` cells; state `s` is cell `(i, j)` with `s = (j-1)*n1 + i`.
- `extra`       : trailing states that are NOT cells (the absorbing "outside" state).
- `extent`      : `(x1lo, x1hi, x2lo, x2hi)` physical bounds, or `nothing` for cell indices.
- `provenance`  : `:name_split` (read off the `_S<n>` token) or `:perfect_square` (inferred).
"""
struct GridGeometry
    shape::Tuple{Int,Int}
    extra::Int
    extent::Union{Nothing,NTuple{4,Float64}}
    provenance::Symbol
end

ncells(g::GridGeometry) = g.shape[1] * g.shape[2]

# Longest-prefix lookup, so `car_parking_S20_I3` finds the `car_parking` entry and a
# more specific future entry wins over a more general one.
function _grid_family(instance_name::AbstractString)
    best = nothing; blen = 0
    for k in keys(_STATE_SPACE_EXTENT)
        startswith(instance_name, k) && length(k) > blen && (best = k; blen = length(k))
    end
    return best
end

_is_grid_family(instance_name::AbstractString) = _grid_family(instance_name) !== nothing

# Physical bounds of the family's state space, or `nothing` for cell-index axes.
function _extent_for(instance_name::AbstractString)
    fam = _grid_family(instance_name)
    return fam === nothing ? nothing : _STATE_SPACE_EXTENT[fam]
end

"""
    grid_geometry(instance_name, num_states) -> (GridGeometry, "") | (nothing, reason)

Decide whether `num_states` states laid out by `instance_name` form a 2-D grid, and if
so with what shape.

Being a grid is a property of how the problem was BUILT — `StateUniformGridSplit` — not
of its state count, so the name has to vouch for it: `instance_name` must belong to a
known grid family (a `_STATE_SPACE_EXTENT` prefix). A state count alone is never enough.
`robot_delta0.1`, a PRISM import with no spatial structure whatsoever, has a state count
that is a perfect square; reshaping it would put unrelated states side by side and read
as noise, which is the failure this guard exists to prevent.

Within a grid family the `_S<n>` token is authoritative — it is the state split the
generator was invoked with — and a mismatch against `num_states` is reported rather than
papered over, because it means the trace and the name disagree about the model. A family
member without that token falls back to an exact perfect square (with or without the one
trailing absorbing state), flagged `:perfect_square` so the figure can say the shape was
inferred.
"""
function grid_geometry(instance_name::AbstractString, num_states::Integer)
    ext = _extent_for(instance_name)
    known = _is_grid_family(instance_name)
    m = match(r"_S(\d+)", String(instance_name))
    if known && m !== nothing
        n = parse(Int, m.captures[1])
        n^2 == num_states - 1 && return GridGeometry((n, n), 1, ext, :name_split), ""
        n^2 == num_states     && return GridGeometry((n, n), 0, ext, :name_split), ""
        return nothing, "name declares a $(n)×$(n) split (expects $(n^2) or $(n^2 + 1) states) " *
                        "but the trace has $num_states"
    end
    if known
        for extra in (1, 0)
            n = isqrt(num_states - extra)
            n > 1 && n^2 == num_states - extra &&
                return GridGeometry((n, n), extra, ext, :perfect_square), ""
        end
        return nothing, "$num_states states: a $(instance_name) grid, but the name carries no " *
                        "_S<n> token and neither $num_states nor $(num_states - 1) is a perfect square"
    end
    return nothing, "$num_states states, no grid geometry — '$instance_name' is not a known " *
                    "uniform-grid family ($(join(sort(collect(keys(_STATE_SPACE_EXTENT))), ", ")))"
end

# --- streaming trace reader ------------------------------------------------

"""
    _read_matrix_head(path, max_rows) -> (wallclock, bellman_updates, frames, num_states)

Read at most `max_rows` data rows from any wide trace CSV — `gap_trace.csv`,
`value_trace_lower.csv`, `update_trace.csv` — all of which share the layout
`wallclock_s,bellman_updates,<prefix>[1],…,<prefix>[n]` (see `../trace.jl`,
`_write_matrix_csv`). `frames[k]` is the full per-state vector of row `k`, whatever
quantity the file holds.

Streaming is not an optimization here, it is the only viable strategy: these files reach
6 GB, and the video only ever shows the first `max_rows` iterations. A short final row —
what a SIGKILL mid-write leaves — is dropped, matching how `adapter.jl` treats a
truncated trace.
"""
function _read_matrix_head(path::AbstractString, max_rows::Integer)
    wall = Float64[]; bu = Int[]; frames = Vector{Float64}[]
    ncol = 0
    open(path, "r") do io
        header = readline(io)
        isempty(header) && return
        ncol = count(==(','), header) + 1
        ncol > 3 || return                      # wallclock + bellman_updates + ≥2 states
        for line in eachline(io)
            length(frames) >= max_rows && break
            fields = split(line, ',')
            length(fields) == ncol || continue   # truncated row
            w = tryparse(Float64, fields[1]); b = tryparse(Float64, fields[2])
            (w === nothing || b === nothing || !isfinite(b)) && continue
            g = Vector{Float64}(undef, ncol - 2)
            ok = true
            for k in 3:ncol
                v = tryparse(Float64, fields[k])
                v === nothing && (ok = false; break)
                g[k-2] = v
            end
            ok || continue
            push!(wall, w); push!(bu, round(Int, b)); push!(frames, g)
        end
    end
    return wall, bu, frames, max(ncol - 2, 0)
end

# --- panels ----------------------------------------------------------------
#
# A video shows up to four views of the SAME state space at the SAME iteration: the two
# bounds the solver drives, their difference, and where the sampler actually looked.
# Each comes from its own trace CSV, written by the same `TraceExecutor`, so row `i` of
# every file is the same iteration.
#
# Which panels a cell can show depends on which traces the sweep was configured to
# record (`[output].traces`). Nothing is required: a results dir holding only
# `gap_trace.csv` — every dir written before the `"value"`/`"updates"` kinds were enabled
# — renders the single gap panel exactly as it always did.

# Draw order, and the trace file each panel reads.
const _PANEL_ORDER = (:lower, :upper, :gap, :updates)
const _PANEL_FILE = (lower   = "value_trace_lower.csv",
                     upper   = "value_trace_upper.csv",
                     gap     = "gap_trace.csv",
                     updates = "update_trace.csv")

const _PANEL_TITLE = (lower   = "lower bound  L",
                      upper   = "upper bound  U",
                      gap     = "gap  U − L",
                      updates = "Bellman backups this iteration")

const _PANEL_CBLABEL = (lower   = "L",
                        upper   = "U",
                        gap     = "upper − lower",
                        updates = "backups")

# Colour for a state that received NO backup this iteration. It has to sit clearly
# outside the updates colormap's gamut at BOTH ends, because the common case is a
# sampler whose every pick has a count of exactly 1: those states land on the bottom of
# the ramp, and against a dark background they were invisible — the panel read as "the
# sampler never ran". A mid-tone neutral separates "untouched" from both the dark low end
# and the bright high end.
const _NO_UPDATE_COLOR = RGBAf(0.55, 0.56, 0.60, 1)

"""
    _load_panels(files, max_frames; want=_PANEL_ORDER)
        -> (; wall, bellman_updates, panels, nstates, nframes, derived_gap)

Read every requested panel that has a file, aligned to a common frame count.

`files` maps a panel name to its CSV path (or `nothing`). `panels` comes back as an
ordered `Vector{Pair{Symbol,Vector{Vector{Float64}}}}` in `_PANEL_ORDER`.

Two alignment rules matter:

- The files are written per iteration by one executor, so row `i` is the same iteration
  in each. A SIGKILL can still leave one file a row shorter than another, so every panel
  is truncated to the shortest — which is also what makes the `bellman_updates` columns
  agree, and that agreement is asserted rather than assumed.
- `gap` is DERIVED as `U − L` when the bounds are present but `gap_trace.csv` is not.
  The gap is definitionally `|L − U|`, so a dir that recorded `"value"` but not `"gap"`
  loses nothing; `derived_gap` says whether that happened, for the caption.

Returns `nothing` when no requested panel yielded a usable row.
"""
function _load_panels(files, max_frames::Integer; want=_PANEL_ORDER)
    raw = Dict{Symbol,Any}()
    for name in want
        path = get(files, name, nothing)
        (path === nothing || !isfile(path)) && continue
        w, b, frames, nst = _read_matrix_head(path, max_frames)
        isempty(frames) && continue
        raw[name] = (wall=w, bu=b, frames=frames, nstates=nst)
    end

    derived_gap = false
    if :gap in want && !haskey(raw, :gap) && haskey(raw, :lower) && haskey(raw, :upper)
        l, u = raw[:lower], raw[:upper]
        n = min(length(l.frames), length(u.frames))
        raw[:gap] = (wall=l.wall[1:n], bu=l.bu[1:n],
                     frames=[abs.(u.frames[i] .- l.frames[i]) for i in 1:n],
                     nstates=min(l.nstates, u.nstates))
        derived_gap = true
    end

    isempty(raw) && return nothing

    nframes = minimum(length(r.frames) for r in values(raw))
    nframes == 0 && return nothing
    nstates = minimum(r.nstates for r in values(raw))

    # Take the clock from a panel that was actually read, never from the derived gap.
    ref = :gap
    for k in _PANEL_ORDER
        haskey(raw, k) && !(k === :gap && derived_gap) && (ref = k; break)
    end
    wall, bu = raw[ref].wall[1:nframes], raw[ref].bu[1:nframes]

    # The executor stamps one `bellman_updates` per iteration and every trace records it,
    # so a disagreement means these files did not come from the same run.
    for (name, r) in raw
        (name === :gap && derived_gap) && continue
        r.bu[1:nframes] == bu ||
            @warn "gap_video: trace files disagree on bellman_updates; frames may be misaligned" panel=name
    end

    panels = Pair{Symbol,Vector{Vector{Float64}}}[]
    for name in _PANEL_ORDER
        haskey(raw, name) || continue
        push!(panels, name => raw[name].frames[1:nframes])
    end
    return (; wall, bellman_updates=bu, panels, nstates, nframes, derived_gap)
end

# --- reach / avoid / initial overlay ----------------------------------------
#
# Three sets are outlined on every frame, each from the source that actually defines it:
#
#   reach, avoid  — the property, from the stored spec JSON (`<instance>.json`)
#   initial (S0)  — the MODEL, not the property. S0 decides what `restrict_to_initial`
#                   converges on, so it is the set a reader most needs to locate, and it
#                   is stored in the `.nc` rather than the spec.
#
# S0 is looked up in `problems.csv` first, where the harness records it (problem_stats.jl)
# — a results dir is then self-contained. Older dirs predate that column, so the fallback
# reads the `initial_states` variable straight out of the netCDF. That costs ~7 ms even on
# a 358 MB model: netCDF is random-access, so nothing but that one small array is read.

# Flat state index -> grid cell, for the masks. `s = (j-1)*n1 + i`; off-grid indices (the
# absorbing state) are dropped.
function _mask_from_indices(indices, geom::GridGeometry)
    n1, n2 = geom.shape
    m = falses(n1, n2)
    for s in indices
        1 <= s <= n1 * n2 || continue
        m[(s-1)%n1+1, (s-1)÷n1+1] = true
    end
    return m
end

# Reach/avoid state indices from a problem's stored spec JSON. The spec lists each state
# as a one-element array (flat index) for a flat IMDP; anything else is left undrawn
# rather than guessed at. The absorbing state is always in `avoid` and is off-grid, so it
# drops out naturally.
function _reach_avoid_masks(spec_path, geom::GridGeometry)
    n1, n2 = geom.shape
    empty = (reach=falses(n1, n2), avoid=falses(n1, n2))
    (spec_path === nothing || !isfile(spec_path)) && return empty
    prop = try
        get(JSON.parsefile(String(spec_path)), "property", nothing)
    catch e
        @warn "gap_video: unreadable spec JSON; drawing without the reach/avoid overlay" spec_path exception=e
        return empty
    end
    prop isa AbstractDict || return empty
    function mask(key)
        entries = get(prop, key, nothing)
        entries isa AbstractVector || return falses(n1, n2)
        return _mask_from_indices((Int(e[1]) for e in entries
                                   if e isa AbstractVector && length(e) == 1), geom)
    end
    return (reach=mask("reach"), avoid=mask("avoid"))
end

# S0 from the `initial_states` column of problems.csv, or `nothing` when the file, the
# row or the column is absent (a dir written before the column existed). An empty string
# is a real answer — the model declares `AllStates()`, so there is no set to outline —
# and returns `Int[]`, which stops the netCDF fallback from running pointlessly. The
# column is read with `string`, not `String`: when every problem has a single initial
# state CSV.jl types it `Int64`, not `String`.
function _initial_states_from_csv(results_dir::AbstractString, instance::AbstractString)
    path = joinpath(results_dir, "problems.csv")
    isfile(path) || return nothing
    try
        df = CSV.read(path, DataFrame)
        ("problem" in names(df) && "initial_states" in names(df)) || return nothing
        for r in eachrow(df)
            String(coalesce(r.problem, "")) == instance || continue
            ismissing(r.initial_states) && return nothing
            return [parse(Int, t) for t in split(string(r.initial_states)) if !isempty(t)]
        end
    catch e
        @warn "gap_video: unreadable problems.csv; falling back to the model file" exception=e
    end
    return nothing
end

# S0 straight from the model's netCDF `initial_states` variable. A zero-length array is
# how `AllStates()` is stored (IntervalMDP `src/Data/intervalmdp.jl`), i.e. "no particular
# initial set", which is drawn as no outline.
function _initial_states_from_model(model_path)
    (model_path === nothing || !isfile(model_path)) && return Int[]
    try
        return NCDataset(String(model_path), "r") do ds
            haskey(ds, "initial_states") ? Int.(ds["initial_states"][:]) : Int[]
        end
    catch e
        @warn "gap_video: could not read initial_states; drawing without the S0 outline" model_path exception=e
        return Int[]
    end
end

"""
    overlay_masks(results_dir, instance_name, geom; problems_root=nothing)

The `(; reach, avoid, initial)` boolean grids outlined on every frame of `instance_name`'s
video. Each is all-`false` when its source is unavailable, so a missing problems tree
costs outlines and nothing else.
"""
function overlay_masks(results_dir::AbstractString, instance_name::AbstractString,
                       geom::GridGeometry;
                       problems_root::Union{Nothing,AbstractString}=nothing)
    ra = _reach_avoid_masks(spec_path_for(results_dir, instance_name; problems_root), geom)
    s0 = _initial_states_from_csv(results_dir, instance_name)
    s0 === nothing &&
        (s0 = _initial_states_from_model(model_path_for(results_dir, instance_name; problems_root)))
    return (reach=ra.reach, avoid=ra.avoid, initial=_mask_from_indices(s0, geom))
end

# Locate one of a problem's stored files. `key` is the `[[problems]].params` entry the
# sweep recorded (an absolute path, authoritative when it still resolves) and `ext` the
# extension to look for under `problems_root` otherwise — a sweep run on the cluster
# recorded `/scratch/...` paths that do not exist on the machine doing the analysis.
function _problem_file_for(results_dir, instance_name, key, ext, problems_root)
    for p in get(_resolved_config(results_dir), "problems", [])
        p isa AbstractDict || continue
        String(get(p, "instance_name", "")) == instance_name || continue
        fp = get(get(p, "params", Dict()), key, nothing)
        fp === nothing || (isfile(String(fp)) && return String(fp))
    end
    root = problems_root === nothing ?
           normpath(joinpath(@__DIR__, "..", "..", "problems")) : String(problems_root)
    local_path = joinpath(root, instance_name, instance_name * ext)
    return isfile(local_path) ? local_path : nothing
end

"""
    spec_path_for(results_dir, instance_name; problems_root=nothing) -> String | nothing

Locate the stored spec JSON for `instance_name` (reach/avoid sets). `nothing` when it
resolves nowhere, in which case the video is drawn without those outlines.
"""
spec_path_for(results_dir::AbstractString, instance_name::AbstractString;
              problems_root::Union{Nothing,AbstractString}=nothing) =
    _problem_file_for(results_dir, instance_name, "spec", ".json", problems_root)

"""
    model_path_for(results_dir, instance_name; problems_root=nothing) -> String | nothing

Locate the stored netCDF model for `instance_name` (the initial states live here, not in
the spec). `nothing` when it resolves nowhere.
"""
model_path_for(results_dir::AbstractString, instance_name::AbstractString;
               problems_root::Union{Nothing,AbstractString}=nothing) =
    _problem_file_for(results_dir, instance_name, "model", ".nc", problems_root)

# --- rendering -------------------------------------------------------------

# Outline the true cells of `mask` on the axis.
#
# The mask is padded with a ring of `false` before contouring, and the coordinate vectors
# extended by one cell to match. Without the ring, a region touching the edge of the state
# space has no transition there for `contour` to trace and its outline comes back open:
# car_parking's S0 is a 2×2 block in the corner of X, so two of its four sides went
# missing and it read as a stray arc rather than a highlighted region. The padding puts
# the closing contour exactly on the outer cell boundary, since it falls halfway between
# the real edge cell's centre and the phantom one's.
function _outline!(ax, xc, yc, mask::AbstractMatrix{Bool}; color, linestyle=:solid,
                   linewidth=2)
    dx = length(xc) > 1 ? xc[2] - xc[1] : one(eltype(xc))
    dy = length(yc) > 1 ? yc[2] - yc[1] : one(eltype(yc))
    xp = vcat(xc[1] - dx, collect(xc), xc[end] + dx)
    yp = vcat(yc[1] - dy, collect(yc), yc[end] + dy)
    padded = zeros(Float64, length(xp), length(yp))
    padded[2:end-1, 2:end-1] .= mask
    return contour!(ax, xp, yp, padded; levels=[0.5], color=color,
                    linewidth=linewidth, linestyle=linestyle)
end

"""
    render_state_video(files, out_path, geom; kwargs...) -> NamedTuple

Render one run's trace CSVs as an mp4: up to four heatmaps of the same 2-D state space,
side by side, advancing together one frame per recorded Bellman iteration.

`files` maps a panel name (`:lower`, `:upper`, `:gap`, `:updates`) to its CSV path;
absent or unreadable files simply drop their panel, so a results dir that recorded only
`gap_trace.csv` yields the one-panel gap video it always did.

Colour scales are LINEAR and, for the three value panels, FIXED across every frame of
every video, so a colour means the same thing across iterations, samples and algorithms.
That fixed scale is also why only the first `max_frames` iterations are shown: once the
gap has collapsed the picture is uniformly dark and carries nothing, and the early
iterations are where the spatial propagation happens.

The `updates` panel is the exception and is scaled PER VIDEO, because the number of
backups a state receives is bounded by the sampler's `k` rather than by anything
intrinsic — a fixed scale would render every sampled variant as a single flat colour.
States with no backup that iteration are drawn as background rather than as the bottom
of the ramp, so the sampler's focus reads at a glance. A full sweep therefore shows a
uniform all-ones panel; that is a correct picture of a sweep, not a bug.

Keyword arguments:
- `max_frames`   : iterations to render, counted from the first (default 600).
- `framerate`    : frames per second (default 15).
- `colorrange`   : fixed linear limits for the gap panel (default `(0.0, 1.0)`).
- `value_range`  : fixed linear limits for the two bound panels (default `(0.0, 1.0)`).
- `update_range` : limits for the updates panel; `nothing` scales it per video.
- `colormap`     : Makie colormap for the value panels (default `:viridis`).
- `panels`       : which panels to draw, in `_PANEL_ORDER` (default: all with data).
- `title`        : headline drawn above the panel grid.
- `masks`        : `(; reach, avoid, initial)` boolean grids to outline, or `nothing`.
- `total_iterations` : the run's full iteration count, for the "first N of M" caption.

Returns `(; path, frames, states, truncated, panels)`, or `nothing` when no panel held a
usable row.
"""
function render_state_video(files, out_path::AbstractString, geom::GridGeometry;
                            max_frames::Integer=600, framerate::Integer=15,
                            colorrange::Tuple{<:Real,<:Real}=(0.0, 1.0),
                            value_range::Tuple{<:Real,<:Real}=(0.0, 1.0),
                            update_range::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
                            colormap=:viridis,
                            panels=_PANEL_ORDER,
                            title::AbstractString="",
                            masks=nothing,
                            total_iterations::Union{Nothing,Integer}=nothing)
    loaded = _load_panels(files, max_frames; want=panels)
    loaded === nothing && return nothing
    nframes, nstates = loaded.nframes, loaded.nstates
    wall, bu = loaded.wall, loaded.bellman_updates

    n1, n2 = geom.shape
    cells = n1 * n2
    nstates >= cells ||
        error("gap_video: trace has $nstates states but the geometry needs $cells")

    # Axis coordinates. Makie's heatmap takes n+1 edges to draw exactly n cells; the
    # contour overlay takes the n cell centres, on the same grid.
    if geom.extent === nothing
        xe, ye = collect(0.5:1.0:(n1 + 0.5)), collect(0.5:1.0:(n2 + 0.5))
        xlab, ylab = "cell index (dim 1)", "cell index (dim 2)"
    else
        x0, x1, y0, y1 = geom.extent
        xe, ye = collect(range(x0, x1; length=n1 + 1)), collect(range(y0, y1; length=n2 + 1))
        xlab, ylab = "x₁", "x₂"
    end
    xc = (xe[1:end-1] .+ xe[2:end]) ./ 2
    yc = (ye[1:end-1] .+ ye[2:end]) ./ 2

    truncated = total_iterations !== nothing && total_iterations > nframes

    # A state with zero backups is drawn as background, not as the bottom of the ramp:
    # on a sampled run almost every state is zero every iteration, and mapping them onto
    # the colormap would bury the handful that were actually touched.
    _reshape(v) = collect(reshape(view(v, 1:cells), n1, n2))
    function _reshape_updates(v)
        m = _reshape(v)
        return map(x -> x <= 0 ? NaN : x, m)
    end

    npanels = length(loaded.panels)
    # One column pair (axis, colorbar) per panel, wrapped to two columns of panels. A
    # single panel keeps the original 760×720 figure so existing videos are unchanged.
    ncols = npanels == 1 ? 1 : 2
    nrows = cld(npanels, ncols)
    fig = Figure(size=(ncols == 1 ? 760 : 1320, 120 + nrows * (npanels == 1 ? 600 : 560)))
    Label(fig[0, 1:(2 * ncols)], title; fontsize=14, font=:bold, tellwidth=false)

    # Per-panel state: the Observable the record loop writes, and how to reshape into it.
    updaters = Tuple{Observable,Function,Vector{Vector{Float64}}}[]
    for (idx, (name, frames)) in enumerate(loaded.panels)
        r, c = fldmod1(idx, ncols)
        # Limits are pinned to the state space rather than autoscaled: `_outline!` pads
        # its contour grid by one cell beyond the edge, and letting that pad widen the
        # axis would float the heatmap inside a margin of blank space.
        ax = Axis(fig[r, 2c - 1]; title=_PANEL_TITLE[name], titlesize=12,
                  xlabel=xlab, ylabel=ylab, aspect=DataAspect(),
                  limits=(xe[1], xe[end], ye[1], ye[end]))

        shaper = name === :updates ? _reshape_updates : _reshape
        obs = Observable(shaper(frames[1]))
        cblabel = _PANEL_CBLABEL[name]
        if name === :updates
            # Observed maximum over the frames actually rendered. A sampler that never
            # draws a state twice has a maximum of 1, so the ramp is widened to keep the
            # range non-degenerate; the label reports the real maximum so a flat panel is
            # not mistaken for a clipped one.
            obs_max = maximum(maximum, frames)
            lo = update_range === nothing ? 1.0 : Float64(update_range[1])
            hi = update_range === nothing ? max(2.0, obs_max) : Float64(update_range[2])
            hm = heatmap!(ax, xe, ye, obs; colormap=:plasma,
                          colorrange=(lo, max(hi, lo + 1)),
                          nan_color=_NO_UPDATE_COLOR)
            cblabel *= @sprintf("  (max %d this video;\ngrey = no backup)", round(Int, obs_max))
        else
            lim = name === :gap ? colorrange : value_range
            hm = heatmap!(ax, xe, ye, obs; colormap=colormap,
                          colorrange=(Float64(lim[1]), Float64(lim[2])))
        end
        Colorbar(fig[r, 2c], hm; label=cblabel, labelsize=10)
        push!(updaters, (obs, shaper, frames))

        # Outlines, drawn once per panel (they never change across frames). Initial
        # states are dashed and in a colour that reads against the whole viridis ramp,
        # because S0 is often a handful of cells in a corner and has to stay findable
        # next to the much larger reach/avoid blocks.
        masks === nothing && continue
        for (m, color, style) in ((masks.reach,   :white,   :solid),
                                  (masks.avoid,   :red,     :solid),
                                  (masks.initial, :magenta, :dash))
            any(m) || continue
            _outline!(ax, xc, yc, m; color=color, linestyle=style)
        end
    end

    legend = String[]
    if masks !== nothing
        any(masks.reach)   && push!(legend, "white = reach")
        any(masks.avoid)   && push!(legend, "red = avoid")
        any(masks.initial) && push!(legend, "magenta dashed = initial (S0)")
    end

    caption = Observable("")
    Label(fig[nrows + 1, 1:(2 * ncols)], caption; fontsize=11, color=:gray25, halign=:left,
          tellwidth=false, justification=:left)
    static = string(
        geom.extra > 0 ? "$(geom.extra) off-grid absorbing state not shown.  " : "",
        isempty(legend) ? "" : join(legend, ", ") * ".  ",
        truncated ? "First $(nframes) of $(total_iterations) iterations." :
                    "$(nframes) iterations (complete).",
        geom.provenance === :perfect_square ? "  Grid shape inferred from the state count." : "",
        loaded.derived_gap ? "  Gap derived as U − L (no gap_trace.csv)." : "",
    )
    _note = _FIG_NOTE[]

    record(fig, out_path, 1:nframes; framerate=framerate) do k
        for (obs, shaper, frames) in updaters
            obs[] = shaper(frames[k])
        end
        caption[] = @sprintf("iteration %d / %d   ·   %d Bellman updates   ·   t = %.3f s\n%s%s",
                             k, nframes, bu[k], wall[k], static,
                             isempty(_note) ? "" : "\n" * _note)
    end
    return (; path=out_path, frames=nframes, states=nstates, truncated,
            panels=first.(loaded.panels))
end

# --- sweep-wide driver -----------------------------------------------------

# One `runs/<cell_id>/sample<k>/` found on disk, with whichever panel trace files it
# holds. Discovery walks the runs tree rather than results.csv so a sweep that was killed
# before writing its final rows still yields every trace it managed to produce;
# results.csv is then consulted only to enrich the caption.
#
# `files` maps each panel name to a path or `nothing`. A sample dir contributes an entry
# as soon as ONE panel file is there, which is what lets a dir recorded before the
# `"value"`/`"updates"` trace kinds existed keep rendering its gap panel.
struct _GapTraceEntry
    csv::String                      # the first available panel file; kept for geometry
    files::NamedTuple
    cell_id::String
    instance::String
    algorithm::String
    sample::Int
end

# Back-compat positional form (no per-panel files): used by tests and by any caller that
# only cares about the identifying fields.
_GapTraceEntry(csv, cell_id, instance, algorithm, sample) =
    _GapTraceEntry(csv, (; gap=csv), cell_id, instance, algorithm, sample)

function _discover_gap_traces(results_dir::AbstractString)
    out = _GapTraceEntry[]
    runs = joinpath(results_dir, "runs")
    isdir(runs) || return out
    for cell in sort(readdir(runs))
        celldir = joinpath(runs, cell)
        isdir(celldir) || continue
        # cell_id is "<instance_name>__<alg_label>"; instance names contain no "__".
        parts = split(cell, "__"; limit=2)
        instance = String(parts[1])
        algorithm = length(parts) == 2 ? String(parts[2]) : ""
        for sample in readdir(celldir)
            sdir = joinpath(celldir, sample)
            isdir(sdir) || continue
            paths = map(_PANEL_ORDER) do panel
                f = joinpath(sdir, _PANEL_FILE[panel])
                isfile(f) ? f : nothing
            end
            files = NamedTuple{_PANEL_ORDER}(paths)
            present = filter(!isnothing, collect(paths))
            isempty(present) && continue
            m = match(r"(\d+)$", sample)
            push!(out, _GapTraceEntry(first(present), files, cell, instance, algorithm,
                                      m === nothing ? 0 : parse(Int, m.captures[1])))
        end
    end
    # Sorted by sample NUMBER, not by directory name: `readdir` is lexicographic, which
    # would order sample10 before sample2 and make "the first sample" depend on how many
    # samples the sweep happened to run.
    return sort!(out; by=e -> (e.cell_id, e.sample))
end

"""
    _select_one_per_cell(entries, sample) -> Vector{_GapTraceEntry}

Keep one trace per `(problem, algorithm)` cell. The samples of a cell differ only in
their random seed, so rendering all of them produces near-identical videos at N× the
cost; one is what the spatial view is actually read for.

`sample === nothing` takes the lowest-numbered sample present, which is stable across
re-runs and degrades gracefully when sample 1 was never written (a cell killed part-way
still contributes whichever sample it did produce). An explicit `sample` takes that index
and nothing else — a cell lacking it contributes no video rather than silently
substituting a different seed, which would make two videos in the same batch
incomparable.
"""
function _select_one_per_cell(entries::Vector{_GapTraceEntry}, sample::Union{Nothing,Integer})
    sample !== nothing && return filter(e -> e.sample == Int(sample), entries)
    best = Dict{String,_GapTraceEntry}()
    for e in entries
        (!haskey(best, e.cell_id) || e.sample < best[e.cell_id].sample) && (best[e.cell_id] = e)
    end
    return sort!(collect(values(best)); by=e -> (e.cell_id, e.sample))
end

# `cell_id => num_iterations` from results.csv, for the "first N of M" caption. Absent
# or unreadable results.csv simply means no totals — never an error.
function _iteration_totals(results_dir::AbstractString)
    totals = Dict{Tuple{String,Int},Int}()
    csv = joinpath(results_dir, "results.csv")
    isfile(csv) || return totals
    try
        df = CSV.read(csv, DataFrame)
        cols = names(df)
        ("cell_id" in cols && "sample_index" in cols && "num_iterations" in cols) || return totals
        for r in eachrow(df)
            (ismissing(r.cell_id) || ismissing(r.sample_index) || ismissing(r.num_iterations)) && continue
            r.num_iterations isa Real && isfinite(r.num_iterations) || continue
            totals[(String(r.cell_id), Int(r.sample_index))] = round(Int, r.num_iterations)
        end
    catch e
        @warn "gap_video: unreadable results.csv; captions will omit iteration totals" exception=e
    end
    return totals
end

"""
    render_gap_videos(results_dir; kwargs...) -> NamedTuple

Render one mp4 per `(problem, algorithm)` cell from the trace CSVs under
`<results_dir>/runs/`, into `<results_dir>/plots/gap_videos/<cell_id>__sample<k>.mp4`.

Each video shows up to four synchronized views of the state space — lower bound, upper
bound, gap, and the per-iteration Bellman-backup count — sourced from
`value_trace_lower.csv`, `value_trace_upper.csv`, `gap_trace.csv` and
`update_trace.csv` respectively. A panel whose file the sweep did not record is simply
not drawn, so a dir holding only `gap_trace.csv` renders the one-panel gap video it
always did. See [`render_state_video`](@ref) and `_load_panels`.

ONE sample per cell is drawn, not all of them — see [`_select_one_per_cell`](@ref). The
filename still carries the sample index, so which seed a video shows is never in doubt.

Only the video is kept — no per-frame images are written.

Problems with no grid geometry are skipped and the reason recorded; see
`grid_geometry`. The returned `log` is a human-readable account of every trace
considered, which `write_manifest` folds into `manifest.txt`.

Keyword arguments:
- `max_frames` / `framerate` / `colorrange` / `colormap` / `value_range` /
  `update_range` : passed to [`render_state_video`](@ref).
- `panels`        : which panels to draw (default: all four, minus those with no data).
- `problems`      : restrict to these instance names (default: all).
- `algorithms`    : restrict to these algorithm labels (default: all).
- `sample`        : which sample index to draw (default: the lowest present in each cell).
- `overwrite`     : re-render a video that already exists (default `false`, so a
                    re-analysis after a crash resumes instead of redoing hours of work).
- `problems_root` : where to look for spec JSONs when the recorded path is unreachable.

Returns `(; rendered, skipped, failed, dir, log)`.
"""
function render_gap_videos(results_dir::AbstractString;
                           max_frames::Integer=600, framerate::Integer=15,
                           colorrange::Tuple{<:Real,<:Real}=(0.0, 1.0),
                           value_range::Tuple{<:Real,<:Real}=(0.0, 1.0),
                           update_range::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
                           colormap=:viridis,
                           panels=_PANEL_ORDER,
                           problems::Union{Nothing,Vector{String}}=nothing,
                           algorithms::Union{Nothing,Vector{String}}=nothing,
                           sample::Union{Nothing,Integer}=nothing,
                           overwrite::Bool=false,
                           problems_root::Union{Nothing,AbstractString}=nothing)
    outdir = joinpath(results_dir, "plots", "gap_videos")
    # A video outlives the results dir it came from just as a figure does, so a PARTIAL
    # sweep has to say so on the frames themselves. `run_analyses` has already set this;
    # recomputing costs one meta.json read and makes the standalone driver honest too.
    try
        _FIG_NOTE[] = figure_note(load_completeness(results_dir))
    catch
        # provenance is a nicety; never lose the videos over it
    end
    entries = _discover_gap_traces(results_dir)
    log = String[]
    if isempty(entries)
        push!(log, "no trace CSV found under $(joinpath(results_dir, "runs")) " *
                   "([output].traces must include at least one of " *
                   "\"gap\", \"value\", \"updates\")")
        @info "gap videos: nothing to render" results_dir
        return (; rendered=0, skipped=0, failed=0, dir=outdir, log)
    end

    # One video per cell. `n_samples` is reported because the difference between "this
    # sweep ran one sample" and "this sweep ran five and you are looking at one of them"
    # changes how a reader should weigh the picture.
    n_samples = length(entries)
    entries = _select_one_per_cell(entries, sample)
    if isempty(entries)
        push!(log, "no sample $(sample) found in any cell (of $n_samples trace(s) on disk)")
        @warn "gap videos: requested sample not present" results_dir sample
        return (; rendered=0, skipped=0, failed=0, dir=outdir, log)
    end
    push!(log, "one sample per (problem, algorithm) cell: $(length(entries)) of " *
               "$n_samples trace(s) on disk" *
               (sample === nothing ? " (lowest sample index in each cell)" : " (sample $sample)"))

    mkpath(outdir)
    totals = _iteration_totals(results_dir)

    # Geometry and the reach/avoid overlay are per-problem, and the overlay costs a JSON
    # parse, so both are resolved once per instance and reused across its samples.
    geoms = Dict{String,Any}()
    maskcache = Dict{String,Any}()

    rendered = skipped = failed = 0
    for e in entries
        problems   === nothing || e.instance  in problems   || continue
        algorithms === nothing || e.algorithm in algorithms || continue
        out = joinpath(outdir, "$(e.cell_id)__sample$(e.sample).mp4")
        if !overwrite && isfile(out)
            skipped += 1
            push!(log, "kept     $(basename(out))  (already rendered; pass overwrite=true to redo)")
            continue
        end

        if !haskey(geoms, e.instance)
            # The trace itself declares the state count (one column per state), so
            # geometry never depends on problems.csv being present.
            nstates = open(io -> (h = readline(io); isempty(h) ? 0 : count(==(','), h) - 1), e.csv)
            geoms[e.instance] = grid_geometry(e.instance, nstates)
        end
        geom, reason = geoms[e.instance]
        if geom === nothing
            skipped += 1
            push!(log, "skipped  $(e.cell_id) sample$(e.sample): $reason")
            continue
        end
        if !haskey(maskcache, e.instance)
            maskcache[e.instance] = overlay_masks(results_dir, e.instance, geom; problems_root)
        end

        title = "$(e.instance)  ·  $(e.algorithm)  ·  sample $(e.sample)"
        try
            r = render_state_video(e.files, out, geom; max_frames, framerate, colorrange,
                                   value_range, update_range, colormap, panels, title,
                                   masks=maskcache[e.instance],
                                   total_iterations=get(totals, (e.cell_id, e.sample), nothing))
            if r === nothing
                failed += 1
                push!(log, "FAILED   $(e.cell_id) sample$(e.sample): no panel holds a usable row")
            else
                rendered += 1
                push!(log, "rendered $(basename(out))  " *
                           "($(r.frames) frames, $(geom.shape[1])×$(geom.shape[2]) grid, " *
                           "panels: $(join(r.panels, '+'))" *
                           (r.truncated ? ", truncated" : "") * ")")
                @info "gap video" out frames=r.frames
            end
        catch err
            failed += 1
            push!(log, "FAILED   $(e.cell_id) sample$(e.sample): $(sprint(showerror, err))")
            @warn "gap video failed" csv=e.csv exception=(err, catch_backtrace())
        end
    end

    @info "gap videos written" dir=outdir rendered skipped failed
    return (; rendered, skipped, failed, dir=outdir, log)
end
