# Per-problem policy grids: the control each solver variant actually settled on, drawn
# over the state space, one panel per (algorithm, sample).
#
# The videos next door (gap_video.jl) show WHERE a sampled variant is still uncertain.
# This file answers the other half of the question — what it DECIDED — and the two are
# read together, which is why the geometry, the reach/avoid/S0 overlays and the
# "untouched" grey are shared with that file rather than reimplemented here.
#
# Source. `runs/<cell_id>/sample<k>/policy.csv`, written by `write_policy_csv`
# (../trace.jl) from the `StationaryStrategy` a `ControlSynthesisSolution` carries. That
# file is new: a results dir produced before `[output].save_policy` existed holds none,
# and is skipped with a recorded reason rather than half-drawn.
#
# The sentinel is the point. GSRDP initialises its strategy array to action 0 and only
# overwrites a state it actually backed up, so on a sampled variant every state the
# sampler never reached still reads 0. Those are drawn in the same neutral grey the
# updates panel uses for "no backup", so a row of the grid reads as coverage at a glance:
# the full sweep is solid colour, a sampler's panel is colour where it looked and grey
# where it did not.
#
# Every sample is drawn, unlike the videos which pick one per cell. A video is minutes of
# rendering off a multi-gigabyte trace; a policy is one small CSV, and the across-seed
# spread within one algorithm's row is most of what the figure is for.

# Colour scales. Actions are a NOMINAL scale — action 7 is not "between" 6 and 8, it is a
# different control input — so the palette has to be categorical. Glasbey is a maximally
# distinct sequence and covers the 25 actions of `robot_2d_S12_I5` without repeats;
# `:tab20` is the fallback for a ColorSchemes that does not carry it.
const _POLICY_PALETTES = (:glasbey_bw_minc_20_n256, :tab20, :viridis)

# Above this many actions, labelling every tick turns the colorbar into a grey smear.
const _POLICY_TICK_LIMIT = 12

# Panel edge in points. Small enough that a 3x5 grid fits on a page, large enough that a
# 30x30 state grid still resolves individual cells.
const _POLICY_PANEL = 250

function _policy_colormap(n::Integer)
    for name in _POLICY_PALETTES
        try
            return cgrad(name, max(n, 2); categorical=true)
        catch
            continue
        end
    end
    return cgrad(:viridis, max(n, 2); categorical=true)
end

# --- discovery --------------------------------------------------------------

# One `runs/<cell_id>/sample<k>/` on disk, with its `policy.csv` if it has one. Discovery
# walks the runs tree rather than results.csv for the same reason the videos do: a sweep
# killed before it wrote its final rows still yields every policy it managed to produce.
#
# `path === nothing` is a sample dir that produced no policy — a run censored by
# `timeout_s` never returns a solution to take one from. Those are kept rather than
# dropped, so the algorithm still occupies a row of the grid: an algorithm that silently
# vanished from a comparison figure is worse than one shown as visibly empty.
struct _PolicyEntry
    path::Union{String,Nothing}
    cell_id::String
    instance::String
    algorithm::String
    sample::Int
end

function _discover_policies(results_dir::AbstractString)
    out = _PolicyEntry[]
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
            isdir(joinpath(celldir, sample)) || continue
            f = joinpath(celldir, sample, "policy.csv")
            m = match(r"(\d+)$", sample)
            push!(out, _PolicyEntry(isfile(f) ? f : nothing, cell, instance, algorithm,
                                    m === nothing ? 0 : parse(Int, m.captures[1])))
        end
    end
    # By sample NUMBER, not directory name: `readdir` is lexicographic and would order
    # sample10 before sample2, putting the grid's columns out of sequence.
    return sort!(out; by=e -> (e.instance, e.algorithm, e.sample))
end

"""
    _read_policy(path) -> Vector{Int} | nothing

The `action` column of a `policy.csv`, indexed by flat state. `nothing` when the file is
unreadable or carries no `action` column — a malformed policy costs its panel, never the
figure.
"""
function _read_policy(path::AbstractString)
    try
        df = CSV.read(path, DataFrame)
        ("action" in names(df)) || return nothing
        nrow(df) == 0 && return nothing
        return Int[ismissing(a) ? 0 : Int(a) for a in df.action]
    catch e
        @warn "policy plot: unreadable policy.csv; its panel will be blank" path exception=e
        return nothing
    end
end

"""
    _num_actions_for(results_dir, instance) -> Int | nothing

The problem's action count from the `num_actions` column of `problems.csv`.

This pins the colour scale to the problem's FULL action set rather than to the actions a
particular run happened to select, which is what makes a colour mean the same thing
across every panel of the grid — and across algorithms, where a sampled variant selecting
fewer distinct actions than the sweep is a finding, not a rescaling.
"""
function _num_actions_for(results_dir::AbstractString, instance::AbstractString)
    path = joinpath(results_dir, "problems.csv")
    isfile(path) || return nothing
    try
        df = CSV.read(path, DataFrame)
        ("problem" in names(df) && "num_actions" in names(df)) || return nothing
        for r in eachrow(df)
            String(coalesce(r.problem, "")) == instance || continue
            ismissing(r.num_actions) && return nothing
            return Int(r.num_actions)
        end
    catch e
        @warn "policy plot: unreadable problems.csv; scaling colours to the observed actions" exception=e
    end
    return nothing
end

# --- rendering --------------------------------------------------------------

"""
    analysis_per_problem_policy(results_dir, plots_dir; problems_root=nothing)
        -> (; rendered, skipped, log)

One figure per problem under `<plots_dir>/per_problem/<instance>_policy.<ext>`: a grid of
policy heatmaps with **one row per algorithm and one column per sample**, colour = the
action index the solver selected for that state.

Eligibility is the videos' gate, `grid_geometry` — a problem is drawn only when its name
vouches for a uniform-grid family. A PRISM import has no spatial structure, and laying it
out as a square would put unrelated states side by side and read as a policy that is
merely noisy. Such problems are skipped with the reason recorded in the returned `log`.

States the solver never assigned an action (the `0` sentinel — see the file header) are
drawn in the neutral grey the videos use for "no backup", not as action 1 and not as
blank space.

`problems_root` is forwarded to [`overlay_masks`](@ref) for locating a problem's spec
when the path recorded by the sweep is unreachable (a cluster run records `/scratch/...`).
"""
function analysis_per_problem_policy(results_dir::AbstractString, plots_dir::AbstractString;
                                     problems_root::Union{Nothing,AbstractString}=nothing)
    entries = _discover_policies(results_dir)
    log = String[]
    if isempty(entries)
        push!(log, "no policy.csv found under $(joinpath(results_dir, "runs")) " *
                   "([output].save_policy must be on, and the dir must have been written " *
                   "by a sweep that had it)")
        return (; rendered=0, skipped=0, log)
    end

    sub = joinpath(plots_dir, "per_problem")
    rendered = skipped = 0
    for instance in sort(unique(e.instance for e in entries))
        ents = filter(e -> e.instance == instance, entries)

        # Load first: the policy itself declares the state count, so geometry never
        # depends on problems.csv being present. An entry with no file, or an unreadable
        # one, stays in the list with `nothing` so it still claims its cell of the grid.
        loaded = Tuple{_PolicyEntry,Union{Nothing,Vector{Int}}}[
            (e, e.path === nothing ? nothing : _read_policy(e.path)) for e in ents
        ]
        withdata = [a for (_, a) in loaded if a !== nothing]
        if isempty(withdata)
            skipped += 1
            push!(log, "skipped  $instance: no readable policy.csv among " *
                       "$(length(ents)) sample dir(s)")
            continue
        end

        # The longest policy, not the first: a sample killed mid-write leaves a short
        # file, and taking its length would either refuse the whole problem or — worse —
        # match some other family's grid. `grid_geometry` validates the count against the
        # instance name anyway, so a genuinely wrong one is refused loudly.
        nstates = maximum(length, withdata)
        geom, reason = grid_geometry(instance, nstates)
        if geom === nothing
            skipped += 1
            push!(log, "skipped  $instance: $reason")
            continue
        end

        try
            n = _render_policy_grid(results_dir, sub, instance, geom, loaded; problems_root)
            rendered += 1
            push!(log, "rendered $(instance)_policy  ($n panel(s), " *
                       "$(geom.shape[1])x$(geom.shape[2]) grid)")
        catch err
            skipped += 1
            push!(log, "FAILED   $instance: $(sprint(showerror, err))")
            @warn "policy plot failed" instance exception=(err, catch_backtrace())
        end
    end

    @info "policy figures written" dir=sub rendered skipped
    return (; rendered, skipped, log)
end

function _render_policy_grid(results_dir, sub, instance, geom::GridGeometry, loaded;
                             problems_root=nothing)
    algs = sort(unique(e.algorithm for (e, _) in loaded))
    samples = sort(unique(e.sample for (e, _) in loaded))
    nrows, ncols = length(algs), length(samples)
    byij = Dict((e.algorithm, e.sample) => a for (e, a) in loaded if a !== nothing)

    n1, n2 = geom.shape
    cells = n1 * n2

    # Colour scale: the problem's full action set, else the largest action anyone picked.
    nactions = _num_actions_for(results_dir, instance)
    if nactions === nothing || nactions < 1
        nactions = maximum(maximum(a) for (_, a) in loaded if a !== nothing; init=1)
    end
    nactions = max(nactions, 1)

    # Axis coordinates — the videos' construction: n+1 edges for the heatmap cells, the n
    # centres for the contour overlay, on the same grid.
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

    masks = overlay_masks(results_dir, instance, geom; problems_root)
    cmap = _policy_colormap(nactions)

    fig = Figure(size=(200 + ncols * _POLICY_PANEL, 140 + nrows * _POLICY_PANEL))
    Label(fig[0, 1:ncols], "$instance  ·  policy (colour = action index)";
          fontsize=14, font=:bold, tellwidth=false)

    hm = nothing
    npanels = 0
    blanks = 0
    for (r, alg) in enumerate(algs)
        # Row identity goes in a rotated label of its own so the y axis can keep carrying
        # the state-space coordinate.
        Label(fig[r, 0], alg; rotation=pi/2, fontsize=11, font=:bold, tellheight=false)
        for (c, s) in enumerate(samples)
            ax = Axis(fig[r, c]; aspect=DataAspect(),
                      limits=(xe[1], xe[end], ye[1], ye[end]),
                      title=(r == 1 ? "sample $s" : ""), titlesize=12,
                      xlabel=(r == nrows ? xlab : ""), ylabel=(c == 1 ? ylab : ""))
            r == nrows || hidexdecorations!(ax; grid=false)
            c == 1     || hideydecorations!(ax; grid=false)

            a = get(byij, (alg, s), nothing)
            if a === nothing || length(a) < cells
                # Keep the cell so the grid does not reflow around a hole, and say why it
                # is empty rather than leaving a reader to guess.
                text!(ax, mean(xe), mean(ye); text="no policy", align=(:center, :center),
                      color=:gray55, fontsize=11)
                blanks += 1
                continue
            end
            # Column-major, s = (j-1)*n1 + i; the trailing absorbing state is off-grid.
            # `<= 0` is the never-assigned sentinel and is drawn as background.
            M = map(x -> x <= 0 ? NaN : Float64(x),
                    collect(reshape(view(a, 1:cells), n1, n2)))
            hm = heatmap!(ax, xe, ye, M; colormap=cmap,
                          colorrange=(0.5, nactions + 0.5),
                          nan_color=_NO_UPDATE_COLOR)
            npanels += 1

            for (m, color, style) in ((masks.reach,   :white,   :solid),
                                      (masks.avoid,   :red,     :solid),
                                      (masks.initial, :magenta, :dash))
                any(m) || continue
                _outline!(ax, xc, yc, m; color=color, linestyle=style)
            end
        end
    end

    if hm !== nothing
        ticks = nactions <= _POLICY_TICK_LIMIT ? (1:nactions) :
                (1:cld(nactions, _POLICY_TICK_LIMIT):nactions)
        Colorbar(fig[1:nrows, ncols + 1], hm; label="action index", labelsize=10,
                 ticks=(collect(ticks), string.(collect(ticks))))
    end

    legend = String[]
    any(masks.reach)   && push!(legend, "white = reach")
    any(masks.avoid)   && push!(legend, "red = avoid")
    any(masks.initial) && push!(legend, "magenta dashed = initial (S0)")

    # Wrapped by hand across a few short lines. Makie's `Label` does not reflow, and a
    # single run of text long enough to say all of this is simply clipped at the figure
    # edge -- which cost the last clause of the caption outright.
    lines = String[
        "rows = algorithm, columns = sample (seed).  " *
        "grey = no action assigned (state never backed up).",
    ]
    l2 = string(isempty(legend) ? "" : join(legend, ", ") * ".  ",
                geom.extra > 0 ? "$(geom.extra) off-grid absorbing state not shown." : "")
    isempty(strip(l2)) || push!(lines, strip(l2))
    l3 = string(blanks > 0 ? "$blanks panel(s) empty: that run returned no policy " *
                             "(what a run censored by timeout_s does).  " : "",
                geom.provenance === :perfect_square ?
                    "Grid shape inferred from the state count." : "")
    isempty(strip(l3)) || push!(lines, strip(l3))
    _save(fig, sub, "$(instance)_policy"; note=join(lines, "\n"))
    return npanels
end
