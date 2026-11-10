# Benchmark chapter: problem clusters from the model metrics. The metrics are selected from the 23
# candidates (PCA_SPEC_ALL, _cluster_common.jl) on z-scores: communality h² ≥ 0.6 on the components
# that explain 75 % of the variance, near-duplicates (|ρ| > 0.85) dropped from the highest h² down.
# PCA of the selected metrics of the 33 problems left after setting the extremes aside
# (CLUSTER_EXCLUDE_GROUPS), Ward clustering on the leading components cut at k = 5, then refined.
#
#   output/benchmark/dataset_cluster_pc_loadings.{pdf,png}
#       the signed loading of every metric on each retained component
#   output/benchmark/dataset_cluster_dendrogram.{pdf,png}
#       Ward dendrogram, with the silhouette of every problem after the refinement
#   output/benchmark/dataset_cluster_k.{pdf,png}
#       choosing k: mean silhouette, merge height and within-cluster sum of squares
#   output/benchmark/dataset_cluster_scatter.{pdf,png}
#       the problems on every pair of retained components, with loading arrows
#   cache/dataset_cluster_{problems,excluded}.csv
#       the clusters, read by the experiments-*.jl scripts (run this script first)
#
# Inputs: results/mixing/ (tools/mixing_stats.jl, s0_horizon.jl, stochasticity_stats.jl,
# spectral_density.jl; see configs/thesis/README.md) and the problems.csv of THESIS_DIR.
# Nothing comes from a solve.

include(joinpath(@__DIR__, "_dataset_common.jl"))
include(joinpath(@__DIR__, "_cluster_common.jl"))

const OUT = "benchmark"   # output/benchmark/

# The analysis as the thesis runs it.
const DC = (;
    # the metrics, selected from all 23 candidates by select_metrics
    select=(; h2min=0.6, rho=0.85, topdown=true),
    rank=false,                                 # z-scores (false) or normal scores (true)
    k=5,                                        # clusters, fixed by hand
    # components oriented so the loading of these metrics is positive (skipped if not used)
    orient=(:log_depth_max, :log_nt, :log_h_reach_s0),
    pc_names=["horizon", "size vs terminal sets", "reach probability vs actions"],
    # Named clusters: each holds exactly one anchor problem (checked below), which fixes the
    # numbering and the colour whatever order Ward returns them in.
    anchors=["coin2_K2" => "long near-deterministic chains",
             "storm_tiny02" => "tiny models",
             "pimdp_0" => "loose lower bounds",
             "package_delivery_S50_I5" => "low-reach abstractions",
             "linear_stochastically_switched_S40" => "dense high-reach abstractions"])
const CLUSTER_ANCHORS = DC.anchors
# the transform before standardising
tf(v) = DC.rank ? normal_score(v) : v
# categorical slots 1–5 (validated); every cluster also has its own marker
const CLUSTER_COLOR = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
const CLUSTER_MARKER = [:utriangle, :circle, :hexagon, :rect, :diamond]

missing_inputs = filter(!isfile, [S0_CSV; joinpath(MIXING_DIR, "mixing_stats.csv"); cluster_inputs])
isempty(missing_inputs) || error("dataset-cluster.jl: $(length(missing_inputs)) mixing files missing, e.g. " *
                                 "$(first(missing_inputs)) (run the tools/ metric scripts, configs/thesis/README.md)")

cm_all = DataFrame([(; problem=p, source=CATEGORY_OF[p], class=CLASS_OF[p], cluster_metrics(p)...)
                    for p in problems.problem])
all(p -> p in cm_all.problem, first.(CLUSTER_EXCLUDE)) || error("dataset-cluster.jl: CLUSTER_EXCLUDE names an unknown problem")
cm = cm_all[.!in(first.(CLUSTER_EXCLUDE)).(cm_all.problem), :]
println("  clusters: $(nrow(cm)) problems, left out: ", join(first.(CLUSTER_EXCLUDE), ", "))

# The metrics, selected from all candidates (select_metrics, _cluster_common.jl)
const SEL = select_metrics(cm, first.(PCA_SPEC_ALL); rank=DC.rank, h2min=DC.select.h2min, rho=DC.select.rho,
                           topdown=DC.select.topdown)
const SPEC = PCA_SPEC_ALL[SEL.keep]
const COLS = first.(SPEC)
const GROUPS = filter(g -> any(s -> s[2] == g, SPEC), ["size", "sets", "stochasticity", "horizon", "spectrum"])
spec_label(i) = latexstring(SPEC[i][3] * " \$" * SPEC[i][4] * "\$")
ncm, npm = nrow(cm), length(COLS)
println("  selected metrics: ", join(String.(COLS), ", "))

# --- PCA --------------------------------------------------------------------------------
Xc = standardise(reduce(hcat, [tf(Float64.(cm[!, c])) for c in COLS]))
pc = pca_of(Xc)
# Signs are arbitrary; orient PC1 along the maximum depth, PC2 along the non-terminal states and
# PC3 along the reach probability.
for (j, c) in enumerate(DC.orient)
    i = findfirst(==(c), COLS)
    if i !== nothing && pc.V[i, j] < 0
        pc.V[:, j] .*= -1; pc.Z[:, j] .*= -1
    end
end
npc = findfirst(>=(CLUSTER_VAR), cumsum(pc.ev))

# --- Ward clustering on the retained components (hclust, cut, refine_kmeans: _cluster_common.jl)
Xk = pc.Z[:, 1:npc]
Dk = distances(Xk)
merges = hclust(Dk, :ward)
heights = reverse(last.(merges))            # heights[k - 1]: the merge from k to k - 1 clusters
sil_k = Dict(k => mean(silhouettes(Dk, cut(merges, k))) for k in CLUSTER_K)
wss_k = [wss(Xk, cut(merges, k)) for k in 1:last(CLUSTER_K)]
krule = argmax(k -> sil_k[k], CLUSTER_KRANGE)   # the best silhouette in the range read
kbest = DC.k
# the other cut marked in the figures: the best silhouette in CLUSTER_KRANGE
kglobal = krule
lab_ward = cut(merges, kbest)
lab0 = refine_kmeans(Xk, lab_ward)
# renumber by the anchors (no anchor moves in the refinement, so one map serves both)
length(CLUSTER_ANCHORS) == kbest || error("dataset-cluster.jl: k = $kbest clusters but $(length(CLUSTER_ANCHORS)) anchors")
relabel = Dict(lab0[findfirst(==(p), cm.problem)] => c for (c, (p, _)) in enumerate(CLUSTER_ANCHORS))
all(p -> lab0[findfirst(==(p), cm.problem)] == lab_ward[findfirst(==(p), cm.problem)], first.(CLUSTER_ANCHORS)) ||
    error("dataset-cluster.jl: an anchor moved in the k-means refinement; update CLUSTER_ANCHORS")
length(relabel) == kbest || error("dataset-cluster.jl: two cluster anchors fall in one cluster; update CLUSTER_ANCHORS")
cm.cluster = [relabel[l] for l in lab0]
cm.ward_cluster = [relabel[l] for l in lab_ward]
refined = findall(cm.cluster .!= cm.ward_cluster)
cm.silhouette = silhouettes(Dk, cm.cluster)
for j in 1:npc
    cm[!, "PC$j"] = pc.Z[:, j]
end
cname(c) = "C$c " * last(CLUSTER_ANCHORS[c])
csize(c) = count(==(c), cm.cluster)

@printf("  clusters: PCA of %d metrics, %d components explain %.0f %% (%s); Ward k = %d (silhouette %s)\n",
        npm, npc, 100sum(pc.ev[1:npc]), join([@sprintf("%.0f", 100e) for e in pc.ev[1:npc]], "/"), kbest,
        join([@sprintf("k%d %.2f", k, sil_k[k]) for k in CLUSTER_K], " "))
for c in 1:kbest
    println("    ", cname(c), " (", csize(c), "): ", join(short_name.(cm.problem[cm.cluster .== c]), ", "))
end
for i in refined
    @printf("    refinement: %s C%d -> C%d\n", cm.problem[i], cm.ward_cluster[i], cm.cluster[i])
end

# --- Figures --------------------------------------------------------------------------------
const CL_AX =(titlesize=8, titlealign=:left, titlefont=:regular, xlabelsize=8, ylabelsize=8,
               xticklabelsize=7, yticklabelsize=7)
cluster_legend!(pos; nbanks=2) =
    Legend(pos, [MarkerElement(marker=CLUSTER_MARKER[c], color=CLUSTER_COLOR[c], strokecolor=INK,
                               strokewidth=0.5, markersize=7) for c in 1:kbest],
           ["$(cname(c)) ($(csize(c)))" for c in 1:kbest]; orientation=:horizontal, nbanks,
           framevisible=false, labelsize=7, tellwidth=false, tellheight=true, colgap=10)

# (1) What each component represents: one bar panel per component, the signed loading of every
# metric (rows shared, by metric group). Red = the metric grows along the component, blue = it
# shrinks; faded where |loading| < 0.25. A loading squared is the metric's share of the
# component (the squares of one component sum to 1).
const PC_NAMES = DC.pc_names
let fig = report_figure(height=40 + 13npm), ys = 1:npm
    rows = reduce(vcat, [[i for i in 1:npm if SPEC[i][2] == g] for g in GROUPS])
    seps = cumsum([count(s -> s[2] == g, SPEC) for g in GROUPS])[1:end-1] .+ 0.5
    strong(v) = abs(v) >= 0.25
    axs = Axis[]
    for j in 1:npc
        L = pc.V[rows, j]
        ax = Axis(fig[1, j + 1]; title=@sprintf("PC%d (%.0f %%)", j, 100pc.ev[j]), subtitle=PC_NAMES[j],
                  subtitlesize=7.5, subtitlecolor=C_INK2, xlabel="loading", yreversed=true,
                  limits=((-0.7, 0.7), (0.4, npm + 0.6)), xgridvisible=false,
                  xticks=([-0.5, -0.25, 0, 0.25, 0.5], ["−0.5", "", "0", "", "0.5"]),
                  ygridvisible=false, CL_AX..., titlealign=:center)
        vlines!(ax, [0]; color=C_CONTEXT, linewidth=0.6)
        vlines!(ax, [-0.25, 0.25]; color=(C_CONTEXT, 0.6), linewidth=0.5, linestyle=:dash)
        hlines!(ax, seps; color=(C_CONTEXT, 0.4), linewidth=0.6)
        barplot!(ax, ys, L; direction=:x, gap=0.3,
                 color=[(v > 0 ? DIVERGING[5] : DIVERGING[1], strong(v) ? 1.0 : 0.3) for v in L])
        hideydecorations!(ax; grid=false)
        push!(axs, ax)
    end
    # metric names left of the first panel, in an axis without decorations sharing the rows
    axl = Axis(fig[1, 1]; yreversed=true, limits=((0, 1), (0.4, npm + 0.6)),
               yticks=(ys, [spec_label(i) for i in rows]), yticklabelsize=7)
    hidexdecorations!(axl); hidespines!(axl); axl.yticksvisible = false; axl.ygridvisible = false
    linkyaxes!(axl, axs...)
    colsize!(fig.layout, 1, Fixed(0))
    colgap!(fig.layout, 1, 0); colgap!(fig.layout, 10)
    save_figure(fig, "dataset_cluster_pc_loadings", OUT)
end

# (2) Dendrogram, with per-problem silhouettes under the leaves.
function dendrogram!(ax, merges, lab)
    n = length(merges) + 1
    order = leaves(merges, length(merges)); xof = zeros(n); xof[order] = 1:n
    pos = Dict{Int,Tuple{Float64,Float64,Int}}()   # node -> (x, height, cluster or -1)
    node(i) = i < 0 ? (xof[-i], 0.0, lab[-i]) : pos[i]
    for (m, (a, b, h)) in enumerate(merges)
        (xa, ha, ca), (xb, hb, cb) = node(a), node(b)
        c = ca == cb ? ca : -1
        pos[m] = ((xa + xb) / 2, h, c)
        for (x0, h0, cc) in ((xa, ha, ca), (xb, hb, cb))
            lines!(ax, [x0, x0, (xa + xb) / 2], [h0, h, h]; linewidth=1,
                   color=cc > 0 && c > 0 ? CLUSTER_COLOR[cc] : C_CONTEXT)
        end
    end
    return order
end
let fig = report_figure(height=330)
    ax = Axis(fig[1, 1]; title="(a) Ward dendrogram on PC1–PC$npc", ylabel="merge height", CL_AX...)
    order = dendrogram!(ax, merges, cm.ward_cluster)
    for (k, style) in (kglobal == kbest ? ((kbest, :dash),) : ((kbest, :dash), (kglobal, :dot)))
        hlines!(ax, [(heights[k - 1] + heights[k]) / 2]; color=INK, linestyle=style, linewidth=0.8)
        text!(ax, length(order) + 0.5, (heights[k - 1] + heights[k]) / 2; text="k = $k", fontsize=7,
              color=C_INK2, align=(:right, :bottom), offset=(0, 1))
    end
    hidexdecorations!(ax); hidespines!(ax, :t, :r, :b)
    xlims!(ax, 0.3, length(order) + 0.7); ylims!(ax, 0, nothing)
    ax2 = Axis(fig[2, 1]; title="(b) silhouette of each problem at k = $kbest, after the refinement",
               ylabel="silhouette",
               xticks=(1:length(order), [short_name(cm.problem[i]) * (i in refined ? " *" : "") for i in order]),
               xticklabelrotation=π / 2, CL_AX...,
               xticklabelsize=6.5)
    barplot!(ax2, 1:length(order), cm.silhouette[order]; color=CLUSTER_COLOR[cm.cluster[order]], gap=0.15)
    hlines!(ax2, [0]; color=INK, linewidth=0.6)
    xlims!(ax2, 0.3, length(order) + 0.7)
    linkxaxes!(ax, ax2)
    cluster_legend!(fig[3, 1]; nbanks=2)
    rowsize!(fig.layout, 2, Relative(0.22))
    rowgap!(fig.layout, 6)
    save_figure(fig, "dataset_cluster_dendrogram", OUT)
end

# (3) Choosing k: mean silhouette, merge height and within-cluster sum of squares.
let fig = report_figure(height=170), ks = collect(CLUSTER_K)
    function mark!(ax)
        vspan!(ax, first(CLUSTER_KRANGE) - 0.5, last(CLUSTER_KRANGE) + 0.5; color=(C_CONTEXT, 0.15))
        vlines!(ax, [kbest]; color=INK, linestyle=:dash, linewidth=0.8)
    end
    ax = Axis(fig[1, 1]; title="(a) mean silhouette (higher is better)", xlabel="clusters k", xticks=ks, CL_AX...)
    mark!(ax)
    scatterlines!(ax, ks, [sil_k[k] for k in ks]; color=INK, markersize=5)
    scatter!(ax, [kbest], [sil_k[kbest]]; color=C_HI, markersize=8)
    text!(ax, kglobal, sil_k[kglobal]; text=DC.k === nothing ? "best overall" : "best in range", fontsize=6.5, color=C_INK2, align=(:left, :top),
          offset=(4, -1))
    ax = Axis(fig[1, 2]; title="(b) merge height from k to k − 1", xlabel="clusters k", xticks=ks, CL_AX...)
    mark!(ax)
    scatterlines!(ax, ks, heights[ks .- 1]; color=INK, markersize=5)
    ax = Axis(fig[1, 3]; title="(c) within-cluster sum of squares", xlabel="clusters k", xticks=1:last(ks), CL_AX...)
    mark!(ax)
    scatterlines!(ax, 1:last(ks), wss_k; color=INK, markersize=5)
    colgap!(fig.layout, 12)
    save_figure(fig, "dataset_cluster_k", OUT)
end

# (4) The clusters on every pair of the retained components, drawn to scale (one PC unit is the
# same length on every axis): PC1 × PC2 on top, PC1 × PC3 under it (shared PC1 axis) and
# PC2 × PC3 beside that (shared PC3 axis). Grey arrows: the metrics with the largest loadings
# in the plane.
# Labels are placed greedily, left to right: per point (or stack of points within 3 pt) the
# first of eight offsets whose box hits neither a marker nor an earlier label; a label that
# fits nowhere is left out (the dendrogram lists every problem).
function stack_label(names_)
    length(names_) == 1 && return only(names_)
    s = first(names_)
    for t in names_
        i = findfirst(k -> k > lastindex(t) || t[k] != s[k], collect(eachindex(s)))
        i === nothing || (s = s[1:prevind(s, collect(eachindex(s))[i])])
    end
    fam = rstrip(s, ['0':'9'..., ' ', '_', 'S', 'K'])           # "tiny0" -> "tiny", "aid S" -> "aid"
    length(fam) >= 3 && return "$fam ×$(length(names_))"
    both = join(names_, ", ")
    return length(both) <= 18 ? both : first(names_) * " +$(length(names_) - 1)"
end
# `blocked_pts` / `blocked` (pt, relative to the lower-left corner): other marks to keep clear of
function place_labels!(ax, x, y, names_, xr, yr, wpt, hpt; fs=5.5, blocked_pts=Tuple{Float64,Float64}[],
                       blocked=NTuple{4,Float64}[])
    P = [((x[i] - xr[1]) / (xr[2] - xr[1]) * wpt, (y[i] - yr[1]) / (yr[2] - yr[1]) * hpt) for i in eachindex(x)]
    done = falses(length(x)); boxes = copy(blocked)
    hits(b, (px, py), r) = b[1] - r < px < b[2] + r && b[3] - r < py < b[4] + r
    overlap(a, b) = a[1] < b[2] && b[1] < a[2] && a[3] < b[4] && b[3] < a[4]
    for i in sortperm(x)
        done[i] && continue
        st = [j for j in eachindex(x) if !done[j] && hypot(P[j][1] - P[i][1], P[j][2] - P[i][2]) < 3]
        done[st] .= true
        txt = stack_label(names_[st])
        w, h = 0.5fs * length(txt), fs
        px, py = mean(first.(P[st])), mean(last.(P[st]))
        for (dx, dy, al) in ((5, 0, (:left, :center)), (-5, 0, (:right, :center)), (0, 5, (:center, :bottom)),
                             (0, -5, (:center, :top)), (4, 4, (:left, :bottom)), (-4, 4, (:right, :bottom)),
                             (4, -4, (:left, :top)), (-4, -4, (:right, :top)))
            x0 = al[1] == :left ? px + dx : al[1] == :right ? px + dx - w : px - w / 2
            y0 = al[2] == :bottom ? py + dy : al[2] == :top ? py + dy - h : py - h / 2
            b = (x0, x0 + w, y0, y0 + h)
            (b[1] < 0 || b[2] > wpt || b[3] < 0 || b[4] > hpt) && continue
            any(j -> hits(b, P[j], 2.5), eachindex(x)) && continue
            any(q -> hits(b, q, 1.0), blocked_pts) && continue
            any(o -> overlap(b, o), boxes) && continue
            push!(boxes, b)
            text!(ax, mean(x[st]), mean(y[st]); text=txt, fontsize=fs, color=C_INK2, align=al, offset=(dx, dy))
            break
        end
    end
    return boxes
end
# loading arrows of the `nshow` metrics longest in the (a, b) plane, scaled so the longest
# spans `frac` of the shorter half-axis; arrows within 12° of each other share one label, placed
# beyond the tip at the first offset whose box clears the markers `mpts` (pt).
# Returns points along the arrows and the label boxes, in pt via `topt`, for place_labels!.
function loading_arrows!(ax, a, b, xr, yr, topt; nshow=4, frac=0.85, mpts=Tuple{Float64,Float64}[])
    pts, boxes = Tuple{Float64,Float64}[], NTuple{4,Float64}[]
    len = hypot.(pc.V[:, a], pc.V[:, b])
    top = sortperm(len; rev=true)[1:nshow]
    s = frac * min(-xr[1], xr[2], -yr[1], yr[2]) / maximum(len)
    θ = Dict(i => atan(pc.V[i, b], pc.V[i, a]) for i in top)
    groups = Vector{Int}[]
    for i in sort(top; by=i -> θ[i])
        if !isempty(groups) && abs(θ[i] - θ[last(groups)[end]]) < deg2rad(12)
            push!(last(groups), i)
        else
            push!(groups, [i])
        end
    end
    for g in groups
        for i in g
            arrows2d!(ax, [0.0], [0.0], [s * pc.V[i, a]], [s * pc.V[i, b]]; color=(C_CONTEXT, 0.9),
                      shaftwidth=0.6, tipwidth=4, tiplength=4)
            append!(pts, [topt(t * s * pc.V[i, a], t * s * pc.V[i, b]) for t in 0.15:0.05:1])
        end
        i = g[argmax(len[g])]
        ex, ey = s * pc.V[i, a], s * pc.V[i, b]
        sym = join([SPEC[j][4] for j in g], ",\\ ")
        # rough size of the label: visible characters of the LaTeX at 6.5 pt
        w, h = 3.6 * length(replace(sym, r"\\[a-zA-Z]+" => "", r"[{}_^ ]" => "")) + 4, 9.0
        (px, py) = topt(ex, ey)
        ux, uy = (ex, ey) ./ hypot(ex, ey)                              # along the arrow, in plot units
        # the candidate box that covers the fewest markers and earlier arrow labels (the first,
        # nearest the tip, on a tie)
        best, nbest = nothing, typemax(Int)
        for d in (2.0, 6.0, 10.0, 14.0, 20.0), side in (0.0, 8.0, -8.0, 16.0, -16.0)
            ox, oy = d * ux - side * uy, d * uy + side * ux
            x0 = ex >= 0 ? px + ox : px + ox - w
            y0 = ey >= 0 ? py + oy : py + oy - h
            bx = (x0, x0 + w, y0, y0 + h)
            nhit = count(q -> bx[1] - 3 < q[1] < bx[2] + 3 && bx[3] - 3 < q[2] < bx[4] + 3, mpts) +
                   2count(o -> bx[1] < o[2] && o[1] < bx[2] && bx[3] < o[4] && o[3] < bx[4], boxes)
            nhit < nbest && ((best, nbest) = ((ox, oy, bx), nhit))
            nhit == 0 && break
        end
        ox, oy, bx = best
        text!(ax, ex, ey; text=latexstring("\$" * sym * "\$"), fontsize=6.5, color=C_INK2,
              align=(ex >= 0 ? :left : :right, ey >= 0 ? :bottom : :top), offset=(ox, oy))
        push!(boxes, bx)
    end
    return pts, boxes
end
let pad(v) = (d = maximum(v) - minimum(v); (minimum(v) - 0.08d, maximum(v) + 0.08d)),
    R = [pad(pc.Z[:, j]) for j in 1:npc], span = [r[2] - r[1] for r in R],
    unit = (TEXTWIDTH_PT - 70) / (span[1] + span[2]),     # pt per PC unit
    fig = report_figure(height=400)
    spots = [((1, 2), (1, 1), "(a)"), ((1, 3), (2, 1), "(b)"), ((2, 3), (2, 2), "(c)")]
    axs = Dict()
    for ((a, b), (r, c), tag) in spots
        ax = Axis(fig[r, c]; title="$tag PC$b vs PC$a", width=unit * span[a], height=unit * span[b],
                  xlabel=@sprintf("PC%d (%.0f %%)", a, 100pc.ev[a]), ylabel=@sprintf("PC%d (%.0f %%)", b, 100pc.ev[b]),
                  CL_AX...)
        axs[(a, b)] = ax
        x, y = pc.Z[:, a], pc.Z[:, b]
        limits!(ax, R[a]..., R[b]...)
        hlines!(ax, [0]; color=(C_CONTEXT, 0.5), linewidth=0.5); vlines!(ax, [0]; color=(C_CONTEXT, 0.5), linewidth=0.5)
        wpt, hpt = unit * span[a], unit * span[b]
        topt(u, v) = ((u - R[a][1]) / span[a] * wpt, (v - R[b][1]) / span[b] * hpt)
        apts, aboxes = loading_arrows!(ax, a, b, R[a], R[b], topt; mpts=[topt(x[i], y[i]) for i in eachindex(x)])
        for k in 1:kbest
            sel = cm.cluster .== k
            scatter!(ax, x[sel], y[sel]; color=CLUSTER_COLOR[k], marker=CLUSTER_MARKER[k], markersize=7,
                     strokecolor=INK, strokewidth=0.5)
        end
        place_labels!(ax, x, y, short_name.(cm.problem), R[a], R[b], wpt, hpt; blocked_pts=apts, blocked=aboxes)
    end
    linkxaxes!(axs[(1, 2)], axs[(1, 3)]); linkyaxes!(axs[(1, 3)], axs[(2, 3)])
    hidexdecorations!(axs[(1, 2)]; grid=false, ticks=false)
    hideydecorations!(axs[(2, 3)]; grid=false, ticks=false)
    Legend(fig[1, 2], [MarkerElement(marker=CLUSTER_MARKER[c], color=CLUSTER_COLOR[c], strokecolor=INK,
                                     strokewidth=0.5, markersize=7) for c in 1:kbest],
           ["$(cname(c)) ($(csize(c)))" for c in 1:kbest]; framevisible=false, labelsize=7,
           tellwidth=false, tellheight=false, halign=:left, valign=:center, rowgap=2)
    colgap!(fig.layout, 8); rowgap!(fig.layout, 6)
    resize_to_layout!(fig)
    save_figure(fig, "dataset_cluster_scatter", OUT)
end

# --- The clusters for the experiments-*.jl scripts ------------------------------------------
mkpath(CACHE_DIR)
CSV.write(CLUSTER_PROBLEMS_CSV,
          select(transform(cm, :cluster => ByRow(c -> last(CLUSTER_ANCHORS[c])) => :cluster_name),
                 :problem, :source, :class, :cluster, :cluster_name, :ward_cluster, :silhouette))
CSV.write(CLUSTER_EXCLUDED_CSV,
          DataFrame([(problem=p, group=g, reason) for (g, (reason, ps)) in CLUSTER_EXCLUDE_GROUPS for p in ps]))
println("  wrote ", relpath(CLUSTER_PROBLEMS_CSV, BENCH_DIR), ", ", relpath(CLUSTER_EXCLUDED_CSV, BENCH_DIR))
