# Shared plumbing for the figure scripts in report-plots/.
#
# Every script starts with `include(joinpath(@__DIR__, "..", "common.jl"))` (directly or through
# a `_*_common.jl`). That activates the benchmark environment, loads BenchmarkAnalysis
# (load_canonical, load_bounds, ...), and provides the report theme and palette plus:
#
#   THESIS_DIR                    results/thesis_combined (see configs/thesis/README.md)
#   results_dir(name)             resolve a results dir under the repo root or results/
#   load_long(dir)                load_canonical, cached in cache/ (trace parsing is slow)
#   report_figure(; width, aspect) a Figure sized in pt for the report's text width
#   save_figure(fig, name, subdir) output/<subdir>/<name>.pdf + .png
#
# Figures go to output/benchmark/ (Benchmark chapter and its appendix) or output/experiments/
# (Experiments chapter and its appendix). Intermediate data shared between scripts goes to
# cache/.

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

# Loaded into Main once per session; make_all.jl then re-includes this file into a fresh
# module per script, which only redefines the cheap helpers below.
if !isdefined(Main, :BenchmarkAnalysis)
    Base.include(Main, joinpath(@__DIR__, "..", "src", "analysis", "BenchmarkAnalysis.jl"))
end
using Main.BenchmarkAnalysis
using CairoMakie
using DataFrames
using Dates
using LaTeXStrings
using Printf
using Serialization
using Statistics

const BENCH_DIR  = normpath(joinpath(@__DIR__, ".."))
const OUTPUT_DIR = joinpath(@__DIR__, "output")
const CACHE_DIR  = joinpath(@__DIR__, "cache")

# Text width of the report in pt. Figures are drawn at their final size, so their 9 pt text
# matches the body text when included without scaling.
const TEXTWIDTH_PT = 426.0   # \textwidth of an A4 12pt report class

# Palette: one highlighted series in blue, everything else as recessive grey context;
# blue↔red diverging with a neutral grey midpoint.
const C_HI      = "#2a78d6"
const C_HI_SOFT = "#86b6ef"
const C_CONTEXT = "#898781"
const C_INK2    = "#52514e"
const C_RULE    = "#c3c2b7"
const C_GRID    = "#e1e0d9"
const DIVERGING = ["#1c5cab", "#6da7ec", "#f0efec", "#ee8f8e", "#b52f2e"]

function report_theme()
    t = Theme(fontsize=9, figure_padding=(4, 8, 4, 4),
              Axis=(xgridcolor=C_GRID, ygridcolor=C_GRID, xgridwidth=0.5, ygridwidth=0.5,
                    spinewidth=0.6, xtickwidth=0.6, ytickwidth=0.6, xticksize=3, yticksize=3,
                    topspinevisible=false, rightspinevisible=false,
                    leftspinecolor=C_RULE, bottomspinecolor=C_RULE,
                    xtickcolor=C_RULE, ytickcolor=C_RULE,
                    titlesize=9, titlefont=:regular, xlabelsize=9, ylabelsize=9,
                    xticklabelsize=8, yticklabelsize=8, xlabelpadding=2, ylabelpadding=2),
              Legend=(framevisible=false, labelsize=8, patchsize=(12, 6), rowgap=0,
                      padding=(2, 2, 2, 2)),
              Colorbar=(labelsize=8, ticklabelsize=8, spinewidth=0.5, ticksize=3))
    return merge(t, theme_latexfonts())
end

# MathTeXEngine (Makie's LaTeX renderer) has \overline but no \underline, which the lower
# bound \underline{p} needs. Register the command and lay it out as a rule just below the
# content's ink; every other expression goes to the original tex_layout. Guarded so that
# make_all.jl's per-script re-includes do not redefine it.
const _MTE = Makie.MathTeXEngine
if !haskey(_MTE.command_definitions, raw"\underline")
    _MTE.command_definitions[raw"\underline"] = (_MTE.TeXExpr(:underline), 1)
    function _MTE.tex_layout(expr::_MTE.TeXExpr, state::_MTE.LayoutState)
        expr.head == :underline || return invoke(_MTE.tex_layout, Tuple{Any, Any}, expr, state)
        content = _MTE.tex_layout(expr.args[1], state)
        lw = _MTE.thickness(state.font_family)
        y = _MTE.bottominkbound(content) - 2lw
        return _MTE.Group([content, _MTE.HLine(_MTE.inkwidth(content), lw)],
                          Point2f[(0, 0), (_MTE.leftinkbound(content), y)])
    end
end

"""
    results_dir(name) -> String

Absolute path of a results directory. `name` may be absolute, relative to the repo root, or a
bare dir name looked up in `results/`.
"""
function results_dir(name::AbstractString)
    isabspath(name) && isdir(name) && return name
    for base in (BENCH_DIR, joinpath(BENCH_DIR, "results"))
        p = joinpath(base, name)
        isdir(p) && return normpath(p)
    end
    error("results_dir: no results directory named $(repr(name)) under $BENCH_DIR " *
          "(run the configs in configs/thesis/ and combine them, see configs/thesis/README.md)")
end

# Every solver run the experiment figures use: the configs/thesis/*.toml runs, combined with
# combine_results.jl.
const THESIS_DIR = results_dir("thesis_combined")

"""
    load_long(dir; thin_rtol=DEFAULT_THIN_RTOL) -> DataFrame

`load_canonical(dir; thin_rtol)`, cached in `report-plots/cache/` (git-ignored). Parsing a
sweep's traces can take many minutes, and the thinned frame is small, so later runs read the
cache. The cache is rebuilt when results.csv is newer than it; delete `cache/` to force a
rebuild.
"""
function load_long(dir::AbstractString; thin_rtol::Real=BenchmarkAnalysis.DEFAULT_THIN_RTOL)
    mkpath(CACHE_DIR)
    path = joinpath(CACHE_DIR, "$(basename(normpath(dir)))__rtol$(thin_rtol).jls")
    if isfile(path) && mtime(path) > mtime(joinpath(dir, "results.csv"))
        return Serialization.deserialize(path)
    end
    println("  parsing traces of $(basename(normpath(dir))) (cached afterwards) …")
    long = load_canonical(dir; thin_rtol=thin_rtol)
    Serialization.serialize(path, long)
    return long
end

"""
    report_figure(; width=1.0, aspect=1.6, height=nothing) -> Figure

A Figure `width × TEXTWIDTH_PT` pt wide, under the report theme. The height is `height` pt if
given, else width / aspect.
"""
function report_figure(; width::Real=1.0, aspect::Real=1.6, height=nothing)
    w = width * TEXTWIDTH_PT
    h = height === nothing ? w / aspect : height
    set_theme!(report_theme())
    return Figure(size=(w, h))
end

"""
    output_dir(subdir) -> String

`output/<subdir>/`, created on first use.
"""
output_dir(subdir::AbstractString) = mkpath(joinpath(OUTPUT_DIR, subdir))

"""
    save_figure(fig, name, subdir) -> path

Write `output/<subdir>/<name>.pdf` (vector) and `<name>.png` (3 px per pt). Returns the PDF
path.
"""
function save_figure(fig, name::AbstractString, subdir::AbstractString)
    dir = output_dir(subdir)
    pdf = joinpath(dir, name * ".pdf")
    CairoMakie.save(pdf, fig; pt_per_unit=1)
    CairoMakie.save(joinpath(dir, name * ".png"), fig; px_per_unit=3)
    println("  wrote ", relpath(pdf, @__DIR__))
    return pdf
end

# Cluster assignment written by scripts/dataset-cluster.jl and read by the experiments scripts:
# the clustered problems (problem, cluster, cluster_name) and the extremes set aside before the
# clustering (problem, group).
const CLUSTER_PROBLEMS_CSV = joinpath(CACHE_DIR, "dataset_cluster_problems.csv")
const CLUSTER_EXCLUDED_CSV = joinpath(CACHE_DIR, "dataset_cluster_excluded.csv")

# y label of every initial-state gap axis (thesis notation)
const GAP_LABEL = L"\max_{s_0 \in S_0} U(s_0) - L(s_0)"
