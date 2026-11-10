# Relative-δ thinning of convergence traces, and a streaming reader for trace CSVs.
#
# A sampled solver fires the trace callback every few Bellman updates, so a long run
# records tens of millions of rows (a single coin2_K4 trajectory `trace.csv` reached
# 6.6 GB / 58M rows). Nothing downstream needs that resolution: every analysis reads the
# trace either as the first crossing of a gap level or as a curve interpolated onto a
# ~100-point grid, and the gap is monotone non-increasing. So a row is kept only when
# its gap has fallen by a relative `rtol` since the last kept row:
#
#   - the first row with a finite gap is kept, and its gaps become the *anchors*;
#   - row i is kept when any key gap satisfies g_i < anchor · (1 − rtol); all of its
#     finite gaps then become the new anchors;
#   - corner rule: when row i is kept, row i − 1 is kept too, so a plateau stays flat
#     under linear interpolation instead of turning into a ramp;
#   - the last row is always kept (budgets, final gap, censoring point);
#   - NaN / unparseable gaps never trigger a keep and never become anchors.
#
# Guarantee: the first kept row at or below any ε lies between the true first crossing
# of ε and that of ε(1 − rtol), and the interpolated curve is within relative `rtol` of
# the original. A trace spanning d decades keeps about 2·d·ln(10)/rtol rows. `rtol = 0`
# keeps every row.
#
# `GapThinner` only makes the anchor decision; the corner and last-row rules belong to
# whoever holds the rows — `TailBuffer` for a stream (the analysis loader and
# `compact_traces.jl`), `TraceExecutor` for the in-process recording (trace.jl).
#
# Plain Base code, so it is included by both `Bench` (for trace.jl) and
# `BenchmarkAnalysis` (which the tests load without `Bench`), and by `compact_traces.jl`
# on its own. The copies never exchange objects.

"""
    GapThinner(rtol, nkeys)

Anchor state for relative-`rtol` thinning over `nkeys` gap columns. See
[`observe!`](@ref).
"""
mutable struct GapThinner
    rtol::Float64
    anchors::Vector{Float64}   # NaN until the key has seen a finite gap
end
GapThinner(rtol::Real, nkeys::Integer) = GapThinner(Float64(rtol), fill(NaN, nkeys))

"""
    observe!(th, gaps) -> Bool

Whether the row with key gaps `gaps` (indexable by `1:nkeys`) is an anchor, i.e. must be
kept. Updates the anchors when it is. Always `true` when `rtol <= 0`.
"""
function observe!(th::GapThinner, gaps)
    keep = th.rtol <= 0
    @inbounds for k in eachindex(th.anchors)
        g = gaps[k]
        isfinite(g) || continue
        a = th.anchors[k]
        (isnan(a) || g < a * (1 - th.rtol)) && (keep = true)
    end
    if keep
        @inbounds for k in eachindex(th.anchors)
            g = gaps[k]
            isfinite(g) && (th.anchors[k] = g)
        end
    end
    return keep
end

"""
    TailBuffer{T}(thinner)

Applies the corner and last-row rules to a stream of rows of type `T`: [`offer!`](@ref)
each row in order, then [`flush_tail!`](@ref) once at the end. `emit(row)` receives the
kept rows, in order.
"""
mutable struct TailBuffer{T}
    th::GapThinner
    tail::Union{Nothing,T}    # the latest row, held back until we know whether it is a corner
end
TailBuffer{T}(th::GapThinner) where {T} = TailBuffer{T}(th, nothing)

function offer!(emit, tb::TailBuffer, row, gaps)
    if observe!(tb.th, gaps)
        tb.tail === nothing || emit(tb.tail)       # the corner before this drop
        emit(row)
        tb.tail = nothing
    else
        tb.tail = row
    end
    return nothing
end

function flush_tail!(emit, tb::TailBuffer)
    tb.tail === nothing || emit(tb.tail)
    tb.tail = nothing
    return nothing
end

# --- streaming trace CSV reader ----------------------------------------------
# The report-facing `trace.csv` is `wallclock_s,bellman_updates,gap_max,gap_mean[,
# initial_gap_max,initial_gap_mean]` (trace.jl, `write_report_trace_csv`). These helpers
# read it one line at a time, so memory is independent of the file size.

# Split `line` on commas into `fields` (reused across calls).
function split_fields!(fields::Vector{SubString{String}}, line::String)
    empty!(fields)
    start = 1
    while true
        i = findnext(',', line, start)
        if i === nothing
            push!(fields, SubString(line, start, lastindex(line)))
            return fields
        end
        push!(fields, SubString(line, start, prevind(line, i)))
        start = i + 1
    end
end

# `tryparse` that maps a failure to `nothing` for any field index out of range.
_field_float(fields, j) = (1 <= j <= length(fields)) ? tryparse(Float64, fields[j]) : nothing

"""
    trace_header_columns(header) -> Dict{String,Int}

Column name → 1-based position for a trace CSV header line.
"""
trace_header_columns(header::AbstractString) =
    Dict{String,Int}(String(strip(n)) => i for (i, n) in enumerate(split(header, ',')))

"""
    last_trace_row(path) -> Union{Nothing, Dict{String,Float64}}

The last complete, fully numeric data row of a trace CSV, keyed by column name, read
from the end of the file rather than parsing all of it. `nothing` when the file has no
such row (empty, header only, or every tail line truncated).
"""
function last_trace_row(path::AbstractString; window::Int=1 << 16)
    open(path, "r") do io
        header = readline(io)
        isempty(header) && return nothing
        cols = split(header, ',')
        ncol = length(cols)
        body_start = position(io)
        sz = filesize(path)
        # Widen the window until it holds at least one complete line, or the whole body.
        w = window
        while true
            from = max(body_start, sz - w)
            seek(io, from)
            chunk = String(read(io))
            lines = split(chunk, '\n'; keepempty=false)
            # The first line of a window that does not start at the body is partial.
            from > body_start && !isempty(lines) && popfirst!(lines)
            for line in Iterators.reverse(lines)
                f = split(line, ',')
                length(f) == ncol || continue
                vals = [tryparse(Float64, x) for x in f]
                any(isnothing, vals) && continue
                return Dict{String,Float64}(String(strip(cols[k])) => vals[k] for k in 1:ncol)
            end
            from == body_start && return nothing
            w *= 4
        end
    end
end
