# Convergence traces: callback functors passed to `solve` on the instrumented run.
# They record per-iteration data so we can study convergence RATE (gap / value vs
# wall-clock and vs bellman-updates), not just total solve time. The GSRDP callback
# comes in two shapes, and the solver picks whichever the callback supports:
#
#   callback(vf::IntervalValueFunction, bellman_updates)
#   callback(vf::IntervalValueFunction, bellman_updates, update_sequence)
#
# The three-argument form additionally reports WHICH states that iteration relaxed,
# which is what `UpdateTrace` records (`update_sequence` is `nothing` on the pre-update
# fire). Everything else ignores it. See `_invoke_callback` in IntervalMDP
# `src/gsrdp.jl` for the arity probe.
#
# The callback runs INSIDE the timed solve, so each trace does the least work it can.
# Scalar traces (`Trace`, `InitialTrace`) reduce the gap to max/mean and keep
# O(iterations) memory; matrix traces (`GapTrace`, `ValueTrace`, and their initial-state
# variants) snapshot full copies and trade memory for fidelity. A `TraceExecutor` fans
# the single solve callback out to a collection of traces, so a run records exactly the
# data it asked for and nothing more.
#
# The interval value function carries lower/upper bounds with `gap(vf) ==
# abs.(vf.lower.current .- vf.upper.current)`. `gap(vf)` allocates a fresh array each
# call (the solver mutates one vf in place, so snapshots MUST copy).
#
# To keep the hot path allocation-free, the executor computes the gap ONCE per iteration
# into a buffer it reuses across iterations, stamps a single wall-clock, and hands both to
# each trace's `record!(trace, vf, gap, wall, bellman_updates, seq)` method. Scalar traces then
# only reduce the shared gap (zero allocation after iteration 1); matrix/value traces copy
# from it (the snapshot copy is inherent). Each trait also keeps its public functor
# `trace(vf, bellman_updates)` as a thin shim (computes its own gap) for standalone use.

abstract type AbstractTrace end

# --- censoring -------------------------------------------------------------

"""
    CellTimeout(elapsed_s, timeout_s, bellman_updates, gap_max)

Thrown by a `TraceExecutor` whose per-solve deadline has passed. A solve is a tight
compute loop that cannot be interrupted from outside, so the trace callback — which
already runs once per Bellman iteration — is the only place a deadline can be enforced.

The throw unwinds out of `solve`; `_run_cell` catches it and emits a `status="timeout"`
row, keeping the partial trace as a censored convergence curve. Timing is therefore
*data* rather than absence: without this, a cluster wall-clock kill leaves no record and
the analysis silently reduces around the hole.
"""
struct CellTimeout <: Exception
    elapsed_s::Float64
    timeout_s::Float64
    bellman_updates::Int
    gap_max::Float64
end

Base.showerror(io::IO, e::CellTimeout) = print(io,
    "CellTimeout: solve exceeded timeout_s=$(e.timeout_s) after $(round(e.elapsed_s, digits=3))s ",
    "($(e.bellman_updates) Bellman updates, gap_max=$(e.gap_max))")

# --- scalar gap traces -----------------------------------------------------

"""
    Trace()

Per-iteration `gap_max`/`gap_mean` over ALL states, reduced inline. Lightweight:
O(iterations) memory.
"""
mutable struct Trace <: AbstractTrace
    wallclock::Vector{Float64}
    bellman_updates::Vector{Int}
    gap_max::Vector{Float64}
    gap_mean::Vector{Float64}
end
Trace() = Trace(Float64[], Int[], Float64[], Float64[])

function record!(t::Trace, vf::IntervalMDP.IntervalValueFunction, g, wall::Float64, bu::Int, seq)
    push!(t.gap_max, maximum(g))
    push!(t.gap_mean, mean(g))
    push!(t.bellman_updates, bu)
    push!(t.wallclock, wall)
    return nothing
end
(t::Trace)(vf::IntervalMDP.IntervalValueFunction, bellman_updates::Integer) =
    record!(t, vf, gap(vf), time_ns() / 1e9, Int(bellman_updates), nothing)

"""
    InitialTrace(; initial_states)

Like `Trace` but reduced over the model's initial states only. Pushes `NaN` when the
model declares no initial states.
"""
mutable struct InitialTrace <: AbstractTrace
    wallclock::Vector{Float64}
    bellman_updates::Vector{Int}
    gap_max::Vector{Float64}
    gap_mean::Vector{Float64}
    initial_states::Vector{Int32}
end
InitialTrace(; initial_states::AbstractVector{<:Integer}=Int32[]) =
    InitialTrace(Float64[], Int[], Float64[], Float64[], Int32.(initial_states))

function record!(t::InitialTrace, vf::IntervalMDP.IntervalValueFunction, g, wall::Float64, bu::Int, seq)
    if isempty(t.initial_states)
        push!(t.gap_max, NaN)
        push!(t.gap_mean, NaN)
    else
        gi = @view g[t.initial_states]
        push!(t.gap_max, maximum(gi))
        push!(t.gap_mean, mean(gi))
    end
    push!(t.bellman_updates, bu)
    push!(t.wallclock, wall)
    return nothing
end
(t::InitialTrace)(vf::IntervalMDP.IntervalValueFunction, bellman_updates::Integer) =
    record!(t, vf, gap(vf), time_ns() / 1e9, Int(bellman_updates), nothing)

# --- full gap-matrix traces ------------------------------------------------

# A fresh, independent `Vector{Float64}` copy of `x` (any AbstractArray). Matrix traces
# MUST copy: the executor reuses one gap buffer across iterations, and `convert` would
# alias it when the buffer is already a `Vector{Float64}`.
_snapshot(x) = copyto!(Vector{Float64}(undef, length(x)), x)

"""
    GapTrace()

Snapshots the entire per-state gap vector every iteration. Heavy: O(iterations × states).
"""
mutable struct GapTrace <: AbstractTrace
    wallclock::Vector{Float64}
    bellman_updates::Vector{Int}
    gaps::Vector{Vector{Float64}}
end
GapTrace() = GapTrace(Float64[], Int[], Vector{Float64}[])

function record!(t::GapTrace, vf::IntervalMDP.IntervalValueFunction, g, wall::Float64, bu::Int, seq)
    push!(t.gaps, _snapshot(g))
    push!(t.bellman_updates, bu)
    push!(t.wallclock, wall)
    return nothing
end
(t::GapTrace)(vf::IntervalMDP.IntervalValueFunction, bellman_updates::Integer) =
    record!(t, vf, gap(vf), time_ns() / 1e9, Int(bellman_updates), nothing)

"""
    InitialGapTrace(; initial_states)

Like `GapTrace` but snapshots only the initial-state slice of the gap vector.
"""
mutable struct InitialGapTrace <: AbstractTrace
    wallclock::Vector{Float64}
    bellman_updates::Vector{Int}
    gaps::Vector{Vector{Float64}}
    initial_states::Vector{Int32}
end
InitialGapTrace(; initial_states::AbstractVector{<:Integer}=Int32[]) =
    InitialGapTrace(Float64[], Int[], Vector{Float64}[], Int32.(initial_states))

function record!(t::InitialGapTrace, vf::IntervalMDP.IntervalValueFunction, g, wall::Float64, bu::Int, seq)
    push!(t.gaps, isempty(t.initial_states) ? Float64[] :
                  _snapshot(@view g[t.initial_states]))
    push!(t.bellman_updates, bu)
    push!(t.wallclock, wall)
    return nothing
end
(t::InitialGapTrace)(vf::IntervalMDP.IntervalValueFunction, bellman_updates::Integer) =
    record!(t, vf, gap(vf), time_ns() / 1e9, Int(bellman_updates), nothing)

# --- value-function traces -------------------------------------------------

"""
    ValueTrace()

Snapshots the lower and upper value-function vectors every iteration. Heavy.
"""
mutable struct ValueTrace <: AbstractTrace
    wallclock::Vector{Float64}
    bellman_updates::Vector{Int}
    lower::Vector{Vector{Float64}}
    upper::Vector{Vector{Float64}}
end
ValueTrace() = ValueTrace(Float64[], Int[], Vector{Float64}[], Vector{Float64}[])

# Value traces snapshot the bounds directly and ignore the shared gap buffer `g`. The
# solver mutates `vf.*.current` in place, so these MUST copy (see `_snapshot`).
function record!(t::ValueTrace, vf::IntervalMDP.IntervalValueFunction, g, wall::Float64, bu::Int, seq)
    push!(t.lower, _snapshot(vf.lower.current))
    push!(t.upper, _snapshot(vf.upper.current))
    push!(t.bellman_updates, bu)
    push!(t.wallclock, wall)
    return nothing
end
(t::ValueTrace)(vf::IntervalMDP.IntervalValueFunction, bellman_updates::Integer) =
    record!(t, vf, nothing, time_ns() / 1e9, Int(bellman_updates), nothing)

"""
    InitialValueTrace(; initial_states)

Like `ValueTrace` but snapshots only the initial-state slice of each bound.
"""
mutable struct InitialValueTrace <: AbstractTrace
    wallclock::Vector{Float64}
    bellman_updates::Vector{Int}
    lower::Vector{Vector{Float64}}
    upper::Vector{Vector{Float64}}
    initial_states::Vector{Int32}
end
InitialValueTrace(; initial_states::AbstractVector{<:Integer}=Int32[]) =
    InitialValueTrace(Float64[], Int[], Vector{Float64}[], Vector{Float64}[], Int32.(initial_states))

function record!(t::InitialValueTrace, vf::IntervalMDP.IntervalValueFunction, g, wall::Float64, bu::Int, seq)
    if isempty(t.initial_states)
        push!(t.lower, Float64[])
        push!(t.upper, Float64[])
    else
        push!(t.lower, _snapshot(@view vf.lower.current[t.initial_states]))
        push!(t.upper, _snapshot(@view vf.upper.current[t.initial_states]))
    end
    push!(t.bellman_updates, bu)
    push!(t.wallclock, wall)
    return nothing
end
(t::InitialValueTrace)(vf::IntervalMDP.IntervalValueFunction, bellman_updates::Integer) =
    record!(t, vf, nothing, time_ns() / 1e9, Int(bellman_updates), nothing)

# --- update-count traces ---------------------------------------------------
#
# WHERE the solver looked, as opposed to what it computed. The GSRDP callback's third
# argument is the iteration's update sequence — exactly the collection `bellman_v!`
# sweeps — so counting occurrences per state gives the number of backups each state
# received in that iteration. Duplicates are real and are kept: a trajectory sampler that
# revisits a state relaxes it twice.
#
# This is the only trace that needs the three-argument callback form
# (IntervalMDP `src/gsrdp.jl`, `_invoke_callback`). Against a solver that only offers the
# two-argument form, every row is zero rather than an error.

"""
    UpdateTrace()

Per-iteration count of Bellman backups per state. Heavy in the same sense as
`GapTrace` — O(iterations × states) — but `Int32` rather than `Float64`, and its CSV is
far smaller because the entries are small integers rather than full-precision floats.

A full sweep records all-ones rows; that is the correct picture of a sweep, not a bug.
"""
mutable struct UpdateTrace <: AbstractTrace
    wallclock::Vector{Float64}
    bellman_updates::Vector{Int}
    counts::Vector{Vector{Int32}}
end
UpdateTrace() = UpdateTrace(Float64[], Int[], Vector{Int32}[])

# Function barrier: `seq` arrives as `Union{Nothing,StateIterator{...}}` (it is `nothing`
# on the pre-update callback), so the tight loop is kept out of the abstractly-typed
# caller. `li` maps the solver's `CartesianIndex` states onto the flat state index the
# trace CSVs are keyed by; for the flat IMDPs this harness runs it is the identity.
function _count_updates!(c::Vector{Int32}, li, seq)
    for s in seq
        i = li[s]
        @inbounds c[i] += one(Int32)
    end
    return c
end

function record!(t::UpdateTrace, vf::IntervalMDP.IntervalValueFunction, g, wall::Float64,
                 bu::Int, seq)
    c = zeros(Int32, length(vf.lower.current))
    seq === nothing || _count_updates!(c, LinearIndices(vf.lower.current), seq)
    push!(t.counts, c)
    push!(t.bellman_updates, bu)
    push!(t.wallclock, wall)
    return nothing
end
(t::UpdateTrace)(vf::IntervalMDP.IntervalValueFunction, bellman_updates::Integer) =
    record!(t, vf, nothing, time_ns() / 1e9, Int(bellman_updates), nothing)
(t::UpdateTrace)(vf::IntervalMDP.IntervalValueFunction, bellman_updates::Integer, seq) =
    record!(t, vf, nothing, time_ns() / 1e9, Int(bellman_updates), seq)

# --- executor --------------------------------------------------------------

"""
    TraceExecutor(traces)

A single `solve` callback that fans out to a collection of traces. It computes the gap
ONCE per iteration into a reused buffer, stamps one wall-clock, and hands both to each
trace's `record!`. Traces are held in a concrete `Tuple` and the per-iteration work runs
behind function barriers, so the scalar-trace path allocates nothing after the first call.

`deadline` is an absolute `time_ns()/1e9` stamp past which the executor throws
`CellTimeout` (`Inf` = never). The check reuses the wall-clock the executor already
stamps, so enforcing a per-solve budget costs one `Float64` compare per iteration.
Note the executor fires on every solve even when it holds no traces at all, which is
what makes the deadline universal rather than conditional on `[output].save_trace`.

`thin_rtol > 0` thins the SCALAR traces (`Trace`, `InitialTrace`) as they record, with
the relative-gap rule of `thinning.jl` (`[output].trace_thin_rtol`). A sampled solver
fires the callback every few Bellman updates, so an unthinned run can hold tens of
millions of rows — gigabytes in memory and on disk — for a curve the analysis reads at
~100 points. The keep/drop decision is made ONCE per callback for both scalar traces,
because `write_report_trace_csv` pairs their rows by index. The newest row is always
stored, so `final_gap`, `final_bellman_updates` and `CellTimeout` are unaffected; the
matrix traces (`GapTrace`, `ValueTrace`, `UpdateTrace`) are never thinned — the videos
need every iteration. `ncalls` counts callbacks, which is what `final_iteration_count`
reads once the record count no longer equals it.
"""
mutable struct TraceExecutor{T<:Tuple}
    traces::T
    needs_gap::Bool
    # The states whose gap the traces read, when every trace that reads the gap is an
    # initial-state trace over these same states; `nothing` means all states. See
    # `_gap_scope`.
    gap_scope::Union{Nothing,Vector{Int}}
    gapbuf::Vector{Float64}   # reused across iterations; sized to the state count on first use
    deadline::Float64         # absolute wall-clock (time_ns()/1e9); Inf disables
    timeout_s::Float64        # the budget `deadline` was derived from; reporting only
    thin::Union{Nothing,GapThinner}   # scalar-trace thinning; `nothing` records every call
    pending::Bool             # the scalar traces' newest row is a not-yet-kept tail
    ncalls::Int               # callbacks received (== iterations + 1)
end
function TraceExecutor(traces; deadline::Real=Inf, timeout_s::Real=Inf, thin_rtol::Real=0.0)
    tup = Tuple(traces)
    thin = thin_rtol > 0 ? GapThinner(thin_rtol, length(tup)) : nothing
    return TraceExecutor{typeof(tup)}(tup, any(_needs_gap, traces), _gap_scope(traces),
                                      Float64[], Float64(deadline), Float64(timeout_s),
                                      thin, false, 0)
end

# Which traces consume the per-state gap (vs. snapshotting the bounds directly).
_needs_gap(::AbstractTrace) = false
_needs_gap(::Union{Trace,InitialTrace,GapTrace,InitialGapTrace}) = true

# The gap is filled in once per iteration for every trace, so it costs O(|S|) per
# iteration even when the traces only read the initial states, which is what
# `[output].traces = []` with `track_initial` records. When every trace that reads the
# gap is an initial-state trace over the same states, only those entries are filled in:
# the traces read `g[initial_states]` and nothing else, so they record the same values.
function _gap_scope(traces)
    readers = [tr for tr in traces if _needs_gap(tr)]
    isempty(readers) && return nothing
    all(tr -> tr isa Union{InitialTrace,InitialGapTrace}, readers) || return nothing
    S0 = readers[1].initial_states
    all(tr -> tr.initial_states == S0, readers) || return nothing
    return Int.(S0)
end

# Fill the reusable gap buffer in place. `l`/`u` are passed positionally so this barrier
# specializes on their concrete array type (the executor's `vf` is abstractly typed),
# keeping the broadcast allocation-free. The buffer is (re)sized via `gap` only when the
# state count changes — i.e. the first call.
function _fill_gap!(buf::Vector{Float64}, l, u, vf)
    length(buf) == length(l) || return collect(Float64, gap(vf))
    @. buf = abs(l - u)
    return buf
end

# `_fill_gap!` restricted to the entries in `scope`; the rest of the buffer is never read.
# The same `abs(l - u)` per entry, so the scoped traces see the same values. (On the very
# first call the full version fills the buffer from `gap(vf) = u - l`, which differs
# from `abs(l - u)` only if `u < l`, i.e. never on the pre-update callback, where the
# bounds are freshly initialized.)
function _fill_gap!(buf::Vector{Float64}, l, u, vf, scope::Vector{Int})
    length(buf) == length(l) || (buf = zeros(Float64, length(l)))
    @inbounds for i in scope
        buf[i] = abs(l[i] - u[i])
    end
    return buf
end
_fill_gap!(buf::Vector{Float64}, l, u, vf, ::Nothing) = _fill_gap!(buf, l, u, vf)

# Barrier: with `traces` a concrete `Tuple` and `g` concretely typed, the loop unrolls to
# static `record!` dispatches — no boxing of `wall`/`bu`, no per-trace dynamic dispatch.
@inline function _record_all!(traces::Tuple, vf, g, wall::Float64, bu::Int, seq)
    for tr in traces
        record!(tr, vf, g, wall, bu, seq)
    end
    return nothing
end

# --- scalar-trace thinning (see the TraceExecutor docstring) -----------------
# The key gap of each trace in the tuple: the newest `gap_max` of a scalar trace, NaN for
# every other trace (a NaN key never triggers a keep). A tuple, so this allocates nothing.
_thin_key(t::Union{Trace,InitialTrace}) = isempty(t.gap_max) ? NaN : @inbounds t.gap_max[end]
_thin_key(::AbstractTrace) = NaN

# Drop the second-to-last row — the previous tail, once the newest row shows it was
# neither an anchor nor a corner. Allocation-free: shift the newest row down, then pop.
function _drop_prev!(v::Vector)
    n = length(v)
    n >= 2 || return v
    @inbounds v[n-1] = v[n]
    pop!(v)
    return v
end
_drop_prev!(t::Union{Trace,InitialTrace}) =
    (_drop_prev!(t.wallclock); _drop_prev!(t.bellman_updates);
     _drop_prev!(t.gap_max); _drop_prev!(t.gap_mean); nothing)
_drop_prev!(::AbstractTrace) = nothing

function _thin_scalars!(t::TraceExecutor, th::GapThinner)
    if observe!(th, map(_thin_key, t.traces))
        t.pending = false            # a keep: the previous tail (if any) is its corner
    else
        t.pending && foreach(_drop_prev!, t.traces)
        t.pending = true
    end
    return nothing
end

# The solver calls whichever arity it supports; defining BOTH is what makes the fork's
# `applicable(callback, vf, bu, seq)` probe pick the three-argument form (IntervalMDP
# `src/gsrdp.jl`, `_invoke_callback`). Against a two-argument-only solver the executor
# still runs — an `UpdateTrace` then simply records zeros.
(t::TraceExecutor)(vf::IntervalMDP.IntervalValueFunction, bellman_updates::Integer) =
    t(vf, bellman_updates, nothing)

function (t::TraceExecutor)(vf::IntervalMDP.IntervalValueFunction,
                            bellman_updates::Integer, seq)
    wall = time_ns() / 1e9
    bu = Int(bellman_updates)
    t.ncalls += 1
    if t.needs_gap
        t.gapbuf = _fill_gap!(t.gapbuf, vf.lower.current, vf.upper.current, vf, t.gap_scope)
        _record_all!(t.traces, vf, t.gapbuf, wall, bu, seq)
    else
        _record_all!(t.traces, vf, nothing, wall, bu, seq)
    end
    th = t.thin
    th === nothing || _thin_scalars!(t, th)
    # Record FIRST, then check: the partial trace's last row is then the state at the
    # moment of censoring, which is what makes it usable as a censored curve.
    wall > t.deadline && _throw_timeout(t, wall, bu)
    return nothing
end

# Out-of-line so the hot path carries only the compare, not the exception construction.
@noinline function _throw_timeout(t::TraceExecutor, wall::Float64, bu::Int)
    gmax, _ = final_gap(t)
    elapsed = isfinite(t.timeout_s) ? wall - (t.deadline - t.timeout_s) : NaN
    throw(CellTimeout(elapsed, t.timeout_s, bu, gmax))
end

# --- factory ---------------------------------------------------------------
# name => (S0) -> AbstractTrace. Initial variants close over the model's initial
# states; all-states variants ignore S0.
const TRACE_BUILDERS = Dict{String,Function}(
    "trace"         => S0 -> Trace(),
    "initial"       => S0 -> InitialTrace(; initial_states=S0),
    "gap"           => S0 -> GapTrace(),
    "initial_gap"   => S0 -> InitialGapTrace(; initial_states=S0),
    "value"         => S0 -> ValueTrace(),
    "initial_value" => S0 -> InitialValueTrace(; initial_states=S0),
    "updates"       => S0 -> UpdateTrace(),
)

"Whether `name` is a recognized trace kind (used by config validation)."
known_trace_kind(name::AbstractString) = haskey(TRACE_BUILDERS, lowercase(strip(String(name))))

"Sorted list of valid trace-kind names, for error messages."
trace_kinds() = sort!(collect(keys(TRACE_BUILDERS)))

"Build a single trace of `kind`, given the model's initial states `S0`."
function build_trace(kind::AbstractString, S0::AbstractVector{<:Integer})
    k = lowercase(strip(String(kind)))
    haskey(TRACE_BUILDERS, k) || error("unknown trace kind '$kind' (expected one of $(trace_kinds()))")
    return TRACE_BUILDERS[k](S0)
end

"""
    build_trace_executor(kinds, S0; track_initial=false, timeout_s=Inf, thin_rtol=0.0) -> TraceExecutor

Build one trace per (de-duplicated) entry of `kinds`. When `track_initial`, the
`"initial"` scalar trace is included even if absent from `kinds` (back-compat).

`timeout_s` starts a per-solve deadline HERE, at construction — callers build the
executor immediately before entering the timed region, so this brackets the solve to
within the setup tail. Any finite value makes the executor throw `CellTimeout` once the
budget is spent; the returned executor still holds everything recorded up to that point.
Only `Inf` disables the deadline — `0` means "no budget left" and censors at the first
callback. (`[benchmark].timeout_s = 0` means *disabled*, but `run_experiment` maps that
to `Inf` before it ever reaches here.)

`thin_rtol > 0` thins the scalar traces as they record (`[output].trace_thin_rtol`; see
the `TraceExecutor` docstring). `0` records every callback.
"""
function build_trace_executor(kinds, S0::AbstractVector{<:Integer}; track_initial::Bool=false,
                              timeout_s::Real=Inf, thin_rtol::Real=0.0)
    names = String[lowercase(strip(String(k))) for k in kinds]
    track_initial && push!(names, "initial")
    seen = String[]
    for n in names
        n in seen || push!(seen, n)
    end
    deadline = isfinite(timeout_s) ? time_ns() / 1e9 + timeout_s : Inf
    return TraceExecutor(AbstractTrace[build_trace(n, S0) for n in seen];
                         deadline=deadline, timeout_s=timeout_s, thin_rtol=thin_rtol)
end

# --- final gap (for quality) -----------------------------------------------
# `(gap_max, gap_mean)` of the last recorded iteration, or `(NaN, NaN)` if the trace
# can't supply one. Used to derive convergence; see `final_gap(::TraceExecutor)`.

final_gap(t::Trace) = isempty(t.gap_max) ? (NaN, NaN) : (t.gap_max[end], t.gap_mean[end])
final_gap(t::InitialTrace) = isempty(t.gap_max) ? (NaN, NaN) : (t.gap_max[end], t.gap_mean[end])

function final_gap(t::Union{GapTrace,InitialGapTrace})
    (isempty(t.gaps) || isempty(t.gaps[end])) && return (NaN, NaN)
    return (maximum(t.gaps[end]), mean(t.gaps[end]))
end

# An update trace carries no value information at all. It still needs a method: the
# `final_gap(::TraceExecutor)` scan below walks every trace in the tuple.
final_gap(::UpdateTrace) = (NaN, NaN)

function final_gap(t::Union{ValueTrace,InitialValueTrace})
    (isempty(t.lower) || isempty(t.lower[end])) && return (NaN, NaN)
    d = abs.(t.lower[end] .- t.upper[end])
    return (maximum(d), mean(d))
end

"""
    final_gap(executor) -> (gap_max, gap_mean)

Final `(gap_max, gap_mean)` from the first trace that can supply a real (non-NaN) one,
preferring the all-states `Trace`, then `InitialTrace`, then the rest. `(NaN, NaN)` if
none recorded a usable gap.
"""
function final_gap(ex::TraceExecutor)
    order(tr) = tr isa Trace ? 0 : tr isa InitialTrace ? 1 : 2
    for tr in sort(collect(AbstractTrace, ex.traces); by=order)
        g = final_gap(tr)
        isnan(g[1]) || return g
    end
    return (NaN, NaN)
end

"Number of iterations recorded by a trace."
nrecorded(t::AbstractTrace) = length(t.bellman_updates)

"Whether any trace in the executor recorded at least one iteration."
has_records(ex::TraceExecutor) = any(nrecorded(tr) > 0 for tr in ex.traces)

"""
    final_bellman_updates(executor) -> Int | missing

The last `bellman_updates` count any trace recorded — how much work a solve had done
when it stopped. This is the ONLY way to obtain the true count: GSRDP accumulates it
internally but `_gsrdp!` returns only the iteration counter `k`, so the solution
carries no update count. Used for every row, censored or not; `missing` when nothing
was recorded (an executor with no traces at all).
"""
function final_bellman_updates(ex::TraceExecutor)
    best = missing
    for tr in ex.traces
        nrecorded(tr) == 0 && continue
        bu = Int(tr.bellman_updates[end])
        (best === missing || bu > best) && (best = bu)
    end
    return best
end

"""
    final_iteration_count(executor) -> Int | missing

Iterations completed. GSRDP fires one callback per iteration plus one before the loop
starts (with `bellman_updates = 0` and no update sequence, before sampling anything),
hence `ncalls - 1`. Verified against completed runs, where this reproduces the
solution's `num_iterations` exactly. The callback count is used rather than a trace's
record count because thinning makes the latter smaller; an executor whose traces were
fed directly (never called) falls back to the longest trace's `nrecorded - 1`.

Used for censored runs, which have no solution to ask. `missing` when nothing was
recorded.
"""
function final_iteration_count(ex::TraceExecutor)
    ex.ncalls > 0 && has_records(ex) && return max(ex.ncalls - 1, 0)
    best = missing
    for tr in ex.traces
        n = nrecorded(tr)
        n == 0 && continue
        (best === missing || n > best) && (best = n)
    end
    return best === missing ? missing : max(best - 1, 0)
end

# --- CSV writers -----------------------------------------------------------
# Wall-clock is zeroed to the run start in every writer.

_zeroed_wall(wallclock) = (w = copy(wallclock); isempty(w) || (w .-= w[1]); w)

"""
    write_trace_csv(trace, path)

Write a single trace to CSV. Scalar traces emit `gap_max`/`gap_mean` columns (NaN rows
dropped); matrix traces emit one column per state.
"""
function write_trace_csv(t::Trace, path::AbstractString)
    mkpath(dirname(path))
    wall = _zeroed_wall(t.wallclock)
    open(path, "w") do io
        println(io, "wallclock_s,bellman_updates,gap_max,gap_mean")
        for i in eachindex(t.bellman_updates)
            isnan(t.gap_max[i]) && continue
            println(io, wall[i], ',', t.bellman_updates[i], ',', t.gap_max[i], ',', t.gap_mean[i])
        end
    end
    return path
end

function write_trace_csv(t::InitialTrace, path::AbstractString)
    mkpath(dirname(path))
    wall = _zeroed_wall(t.wallclock)
    open(path, "w") do io
        println(io, "wallclock_s,bellman_updates,initial_gap_max,initial_gap_mean")
        for i in eachindex(t.bellman_updates)
            isnan(t.gap_max[i]) && continue
            println(io, wall[i], ',', t.bellman_updates[i], ',', t.gap_max[i], ',', t.gap_mean[i])
        end
    end
    return path
end

# Wide matrix CSV: header `wallclock_s,bellman_updates,<prefix>[1],...,<prefix>[n]`,
# one row per iteration. `n` is taken from the first snapshot.
function _write_matrix_csv(path, wallclock, bellman_updates, snapshots, prefix)
    mkpath(dirname(path))
    wall = _zeroed_wall(wallclock)
    n = isempty(snapshots) ? 0 : length(snapshots[1])
    open(path, "w") do io
        print(io, "wallclock_s,bellman_updates")
        for k in 1:n
            print(io, ',', prefix, '[', k, ']')
        end
        println(io)
        for i in eachindex(bellman_updates)
            print(io, wall[i], ',', bellman_updates[i])
            for v in snapshots[i]
                print(io, ',', v)
            end
            println(io)
        end
    end
    return path
end

write_trace_csv(t::GapTrace, path::AbstractString) =
    _write_matrix_csv(path, t.wallclock, t.bellman_updates, t.gaps, "gap")
write_trace_csv(t::InitialGapTrace, path::AbstractString) =
    _write_matrix_csv(path, t.wallclock, t.bellman_updates, t.gaps, "gap")
write_trace_csv(t::UpdateTrace, path::AbstractString) =
    _write_matrix_csv(path, t.wallclock, t.bellman_updates, t.counts, "n")

# Value traces write two files: `*_lower.csv` and `*_upper.csv`. `path` is the base
# (e.g. ".../value_trace.csv"); the suffix is inserted before the extension.
function _bound_path(path, suffix)
    base, ext = splitext(String(path))
    return base * suffix * ext
end

function write_trace_csv(t::Union{ValueTrace,InitialValueTrace}, path::AbstractString)
    _write_matrix_csv(_bound_path(path, "_lower"), t.wallclock, t.bellman_updates, t.lower, "v")
    _write_matrix_csv(_bound_path(path, "_upper"), t.wallclock, t.bellman_updates, t.upper, "v")
    return path
end

"""
    write_report_trace_csv(executor, path)

Write the report-facing legacy trace CSV: `wallclock_s,bellman_updates,gap_max,gap_mean`
from the all-states `Trace`, plus `initial_gap_max,initial_gap_mean` when an
`InitialTrace` is present. Returns `path`, or `nothing` when the executor holds no
scalar `Trace`/`InitialTrace` to source these columns from.
"""
function write_report_trace_csv(ex::TraceExecutor, path::AbstractString)
    base = nothing; init = nothing
    for tr in ex.traces
        tr isa Trace && (base = tr)
        tr isa InitialTrace && (init = tr)
    end
    (base === nothing && init === nothing) && return nothing
    mkpath(dirname(path))

    # Drive rows off whichever scalar trace exists (they share iteration count/order).
    src = base === nothing ? init : base
    wall = _zeroed_wall(src.wallclock)
    have_initial = init !== nothing
    open(path, "w") do io
        if have_initial
            println(io, "wallclock_s,bellman_updates,gap_max,gap_mean,initial_gap_max,initial_gap_mean")
        else
            println(io, "wallclock_s,bellman_updates,gap_max,gap_mean")
        end
        for i in eachindex(src.bellman_updates)
            gmax  = base === nothing ? NaN : base.gap_max[i]
            gmean = base === nothing ? NaN : base.gap_mean[i]
            # When only the initial trace exists, populate gap_max/gap_mean from it so the
            # report still has a curve to draw.
            if base === nothing
                gmax = init.gap_max[i]; gmean = init.gap_mean[i]
            end
            (isnan(gmax) && (!have_initial || isnan(init.gap_max[i]))) && continue
            if have_initial
                println(io, wall[i], ',', src.bellman_updates[i], ',', gmax, ',', gmean, ',',
                        init.gap_max[i], ',', init.gap_mean[i])
            else
                println(io, wall[i], ',', src.bellman_updates[i], ',', gmax, ',', gmean)
            end
        end
    end
    return path
end

# Filename for each heavy trace's own CSV (written alongside the report-facing
# trace.csv). Scalar Trace/InitialTrace are captured by the report CSV, not here.
trace_filename(::GapTrace)          = "gap_trace.csv"
trace_filename(::InitialGapTrace)   = "initial_gap_trace.csv"
trace_filename(::ValueTrace)        = "value_trace.csv"
trace_filename(::InitialValueTrace) = "initial_value_trace.csv"
trace_filename(::UpdateTrace)       = "update_trace.csv"

# --- policy -----------------------------------------------------------------
#
# The solved strategy, written once per completed sample. Every benchmark problem is a
# `ControlSynthesisProblem` (`file_sources.jl` builds them with `control_synthesis=true`),
# so `solve` returns a `ControlSynthesisSolution` carrying the policy GSRDP converged on —
# which the harness otherwise drops on the floor when the cell returns.
#
# Unlike the trace matrices this is one row per STATE, not per iteration: a few kilobytes
# against their gigabytes, which is why `[output].save_policy` defaults to on.
#
# The sentinel matters. `_gsrdp_strategy_cache` (IntervalMDP `src/gsrdp.jl`) fills the
# strategy array with action 0 and `extract_strategy!` only overwrites a state the solver
# actually backed up, so on a sampled variant every state the sampler never looked at
# still reads 0 on the way out. That is recorded verbatim rather than remapped or
# dropped — it is exactly what the policy figure exists to show.

# The action-variable shape of a problem's model, for flattening an action tuple to a
# single index. `nothing` when the model does not answer, which sends the writer down the
# 1-D path (correct for every flat IMDP, i.e. all of them today).
function _policy_action_shape(bp)
    try
        return IntervalMDP.action_shape(IntervalMDP.system(bp.problem))
    catch
        return nothing
    end
end

# How many rows a policy should have: one per state of the MODEL, which is not the same
# as one per entry of the strategy array.
#
# The strategy is sized by `source_shape`, the states the solver makes a DECISION in; the
# value function, and therefore every other per-state artifact in the run dir
# (`gap_trace.csv`, `value_trace_*.csv`), is sized by `num_states`. For the grid
# abstractions those differ by exactly the appended absorbing "left the region" state:
# `car_parking_S10_I3` has `num_states = 101` and `source_shape = (100,)`, and the
# absorbing state is index 101, i.e. last. Padding the tail with action 0 therefore keeps
# row `s` = state `s` while making policy.csv the same width as its neighbours, so the
# files can be read side by side without an off-by-one per problem family.
#
# Only for a 1-D `source_shape`. A factored or product model's states are not a flat
# range and padding a tail would be meaningless, so those write exactly what the strategy
# array holds.
function _policy_num_rows(bp, nstrategy::Integer)
    try
        mp = IntervalMDP.system(bp.problem)
        length(IntervalMDP.source_shape(mp)) == 1 || return nstrategy
        ns = Int(IntervalMDP.num_states(mp))
        return ns > nstrategy ? ns : nstrategy
    catch
        return nstrategy
    end
end

"""
    write_policy_csv(sol, bp, path) -> String | nothing

Write the stationary policy of a solved `ControlSynthesisSolution` to `path` as
`state,action[,a1,…,aN]`, one row per state in flat state order.

`aᵢ` are the action tuple's components verbatim; `action` is their flattened index into
the model's action shape (just `a1` for a flat IMDP, where `N == 1`). An action of `0`
means no action was selected for that state — either the solver never backed it up (see
the note above) or it is one of the decision-free absorbing states past the end of the
strategy array (see [`_policy_num_rows`](@ref)) — and is written as `0` in every column.

The file has one row per state of the model, matching the column count of the per-state
trace matrices written alongside it.

Returns `nothing`, without raising, when there is no policy to write: a
`VerificationProblem` solve, a `NoStrategy`, a time-varying strategy, or any failure to
read the solution. A policy is a nice-to-have; it must never cost the cell its row.
"""
function write_policy_csv(sol, bp, path::AbstractString)
    # A `VerificationSolution` carries no strategy at all. That is a configuration
    # choice (`control_synthesis = false`), not a fault, so it returns quietly rather
    # than warning once per sample.
    (sol !== nothing && hasproperty(sol, :strategy)) || return nothing
    try
        σ = IntervalMDP.strategy(sol)
        σ isa IntervalMDP.StationaryStrategy || return nothing
        A = σ.strategy
        isempty(A) && return nothing

        # Flatten an action tuple to one index. Any zero component means "no action
        # selected", and a shape the model would not give us leaves the 1-D case exact
        # and the multi-variable case reported per component only.
        as = _policy_action_shape(bp)
        N = length(first(A))
        lin = (as !== nothing && length(as) == N && N > 1) ? LinearIndices(Tuple(as)) : nothing
        flatten(t) = any(<=(0), t) ? 0 :
                     N == 1        ? Int(t[1]) :
                     lin !== nothing ? Int(lin[CartesianIndex(Int.(t))]) : 0

        mkpath(dirname(path))
        open(path, "w") do io
            print(io, "state,action")
            if N > 1
                for k in 1:N
                    print(io, ",a", k)
                end
            end
            println(io)
            # Linear indexing, so row s is state s whatever shape the array carries; the
            # tail past the strategy array is the decision-free absorbing state(s), which
            # get the same 0 as a state the solver never reached. Both mean "no action
            # here", and the figure draws them identically.
            zeroed = N > 1 ? ntuple(_ -> Int32(0), N) : (Int32(0),)
            for s in 1:_policy_num_rows(bp, length(A))
                t = s <= length(A) ? A[s] : zeroed
                print(io, s, ',', flatten(t))
                if N > 1
                    for v in t
                        print(io, ',', Int(v))
                    end
                end
                println(io)
            end
        end
        return path
    catch e
        @warn "failed to write policy.csv" path exception=(e, catch_backtrace())
        return nothing
    end
end
