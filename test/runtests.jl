# Synthetic-data tests for the BenchmarkAnalysis suite. These run end-to-end on
# generated data conforming to the canonical schema BEFORE any real data is touched.
# The module is included directly (not via Bench) so the tests do not pull in the
# heavy solver stack.

using Test
using DataFrames
using Statistics
using Random
using CSV
using JSON
import NCDatasets

include(joinpath(@__DIR__, "..", "src", "analysis", "BenchmarkAnalysis.jl"))
using .BenchmarkAnalysis
const BA = BenchmarkAnalysis

# Validation core (TrialBounds / validate_bounds / write_validation_report) is plain
# code with no solver dependency, so include it directly — its only IntervalMDP use is
# inside `extract_trial_bounds`, which these tests do not exercise.
include(joinpath(@__DIR__, "..", "src", "validation.jl"))

const REF = BA.REFERENCE

# --- synthetic canonical long-format builder -------------------------------
# trace for one run: gap decreases and (when solved) hits exactly `eps` at wall == ts.
function _run_rows!(long, alg, prob, ntrans, seed, ts, eps; tnoise=0.0, rng=Random.GLOBAL_RNG)
    if isinf(ts)                                   # right-censored (never reaches eps)
        ws = [0.0, 0.5, 1.0]; gaps = [1.0, 0.5, 0.1]
    else
        f = 1.0 + tnoise * randn(rng)
        ws = [0.0, 0.5ts, ts, 1.4ts] .* f
        gaps = [1.0, sqrt(eps), eps, 0.5eps]
    end
    for (i, (w, g)) in enumerate(zip(ws, gaps))
        push!(long, (alg, prob, ntrans, ntrans, seed, i, w,
                     round(Int, w * 1000), g, !isinf(ts), eps, 0.0, 0.0))
    end
end

function make_long(times::Dict; eps, ntrans::Dict, seeds=3, tnoise=0.0, seed0=1234)
    rng = MersenneTwister(seed0)
    long = DataFrame(algorithm=String[], problem=String[], n_trans=Union{Int,Missing}[],
                     num_states=Union{Int,Missing}[], seed=Int[], sample_idx=Int[],
                     wall_time=Float64[], bellman_updates=Int[], gap=Float64[],
                     run_converged=Bool[], convergence_eps=Float64[],
                     time_budget=Float64[], update_budget=Float64[])
    for (alg, pm) in times, (prob, ts) in pm, s in 1:seeds
        _run_rows!(long, alg, prob, ntrans[prob], s, ts, eps; tnoise=tnoise, rng=rng)
    end
    # per-problem censoring budgets, as load_canonical would set them
    for g in groupby(long, :problem)
        idx = long.problem .== g.problem[1]
        long.time_budget[idx]   .= maximum(g.wall_time)
        long.update_budget[idx] .= maximum(g.bellman_updates)
    end
    return long
end

const EPS = 1e-3
const PROBS = ["p$i" for i in 1:8]
const NTRANS = Dict(p => 100 * 2^(i - 1) for (i, p) in enumerate(PROBS))

# reference: 1s everywhere; "sym": symmetric times (2x / 0.5x); "slow": 4x slower.
sym_t(i) = isodd(i) ? 2.0 : 0.5
const TIMES = Dict(
    REF                       => Dict(p => 1.0 for p in PROBS),
    "gsrdp_sym"               => Dict(p => sym_t(i) for (i, p) in enumerate(PROBS)),
    "gsrdp_slow"              => Dict(p => 4.0 for p in PROBS),
)

@testset "BenchmarkAnalysis synthetic suite" begin
    long = make_long(TIMES; eps=EPS, ntrans=NTRANS, seeds=3, tnoise=0.01)
    rs = BA.run_scalars(long, EPS)
    l1 = BA.level1(rs)

    @testset "normalized reference curve hits gap ≈ ε at τ ≈ 1" begin
        # noise-free single-seed reference run, normalized by its own solve_time
        l = make_long(Dict(REF => Dict("p1" => 1.0)); eps=EPS, ntrans=Dict("p1" => 100),
                      seeds=1, tnoise=0.0)
        Tref = BA.first_crossing(sort(l, :wall_time).wall_time, sort(l, :wall_time).gap, EPS)
        scaled = copy(l); scaled.wall_time = scaled.wall_time ./ Tref
        g_at_1 = BA.interp_gap(scaled, [1.0]; xcol=:wall_time)[1]
        @test isapprox(g_at_1, EPS; rtol=1e-6)
    end

    @testset "cactus line monotone, length == #solved" begin
        dir = mktempdir()
        BA.analysis_cactus(l1, dir; reference=REF)
        ys = sort(collect(skipmissing(l1[l1.algorithm .== "gsrdp_sym", :time_mean])))
        @test issorted(ys)                          # cumulative cactus line is non-decreasing
        n_solved = sum(l1[l1.algorithm .== "gsrdp_sym", :n_solved] .> 0)
        @test length(ys) == n_solved == length(PROBS)   # sym solves every problem
        @test isfile(joinpath(dir, "cactus.png"))
    end

    @testset "cactus whiskers survive seed-std > mean on the log axis" begin
        dir = mktempdir()
        wide = copy(l1)
        wide.time_std = 10 .* coalesce.(wide.time_mean, 0.0)   # every low end would go negative
        wide.upd_std  = 10 .* coalesce.(wide.upd_mean, 0.0)
        BA.analysis_cactus(wide, dir; reference=REF)
        @test isfile(joinpath(dir, "cactus.png"))
    end

    @testset "censoring: timeouts excluded, counts correct" begin
        times = Dict(REF => Dict(p => 1.0 for p in PROBS),
                     "gsrdp_part" => Dict(p => (i <= 5 ? 2.0 : Inf) for (i, p) in enumerate(PROBS)))
        l = make_long(times; eps=EPS, ntrans=NTRANS, seeds=2, tnoise=0.0)
        r = BA.run_scalars(l, EPS)
        part = filter(x -> x.algorithm == "gsrdp_part", r)
        @test count(==(:timeout), part.status) == 3 * 2     # 3 problems × 2 seeds censored
        @test count(==(:converged), part.status) == 5 * 2
        @test all(ismissing, filter(x -> x.status == :timeout, part).solve_time)
        ll1 = BA.level1(r)
        @test sum(ll1[ll1.algorithm .== "gsrdp_part", :n_solved]) == 5 * 2   # 5 problems × 2 seeds
    end

    @testset "full suite runs without error" begin
        dir = mktempdir()
        BA.analysis_normalized_gap_curves(long, l1, dir; reference=REF, eps=EPS)
        BA.analysis_per_problem_gap(long, dir; problems=BA.default_problem_subset(l1))
        BA.analysis_scatter_vs_reference(rs, dir; reference=REF)
        _, fit, fit_u = BA.analysis_vs_transitions(rs, dir)
        @test isfile(joinpath(dir, "normalized_gap_curves.png"))
        @test isfile(joinpath(dir, "scatter_vs_reference.png"))
        @test isfile(joinpath(dir, "vs_transitions.png"))
        @test nrow(fit) == length(unique(rs.algorithm))
        @test nrow(fit_u) == length(unique(rs.algorithm))
    end
end

# --- solution-based validation (TrialBounds / validate_bounds) -------------
# Builds TrialBounds directly (no solver); checks the cross-trial overlap logic,
# scope restriction, tolerance boundary, and the pairwise≡extrema equivalence.
function _bounds(inst, label, lower, upper; scope=Int32[], eps=1e-3, sample=1, seed=missing)
    lo = Float64.(lower); up = Float64.(upper)
    g = isempty(scope) ? (up .- lo) : (up .- lo)[scope]
    gmax = maximum(g)
    return TrialBounds(inst, label, label, sample, seed, lo, up, Int32.(scope), eps, gmax, gmax < eps)
end

# Brute-force pairwise form of check #2, for equivalence testing.
function _pairwise_consistent(tbs, tol)
    for i in eachindex(tbs), j in eachindex(tbs)
        sc = tbs[i].scope
        li = isempty(sc) ? tbs[i].lower : tbs[i].lower[sc]
        ui = isempty(sc) ? tbs[i].upper : tbs[i].upper[sc]
        lj = isempty(sc) ? tbs[j].lower : tbs[j].lower[sc]
        uj = isempty(sc) ? tbs[j].upper : tbs[j].upper[sc]
        (all(li .<= uj .+ tol) && all(ui .>= lj .- tol)) || return false
    end
    return true
end

@testset "solution-based validation" begin
    @testset "overlapping intervals are consistent + converged" begin
        tbs = Dict("p" => [
            _bounds("p", "a", [0.0, 0.5], [5e-4, 0.5005]),
            _bounds("p", "b", [3e-4, 0.5002], [8e-4, 0.5007]),
        ])
        tvals, pvals = validate_bounds(tbs)
        @test all(t -> t.bounds_consistent, tvals)
        @test all(t -> t.converged, tvals)
        @test length(pvals) == 1 && pvals[1].consistent
        @test pvals[1].n_converged == 2
    end

    @testset "disjoint intervals flagged inconsistent (both trials)" begin
        tbs = Dict("p" => [
            _bounds("p", "a", [0.0, 0.5], [5e-4, 0.5005]),
            _bounds("p", "b", [0.01, 0.5], [0.011, 0.5005]),   # state 1 lower far above a's upper
        ])
        tvals, pvals = validate_bounds(tbs)
        @test !pvals[1].consistent
        @test count(t -> !t.bounds_consistent, tvals) == 2     # disagreement is symmetric
        @test all(t -> t.bounds_violation > 0, tvals)
    end

    @testset "scope restriction ignores out-of-scope disagreement" begin
        # Trials agree on state 1, disagree wildly on state 2.
        mk(scope) = Dict("p" => [
            _bounds("p", "a", [0.0, 0.0], [1e-4, 1e-4]; scope=scope),
            _bounds("p", "b", [0.0, 9.0], [1e-4, 9.0]; scope=scope),
        ])
        _, p_init = validate_bounds(mk(Int32[1]))              # scope = state 1 only
        _, p_all  = validate_bounds(mk(Int32[]))               # all states
        @test p_init[1].consistent
        @test !p_all[1].consistent
    end

    @testset "tolerance boundary (2·eps)" begin
        eps = 1e-3; tol = BOUNDS_TOL_FACTOR * eps
        # b's lower sits exactly tol above a's upper ⇒ within tolerance.
        within = Dict("p" => [
            _bounds("p", "a", [0.0], [0.0]; eps=eps),
            _bounds("p", "b", [tol], [tol]; eps=eps),
        ])
        beyond = Dict("p" => [
            _bounds("p", "a", [0.0], [0.0]; eps=eps),
            _bounds("p", "b", [tol + 1e-6], [tol + 1e-6]; eps=eps),
        ])
        @test validate_bounds(within)[2][1].consistent
        @test !validate_bounds(beyond)[2][1].consistent
    end

    @testset "extrema form ≡ pairwise form" begin
        eps = 1e-3; tol = BOUNDS_TOL_FACTOR * eps
        cases = [
            [_bounds("p","a",[0.0,0.5,1.0],[1e-4,0.5004,1.0]),
             _bounds("p","b",[2e-4,0.5001,0.999],[5e-4,0.5006,1.0]),
             _bounds("p","c",[1e-4,0.4999,0.9995],[4e-4,0.5003,1.0])],
            [_bounds("p","a",[0.0],[1e-4]), _bounds("p","b",[0.02],[0.021])],  # disjoint
        ]
        for tbs in cases
            _, p = validate_bounds(Dict("p" => tbs))
            @test p[1].consistent == _pairwise_consistent(tbs, tol)
        end
    end

    @testset "write_validation_report emits csv + json" begin
        dir = mktempdir()
        tbs = Dict("p" => [
            _bounds("p", "a", [0.0, 0.5], [5e-4, 0.5005]; seed=11),
            _bounds("p", "b", [3e-4, 0.5002], [8e-4, 0.5007]; seed=22),
        ])
        tvals, pvals = validate_bounds(tbs)
        csv_path, json_path = write_validation_report(dir, tvals, pvals)
        @test isfile(csv_path) && isfile(json_path)
        df = CSV.read(csv_path, DataFrame)
        @test nrow(df) == 2
        @test Set(names(df)) ⊇ Set(["instance_name","cell_id","alg_label","sample_index",
                                    "gap_max","converged","bounds_violation","bounds_consistent"])
        @test all(df.scope .== "all")
        summary = JSON.parsefile(json_path)
        @test length(summary) == 1 && summary[1]["n_trials"] == 2
    end
end

# ===========================================================================
# Robustness against a killed / censored sweep
# ===========================================================================
# Everything above runs on synthetic in-memory frames. These build real results
# directories on disk, because the failure mode being tested is about FILES: a sweep
# that dies part-way leaves a directory with pieces missing, and the loader used to
# hard-error on exactly those pieces.

# Write a trace.csv in the schema `load_canonical` expects. `initial_gaps` adds the
# initial-state columns a sweep with `[output].track_initial = true` records.
function _write_trace(path, ws, bus, gaps; initial_gaps=nothing)
    mkpath(dirname(path))
    df = DataFrame(wallclock_s=ws, bellman_updates=bus, gap_max=gaps, gap_mean=gaps ./ 2)
    if initial_gaps !== nothing
        df.initial_gap_max  = initial_gaps
        df.initial_gap_mean = initial_gaps ./ 2
    end
    CSV.write(path, df)
    return path
end

# Build a results dir. `cells` is a vector of NamedTuples describing one results.csv row
# each; `expected` is the cell list meta.json claims the sweep would run.
function _make_results_dir(cells; expected=nothing, eps=1e-3, problems_csv=true,
                           datetime_end=true, timeout_s=0.0, restrict_to_initial=false)
    dir = mktempdir()
    rows = NamedTuple[]
    for c in cells
        tp = missing
        if get(c, :trace, nothing) !== nothing
            tp = joinpath("runs", c.cell_id, "sample$(c.sample)", "trace.csv")
            ws, bus, gaps = c.trace
            _write_trace(joinpath(dir, tp), ws, bus, gaps;
                         initial_gaps=get(c, :initial_gaps, nothing))
        end
        push!(rows, (cell_id=c.cell_id, problem="src", instance_name=c.problem,
                     alg_label=c.alg, seed=c.seed, sample_index=c.sample,
                     convergence_eps=eps, converged=get(c, :converged, true),
                     trace_path=tp, status=c.status))
    end
    isempty(rows) || CSV.write(joinpath(dir, "results.csv"), DataFrame(rows))

    if problems_csv
        probs = unique([c.problem for c in cells])
        CSV.write(joinpath(dir, "problems.csv"),
                  DataFrame(problem=probs, num_states=fill(10, length(probs)),
                            num_transitions=[100 * 2^i for i in eachindex(probs)],
                            status=fill("ok", length(probs))))
    end

    meta = Dict{String,Any}("experiment" => "t", "n_cells" => 0, "samples" => 1)
    if expected !== nothing
        meta["cells"] = expected
        meta["n_cells"] = length(expected)
    end
    datetime_end && (meta["datetime_end"] = "2026-01-01 00:00:00")
    open(io -> JSON.print(io, meta, 2), joinpath(dir, "meta.json"), "w")

    if timeout_s > 0 || restrict_to_initial
        open(joinpath(dir, "resolved_config.toml"), "w") do io
            timeout_s > 0 && println(io, "[benchmark]\ntimeout_s = $timeout_s")
            restrict_to_initial && println(io, "[property]\nrestrict_to_initial = true")
        end
    end
    return dir
end

# One ordinary converged cell, as a baseline ingredient for the fixtures below.
_ok_cell(; cell_id="p1__a", problem="p1", alg="a", seed=1, sample=1, eps=1e-3) =
    (cell_id=cell_id, problem=problem, alg=alg, seed=seed, sample=sample, status="ok",
     trace=([0.0, 0.5, 1.0], [0, 500, 1000], [1.0, 0.03, eps / 2]))

@testset "adapter robustness & completeness" begin

    @testset "missing results.csv returns the full empty schema" begin
        dir = mktempdir()
        long = BA.load_canonical(dir)     # must not throw
        @test nrow(long) == 0
        @test Set(names(long)) ⊇ Set(["algorithm", "problem", "n_trans", "num_states",
                                      "seed", "sample_idx", "wall_time", "bellman_updates",
                                      "gap", "run_converged", "convergence_eps", "censored",
                                      "time_budget", "update_budget"])
        # The downstream reduction indexes every one of those columns; an early return
        # that dropped time_budget/update_budget used to KeyError here.
        @test nrow(BA.run_scalars(long, 1e-3)) == 0

        out = BA.run_analyses(dir)
        mf = joinpath(dir, "manifest.txt")
        @test isfile(mf)                                  # written despite having no data
        @test occursin("INCOMPLETE", read(mf, String))
    end

    @testset "missing problems.csv: no n_trans, but figures and fit CSVs still written" begin
        dir = _make_results_dir([_ok_cell(), _ok_cell(cell_id="p1__b", alg="b", seed=2)];
                                problems_csv=false)
        long = BA.load_canonical(dir)
        @test nrow(long) > 0
        @test all(ismissing, long.n_trans)

        BA.run_analyses(dir; reference_label="a")
        @test isfile(joinpath(dir, "plots", "vs_transitions.png"))
        fit = CSV.read(joinpath(dir, "plots", "time_vs_transitions_fit.csv"), DataFrame)
        # The headline scaling exponent is genuinely unavailable — the point is that the
        # CSV says so rather than being an unexplained empty file.
        @test "n_missing_ntrans" in names(fit)
        @test occursin("ABSENT", read(joinpath(dir, "manifest.txt"), String))
    end

    @testset "truncated trace.csv: keep what parsed, count the rest, keep other traces" begin
        dir = _make_results_dir([_ok_cell(), _ok_cell(cell_id="p1__b", alg="b", seed=2)])
        bad = joinpath(dir, "runs", "p1__a", "sample1", "trace.csv")

        # (a) a final line cut mid-row, as a SIGKILL leaves it
        open(bad, "a") do io; print(io, "1.5,1500"); end
        long = BA.load_canonical(dir)
        @test nrow(long) > 0                              # no throw
        @test "b" in long.algorithm                       # the healthy trace still loaded
        @test BA.load_diagnostics().traces_unreadable == 1
        @test all(isfinite, long.wall_time)               # no `missing` leaked through

        # (b) a truncated header — the file is not a trace at all
        write(bad, "wallclo")
        long = BA.load_canonical(dir)
        @test "b" in long.algorithm
        @test !("a" in long.algorithm)
        @test BA.load_diagnostics().traces_unreadable == 1

        # (c) an empty file
        write(bad, "")
        @test "b" in BA.load_canonical(dir).algorithm
    end

    @testset "trace_path pointing at a file that was never written" begin
        dir = _make_results_dir([_ok_cell(), _ok_cell(cell_id="p1__b", alg="b", seed=2)])
        rm(joinpath(dir, "runs", "p1__a", "sample1", "trace.csv"))
        long = BA.load_canonical(dir)
        @test BA.load_diagnostics().traces_missing == 1
        @test "b" in long.algorithm
    end

    @testset "absent reference falls back and says so" begin
        dir = _make_results_dir([_ok_cell(cell_id="p1__b", alg="b", seed=1)])
        out = BA.run_analyses(dir; reference_label="not_a_real_algorithm")
        @test out.reference == "b"
        @test occursin("FALLBACK", read(joinpath(dir, "manifest.txt"), String))
    end

    @testset "normalized_gap_curves with an all-missing reference excludes everything" begin
        long = make_long(TIMES; eps=EPS, ntrans=NTRANS, seeds=2)
        l1 = BA.level1(BA.run_scalars(long, EPS))
        # Blank the reference's solve times: the shape left when the reference cell of
        # every problem was killed.
        allowmissing!(l1, [:time_mean, :upd_mean])
        l1[l1.algorithm .== REF, :time_mean] .= missing
        l1[l1.algorithm .== REF, :upd_mean]  .= missing
        dir = mktempdir()
        _, info = BA.analysis_normalized_gap_curves(long, l1, dir; reference=REF, eps=EPS)
        @test info.excluded == length(unique(long.problem))
        @test isfile(joinpath(dir, "normalized_gap_curves.png"))   # still saved
    end

    @testset "timeout rows: censored runs load, non-run ones are only in the manifest" begin
        eps = 1e-3
        cells = [
            _ok_cell(cell_id="p1__a", seed=1, sample=1),
            _ok_cell(cell_id="p1__a", seed=2, sample=2),
            _ok_cell(cell_id="p1__a", seed=3, sample=3),
            # censored mid-solve: real data, but the gap never reaches eps
            (cell_id="p1__a", problem="p1", alg="a", seed=4, sample=4, status="timeout",
             converged=false, trace=([0.0, 0.5], [0, 500], [1.0, 0.2])),
            # never started: no trace at all, so invisible to every figure
            (cell_id="p1__a", problem="p1", alg="a", seed=5, sample=5,
             status="timeout:not_run", converged=false, trace=nothing),
        ]
        dir = _make_results_dir(cells; expected=["p1__a"], eps=eps, timeout_s=30.0)

        long = BA.load_canonical(dir)
        @test length(unique(long.seed)) == 4        # 4 runs with data, not 5
        @test count(long.censored) > 0
        # timeout_s is the censoring budget once the sweep declares one, replacing the
        # max-observed fallback that a killed run would have distorted.
        @test all(long.time_budget .== 30.0)

        rs = BA.run_scalars(long, eps)
        @test nrow(rs) == 4
        @test count(rs.censored) == 1
        @test rs[rs.censored, :status] == [:timeout]

        l1 = BA.level1(rs)
        @test l1.n_seeds == [4] && l1.n_solved == [3] && l1.n_censored == [1]

        BA.run_analyses(dir; reference_label="a")
        txt = read(joinpath(dir, "manifest.txt"), String)
        @test occursin("timeout:not_run=1", txt)
        @test occursin("censored=1", txt)
    end

    @testset "derived timeout is distinguished from a harness timeout" begin
        # A run that converged to its own eps but not to a stricter analysis eps is
        # :timeout by derivation only — it was never censored.
        dir = _make_results_dir([_ok_cell()])
        long = BA.load_canonical(dir)
        rs = BA.run_scalars(long, 1e-9)             # far stricter than the trace reaches
        @test rs.status == [:timeout]
        @test rs.censored == [false]
    end

    @testset "missing cells are named" begin
        expected = ["p$(i)__a" for i in 1:9]
        cells = [_ok_cell(cell_id="p$(i)__a", problem="p$i", seed=i) for i in 1:6]
        dir = _make_results_dir(cells; expected=expected, datetime_end=false)

        c = BA.load_completeness(dir)
        @test Set(c.missing_cells) == Set(["p7__a", "p8__a", "p9__a"])
        @test c.n_cells == 9
        @test !c.complete                            # no datetime_end => hard kill

        BA.run_analyses(dir; reference_label="a")
        txt = read(joinpath(dir, "manifest.txt"), String)
        @test occursin("SWEEP INCOMPLETE", txt)
        @test occursin("9 expected, 6 present, 3 MISSING", txt)
        @test all(occursin(id, txt) for id in ["p7__a", "p8__a", "p9__a"])
    end

    @testset "a complete sweep gets no incomplete banner" begin
        dir = _make_results_dir([_ok_cell()]; expected=["p1__a"])
        c = BA.load_completeness(dir)
        @test c.complete && isempty(c.missing_cells)
        @test BA.figure_note(c) == ""
        BA.run_analyses(dir; reference_label="a")
        @test !occursin("SWEEP INCOMPLETE", read(joinpath(dir, "manifest.txt"), String))
    end

    # A sweep that converged on its initial states terminates on the INITIAL-state gap,
    # so that is the gap its ε refers to and the one every figure must measure. Reading
    # `gap_max` there made a converged run look right-censored.
    @testset "gap basis follows the sweep's restrict_to_initial" begin
        eps = 1e-3
        # global gap never crosses eps; the initial-state gap does
        cell = (cell_id="p1__a", problem="p1", alg="a", seed=1, sample=1, status="ok",
                trace=([0.0, 0.5, 1.0], [0, 500, 1000], [1.0, 0.5, 0.4]),
                initial_gaps=[1.0, 0.03, eps / 2])

        @testset "restricted sweep reads initial_gap_max" begin
            dir = _make_results_dir([cell]; eps=eps, restrict_to_initial=true)
            long = BA.load_canonical(dir)
            @test long.gap == [1.0, 0.03, eps / 2]
            @test BA.load_diagnostics().gap_basis === :initial

            rs = BA.run_scalars(long, eps)
            @test rs.status == [:converged]              # was :timeout on the global gap
            @test rs.solve_time == [1.0]

            c = BA.load_completeness(dir)
            @test occursin("initial-state gap", join(BA.completeness_lines(c), "\n"))
            @test occursin("gap basis", BA.figure_note(c))
        end

        @testset "restricted sweep without track_initial falls back per trace" begin
            plain = (cell_id="p1__b", problem="p1", alg="b", seed=1, sample=1, status="ok",
                     trace=([0.0, 0.5, 1.0], [0, 500, 1000], [1.0, 0.5, 0.4]))
            dir = _make_results_dir([cell, plain]; eps=eps, restrict_to_initial=true)
            long = BA.load_canonical(dir)
            @test long[long.algorithm .== "a", :gap] == [1.0, 0.03, eps / 2]
            @test long[long.algorithm .== "b", :gap] == [1.0, 0.5, 0.4]
            @test BA.load_diagnostics().gap_basis === :mixed
            @test occursin("all-states gap for the runs that recorded none",
                           join(BA.completeness_lines(BA.load_completeness(dir)), "\n"))
        end

        # A model that declares no initial states records `initial_gap_max = NaN` for
        # every iteration; `extract_quality` falls back to all states there, so the
        # analysis must too rather than plotting a column of NaN.
        @testset "restricted sweep whose models have no initial states falls back" begin
            nan_cell = merge(cell, (initial_gaps=[NaN, NaN, NaN],))
            dir = _make_results_dir([nan_cell]; eps=eps, restrict_to_initial=true)
            long = BA.load_canonical(dir)
            @test long.gap == [1.0, 0.5, 0.4]
            @test BA.load_diagnostics().gap_basis === :global_fallback
            @test BA.load_diagnostics().traces_unreadable == 0
            c = BA.load_completeness(dir)
            txt = join(BA.completeness_lines(c), "\n")
            @test occursin("all-states gap", txt) && occursin("no run recorded an initial-state gap", txt)
            # The plotted quantity is the ordinary all-states gap, so the figures need
            # no stamp — the manifest carries the explanation.
            @test BA.figure_note(c) == ""
        end

        @testset "unrestricted sweep keeps the all-states gap" begin
            for dir in (_make_results_dir([cell]; eps=eps),                          # no config
                        _make_results_dir([cell]; eps=eps, timeout_s=30.0))          # config, no key
                long = BA.load_canonical(dir)
                @test long.gap == [1.0, 0.5, 0.4]
                @test BA.load_diagnostics().gap_basis === :global
                c = BA.load_completeness(dir)
                @test occursin("all-states gap", join(BA.completeness_lines(c), "\n"))
                @test !occursin("gap basis", BA.figure_note(c))
            end
        end
    end

    @testset "real partial run directories in-tree" begin
        # results/smoke_2026-09-16_09-54-18 predates the expected-cell list, so it can
        # only report row counts — but it must still load and report INCOMPLETE rather
        # than throwing on its absent problems.csv.
        for name in readdir(joinpath(@__DIR__, "..", "results"); join=true)
            isdir(name) && isfile(joinpath(name, "results.csv")) || continue
            meta_path = joinpath(name, "meta.json")
            isfile(meta_path) || continue
            haskey(JSON.parsefile(meta_path), "datetime_end") && continue   # finished run
            BA.load_canonical(name)                       # must not throw
            c = BA.load_completeness(name)
            @test !c.complete
            @test !isempty(BA.completeness_lines(c))
            @test occursin("INCOMPLETE", join(BA.completeness_lines(c), "\n"))
        end
    end
end

# ---------------------------------------------------------------------------
# Gap videos (src/analysis/gap_video.jl)
# ---------------------------------------------------------------------------
# The geometry rule is the part with real judgement in it: it decides whether a flat gap
# vector may be drawn as a 2-D picture at all. Getting it wrong is not a crash, it is a
# plausible-looking figure that means nothing, so it is pinned here rather than left to
# the eye. Rendering itself is exercised end-to-end on one small synthetic trace.

# A wide gap_trace.csv with `nstates` columns and `nrows` iterations, whose gap decays
# uniformly — enough to drive a render, not meant to look like a real solve.
function _write_gap_trace(path, nstates, nrows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "wallclock_s,bellman_updates,", join(("gap[$k]" for k in 1:nstates), ','))
        for i in 1:nrows
            print(io, (i - 1) * 0.01, ',', (i - 1) * 100)
            for k in 1:nstates
                print(io, ',', max(0.0, 1.0 - 0.1 * (i - 1) - 1e-4 * k))
            end
            println(io)
        end
    end
    return path
end

# Any wide trace CSV: `wallclock_s,bellman_updates,<prefix>[1],…`. `val(i, k)` is the
# entry for row `i`, state `k`.
function _write_matrix_trace(path, prefix, nstates, nrows, val)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "wallclock_s,bellman_updates,", join(("$prefix[$k]" for k in 1:nstates), ','))
        for i in 1:nrows
            print(io, (i - 1) * 0.01, ',', (i - 1) * 100)
            for k in 1:nstates
                print(io, ',', val(i, k))
            end
            println(io)
        end
    end
    return path
end

# ===========================================================================
# Trace thinning (src/thinning.jl) and the streaming loader built on it
# ===========================================================================
# A sampled arm records a row every few Bellman updates — one real trace.csv held 58M
# rows (6.6 GB) — and loading every row is what used to OOM the machine. The loader now
# keeps a row only when the gap has fallen by a relative `rtol`; these pin the guarantee
# that makes that safe: crossings and interpolated curves move by at most `rtol`.

# Kept row indices of `gaps` under the thinning rule (anchor + corner + last row).
function _thin_idx(gaps, rtol)
    kept = Int[]
    tb = BA.TailBuffer{Int}(BA.GapThinner(rtol, 1))
    for (i, g) in enumerate(gaps)
        BA.offer!(k -> push!(kept, k), tb, i, (g,))
    end
    BA.flush_tail!(k -> push!(kept, k), tb)
    return kept
end

# Monotone gap from 1 to ~1e-5 over `n` rows: plateaus of 997 rows (so the corner rule
# matters), each with a slow sub-`rtol` creep, as a sampled solver's trace looks.
_plateau_gaps(n) = [exp(-11.5 * (i ÷ 997 * 997) / n) * (1 - 1e-7 * (i % 997)) for i in 1:n]

@testset "trace thinning" begin
    n = 200_000; rtol = 1e-3
    gaps = _plateau_gaps(n)
    @assert issorted(gaps; rev=true)

    @testset "GapThinner / TailBuffer rule" begin
        kept = _thin_idx(gaps, rtol)
        @test first(kept) == 1 && last(kept) == n
        @test issorted(kept) && allunique(kept)
        @test length(kept) < n ÷ 50
        @test _thin_idx(gaps, 0.0) == collect(1:n)            # rtol = 0 keeps every row

        # first crossing of any ε lands in [T(ε), T(ε(1 − rtol))]
        for eps in (0.5, 0.1, 1e-2, 1e-3, 1e-4, 2e-5)
            t_true = findfirst(<=(eps), gaps)
            t_thin = kept[findfirst(k -> gaps[k] <= eps, kept)]
            t_lo   = findfirst(<=(eps * (1 - rtol)), gaps)
            @test t_true <= t_thin
            t_lo === nothing || @test t_thin <= t_lo
        end
        # between consecutive kept rows the gap never leaves [g_a(1 − rtol), g_a]: linear
        # interpolation of the kept rows is within relative rtol everywhere
        ok = true
        for (a, b) in zip(kept[1:end-1], kept[2:end]), j in a+1:b-1
            ok &= gaps[a] * (1 - rtol) <= gaps[j] <= gaps[a]
        end
        @test ok

        # NaN gaps neither trigger a keep nor become anchors
        th = BA.GapThinner(0.1, 2)
        @test BA.observe!(th, (1.0, NaN))
        @test !BA.observe!(th, (NaN, NaN))
        @test !BA.observe!(th, (0.95, NaN))
        @test BA.observe!(th, (0.85, NaN))
        @test BA.observe!(th, (0.85, 0.5))                   # a second key's first value
    end

    @testset "load_canonical streams and thins, faithfully" begin
        ws  = collect(range(0.0, 100.0; length=n))
        bus = collect(0:6:6(n-1))
        cells = [(cell_id="p1__a", problem="p1", alg="a", seed=1, sample=1, status="ok",
                  trace=(ws, bus, gaps), initial_gaps=gaps ./ 2)]
        dir = _make_results_dir(cells; restrict_to_initial=true)

        full = BA.load_canonical(dir; thin_rtol=0)
        @test nrow(full) == n
        @test BA.load_diagnostics().rows_kept == n

        long = BA.load_canonical(dir)                         # default rtol = 1e-3
        d = BA.load_diagnostics()
        @test d.rows_read == n && d.rows_kept == nrow(long) && d.thin_rtol == 1e-3
        @test d.gap_basis === :initial && d.traces_unreadable == 0
        @test nrow(long) < n ÷ 50
        @test first(long.sample_idx) == 1 && last(long.sample_idx) == n   # original rows
        @test long.gap == gaps[long.sample_idx] ./ 2                       # initial basis
        @test long.wall_time == ws[long.sample_idx]
        @test long.update_budget[1] == bus[end] && long.time_budget[1] == ws[end]

        # the solve points every analysis reads move by at most one rtol step in ε
        for eps in (1e-2, 1e-3, 1e-4)
            st(l, e) = only(BA.run_scalars(l, e).solve_updates)
            @test st(full, eps) <= st(long, eps) <= st(full, eps * (1 - 1e-3))
        end
        @test occursin("trace rows    : $n read",
                       join(BA.completeness_lines(BA.load_completeness(dir)), "\n"))
    end

    @testset "last_trace_row reads the tail only" begin
        dir = mktempdir()
        p = _write_trace(joinpath(dir, "trace.csv"), collect(0.0:1.0:9999.0),
                         collect(0:9999), _plateau_gaps(10_000))
        r = BA.last_trace_row(p)
        @test r["bellman_updates"] == 9999 && r["wallclock_s"] == 9999.0
        # a window shorter than one row holds no complete line, so it has to widen
        @test BA.last_trace_row(p; window=8)["bellman_updates"] == 9999
        open(p, "a") do io; print(io, "10000.0,10000"); end   # truncated final line
        @test BA.last_trace_row(p)["bellman_updates"] == 9999
        write(p, "wallclock_s,bellman_updates,gap_max,gap_mean\n")
        @test BA.last_trace_row(p) === nothing                # header only
        write(p, "")
        @test BA.last_trace_row(p) === nothing
    end
end

@testset "algorithm groups" begin
    @testset "parse_group_spec" begin
        @test BA.parse_group_spec("eps=traj_centre, traj_eps*,,") ==
              ("eps" => ["traj_centre", "traj_eps*"])
        @test_throws ErrorException BA.parse_group_spec("traj_centre,traj_eps*")  # no '='
        @test_throws ErrorException BA.parse_group_spec("a/b=x")                   # not a dir name
        @test_throws ErrorException BA.parse_group_spec("eps=,")                   # no patterns
    end

    @testset "load_group_file keeps file order" begin
        f = joinpath(mktempdir(), "groups.toml")
        write(f, "zeta = [\"traj_k*\"]\nalpha = [\"traj_centre\", \"traj_eps0?0\"]\n")
        @test BA.load_group_file(f) ==
              ["zeta" => ["traj_k*"], "alpha" => ["traj_centre", "traj_eps0?0"]]
        write(f, "bad = \"traj_k1\"\n")
        @test_throws ErrorException BA.load_group_file(f)
    end

    @testset "resolve_groups" begin
        algs = ["ref", "traj_centre", "traj_eps002", "traj_eps040", "traj_k1", "traj_k16", "x.y"]
        g = @test_logs (:warn, r"matched no algorithm") (:warn, r"no algorithm besides") BA.resolve_groups(
            ["eps" => ["traj_centre", "traj_eps*"],
             "k"   => ["traj_centre", "traj_k?", "nope*"],     # overlaps "eps" on traj_centre
             "lit" => ["x.y"],                                 # '.' is literal, not a wildcard
             "empty" => ["ref"]],                               # reference only -> skipped
            algs, "ref")
        @test [x.name for x in g] == ["eps", "k", "lit"]
        @test g[1].algorithms == ["ref", "traj_centre", "traj_eps002", "traj_eps040"]
        @test g[2].algorithms == ["ref", "traj_centre", "traj_k1"]   # `?` is one character
        @test g[2].unmatched == ["nope*"]
        @test g[3].algorithms == ["ref", "x.y"]
        @test isempty(BA.resolve_groups(["xy" => ["x?y"]], ["xzz"], "ref"))
        @test_throws ErrorException BA.resolve_groups(["a" => ["ref"], "a" => ["ref"]], algs, "ref")
    end

    @testset "run_analyses draws the four figures per group" begin
        cells = [_ok_cell(cell_id="p$(i)__$a", problem="p$i", alg=a, seed=i)
                 for i in 1:3 for a in ("ref", "fast1", "fast2", "slow")]
        dir = _make_results_dir(cells)
        out = BA.run_analyses(dir; reference_label="ref", image_formats=["png"],
                              groups=["fast" => ["fast*"], "mix" => ["fast1", "slow"]])
        for name in ("fast", "mix"), fig in ("cactus", "normalized_gap_curves",
                                             "scatter_vs_reference", "vs_transitions")
            @test isfile(joinpath(dir, "plots", "groups", name, fig * ".png"))
        end
        @test isfile(joinpath(dir, "plots", "groups", "fast", "time_vs_transitions_fit.csv"))
        fit = CSV.read(joinpath(dir, "plots", "groups", "mix", "time_vs_transitions_fit.csv"), DataFrame)
        @test Set(fit.algorithm) == Set(["ref", "fast1", "slow"])
        @test isfile(joinpath(dir, "plots", "cactus.png"))              # full suite still ran
        txt = read(joinpath(dir, "manifest.txt"), String)
        @test occursin("algorithm groups", txt)
        @test occursin("algorithms: fast1, fast2, ref", txt)
        @test BA._FIG_NOTE[] == ""                                      # caption restored

        dir2 = _make_results_dir(cells)
        BA.run_analyses(dir2; reference_label="ref", image_formats=["png"],
                        groups=["fast" => ["fast*"]], only_groups=true)
        @test isfile(joinpath(dir2, "plots", "groups", "fast", "cactus.png"))
        @test !isfile(joinpath(dir2, "plots", "cactus.png"))
        @test !isfile(joinpath(dir2, "manifest.txt"))       # the full suite's is left alone
        @test occursin("algorithm groups",
                       read(joinpath(dir2, "plots", "groups", "manifest.txt"), String))
        @test_throws ErrorException BA.run_analyses(dir2; only_groups=true)
    end
end

@testset "gap videos" begin

    @testset "geometry: the _S<n> token is authoritative" begin
        g, why = BA.grid_geometry("car_parking_S20_I3", 401)
        @test g !== nothing && isempty(why)
        @test g.shape == (20, 20)
        @test g.extra == 1                       # the absorbing "outside" state
        @test g.provenance === :name_split
        @test g.extent == (-10.0, 10.0, -10.0, 10.0)

        # A model with no absorbing state is equally valid.
        g2, _ = BA.grid_geometry("van_der_pol_S40_I10", 1600)
        @test g2.shape == (40, 40) && g2.extra == 0

        # Name and trace disagreeing is reported, never reshaped into something else.
        g3, why3 = BA.grid_geometry("car_parking_S20_I3", 999)
        @test g3 === nothing
        @test occursin("20×20", why3) && occursin("999", why3)
    end

    @testset "geometry: state ordering is column-major over the split" begin
        # The whole picture rests on `s = (j-1)*n + i`. car_parking_S20_I3 splits
        # X = [-10,10]² into 20 cells per axis (width 1.0), and its reach region
        # [4,10]×[-4,0] is therefore cells i = 15…20, j = 7…10 — states 135…140 for
        # j = 7. The stored spec is the independent witness.
        spec = joinpath(@__DIR__, "..", "problems", "car_parking_S20_I3",
                        "car_parking_S20_I3.json")
        if isfile(spec)
            reach = Set(Int(e[1]) for e in JSON.parsefile(spec)["property"]["reach"])
            expected = Set((j - 1) * 20 + i for i in 15:20, j in 7:10)
            @test reach == expected
        end
    end

    @testset "geometry: a non-grid model is refused, whatever its state count" begin
        # PRISM imports have no spatial structure. `robot_delta0.1` is the trap: 100 is a
        # perfect square, so a count-only rule would happily draw it as 10×10.
        for (name, n) in (("coin2_K4", 272), ("robot_delta0.1", 100), ("coin4_K2", 65))
            g, why = BA.grid_geometry(name, n)
            @test g === nothing
            @test occursin("not a known uniform-grid family", why)
        end
    end

    @testset "overlay: flat state index -> grid cell is column-major" begin
        geom, _ = BA.grid_geometry("car_parking_S20_I3", 401)
        # car_parking_S20_I3's stored S0 is [1, 2, 21, 22] — cells (1,1), (2,1), (1,2),
        # (2,2), the 2×2 corner block covering [-10,-8]². If the mapping were row-major
        # this would come out as (1,1), (1,2), (2,1), (2,2)'s transpose and the highlight
        # would sit in the wrong corner for any non-square region.
        m = BA._mask_from_indices([1, 2, 21, 22], geom)
        @test findall(m) == [CartesianIndex(1, 1), CartesianIndex(2, 1),
                             CartesianIndex(1, 2), CartesianIndex(2, 2)]
        @test count(m) == 4

        # The absorbing state (401) is off-grid and must drop out, not wrap around.
        @test count(BA._mask_from_indices([401], geom)) == 0
        @test count(BA._mask_from_indices([0, -3, 100_000], geom)) == 0
    end

    @testset "initial states: problems.csv first, model file as fallback" begin
        dir = mktempdir()
        geom, _ = BA.grid_geometry("car_parking_S20_I3", 401)

        # No problems.csv at all -> nothing (so the caller falls through to the model).
        @test BA._initial_states_from_csv(dir, "car_parking_S20_I3") === nothing

        # A dir written before the column existed -> also nothing, not an error.
        write(joinpath(dir, "problems.csv"), "problem,num_states\ncar_parking_S20_I3,401\n")
        @test BA._initial_states_from_csv(dir, "car_parking_S20_I3") === nothing

        # With the column, the space-separated list round-trips.
        write(joinpath(dir, "problems.csv"),
              "problem,num_states,initial_states\ncar_parking_S20_I3,401,1 2 21 22\np2,9,\n")
        @test BA._initial_states_from_csv(dir, "car_parking_S20_I3") == [1, 2, 21, 22]
        @test BA._initial_states_from_csv(dir, "absent_problem") === nothing

        # One initial state per problem (every PRISM model): CSV.jl types the column
        # Int64, which used to throw and log "unreadable problems.csv" on every render.
        write(joinpath(dir, "problems.csv"),
              "problem,num_states,initial_states\ncar_parking_S20_I3,401,121\np2,9,3\n")
        @test (@test_logs BA._initial_states_from_csv(dir, "car_parking_S20_I3")) == [121]

        # The netCDF fallback reads only the `initial_states` variable.
        nc = joinpath(dir, "m.nc")
        NCDatasets.NCDataset(nc, "c") do ds
            NCDatasets.defDim(ds, "initial_states", 3)
            NCDatasets.defVar(ds, "initial_states", Int32[7, 8, 27], ("initial_states",))
        end
        @test BA._initial_states_from_model(nc) == [7, 8, 27]
        @test BA._initial_states_from_model(nothing) == Int[]
        @test BA._initial_states_from_model(joinpath(dir, "nope.nc")) == Int[]

        # A model with no such variable is "no declared S0", not a failure.
        nc2 = joinpath(dir, "empty.nc")
        NCDatasets.NCDataset(nc2, "c") do ds
            NCDatasets.defDim(ds, "x", 1)
        end
        @test BA._initial_states_from_model(nc2) == Int[]
    end

    @testset "overlay_masks: all three sets, degrading to none when absent" begin
        dir = mktempdir()
        geom, _ = BA.grid_geometry("car_parking_S20_I3", 401)

        # Nothing on disk at all: every mask empty, no throw.
        bare = BA.overlay_masks(dir, "car_parking_S20_I3", geom; problems_root=dir)
        @test !any(bare.reach) && !any(bare.avoid) && !any(bare.initial)

        proot = joinpath(dir, "problems", "car_parking_S20_I3"); mkpath(proot)
        # Reach/avoid come from the spec JSON; the absorbing state 401 is in avoid and
        # must not land on the grid.
        write(joinpath(proot, "car_parking_S20_I3.json"),
              JSON.json(Dict("property" => Dict("reach" => [[135], [136]],
                                                "avoid" => [[401], [215]]))))
        write(joinpath(dir, "problems.csv"),
              "problem,initial_states\ncar_parking_S20_I3,1 2 21 22\n")

        m = BA.overlay_masks(dir, "car_parking_S20_I3", geom;
                             problems_root=joinpath(dir, "problems"))
        @test count(m.reach) == 2 && m.reach[15, 7] && m.reach[16, 7]
        @test count(m.avoid) == 1 && m.avoid[15, 11]     # 401 dropped as off-grid
        @test count(m.initial) == 4 && m.initial[1, 1] && m.initial[2, 2]
    end

    @testset "reads only the head of a long trace" begin
        dir = mktempdir()
        csv = _write_gap_trace(joinpath(dir, "gap_trace.csv"), 26, 500)
        wall, bu, frames, nstates = BA._read_matrix_head(csv, 12)
        @test length(frames) == 12          # stopped early; the other 488 rows unread
        @test nstates == 26
        @test length(wall) == 12 && length(bu) == 12
        @test all(f -> length(f) == 26, frames)
        @test bu[3] == 200
    end

    @testset "a truncated final row is dropped, not fatal" begin
        dir = mktempdir()
        csv = _write_gap_trace(joinpath(dir, "gap_trace.csv"), 26, 5)
        open(csv, "a") do io; print(io, "0.06,600,0.1,0.2"); end   # short row, as a SIGKILL leaves
        _, _, frames, _ = BA._read_matrix_head(csv, 100)
        @test length(frames) == 5
    end

    @testset "panels: each one comes from its own trace file" begin
        dir = mktempdir()
        _write_matrix_trace(joinpath(dir, "value_trace_lower.csv"), "v", 4, 6,
                            (i, k) -> 0.1 * i)
        _write_matrix_trace(joinpath(dir, "value_trace_upper.csv"), "v", 4, 6,
                            (i, k) -> 0.1 * i + 0.25)
        _write_gap_trace(joinpath(dir, "gap_trace.csv"), 4, 6)
        _write_matrix_trace(joinpath(dir, "update_trace.csv"), "n", 4, 6,
                            (i, k) -> k == mod1(i, 4) ? 2 : 0)

        files = (; lower=joinpath(dir, "value_trace_lower.csv"),
                   upper=joinpath(dir, "value_trace_upper.csv"),
                   gap=joinpath(dir, "gap_trace.csv"),
                   updates=joinpath(dir, "update_trace.csv"))
        r = BA._load_panels(files, 100)
        @test first.(r.panels) == [:lower, :upper, :gap, :updates]   # _PANEL_ORDER
        @test r.nframes == 6 && r.nstates == 4
        @test !r.derived_gap                      # a real gap_trace.csv was present
        @test r.bellman_updates[3] == 200
        upd = last(r.panels[4])
        @test upd[2] == [0.0, 2.0, 0.0, 0.0]      # row 2 backs up state 2, twice

        # `panels` selects a subset, still in draw order.
        sub = BA._load_panels(files, 100; want=(:gap, :updates))
        @test first.(sub.panels) == [:gap, :updates]
    end

    @testset "panels: a gap-only dir still renders, a value-only dir derives the gap" begin
        dir = mktempdir()
        _write_gap_trace(joinpath(dir, "gap_trace.csv"), 4, 3)
        only_gap = BA._load_panels((; lower=nothing, upper=nothing,
                                      gap=joinpath(dir, "gap_trace.csv"), updates=nothing), 100)
        @test first.(only_gap.panels) == [:gap]
        @test !only_gap.derived_gap

        # Bounds but no gap file: the gap is |U - L|, so nothing is lost.
        d2 = mktempdir()
        _write_matrix_trace(joinpath(d2, "value_trace_lower.csv"), "v", 3, 4, (i, k) -> 0.2)
        _write_matrix_trace(joinpath(d2, "value_trace_upper.csv"), "v", 3, 4, (i, k) -> 0.7)
        r = BA._load_panels((; lower=joinpath(d2, "value_trace_lower.csv"),
                               upper=joinpath(d2, "value_trace_upper.csv"),
                               gap=nothing, updates=nothing), 100)
        @test first.(r.panels) == [:lower, :upper, :gap]
        @test r.derived_gap
        @test all(≈(0.5), last(r.panels[3])[1])

        # Nothing on disk at all is "no video", not an error.
        @test BA._load_panels((; lower=nothing, upper=nothing, gap=nothing, updates=nothing),
                              100) === nothing
    end

    @testset "panels: a short file truncates the rest rather than misaligning them" begin
        # A SIGKILL mid-write can leave one trace shorter than its siblings. Frames are
        # matched by row index, so every panel has to stop where the shortest one does.
        dir = mktempdir()
        _write_gap_trace(joinpath(dir, "gap_trace.csv"), 4, 10)
        _write_matrix_trace(joinpath(dir, "update_trace.csv"), "n", 4, 3, (i, k) -> 1)
        r = BA._load_panels((; lower=nothing, upper=nothing,
                              gap=joinpath(dir, "gap_trace.csv"),
                              updates=joinpath(dir, "update_trace.csv")), 100)
        @test r.nframes == 3
        @test all(length(last(pr)) == 3 for pr in r.panels)
    end

    @testset "discovery: a sample dir contributes whichever panel files it holds" begin
        dir = mktempdir()
        sdir = joinpath(dir, "runs", "car_parking_S20_I3__gsrdp_exhaustive_state", "sample2")
        _write_gap_trace(joinpath(sdir, "gap_trace.csv"), 4, 2)
        _write_matrix_trace(joinpath(sdir, "update_trace.csv"), "n", 4, 2, (i, k) -> 1)

        es = BA._discover_gap_traces(dir)
        @test length(es) == 1
        e = only(es)
        @test e.instance == "car_parking_S20_I3"
        @test e.algorithm == "gsrdp_exhaustive_state"
        @test e.sample == 2
        @test e.files.gap !== nothing && e.files.updates !== nothing
        @test e.files.lower === nothing && e.files.upper === nothing

        # A sample dir with no trace file at all contributes nothing.
        mkpath(joinpath(dir, "runs", "p__a", "sample1"))
        @test length(BA._discover_gap_traces(dir)) == 1
    end

    @testset "one sample per cell is picked by NUMBER, not directory order" begin
        # `readdir` is lexicographic, so sample10 sorts before sample2. Picking "the
        # first sample" off that order would make the chosen seed depend on how many
        # samples the sweep happened to run.
        e(cell, k) = BA._GapTraceEntry("/x", cell, "p", "a", k)
        entries = [e("c1", 10), e("c1", 2), e("c1", 1), e("c2", 3), e("c2", 7)]

        picked = BA._select_one_per_cell(entries, nothing)
        @test length(picked) == 2
        @test Dict(p.cell_id => p.sample for p in picked) == Dict("c1" => 1, "c2" => 3)

        # An explicit index takes that sample and nothing else: a cell without it
        # contributes no video rather than a different seed, which would make two
        # videos in the same batch incomparable.
        picked3 = BA._select_one_per_cell(entries, 3)
        @test [p.cell_id for p in picked3] == ["c2"]
        @test isempty(BA._select_one_per_cell(entries, 99))
    end

    @testset "renders one mp4 per problem×algorithm cell, and skips what it cannot draw" begin
        dir = mktempdir()
        for (cell, nstates, samples) in (("car_parking_S20_I3__alg_a", 401, (2, 3)),
                                         ("car_parking_S20_I3__alg_b", 401, (1,)),
                                         ("coin2_K4__alg_a", 272, (1,)))
            for k in samples
                _write_gap_trace(joinpath(dir, "runs", cell, "sample$k", "gap_trace.csv"),
                                 nstates, 6)
            end
        end

        r = BA.render_gap_videos(dir; max_frames=4, framerate=2)
        # 4 traces on disk, 3 cells, 2 of them drawable => 2 videos.
        @test r.rendered == 2 && r.failed == 0
        vids = joinpath(dir, "plots", "gap_videos")
        # alg_a has samples 2 and 3; the lowest present wins, and sample3 is never drawn.
        @test isfile(joinpath(vids, "car_parking_S20_I3__alg_a__sample2.mp4"))
        @test !isfile(joinpath(vids, "car_parking_S20_I3__alg_a__sample3.mp4"))
        @test isfile(joinpath(vids, "car_parking_S20_I3__alg_b__sample1.mp4"))
        @test !isfile(joinpath(vids, "coin2_K4__alg_a__sample1.mp4"))
        @test any(l -> occursin("coin2_K4", l) && occursin("skipped", l), r.log)
        # The header reports the SELECTION (3 cells out of 4 traces on disk); the
        # coin2_K4 cell is selected and only then skipped for want of a grid.
        @test any(l -> occursin("one sample per", l) && occursin("3 of 4", l), r.log)

        # Re-invoking keeps finished work instead of redoing it.
        r2 = BA.render_gap_videos(dir; max_frames=4, framerate=2)
        @test r2.rendered == 0
        @test any(l -> occursin("already rendered", l), r2.log)

        # An explicit sample index overrides the default pick.
        r3 = BA.render_gap_videos(dir; max_frames=4, framerate=2, sample=3)
        @test r3.rendered == 1
        @test isfile(joinpath(vids, "car_parking_S20_I3__alg_a__sample3.mp4"))
    end

    @testset "opt-in: off by default, on via resolved_config, reported in the manifest" begin
        dir = _make_results_dir([_ok_cell()])
        _write_gap_trace(joinpath(dir, "runs", "car_parking_S20_I3__a", "sample1",
                                  "gap_trace.csv"), 401, 6)

        BA.run_analyses(dir; reference_label="a")
        @test !isdir(joinpath(dir, "plots", "gap_videos"))     # never by default

        BA.run_analyses(dir; reference_label="a", gap_videos=true,
                        gap_video_max_frames=3, gap_video_framerate=2)
        @test isfile(joinpath(dir, "plots", "gap_videos",
                              "car_parking_S20_I3__a__sample1.mp4"))
        @test occursin("gap videos", read(joinpath(dir, "manifest.txt"), String))

        # `[output].gap_videos` in the sweep's own config is honoured by a later
        # standalone re-analysis, which is how a cluster run gets its videos.
        cfg = joinpath(dir, "resolved_config.toml")
        open(cfg, "a") do io
            println(io, "\n[output]\ngap_videos = true\ngap_video_max_frames = 3\n",
                    "gap_video_framerate = 2")
        end
        @test BA._gap_videos_enabled(dir, nothing)
        @test BA._gap_videos_enabled(dir, false) == false      # explicit caller wins
        @test BA._gap_video_setting(dir, nothing, "gap_video_max_frames", 600) == 3
        @test BA._gap_video_setting(dir, 99, "gap_video_max_frames", 600) == 99
    end
end

# ===========================================================================
# Policy figures (src/analysis/policy_plot.jl)
# ===========================================================================
#
# The policy grid shares the videos' geometry and overlays, so what is worth pinning here
# is what is NEW: discovery of `policy.csv`, the action-0 sentinel surviving the round
# trip, the colour scale coming from the problem's full action set rather than the
# observed one, and a results dir that predates `[output].save_policy` degrading to a
# recorded reason rather than an error.

# Write a synthetic policy.csv for one (cell, sample). `zeros` marks which states carry
# the never-assigned sentinel.
function _write_policy(dir, cell, sample, nstates; nactions=9, zeros=Int[])
    d = joinpath(dir, "runs", cell, "sample$(sample)")
    mkpath(d)
    open(joinpath(d, "policy.csv"), "w") do io
        println(io, "state,action")
        for s in 1:nstates
            println(io, s, ',', s in zeros ? 0 : mod1(s, nactions))
        end
    end
    return joinpath(d, "policy.csv")
end

@testset "policy figures" begin
    @testset "discovery: cell_id split and numeric sample ordering" begin
        dir = mktempdir()
        # sample10 must sort AFTER sample2 — `readdir` is lexicographic and would put the
        # grid's columns out of sequence.
        for k in (1, 2, 10)
            _write_policy(dir, "car_parking_S10_I3__gsrdp_priority_gap_k10", k, 101)
        end
        _write_policy(dir, "car_parking_S10_I3__gsrdp_exhaustive_state", 1, 101)

        es = BA._discover_policies(dir)
        @test length(es) == 4
        @test all(e -> e.path !== nothing, es)
        @test all(e -> e.instance == "car_parking_S10_I3", es)
        # Instance names contain no "__", so the split must keep the whole alg label.
        @test Set(e.algorithm for e in es) ==
              Set(["gsrdp_exhaustive_state", "gsrdp_priority_gap_k10"])
        sampled = [e.sample for e in es if e.algorithm == "gsrdp_priority_gap_k10"]
        @test sampled == [1, 2, 10]

        # A sample dir with no policy.csv — a censored run, or a sweep with save_policy
        # off — is still discovered, with a `nothing` path, so its algorithm keeps a row
        # in the grid instead of disappearing from the comparison.
        mkpath(joinpath(dir, "runs", "car_parking_S10_I3__gsrdp_trajectory_gap_weighted", "sample1"))
        es2 = BA._discover_policies(dir)
        @test length(es2) == 5
        @test count(e -> e.path === nothing, es2) == 1
        @test only(e.algorithm for e in es2 if e.path === nothing) == "gsrdp_trajectory_gap_weighted"
    end

    @testset "reading: the action-0 sentinel survives verbatim" begin
        dir = mktempdir()
        p = _write_policy(dir, "car_parking_S10_I3__algA", 1, 101; zeros=[3, 7, 101])
        a = BA._read_policy(p)
        @test length(a) == 101
        @test a[3] == 0 && a[7] == 0 && a[101] == 0   # never assigned, not remapped to 1
        @test a[1] == 1 && a[9] == 9 && a[10] == 1    # mod1 over 9 actions

        # A malformed file costs its panel, never the figure.
        bad = joinpath(dir, "runs", "car_parking_S10_I3__algA", "sample1", "broken.csv")
        write(bad, "state,value\n1,2\n")
        @test BA._read_policy(bad) === nothing
        @test BA._read_policy(joinpath(dir, "does_not_exist.csv")) === nothing
    end

    @testset "colour scale: the problem's full action set, not the observed one" begin
        dir = mktempdir()
        write(joinpath(dir, "problems.csv"),
              "problem,source,num_states,num_actions,status\n" *
              "car_parking_S10_I3,intervalmdp_file,101,9,ok\n")
        @test BA._num_actions_for(dir, "car_parking_S10_I3") == 9
        @test BA._num_actions_for(dir, "absent_problem") === nothing
        # A dir written before the column existed, and one with no problems.csv at all,
        # both fall back rather than erroring.
        write(joinpath(dir, "problems.csv"), "problem,num_states\ncar_parking_S10_I3,101\n")
        @test BA._num_actions_for(dir, "car_parking_S10_I3") === nothing
        @test BA._num_actions_for(mktempdir(), "car_parking_S10_I3") === nothing
    end

    @testset "render: one figure per grid problem, non-grid problems refused" begin
        dir = mktempdir()
        BA._IMAGE_FORMATS[] = ["png"]
        write(joinpath(dir, "problems.csv"),
              "problem,source,num_states,num_actions,status\n" *
              "car_parking_S10_I3,intervalmdp_file,101,9,ok\n" *
              "robot_delta0.1,prism,100,4,ok\n")

        # 100 cells + 1 absorbing. The sampled variant leaves holes; the sweep does not.
        for k in 1:2
            _write_policy(dir, "car_parking_S10_I3__gsrdp_exhaustive_state", k, 101)
            _write_policy(dir, "car_parking_S10_I3__gsrdp_priority_gap_k10", k, 101;
                          zeros=collect(1:2:100))
        end
        # Present for sample 1 only: the grid must keep the hole, not reflow.
        _write_policy(dir, "car_parking_S10_I3__gsrdp_trajectory_gap_weighted", 1, 101)
        # A PRISM import whose state count IS a perfect square — the trap the geometry
        # gate exists for.
        _write_policy(dir, "robot_delta0.1__gsrdp_exhaustive_state", 1, 100; nactions=4)

        r = BA.analysis_per_problem_policy(dir, joinpath(dir, "plots"))
        @test r.rendered == 1
        @test r.skipped == 1
        @test isfile(joinpath(dir, "plots", "per_problem", "car_parking_S10_I3_policy.png"))
        @test !isfile(joinpath(dir, "plots", "per_problem", "robot_delta0.1_policy.png"))
        @test any(l -> occursin("robot_delta0.1", l) &&
                       occursin("not a known uniform-grid family", l), r.log)
        # 3 algorithms x 2 samples, but trajectory_gap_weighted only ran sample 1 => 5 panels.
        @test any(l -> occursin("car_parking_S10_I3_policy", l) && occursin("5 panel", l), r.log)
    end

    @testset "render: a dir with no policies records the reason, does not throw" begin
        dir = mktempdir()
        mkpath(joinpath(dir, "runs", "car_parking_S10_I3__algA", "sample1"))
        # Every sample dir is discovered, but none holds a policy: the problem is skipped
        # with the reason, and no figure is written.
        r = BA.analysis_per_problem_policy(dir, joinpath(dir, "plots"))
        @test r.rendered == 0 && r.skipped == 1
        @test any(l -> occursin("no readable policy.csv", l), r.log)
        @test !isdir(joinpath(dir, "plots", "per_problem"))

        # And with no runs/ tree at all (an analysis pointed at the wrong directory).
        empty_dir = mktempdir()
        r2 = BA.analysis_per_problem_policy(empty_dir, joinpath(empty_dir, "plots"))
        @test r2.rendered == 0 && r2.skipped == 0
        @test any(l -> occursin("no policy.csv", l), r2.log)
        @test !isdir(joinpath(empty_dir, "plots"))
    end

    @testset "render: no problems.csv falls back to the observed action range" begin
        # The colour scale normally comes from problems.csv. Without it the figure must
        # still render off the actions actually present — and the fallback has to step
        # over the entries that carry no policy at all.
        dir = mktempdir()
        BA._IMAGE_FORMATS[] = ["png"]
        _write_policy(dir, "car_parking_S10_I3__algA", 1, 101; nactions=4)
        mkpath(joinpath(dir, "runs", "car_parking_S10_I3__algB", "sample1"))  # censored
        @test BA._num_actions_for(dir, "car_parking_S10_I3") === nothing
        r = BA.analysis_per_problem_policy(dir, joinpath(dir, "plots"))
        @test r.rendered == 1 && r.skipped == 0
        @test !any(l -> occursin("FAILED", l), r.log)
        @test isfile(joinpath(dir, "plots", "per_problem", "car_parking_S10_I3_policy.png"))
    end

    @testset "render: a policy shorter than the grid blanks its panel, keeps the figure" begin
        dir = mktempdir()
        BA._IMAGE_FORMATS[] = ["png"]
        write(joinpath(dir, "problems.csv"),
              "problem,source,num_states,num_actions,status\n" *
              "car_parking_S10_I3,intervalmdp_file,101,9,ok\n")
        _write_policy(dir, "car_parking_S10_I3__algA", 1, 101)
        # Truncated by a kill mid-write: fewer rows than the geometry needs.
        _write_policy(dir, "car_parking_S10_I3__algB", 1, 40)
        r = BA.analysis_per_problem_policy(dir, joinpath(dir, "plots"))
        @test r.rendered == 1
        @test any(l -> occursin("1 panel", l), r.log)   # algB drew nothing
        @test isfile(joinpath(dir, "plots", "per_problem", "car_parking_S10_I3_policy.png"))
    end
end

# Merging several results directories into one (src/combine.jl). Synthetic mini results
# dirs only -- no solver, no plotting -- so it keeps the same "runs before real data" shape
# as everything above.
include(joinpath(@__DIR__, "combine.jl"))

# Resolver tests for src/registry.jl. Kept in their own file because, unlike
# everything above, they need IntervalMDP itself — the resolvers build real
# strategy objects, and asserting against anything less would not catch the
# library changing under them.
include(joinpath(@__DIR__, "registry_tests.jl"))
