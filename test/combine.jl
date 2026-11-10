# Tests for src/combine.jl -- merging several results directories into one.
#
# Built on synthetic mini results dirs (a handful of CSV rows and stub trace files, no
# solving) so they run in a second and, like everything else in runtests.jl, never pull in
# the solver stack. `analyses=false` throughout: the analysis suite has its own coverage
# above, and a plotting pass here would only make the tests slow.

# combine.jl calls BenchmarkAnalysis.REFERENCE / .run_analyses, both already in scope from
# runtests.jl's include, and uses TOML/Dates/BenchmarkTools that the suite does not.
using TOML
using Dates
using BenchmarkTools
include(joinpath(@__DIR__, "..", "src", "combine.jl"))

const _RESULTS_COLS = ["cell_id", "problem", "instance_name", "source", "num_states",
                       "num_actions", "algorithm", "alg_label", "alg_kind", "bellman",
                       "sampling", "sampling_k", "convergence_eps", "sample_index", "seed",
                       "time_min_s", "time_median_s", "time_mean_s", "time_max_s",
                       "time_std_s", "n_samples", "allocs", "memory_bytes", "gctime_min_s",
                       "num_iterations", "bellman_updates", "final_gap_max",
                       "final_gap_mean", "converged", "trace_path", "ref_max_diff",
                       "ref_mean_diff", "ref_max_diff_initial", "agrees", "status", "error",
                       "error_type", "wall_total_s", "timestamp"]

"One results row, with only the fields the merge actually reads spelled out."
function _row(; cell_id, instance_name, alg_label, sample_index, seed, status="ok")
    d = Dict{String,String}(c => "" for c in _RESULTS_COLS)
    d["cell_id"] = cell_id
    d["problem"] = "intervalmdp_file"
    d["instance_name"] = instance_name
    d["source"] = "intervalmdp_file"
    d["num_states"] = "101"
    d["num_actions"] = "9"
    d["algorithm"] = "gsrdp"
    d["alg_label"] = alg_label
    d["alg_kind"] = "gsrdp"
    d["convergence_eps"] = "1.0e-5"
    d["sample_index"] = sample_index === nothing ? "" : string(sample_index)
    d["seed"] = seed === nothing ? "" : string(seed)
    d["converged"] = "true"
    d["trace_path"] = sample_index === nothing ? "" :
                      "runs/$cell_id/sample$sample_index/trace.csv"
    d["status"] = status
    d["timestamp"] = "2026-09-22 18:04:43"
    return d
end

"""
    _make_results_dir(dir; name, cells, samples, seeds, ...)

Write a minimal but complete results directory: results.csv, problems.csv,
resolved_config.toml, meta.json and one stub `runs/<cell>/sample<k>/trace.csv` per sample.
`cells` is a vector of `(instance_name, alg_label)`.
"""
function _make_results_dir(dir; name="smoke", cells, samples=2, seeds=nothing,
                           host="jyeh", eps=1.0e-5, timeout=900.0, extra_rows=[])
    mkpath(joinpath(dir, "runs"))
    seedvals = seeds === nothing ? collect(1:samples) : seeds
    rows = Dict{String,String}[]
    cell_ids = String[]
    for (inst, alg) in cells
        cid = "$(inst)__$(alg)"
        push!(cell_ids, cid)
        for k in 1:samples
            push!(rows, _row(cell_id=cid, instance_name=inst, alg_label=alg,
                             sample_index=k, seed=seedvals[k]))
            sd = joinpath(dir, "runs", cid, "sample$k")
            mkpath(sd)
            write(joinpath(sd, "trace.csv"), "wallclock_s,gap_max\n0.0,1.0\n1.0,$eps\n")
            write(joinpath(sd, "gap_trace.csv"), "s1,s2\n1.0,1.0\n$eps,$eps\n")
        end
    end
    append!(rows, extra_rows)
    CSV.write(joinpath(dir, "results.csv"),
              DataFrame([[r[c] for r in rows] for c in _RESULTS_COLS], _RESULTS_COLS))

    insts = unique(first.(cells))
    CSV.write(joinpath(dir, "problems.csv"), DataFrame(
        problem=insts, source=fill("intervalmdp_file", length(insts)),
        num_states=fill(101, length(insts)), num_actions=fill(9, length(insts)),
        num_maybe_states=fill(88, length(insts)), num_transitions=fill(37404, length(insts)),
        initial_states=fill("1", length(insts)), status=fill("ok", length(insts))))

    open(io -> TOML.print(io, Dict{String,Any}(
        "name" => name,
        "benchmark" => Dict{String,Any}("samples" => samples, "warmup" => true,
                                        "seeds" => seedvals, "timeout_s" => timeout,
                                        "wall_budget_s" => 0.0, "wall_reserve_s" => 300.0),
        "output" => Dict{String,Any}("save_raw" => true, "save_trace" => true,
                                     "save_policy" => true, "track_initial" => true,
                                     "traces" => ["trace", "gap"],
                                     "reference_label" => last(first(cells)),
                                     "results_root" => "results",
                                     "image_formats" => ["png"], "analysis" => true,
                                     "gap_videos" => false, "gap_video_max_frames" => 600,
                                     "gap_video_framerate" => 15),
        "problems" => [Dict{String,Any}("name" => "intervalmdp_file", "instance_name" => i,
                                        "params" => Dict{String,Any}("model" => "$i.nc"))
                       for i in insts],
        "algorithms" => [Dict{String,Any}("name" => "gsrdp", "label" => a,
                                          "params" => Dict{String,Any}("sampling" => a))
                         for a in unique(last.(cells))],
        "property" => Dict{String,Any}("convergence_eps" => eps,
                                       "restrict_to_initial" => true),
    ); sorted=true), joinpath(dir, "resolved_config.toml"), "w")

    open(io -> JSON.print(io, Dict{String,Any}(
        "experiment" => name, "datetime" => "2026-09-22 18:04:34",
        "datetime_end" => "2026-09-22 18:07:38", "hostname" => host, "cpu" => "test-cpu",
        "ncpu" => 20, "julia_threads" => 20, "blas_threads" => 10,
        "julia_version" => "1.12.5", "intervalmdp_version" => "0.7.0",
        "git_commit" => "abc123", "git_dirty" => false,
        "cells" => cell_ids, "n_cells" => length(cell_ids), "n_cells_run" => length(cell_ids),
        "n_ok" => length(cell_ids), "n_error" => 0, "n_skipped" => 0, "n_timeout" => 0,
        "samples" => samples, "stopped_early" => false, "stop_reason" => ""), 2),
        joinpath(dir, "meta.json"), "w")
    return dir
end

_res(dir) = CSV.read(joinpath(dir, "results.csv"), DataFrame; types=String)

@testset "combine_results" begin

    @testset "union of algorithms, pooled shared arm" begin
        root = mktempdir()
        a = _make_results_dir(joinpath(root, "smoke_A");
                              cells=[("prob1", "ref"), ("prob1", "algA")], samples=2,
                              seeds=[11, 12])
        b = _make_results_dir(joinpath(root, "smoke_B");
                              cells=[("prob1", "ref"), ("prob1", "algB")], samples=3,
                              seeds=[21, 22, 23])
        out = combine_results([a, b]; out=joinpath(root, "merged"), analyses=false)
        df = _res(out)

        # Pure concatenation: nothing is dropped and nothing is invented.
        @test nrow(df) == 4 + 6
        @test Set(df.alg_label) == Set(["ref", "algA", "algB"])

        # Per cell, sample_index is contiguous 1..N. The shared arm pooled 2 + 3.
        for cell in unique(df.cell_id)
            idx = sort(parse.(Int, df[df.cell_id .== cell, :sample_index]))
            @test idx == collect(1:length(idx))
        end
        @test count(==("prob1__ref"), df.cell_id) == 5
        # Pooling keeps every seed, from both sources.
        @test Set(df[df.cell_id .== "prob1__ref", :seed]) ==
              Set(["11", "12", "21", "22", "23"])

        # Every rewritten trace_path resolves, and points at its own renumbered directory.
        for r in eachrow(df)
            @test isfile(joinpath(out, r.trace_path))
            @test r.trace_path == "runs/$(r.cell_id)/sample$(r.sample_index)/trace.csv"
        end
        # The heavy traces came along too.
        @test isfile(joinpath(out, "runs", "prob1__ref", "sample5", "gap_trace.csv"))

        meta = JSON.parsefile(joinpath(out, "meta.json"))
        @test Set(String.(meta["cells"])) ==
              Set(["prob1__ref", "prob1__algA", "prob1__algB"])
        @test meta["combined"] === true
        @test length(meta["combined_from"]) == 2
        @test haskey(meta, "datetime_end")      # load_completeness reads this as "finished"
        @test meta["samples"] == 5

        cfg = TOML.parsefile(joinpath(out, "resolved_config.toml"))
        @test Set(String[String(x["label"]) for x in cfg["algorithms"]]) ==
              Set(["ref", "algA", "algB"])
        @test cfg["property"]["convergence_eps"] == 1.0e-5
        @test cfg["benchmark"]["timeout_s"] == 900.0
        @test cfg["output"]["reference_label"] == "ref"

        txt = read(joinpath(out, COMBINE_SOURCES_FILE), String)
        @test occursin("smoke_A", txt) && occursin("smoke_B", txt)
        @test occursin("prob1__ref", txt)
        @test occursin("Pooled cells", txt)
    end

    @testset "the combined name is readable" begin
        # Parts of one split sweep are named for their part; joining six of those with "+"
        # produces a directory name nobody can read or type.
        @test _shared_name(["grid_search_1_part$i" for i in 1:6]) == "grid_search_1"
        @test _shared_name(["smoke", "smoke"]) == "smoke"
        @test _shared_name(["smoke"]) == "smoke"
        @test _shared_name(["alpha", "beta"]) == "alpha+beta"       # no useful prefix
        @test _shared_name(["ab", "ac"]) == "ab+ac"                 # prefix too short to name a run

        root = mktempdir()
        a = _make_results_dir(joinpath(root, "A"); name="grid_search_1_part1",
                              cells=[("prob1", "ref")], samples=1, seeds=[1])
        b = _make_results_dir(joinpath(root, "B"); name="grid_search_1_part2",
                              cells=[("prob1", "algB")], samples=1, seeds=[2])
        out = combine_results([a, b]; analyses=false)
        @test startswith(basename(out), "grid_search_1_combined_")
    end

    @testset "union of problems" begin
        root = mktempdir()
        a = _make_results_dir(joinpath(root, "smoke_A");
                              cells=[("prob1", "ref")], samples=2, seeds=[1, 2])
        b = _make_results_dir(joinpath(root, "smoke_B");
                              cells=[("prob2", "ref")], samples=2, seeds=[1, 2])
        out = combine_results([a, b]; out=joinpath(root, "merged"), analyses=false)
        df = _res(out)
        @test Set(df.instance_name) == Set(["prob1", "prob2"])
        # Disjoint cells: neither is renumbered, and the seeds do not collide.
        @test sort(df[df.cell_id .== "prob2__ref", :sample_index]) == ["1", "2"]

        probs = CSV.read(joinpath(out, "problems.csv"), DataFrame)
        @test sort(String.(probs.problem)) == ["prob1", "prob2"]
        cfg = TOML.parsefile(joinpath(out, "resolved_config.toml"))
        @test Set(String[String(x["instance_name"]) for x in cfg["problems"]]) ==
              Set(["prob1", "prob2"])
        # Same seeds, different cells -- not a repeat measurement.
        @test occursin("None.", read(joinpath(out, COMBINE_SOURCES_FILE), String))
    end

    @testset "repeated (cell, seed) blocks unless dropped or forced" begin
        # Two runs of the same cell against the same pinned seeds -- the shape a split sweep
        # leaves when it re-runs a reference arm in every part. Pooling them is not a
        # statistics nicety: `run_scalars` groups by (algorithm, problem, seed) with no
        # sample identity, so the two traces interleave and the reported solve time is read
        # off the mixture. Hence blocking rather than a warning.
        root = mktempdir()
        mk(tag) = _make_results_dir(joinpath(root, tag);
                                    cells=[("prob1", "ref")], samples=2, seeds=[7, 8])
        a, b = mk("smoke_A"), mk("smoke_B")

        @test_throws ErrorException combine_results([a, b]; out=joinpath(root, "kept"),
                                                    analyses=false)
        @test !ispath(joinpath(root, "kept"))            # refused before anything was written

        # The message has to name the way out, not just the problem.
        msg = try
            combine_results([a, b]; out=joinpath(root, "kept2"), analyses=false); ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("--dedupe-seeds", msg)
        @test occursin("interleave", msg)

        # Dropping the repeats: the intended fix.
        out2 = combine_results([a, b]; out=joinpath(root, "deduped"), analyses=false,
                               dedupe_seeds=true)
        df2 = _res(out2)
        @test nrow(df2) == 2
        @test sort(df2.seed) == ["7", "8"]
        txt2 = read(joinpath(out2, COMBINE_SOURCES_FILE), String)
        @test occursin("--dedupe-seeds", txt2)
        @test !occursin("FORCED PAST", txt2)

        # Forcing keeps every row, and says so in the write-up.
        out3 = combine_results([a, b]; out=joinpath(root, "forced"), analyses=false,
                               force=true)
        @test nrow(_res(out3)) == 4
        @test occursin("FORCED PAST", read(joinpath(out3, COMBINE_SOURCES_FILE), String))
    end

    @testset "rows without a sample index pass through" begin
        root = mktempdir()
        skipped = _row(cell_id="prob1__algA", instance_name="prob1", alg_label="algA",
                       sample_index=nothing, seed=nothing, status="skipped:finite_horizon")
        a = _make_results_dir(joinpath(root, "smoke_A"); cells=[("prob1", "ref")],
                              samples=1, seeds=[1], extra_rows=[skipped])
        b = _make_results_dir(joinpath(root, "smoke_B"); cells=[("prob1", "algA")],
                              samples=1, seeds=[2])
        out = combine_results([a, b]; out=joinpath(root, "merged"), analyses=false)
        df = _res(out)
        # The cell that was skipped in A and ran in B keeps BOTH records.
        @test nrow(df) == 3
        @test count(==("skipped:finite_horizon"), df.status) == 1
        algA = df[(df.cell_id .== "prob1__algA") .& (df.status .== "ok"), :]
        @test nrow(algA) == 1 && algA[1, :sample_index] == "1"
    end

    @testset "blocking incompatibilities" begin
        root = mktempdir()
        a = _make_results_dir(joinpath(root, "smoke_A"); cells=[("prob1", "ref")],
                              samples=1, seeds=[1])

        # A different tolerance is a different measurement.
        b = _make_results_dir(joinpath(root, "smoke_eps"); cells=[("prob1", "algB")],
                              samples=1, seeds=[2], eps=1.0e-6)
        @test_throws ErrorException combine_results([a, b]; out=joinpath(root, "x1"),
                                                    analyses=false)
        out = combine_results([a, b]; out=joinpath(root, "x1f"), analyses=false, force=true)
        @test occursin("convergence_eps", read(joinpath(out, COMBINE_SOURCES_FILE), String))

        # A different censoring budget would mislabel which points are censored.
        c = _make_results_dir(joinpath(root, "smoke_to"); cells=[("prob1", "algC")],
                              samples=1, seeds=[3], timeout=60.0)
        @test_throws ErrorException combine_results([a, c]; out=joinpath(root, "x2"),
                                                    analyses=false)

        # One label meaning two different algorithms would merge them into one cell.
        d = _make_results_dir(joinpath(root, "smoke_dup"); cells=[("prob1", "ref")],
                              samples=1, seeds=[4])
        cfgp = joinpath(d, "resolved_config.toml")
        cfg = TOML.parsefile(cfgp)
        cfg["algorithms"][1]["params"]["sampling"] = "something_else"
        open(io -> TOML.print(io, cfg; sorted=true), cfgp, "w")
        @test_throws ErrorException combine_results([a, d]; out=joinpath(root, "x3"),
                                                    analyses=false)

        @test_throws ErrorException combine_results([a]; out=joinpath(root, "x4"),
                                                    analyses=false)
        @test_throws ErrorException combine_results([a, a]; out=joinpath(root, "x5"),
                                                    analyses=false)
    end

    @testset "differing host is a caveat, not a blocker" begin
        root = mktempdir()
        a = _make_results_dir(joinpath(root, "smoke_A"); cells=[("prob1", "ref")],
                              samples=1, seeds=[1], host="laptop")
        b = _make_results_dir(joinpath(root, "smoke_B"); cells=[("prob1", "algB")],
                              samples=1, seeds=[2], host="cluster-node-3")
        out = combine_results([a, b]; out=joinpath(root, "merged"), analyses=false)
        txt = read(joinpath(out, COMBINE_SOURCES_FILE), String)
        @test occursin("different hostname", txt)
        @test occursin("comparable within one machine", txt)
        # meta.json must not claim a single host for a directory that has two.
        @test JSON.parsefile(joinpath(out, "meta.json"))["hostname"] == "mixed"
    end

    @testset "sources are untouched and files are shared" begin
        root = mktempdir()
        a = _make_results_dir(joinpath(root, "smoke_A"); cells=[("prob1", "ref")],
                              samples=1, seeds=[1])
        b = _make_results_dir(joinpath(root, "smoke_B"); cells=[("prob1", "algB")],
                              samples=1, seeds=[2])
        before = Dict(p => (stat(p).mtime, read(p, String))
                      for p in (joinpath(a, "results.csv"), joinpath(b, "results.csv")))
        out = combine_results([a, b]; out=joinpath(root, "merged"), analyses=false)
        for (p, (mt, txt)) in before
            @test stat(p).mtime == mt
            @test read(p, String) == txt
        end
        # :link shares the inode rather than duplicating a multi-gigabyte trace.
        src = joinpath(a, "runs", "prob1__ref", "sample1", "gap_trace.csv")
        dst = joinpath(out, "runs", "prob1__ref", "sample1", "gap_trace.csv")
        @test stat(src).inode == stat(dst).inode

        out2 = combine_results([a, b]; out=joinpath(root, "copied"), analyses=false,
                               mode=:copy)
        dst2 = joinpath(out2, "runs", "prob1__ref", "sample1", "gap_trace.csv")
        @test stat(src).inode != stat(dst2).inode
        @test read(dst2, String) == read(src, String)
    end

    # --- a merged directory absorbs ---------------------------------------
    # Re-running as more sweeps finish must grow ONE directory rather than spawning a chain
    # of them, so a merged source becomes the target instead of a fresh timestamped dir.

    @testset "merged + normal is added into the merged directory" begin
        root = mktempdir()
        a = _make_results_dir(joinpath(root, "smoke_A");
                              cells=[("prob1", "ref"), ("prob1", "algA")], samples=2,
                              seeds=[11, 12])
        b = _make_results_dir(joinpath(root, "smoke_B");
                              cells=[("prob1", "ref"), ("prob1", "algB")], samples=2,
                              seeds=[21, 22])
        c = _make_results_dir(joinpath(root, "smoke_C");
                              cells=[("prob1", "ref"), ("prob1", "algC")], samples=2,
                              seeds=[31, 32])

        merged = combine_results([a, b]; out=joinpath(root, "merged"), analyses=false)
        before = _res(merged)
        @test nrow(before) == 8

        # No `out`: the merged directory is the target, whichever position it is given in.
        again = combine_results([merged, c]; analyses=false)
        @test again == merged                                   # no new directory
        @test !any(d -> occursin("_combined_", d), readdir(root))

        df = _res(merged)
        @test nrow(df) == 12                                    # 8 + 4, nothing lost
        @test Set(df.alg_label) == Set(["ref", "algA", "algB", "algC"])

        # The samples already here kept their numbers; C's continue from the end.
        ref = df[df.cell_id .== "prob1__ref", :]
        @test sort(parse.(Int, ref.sample_index)) == collect(1:6)
        for r in eachrow(before)
            now = df[(df.cell_id .== r.cell_id) .& (df.seed .== r.seed), :]
            @test nrow(now) == 1
            @test now[1, :sample_index] == r.sample_index       # pinned, not renumbered
        end
        for r in eachrow(df)
            @test isfile(joinpath(merged, r.trace_path))
        end

        meta = JSON.parsefile(joinpath(merged, "meta.json"))
        @test Set(String.(meta["cells"])) ==
              Set(["prob1__ref", "prob1__algA", "prob1__algB", "prob1__algC"])
        # combined_from flattens to the underlying SWEEPS, never the merged dir itself.
        @test Set(String.(meta["combined_from"])) == Set([a, b, c])

        # The log grew by one entry and COMBINED_FROM.md still names the first pass's runs.
        log = JSON.parsefile(joinpath(merged, COMBINE_LOG_FILE))
        @test length(log) == 2
        @test log[2]["in_place"] === true
        txt = read(joinpath(merged, COMBINE_SOURCES_FILE), String)
        @test all(t -> occursin(t, txt), ["smoke_A", "smoke_B", "smoke_C"])
        @test occursin("Merge 2", txt)
        @test occursin("added into this directory", txt)
    end

    @testset "merged directory absorbs whatever position it is given in" begin
        root = mktempdir()
        a = _make_results_dir(joinpath(root, "smoke_A"); cells=[("prob1", "ref")],
                              samples=1, seeds=[1])
        b = _make_results_dir(joinpath(root, "smoke_B"); cells=[("prob1", "algB")],
                              samples=1, seeds=[2])
        c = _make_results_dir(joinpath(root, "smoke_C"); cells=[("prob1", "algC")],
                              samples=1, seeds=[3])
        merged = combine_results([a, b]; out=joinpath(root, "merged"), analyses=false)
        # Normal first, merged second: the merged one still absorbs.
        @test combine_results([c, merged]; analyses=false) == merged
        @test Set(_res(merged).alg_label) == Set(["ref", "algB", "algC"])
    end

    @testset "merged + merged goes into the first" begin
        root = mktempdir()
        mk(tag, alg, seed) = _make_results_dir(joinpath(root, tag);
                                               cells=[("prob1", alg)], samples=1, seeds=[seed])
        m1 = combine_results([mk("A", "ref", 1), mk("B", "algB", 2)];
                             out=joinpath(root, "m1"), analyses=false)
        m2 = combine_results([mk("C", "algC", 3), mk("D", "algD", 4)];
                             out=joinpath(root, "m2"), analyses=false)

        @test combine_results([m1, m2]; analyses=false) == m1
        @test Set(_res(m1).alg_label) == Set(["ref", "algB", "algC", "algD"])
        @test nrow(_res(m2)) == 2                               # m2 untouched
        # m2's own history came along: all four sweeps are named, not just m2.
        txt = read(joinpath(m1, COMBINE_SOURCES_FILE), String)
        @test all(t -> occursin("`$t`", txt), ["A", "B", "C", "D"])
        # m1's own merge, m2's merge folded in as history, then this one.
        log = JSON.parsefile(joinpath(m1, COMBINE_LOG_FILE))
        @test length(log) == 3
        @test String(log[2]["inherited_from"]) == basename(m2)
        @test occursin("history of `$(basename(m2))`, folded in here", txt)
    end

    @testset "a merged dir predating the log keeps its history" begin
        # combine_log.json is newer than the first merged directories. Merging into one of
        # those must reconstruct an entry from meta.json rather than silently forgetting
        # the runs it was built from.
        root = mktempdir()
        a = _make_results_dir(joinpath(root, "smoke_A"); cells=[("prob1", "ref")],
                              samples=2, seeds=[1, 2])
        b = _make_results_dir(joinpath(root, "smoke_B"); cells=[("prob1", "ref")],
                              samples=2, seeds=[3, 4])
        c = _make_results_dir(joinpath(root, "smoke_C"); cells=[("prob1", "algC")],
                              samples=1, seeds=[5])
        merged = combine_results([a, b]; out=joinpath(root, "merged"), analyses=false)
        rm(joinpath(merged, COMBINE_LOG_FILE))              # as if built before the log existed

        combine_results([merged, c]; analyses=false)
        log = JSON.parsefile(joinpath(merged, COMBINE_LOG_FILE))
        @test length(log) == 2
        @test Set(String[String(s["tag"]) for s in log[1]["sources"]]) ==
              Set(["smoke_A", "smoke_B"])

        txt = read(joinpath(merged, COMBINE_SOURCES_FILE), String)
        @test all(t -> occursin(t, txt), ["smoke_A", "smoke_B", "smoke_C"])
        # What it cannot know, it says it cannot know -- rather than printing "(0)".
        @test occursin("not recorded", txt)
        @test !occursin("Problems (0)", txt)
        @test occursin("cannot be listed in full", txt)     # the pooled set is unknowable here
        @test occursin("predates", txt)
        # Re-adding a run is still refused: combined_from survived in meta.json.
        @test_throws ErrorException combine_results([merged, b]; analyses=false)
    end

    @testset "rerender_sources_file rebuilds the prose from the log" begin
        root = mktempdir()
        a = _make_results_dir(joinpath(root, "smoke_A"); cells=[("prob1", "ref")],
                              samples=1, seeds=[1])
        b = _make_results_dir(joinpath(root, "smoke_B"); cells=[("prob1", "algB")],
                              samples=1, seeds=[2])
        merged = combine_results([a, b]; out=joinpath(root, "merged"), analyses=false)
        before = read(joinpath(merged, COMBINE_SOURCES_FILE), String)
        write(joinpath(merged, COMBINE_SOURCES_FILE), "clobbered")
        @test rerender_sources_file(merged) == joinpath(merged, COMBINE_SOURCES_FILE)
        @test read(joinpath(merged, COMBINE_SOURCES_FILE), String) == before

        # Not a combined directory: nothing to render from.
        @test_throws ErrorException rerender_sources_file(a)
    end

    @testset "out forces a fresh directory even with a merged source" begin
        root = mktempdir()
        a = _make_results_dir(joinpath(root, "smoke_A"); cells=[("prob1", "ref")],
                              samples=1, seeds=[1])
        b = _make_results_dir(joinpath(root, "smoke_B"); cells=[("prob1", "algB")],
                              samples=1, seeds=[2])
        c = _make_results_dir(joinpath(root, "smoke_C"); cells=[("prob1", "algC")],
                              samples=1, seeds=[3])
        merged = combine_results([a, b]; out=joinpath(root, "merged"), analyses=false)
        fresh = combine_results([merged, c]; out=joinpath(root, "fresh"), analyses=false)
        @test fresh == joinpath(root, "fresh")
        @test nrow(_res(merged)) == 2                           # the merged dir is untouched
        @test Set(_res(fresh).alg_label) == Set(["ref", "algB", "algC"])
    end

    @testset "refuses to re-add a run already inside the target" begin
        root = mktempdir()
        a = _make_results_dir(joinpath(root, "smoke_A"); cells=[("prob1", "ref")],
                              samples=1, seeds=[1])
        b = _make_results_dir(joinpath(root, "smoke_B"); cells=[("prob1", "algB")],
                              samples=1, seeds=[2])
        merged = combine_results([a, b]; out=joinpath(root, "merged"), analyses=false)
        # b is already in there: adding it again would double-count its samples.
        @test_throws ErrorException combine_results([merged, b]; analyses=false)
        @test nrow(_res(merged)) == 2                           # and nothing was written
    end

    @testset "refuses a non-empty output directory" begin
        root = mktempdir()
        a = _make_results_dir(joinpath(root, "smoke_A"); cells=[("prob1", "ref")],
                              samples=1, seeds=[1])
        b = _make_results_dir(joinpath(root, "smoke_B"); cells=[("prob1", "algB")],
                              samples=1, seeds=[2])
        occupied = joinpath(root, "occupied")
        mkpath(occupied); write(joinpath(occupied, "keep.txt"), "do not clobber")
        @test_throws ErrorException combine_results([a, b]; out=occupied, analyses=false)
        @test read(joinpath(occupied, "keep.txt"), String) == "do not clobber"
    end
end
