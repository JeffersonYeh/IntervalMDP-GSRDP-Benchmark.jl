#!/usr/bin/env julia
# Regenerate every thesis figure: runs each scripts/*.jl (skipping files that
# start with `_`) in its own module, so one failing script does not stop the rest.
#
#   julia report-plots/make_all.jl              # all scripts
#   julia report-plots/make_all.jl baseline     # only scripts whose name contains a pattern
#
# A single script can also be run on its own:  julia report-plots/scripts/<name>.jl

include(joinpath(@__DIR__, "common.jl"))

# dataset-cluster.jl writes the clusters (cache/) that every experiments-*.jl script reads, so it
# runs first; the rest run alphabetically.
const RUN_FIRST = ["dataset-cluster.jl"]
scripts = sort(filter(f -> endswith(f, ".jl") && !startswith(f, "_"),
                      readdir(joinpath(@__DIR__, "scripts"))); by=f -> (f in RUN_FIRST ? 0 : 1, f))
isempty(ARGS) || filter!(f -> any(p -> occursin(p, f), ARGS), scripts)
isempty(scripts) && (println("no scripts to run"); exit(0))

failed = String[]
for f in scripts
    println("── ", f)
    flush(stdout)
    m = Module(Symbol(splitext(f)[1]))
    Core.eval(m, :(include(p) = Base.include($m, p)))  # bare modules lack `include`
    try
        t = @elapsed Base.include(m, joinpath(@__DIR__, "scripts", f))
        @printf("   done in %.1f s\n", t)
    catch e
        push!(failed, f)
        @error "report-plots: $f failed" exception=(e, catch_backtrace())
    end
end

println()
println(length(scripts) - length(failed), "/", length(scripts), " scripts succeeded")
isempty(failed) || (println("failed: ", join(failed, ", ")); exit(1))
