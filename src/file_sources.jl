# Problem sources loaded from external files, via IntervalMDP's existing readers.
# These are the "various sources" extension point: the file PARSING already
# exists in the library, so each builder is a thin wrapper. (Converting a
# *continuous* PRISM model to an IMDP via IMDPAbstractions is the deferred part —
# marked TODO below.) Not exercised by the smoke config; errors are isolated
# per-cell, so a misconfigured path fails only its own cells.

function _wrap_file_problem(name, problem, source::Symbol, params, srcfile)
    md = Dict{String,Any}("source_file" => String(srcfile))
    try
        merge!(md, safe_size_metadata(system(problem)))
    catch
    end
    return BenchmarkProblem(String(name), problem, source, md, Dict{String,Any}(params))
end

"PRISM explicit-format files. `file` is the path WITHOUT extension (reader appends .sta/.tra/.lab/.pctl)."
function build_prism(params::AbstractDict)
    haskey(params, "file") || error("prism builder requires 'file' (path without extension)")
    base = String(params["file"])
    problem = IntervalMDP.Data.read_prism_file(base)
    # TODO(extension): if params["mode"]=="abstract", run the (out-of-scope) continuous
    # PRISM model -> dynamics -> IMDPAbstractions pipeline before wrapping.
    return _wrap_file_problem(_basename_noext(base), problem, :prism, params, base)
end
register_problem!("prism", build_prism)

"bmdp-tool `.txt` files."
function build_bmdp_tool(params::AbstractDict)
    haskey(params, "file") || error("bmdp_tool builder requires 'file' (.txt path)")
    path = String(params["file"])
    problem = IntervalMDP.Data.read_bmdp_tool_file(path)
    return _wrap_file_problem(_basename_noext(path), problem, :bmdp_tool, params, path)
end
register_problem!("bmdp_tool", build_bmdp_tool)

"Native IntervalMDP.jl model+spec files (netCDF sparse format)."
function build_intervalmdp_file(params::AbstractDict)
    haskey(params, "model") && haskey(params, "spec") ||
        error("intervalmdp_file builder requires 'model' and 'spec' paths")
    problem = IntervalMDP.Data.read_intervalmdp_jl(String(params["model"]), String(params["spec"]);
                                  control_synthesis=Bool(get(params, "control_synthesis", true)))
    return _wrap_file_problem(_basename_noext(params["model"]), problem, :intervalmdp_file,
                              params, params["model"])
end
register_problem!("intervalmdp_file", build_intervalmdp_file)
