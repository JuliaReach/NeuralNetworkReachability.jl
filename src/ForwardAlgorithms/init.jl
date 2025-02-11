# optional dependencies
@static if !isdefined(Base, :get_extension)
    using Requires
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require IntervalConstraintProgramming = "138f1668-1576-5ad7-91b9-7425abbf3153" include("../ext/IntervalConstraintProgrammingExt.jl")
        @require ReachabilityAnalysis = "1e97bd63-91d1-579d-8e8d-501d2b57c93f" include("../ext/ReachabilityAnalysisExt.jl")
    end
end
