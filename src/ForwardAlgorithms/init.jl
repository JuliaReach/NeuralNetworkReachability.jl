# optional dependencies
function __init__()
    @require IntervalConstraintProgramming = "138f1668-1576-5ad7-91b9-7425abbf3153" begin
        # nothing
    end
    @require ReachabilityAnalysis = "1e97bd63-91d1-579d-8e8d-501d2b57c93f" begin
        include("init_ReachabilityAnalysis.jl")
    end
end
