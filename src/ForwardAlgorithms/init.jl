# optional dependencies
function __init__()
    @require IntervalConstraintProgramming = "138f1668-1576-5ad7-91b9-7425abbf3153" (nothing,)

    @require ReachabilityAnalysis = "1e97bd63-91d1-579d-8e8d-501d2b57c93f" begin
        include("init_ReachabilityAnalysis.jl")

        @require TaylorModels = "314ce334-5f6e-57ae-acf6-00b6e903104a" begin
            include("init_TaylorModels.jl")

            @require TaylorIntegration = "92b13dbe-c966-51a2-8445-caca9f8a7d42" begin
                include("init_TaylorIntegration.jl")
            end
        end
    end
end
