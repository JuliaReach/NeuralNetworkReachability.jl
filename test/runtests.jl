using Test, NeuralNetworkReachability
using ControllerFormats, LazySets

# auxiliary code to skip expensive tests
begin
    __test_short = haskey(ENV, "JULIA_PKGEVAL")

    macro ts(arg)
        if !__test_short
            quote
                $(esc(arg))
            end
        end
    end

    macro tv(v1, v2)
        if __test_short
            return v1
        else
            return @eval vcat($v1, $v2)
        end
    end
end

include("example_networks.jl")

@testset "Optional dependencies (not loaded)" begin
    include("optional_dependencies_not_loaded.jl")
end

# load optional dependencies
import Polyhedra, CDDLib, Optim
@ts import IntervalConstraintProgramming, ReachabilityAnalysis

@testset "Util" begin
    include("Util/Util.jl")
end
@testset "ForwardAlgorithms" begin
    include("ForwardAlgorithms/forward.jl")
end
@testset "BackwardAlgorithms" begin
    struct DummyBackward <: NeuralNetworkReachability.BackwardAlgorithms.BackwardAlgorithm
    end

    include("BackwardAlgorithms/simplify_sets.jl")
    include("BackwardAlgorithms/PartitioningLeakyReLU.jl")
    include("BackwardAlgorithms/backward.jl")
end
@testset "BidirectionalAlgorithms" begin
    include("BidirectionalAlgorithms/bidirectional.jl")
end
@testset "AISoLA 2023" begin
    include("AISoLA2023/BoxAffineMap.jl")
    @testset "AISoLA 2023" begin
        include("AISoLA2023/motivation.jl")
    end
    @testset "AISoLA 2023" begin
        include("AISoLA2023/parabola.jl")
    end
    @testset "AISoLA 2023" begin
        include("AISoLA2023/leaky_relu.jl")
    end
    @testset "AISoLA 2023" begin
        include("AISoLA2023/forward_backward.jl")
    end
end

@ts include("quality_assurance.jl")
