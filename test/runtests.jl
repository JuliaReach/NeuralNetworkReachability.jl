using Test, NeuralNetworkReachability
using ControllerFormats, LazySets

include("example_networks.jl")

@testset "Optional dependencies (not loaded)" begin
    include("optional_dependencies_not_loaded.jl")
end

# load optional dependencies
import IntervalConstraintProgramming, ReachabilityAnalysis, Polyhedra, CDDLib, Optim

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

include("Aqua.jl")
