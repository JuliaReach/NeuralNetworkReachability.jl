using NeuralNetworkReachability.Util: ConvSet

@testset "ConvSet" begin
    X = BallInf(zeros(12), 1.0)
    ConvSet(X, (1, 2, 6))
    ConvSet(X, (2, 2, 3))
    @test_throws ArgumentError ConvSet(X, (0, 1, 12))
    @test_throws ArgumentError ConvSet(X, (2, 2, 2))
    @test_throws ArgumentError ConvSet(X, (0, 0, 0))
end
