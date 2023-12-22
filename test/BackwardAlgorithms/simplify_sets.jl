@testset "Set simplification" begin
    # simplify_set
    using NeuralNetworkReachability.BackwardAlgorithms: simplify_set
    X = BallInf([0.0], 0.0)
    @test simplify_set(X) == X
    X = HPolyhedron([HalfSpace([1.0], -1.0), HalfSpace([-1.0], -1.0)])
    @test simplify_set(X) == EmptySet(1)
    X = HPolyhedron([HalfSpace([1.0], 1.0), HalfSpace([-1.0], 1.0)])
    @test simplify_set(X) == X

    # simplify_union
    using NeuralNetworkReachability.BackwardAlgorithms: simplify_union
    @test simplify_union([]) == EmptySet(1)
    @test simplify_union([]; n=3) == EmptySet(3)
    @test simplify_union([X]) == X
    @test simplify_union([X, X]) == UnionSetArray([X, X])
end
