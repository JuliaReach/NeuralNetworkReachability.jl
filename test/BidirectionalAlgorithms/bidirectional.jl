@testset "Bidirectional with initial point" begin
    algoB = BoxBidirectional()
    algoP = PolyhedraBidirectional()
    algoPB = SimpleBidirectional(ConcreteForward(), BoxBackward())
    algoBP = SimpleBidirectional(BoxForward(), PolyhedraBackward())

    N = example_network_222()

    x = [1.0, 1.0]
    @test forward(x, N) == [-5.0, 5.0]

    # initial point
    X = Singleton(x)
    # empty preimage
    Y = Singleton([5.0, 5.0])
    for algo in (algoB, algoP, algoPB, algoBP)
        X2, Y2 = bidirectional(X, Y, N, algo)
        @test X2 == EmptySet(2)

        res = bidirectional(X, Y, N, algo; get_intermediate_results=true)
        @test res[1] == EmptySet(2)
    end
    # nonempty preimage
    Y = HalfSpace([1.0, -1.0], 0.0)
    for algo in (algoB, algoP, algoPB, algoBP)
        X2, Y2 = bidirectional(X, Y, N, algo)
        @test isequivalent(X, X2) && !isdisjoint(Y, Y2)

        res = bidirectional(X, Y, N, algo; get_intermediate_results=true)
        @test !isdisjoint(Y, res[end]) && isequivalent(X, res[1])
    end
end

@testset "Bidirectional with initial set" begin
    algoB = BoxBidirectional()
    algoP = PolyhedraBidirectional()
    algoPB = SimpleBidirectional(ConcreteForward(), BoxBackward())
    algoBP = SimpleBidirectional(BoxForward(), PolyhedraBackward())

    N = example_network_222()

    x = [1.0, 1.0]
    X = BallInf(x, 1.0)
    # empty preimage
    Y = Singleton([5.0, 5.0])
    for algo in (algoB, algoP, algoPB, algoBP)
        X2, Y2 = bidirectional(X, Y, N, algo)
        @test X2 == EmptySet(2)

        res = bidirectional(X, Y, N, algo; get_intermediate_results=true)
        @test res[1] == EmptySet(2)
    end
    # nonempty preimage
    Y = HalfSpace([1.0, -1.0], 0.0)
    for algo in (algoB, algoP, algoBP, algoPB)
        X2, Y2 = bidirectional(X, Y, N, algo)
        @test isequivalent(X, X2) && !isdisjoint(Y, Y2)

        res = bidirectional(X, Y, N, algo; get_intermediate_results=true)
        @test isequivalent(X, res[1]) && !isdisjoint(Y, res[end])
    end
end

@testset "Bidirectional with unusual empty intermediate set" begin
    # the network is the identity for the diagonal line segment, but the box
    # algorithm turns it into a box; the exact backward algorithm detects
    # emptiness before the activation function
    W = [1.0 0; 0 1]
    b = [0.0, 0]
    N = FeedforwardNetwork([DenseLayerOp(W, b, ReLU())])
    X = LineSegment([1.0, 1], [2.0, 2])
    Y = Singleton([1.0, 2])
    algo = SimpleBidirectional(BoxForward(), PolyhedraBackward())

    X2, Y2 = bidirectional(X, Y, N, algo)
    @test isequivalent(Y, Y2) && X2 == EmptySet(2)

    X2, Y2 = bidirectional(X, Y, N, algo; get_intermediate_results=true)
    @test isequivalent(Y, Y2) && X2 == EmptySet(2)
end
