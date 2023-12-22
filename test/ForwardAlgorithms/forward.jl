@testset "Forward affine map" begin
    W = [2.0 3; 4 5; 6 7]
    b = [1.0, 2, 3]
    x = [1.0, 2]
    X = BallInf(x, 0.1)

    @test forward(x, W, b, DefaultForward()) == [9.0, 16, 23]

    for algo in (DefaultForward(), ConcreteForward(), LazyForward(),
                 BoxForward(), BoxForward(LazyForward()),
                 SplitForward(DefaultForward()), DeepZ(), Verisig(nothing))
        Y = concretize(forward(X, W, b, algo))
        @test isequivalent(Y, affine_map(W, X, b))
    end
end

@testset "Forward Id activation" begin
    x = [1.0, 2]
    X = BallInf(x, 0.1)

    @test forward(x, Id(), DefaultForward()) == Id()(x)

    # exact algorithms
    for algo in (DefaultForward(), ConcreteForward(), LazyForward(),
                 BoxForward(), BoxForward(LazyForward()),
                 SplitForward(DefaultForward()), DeepZ(), Verisig(nothing))
        @test isequivalent(forward(X, Id(), algo), X)
    end
end

@testset "Forward ReLU activation" begin
    x1 = [1.0, 2]
    x2 = [1.0, -2]
    X1 = BallInf(x1, 0.1)
    X2 = BallInf(x2, 0.1)
    X3 = LineSegment([0.0, -1], [2.0, 1])

    Y2 = LineSegment([0.9, 0], [1.1, 0])
    Y3 = UnionSet(LineSegment([0.0, 0], [1.0, 0]), LineSegment([1.0, 0], [2.0, 1]))
    @test forward(x1, ReLU(), DefaultForward()) == ReLU()(x1)
    @test forward(x2, ReLU(), DefaultForward()) == ReLU()(x2)

    # exact algorithms
    for algo in (ConcreteForward(), LazyForward())
        @test isequivalent(concretize(forward(X1, ReLU(), algo)), X1)
        @test isequivalent(concretize(forward(X2, ReLU(), algo)), Y2)
        # equivalence check between unions (not available out of the box)
        @test Y3 ⊆ concretize(forward(X3, ReLU(), algo))
        Z3 = concretize(forward(X3, ReLU(), algo))
        @test Z3 isa UnionSetArray && length(Z3) == 2 &&
              (isequivalent(Z3[1], Y3[1]) || isequivalent(Z3[1], Y3[2])) &&
              (isequivalent(Z3[2], Y3[1]) || isequivalent(Z3[2], Y3[2]))
    end

    # approximate algorithms
    for algo in (BoxForward(), BoxForward(LazyForward()), DeepZ())
        @test isequivalent(concretize(forward(X1, ReLU(), algo)), X1)
        @test isequivalent(concretize(forward(X2, ReLU(), algo)), Y2)
        @test Y3 ⊆ concretize(forward(X3, ReLU(), algo))
    end

    # algorithms currently not supporting ReLU activation
    for algo in (SplitForward(ConcreteForward()),)
        @test_broken forward(X1, ReLU(), algo)
    end

    # algorithms not supporting ReLU activation
    for algo in (DefaultForward(), Verisig(nothing))
        @test_throws ArgumentError forward(X1, ReLU(), algo)
    end
end

@testset "Forward sigmoid activation" begin
    x1 = [1.0, 2]
    x2 = [1.0, -2]
    X1 = BallInf(x1, 0.1)
    X2 = BallInf(x2, 0.1)
    X3 = LineSegment([0.0, -1], [2.0, 1])

    Y1 = VPolytope(convex_hull([forward(x, Sigmoid(), DefaultForward()) for x in vertices(X1)]))
    @test forward(x1, Sigmoid(), DefaultForward()) == Sigmoid()(x1)

    # approximate algorithms
    for algo in (BoxForward(), BoxForward(LazyForward()), DeepZ())
        @test Y1 ⊆ forward(X1, Sigmoid(), algo)
    end

    # DeepZ() has a special case for flat sets
    for algo in (DeepZ(),)
        x2 = [1.0, 1.0]
        X2 = BallInf(x2, 0.0)
        Y = forward(X2, Sigmoid(), algo)
        @test isequivalent(Singleton(Sigmoid()(x2)), concretize(Y))
        X3 = LineSegment(x2, [1.0, 2.0])
        Y = forward(X3, Sigmoid(), algo)
        Y_exact3 = VPolygon(convex_hull([Sigmoid()(x2) for x in vertices(X3)]))
        @test Y_exact3 ⊆ concretize(Y)
    end

    # algorithms not supporting sigmoid activation
    for algo in (DefaultForward(), ConcreteForward(), LazyForward(),
                 SplitForward(ConcreteForward()), Verisig(nothing))
        @test_throws ArgumentError forward(X1, Sigmoid(), algo)
    end
end

@testset "Forward tanh activation" begin
    x = [1.0, 2]
    X = BallInf(x, 0.1)

    Y = VPolytope(convex_hull([forward(x, Tanh(), DefaultForward()) for x in vertices(X)]))
    @test forward(x, Tanh(), DefaultForward()) == Tanh()(x)

    # approximate algorithms
    for algo in (BoxForward(), BoxForward(LazyForward()), DeepZ())
        @test Y ⊆ forward(X, Tanh(), algo)
    end

    # algorithms not supporting tanh activation
    for algo in (DefaultForward(), ConcreteForward(), LazyForward(),
                 SplitForward(ConcreteForward()), Verisig(nothing))
        @test_throws ArgumentError forward(X, Tanh(), algo)
    end
end

@testset "Forward LeakyReLU activation" begin
    x = [1.0, 2]
    X = BallInf(x, 0.1)

    Y = VPolytope(convex_hull([forward(x, LeakyReLU(0.1), DefaultForward()) for x in vertices(X)]))
    @test forward(x, LeakyReLU(0.1), DefaultForward()) == LeakyReLU(0.1)(x)

    # approximate algorithms
    for algo in (BoxForward(), BoxForward(LazyForward()))
        @test Y ⊆ forward(X, LeakyReLU(0.1), algo)
    end

    # algorithms currently not supporting leaky-ReLU activation
    for algo in (ConcreteForward(), LazyForward(), SplitForward(ConcreteForward()))
        @test_broken forward(X, LeakyReLU(0.1), algo)
    end

    # algorithms not supporting leaky-ReLU activation
    for algo in (DefaultForward(), DeepZ(), Verisig(nothing))
        @test_throws ArgumentError forward(X, LeakyReLU(0.1), algo)
    end
end

@testset "Forward layer" begin
    W = [2.0 3; 4 5; 6 7]
    b = [1.0, 2, 3]
    L = DenseLayerOp(W, b, Id())
    x = [1.0, 2]
    X = BallInf(x, 0.1)

    @test forward(x, L, DefaultForward()) == [9.0, 16, 23]

    for algo in (DefaultForward(), ConcreteForward(), LazyForward(),
                 BoxForward(), BoxForward(LazyForward()),
                 SplitForward(DefaultForward()), DeepZ(), Verisig(nothing))
        Y = concretize(forward(X, L, algo))
        @test isequivalent(Y, affine_map(W, X, b))
    end
end

@testset "Forward network" begin
    W1 = [2.0 3; 4 5; 6 7]
    b1 = [1.0, 2, 3]
    L1 = DenseLayerOp(W1, b1, Id())
    W2 = [1.0 1 1; 2 2 2]
    b2 = [-1.0, -2]
    L2 = DenseLayerOp(W2, b2, Id())
    N = FeedforwardNetwork([L1, L2])
    x = [1.0, 2]
    X = BallInf(x, 0.1)

    Y = affine_map(W2, affine_map(W1, X, b1), b2)
    @test forward(x, N, DefaultForward()) == [47.0, 94]

    # exact algorithms (in this case)
    for algo in (ConcreteForward(), LazyForward(), BoxForward(),
                 BoxForward(LazyForward()), DeepZ(), Verisig())
        @test isequivalent(concretize(forward(X, N, algo)), Y)
    end
    # approximate algorithms
    for algo in (SplitForward(ConcreteForward()),)
        @test Y ⊆ forward(X, N, algo)
    end
    for algo in (DefaultForward(),)
        @test_throws ArgumentError forward(X, N, algo)
    end
end

@testset "Forward network singleton" begin
    # singleton computation uses vector and hence works for each algorithm
    for (N, y) in ((example_network_222(ReLU()), [-5.0, 5.0]),
                   (example_network_232(ReLU()), [-8.0, 8.0]),
                   (example_network_222(Sigmoid()), [-2.0179862099620918, 2.0179862099620918]),
                   (example_network_232(Sigmoid()), [-4.211161945852107, 4.211161945852107]),
                   (example_network_222(Tanh()), [-0.000670700260932966, 0.000670700260932966]),
                   (example_network_232(Tanh()), [-2.2854531681282273, 2.2854531681282273]),
                   (example_network_222(LeakyReLU(0.1)), [-4.2, 4.2]),
                   (example_network_232(LeakyReLU(0.1)), [-7.2, 7.2]))
        x = Singleton([1.0, 1.0])

        # forward with singleton
        yv = forward(element(x), N)
        @test yv == y
        for algo in (DefaultForward(), ConcreteForward(), LazyForward(),
                     BoxForward(), BoxForward(LazyForward()),
                     SplitForward(DefaultForward()), DeepZ(), Verisig(nothing))
            ys = forward(x, N, algo)
            @test isequivalent(concretize(ys), Singleton(y))
        end

        # _forward_store with singleton
        using NeuralNetworkReachability.ForwardAlgorithms: _forward_store
        yv = _forward_store(element(x), N, DefaultForward())
        @test length(yv) == length(N)
        @test yv[end][2] == y
        for algo in (DefaultForward(), ConcreteForward(), LazyForward(),
                     BoxForward(), BoxForward(LazyForward()),
                     SplitForward(DefaultForward()), DeepZ(), Verisig(nothing))
            ys = _forward_store(x, N, algo)
            @test length(ys) == length(N)
            for i in eachindex(ys)
                for j in 1:2
                    @test isequivalent(concretize(ys[i][j]), Singleton(yv[i][j]))
                end
            end
        end
    end
end

@testset "Forward ReLU network" begin
    N = example_network_232(ReLU())
    X = BallInf([1.0, 1.0], 0.1)
    Y_exact = LineSegment([-9.7, 9.7], [-6.3, 6.3])

    # exact algorithms (in this case)
    for algo in (ConcreteForward(), LazyForward(), DeepZ())
        Y = forward(X, N, algo)
        @test isequivalent(concretize(Y), Y_exact)
    end

    # approximate algorithms
    for algo in (BoxForward(), BoxForward(LazyForward()), SplitForward(ConcreteForward()))
        Y = forward(X, N, algo)
        @test Y_exact ⊆ concretize(Y)
    end

    # algorithms not supporting ReLU activation
    for algo in (DefaultForward(), Verisig())
        @test_throws ArgumentError forward(X, N, algo)
    end
end

@testset "Forward leaky-ReLU network" begin
    N = example_network_232(LeakyReLU(0.1))
    X = BallInf([1.0, 1.0], 0.1)
    Y_exact = LineSegment([-8.92, 8.92], [-5.48, 5.48])

    # approximate algorithms
    for algo in (BoxForward(), BoxForward(LazyForward()))
        Y = forward(X, N, algo)
        @test Y_exact ⊆ concretize(Y)
    end

    # algorithms currently not supporting leaky-ReLU activation
    for algo in (ConcreteForward(), LazyForward(),
                 SplitForward(ConcreteForward()), DeepZ())
        @test_broken forward(X, N, algo)
    end

    # algorithms not supporting ReLU activation
    for algo in (DefaultForward(), Verisig())
        @test_throws ArgumentError forward(X, N, algo)
    end

    # algorithms currently not supporting unbounded inputs
    X = Line([1.0, 1], 1.0)
    for algo in (BoxForward(), BoxForward(LazyForward()))
        @test_broken forward(X, N, algo)
    end
end

@testset "Forward sigmoid and tanh networks" begin
    for act in (Sigmoid(), Tanh())
        N = example_network_232(act)
        X = BallInf([1.0, 1.0], 0.1)
        # in this case the convex enclosure is exact
        Y_exact = VPolygon(convex_hull([forward(x, N) for x in vertices(X)]))

        # approximate algorithms
        for algo in (BoxForward(), BoxForward(LazyForward()), SplitForward(DeepZ()),
                     DeepZ())
            Y = forward(X, N, algo)
            @test Y_exact ⊆ concretize(Y)
        end

        # Verisig result has a special type
        Y = forward(X, N, Verisig())
        if act == Sigmoid()
            @test Y_exact ⊆ overapproximate(Y, Zonotope)
        elseif act == Tanh()
            # this is a known case where the algorithm is unsound
            @test_broken Y_exact ⊆ overapproximate(Y, Zonotope)
        else
            error("unexpected case")
        end

        # algorithms not supporting sigmoid activation
        for algo in (DefaultForward(), ConcreteForward(), LazyForward(),
                     SplitForward(ConcreteForward()))
            @test_throws ArgumentError forward(X, N, algo)
        end
    end
end

@testset "Forward sigmoid multiple hidden layers" begin
    N = example_network_1221(Sigmoid())
    X = Interval(1.0, 2.0)
    # in this case the convex enclosure is exact
    ch = convex_hull([forward(x, N) for x in vertices(X)])
    Y_exact = Interval(ch[1][1], ch[2][1])

    # approximate algorithms
    for algo in (BoxForward(), BoxForward(LazyForward()), SplitForward(DeepZ()),
                 DeepZ())
        Y = forward(X, N, algo)
        @test Y_exact ⊆ concretize(Y)
    end

    # not supported yet
    @test_broken forward(X, N, Verisig())
end
