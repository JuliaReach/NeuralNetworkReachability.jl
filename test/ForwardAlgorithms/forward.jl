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

@testset "Forward layer PolyZonoForward" begin
    W = [1/2 -1/4; 0 1/2]  # second row is flipped in the paper
    b = [-2.0, 1]
    L = DenseLayerOp(W, b, ReLU())
    x = [4.0, 0]
    PZ = convert(SparsePolynomialZonotope, Hyperrectangle(x, [1.0, 2]))

    @test forward(x, L, DefaultForward()) == [0.0, 1]

    algo = PolyZonoForward(; reduced_order=4)

    # affine map
    PZ2 = forward(PZ, W, b, algo)
    PZ3 = SparsePolynomialZonotope([0.0, 1], [0.5 -0.5; 0 1], zeros(Float64, 2, 0), [1 0; 0 1])
    @test PZ2 == PZ3

    # Id
    @test forward(PZ, DenseLayerOp(W, b, Id()), algo) == PZ3

    # ReLU with automatic quadratic approximation
    PZ4 = forward(PZ, L, algo)
    @test PZ4 == forward(PZ2, ReLU(), algo)

    # ReLU with fixed quadratic approximation
    mutable struct PaperQuadratic <: ForwardAlgorithms.QuadraticApproximation
        count::Bool
    end
    @test ForwardAlgorithms._order(PaperQuadratic(true)) == 2
    function ForwardAlgorithms._polynomial_approximation(act, l, u, algo::PaperQuadratic)
        if algo.count
            algo.count = false
            return (0.25, 0.5, 0.25)
        else
            return (0.0, 1.0, 0.0)
        end
    end
    PZ5 = forward(PZ, L,
                  PolyZonoForward(; polynomial_approximation=PaperQuadratic(true), reduced_order=4))
    @test PZ5 == SparsePolynomialZonotope([1 / 8, 1], [1/4 -1/4 1/16 1/16 -1/8; 0 1 0 0 0],
                                          hcat([1 / 8, 0]), [1 0 2 0 1; 0 1 0 2 1])
    # fixed approximation is more precise in this case
    @test overapproximate(PZ5, Zonotope) ⊆ overapproximate(PZ4, Zonotope)

    # ReLU for purely negative set
    PZ = convert(SparsePolynomialZonotope, Hyperrectangle([-2.0, -2], [1.0, 1]))
    PZ2 = forward(PZ, ReLU(), algo)
    @test_broken isequivalent(PZ2, Singleton([0.0, 0]))  # not available, so check implicitly below
    @test center(PZ2) == [0.0, 0] && isempty(genmat_dep(PZ2)) && isempty(genmat_indep(PZ2))
    # ReLU for purely nonnegative set
    PZ = convert(SparsePolynomialZonotope, Hyperrectangle([2.0, 2], [1.0, 1]))
    @test PZ == forward(PZ, ReLU(), algo)

    # ReLU with other approximations
    PZ = convert(SparsePolynomialZonotope, Hyperrectangle(x, [1.0, 2]))
    for (act, pa) in [(ReLU(), ForwardAlgorithms.ClosedFormQuadratic()),
                      (Sigmoid(), ForwardAlgorithms.TaylorExpansionQuadratic()),
                      (Tanh(), ForwardAlgorithms.TaylorExpansionQuadratic())]
        algo2 = PolyZonoForward(; polynomial_approximation=pa, reduced_order=4)
        @test_throws ArgumentError forward(PZ, DenseLayerOp(W, b, act), algo2)
    end

    # ReLU with default algorithm (not fully available yet)
    mutable struct DummyApproximation <: ForwardAlgorithms.PolynomialApproximation end
    ForwardAlgorithms._order(::DummyApproximation) = 1
    ForwardAlgorithms._hq(::DummyApproximation, h, q) = (h, q)
    @test_broken forward(PZ, ReLU(),
                         PolyZonoForward(; polynomial_approximation=DummyApproximation(),
                                         reduced_order=4))
    PZ = convert(SparsePolynomialZonotope, Hyperrectangle([-2.0, -2], [1.0, 1]))
    @test_broken forward(PZ, ReLU(),
                         PolyZonoForward(; polynomial_approximation=DummyApproximation(),
                                         reduced_order=4))

    # Sigmoid / Tanh
    @test_throws ArgumentError forward(PZ, DenseLayerOp(W, b, Sigmoid()), algo)
    @test_throws ArgumentError forward(PZ, DenseLayerOp(W, b, Tanh()), algo)
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
    for algo in @tv [ConcreteForward(), LazyForward(), BoxForward(),
                     BoxForward(LazyForward()), DeepZ()] [Verisig()]
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
    @ts for algo in (DefaultForward(), Verisig())
        @test_throws ArgumentError forward(X, N, algo)
    end
end

@ts @testset "AI² ReLU example" begin
    N = example_network_AI2()
    W = N.layers[1].weights
    b = N.layers[1].bias

    H = Hyperrectangle(; low=[0.0, 1.0], high=[2.0, 3.0])
    Z = Zonotope([1.0, 2.0], [0.5 0.5 0.0; 0.5 0.0 0.5])
    P = HPolytope([HalfSpace([-2.0, 1.0], 1.0), HalfSpace([1.0, 1.0], 4.0),
                   HalfSpace([0.0, -1.0], -1.0), HalfSpace([1.0, -1.0], 0.0)])

    # affine map
    @test isequivalent(forward(H, W, b, AI2Box()),
                       Hyperrectangle([0.0, 2.0], [3.0, 1.0]))
    @test isequivalent(forward(Z, W, b, AI2Zonotope()),
                       Zonotope([0.0, 2.0], [0.5 1.0 -0.5; 0.5 0.0 0.5]))
    @test isequivalent(forward(P, W, b, AI2Polytope()),
                       VPolygon([[2.0, 2.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, 3.0]]))

    # ReLU activation
    Z2 = Zonotope([2.0, 2.0], [0.5 0.5 0.0; 0.5 0.0 0.5])  # all nonnegative
    @test forward(Z2, ReLU(), AI2Zonotope()) == Z2
    Z2 = Zonotope([-2.0, -2.0], [0.5 0.5 0.0; 0.5 0.0 0.5])  # all nonpositive
    @test isequivalent(forward(Z2, ReLU(), AI2Zonotope()), Singleton(zeros(2)))

    # network with ReLU activation
    @test isequivalent(forward(H, N, AI2Box()), Hyperrectangle(; low=[0.0, 1.0], high=[3.0, 3.0]))
    # zonotope implementation is less precise than in the paper
    @test ⊆(Zonotope([0.5, 2.0], [0.5 0.5 -0.5; 0.0 0.5 0.5]), forward(Z, N, AI2Zonotope()))
    @test isequivalent(forward(P, N, AI2Polytope()),
                       VPolygon([[0.0, 3.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]]))
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
    @ts for algo in (DefaultForward(), Verisig())
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
        @ts begin
            Y = forward(X, N, Verisig())
            if act == Sigmoid()
                @test Y_exact ⊆ overapproximate(Y, Zonotope)
            elseif act == Tanh()
                # this is a known case where the algorithm is unsound
                @test_broken Y_exact ⊆ overapproximate(Y, Zonotope)
            else
                error("unexpected case")
            end
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

@testset "Forward flattening layer" begin
    S = Singleton(1:8)
    dims = (2, 2, 2)
    cs = ConvSet(S, dims)
    @test forward(cs, FlattenLayerOp()) == S
end

@testset "AI² max-pooling example" begin
    L = MaxPoolingLayerOp(2, 2)
    M = [0 1 3 -2; 2 -4 0 1; 2 -3 0 1; -1 5 2 3]
    S_in = Singleton(vec(M))
    dims_in = (4, 4, 1)
    S_perm = Singleton([0, 1, 2, -4, 3, -2, 0, 1, 2, -3, -1, 5, 0, 1, 2, 3])
    S_out = Singleton([2, 3, 5, 3])
    dims_out = (2, 2, 1)
    csS = ConvSet(S_in, dims_in)
    csH = ConvSet(convert(Hyperrectangle, S_in), dims_in)
    csZ = ConvSet(convert(Zonotope, S_in), dims_in)
    csP = ConvSet(convert(HPolytope, S_in), dims_in)

    # permutation
    using NeuralNetworkReachability.ForwardAlgorithms: _forward_AI2_Maxpool_normalize
    @test _forward_AI2_Maxpool_normalize(csS, L) == S_perm
    @test isequivalent(_forward_AI2_Maxpool_normalize(csH, L), S_perm)
    @test isequivalent(_forward_AI2_Maxpool_normalize(csZ, L), S_perm)
    @test isequivalent(_forward_AI2_Maxpool_normalize(csP, L), S_perm)

    # AI2Box
    for cs in (csS, csH)
        cs2 = forward(cs, L, AI2Box())
        @test isequivalent(cs2.set, S_out)
        @test cs2.dims == dims_out
    end

    # AI2Zonotope
    for cs in (csS, csZ)
        @test_throws ArgumentError forward(csS, L, AI2Zonotope())
        continue
        cs2 = forward(cs, L, AI2Zonotope())
        @test isequivalent(cs2.set, S_out)
        @test cs2.dims == dims_out
    end

    # AI2Polytope
    for cs in (csS, csP)
        @test_throws ArgumentError forward(csS, L, AI2Polytope())
        continue
        cs2 = forward(cs, L, AI2Polytope())
        @test isequivalent(cs2.set, S_out)
        @test cs2.dims == dims_out
    end
end
