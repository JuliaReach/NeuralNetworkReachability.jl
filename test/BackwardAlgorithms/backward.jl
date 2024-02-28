@testset "PolyhedraBackward algorithm" begin
    # invalid input: 2D but not a polyhedron or union
    Y = Ball2([-1.0, 1.0], 1.0)
    @test_throws AssertionError backward(Y, ReLU(), PolyhedraBackward())
end

@testset "Backward affine map" begin
    # all algorithms
    for algo in (BoxBackward(), PolyhedraBackward())
        # invalid input: incompatible matrix/vector dimensions in affine map
        Y = Singleton([1.0, 1.0])
        @test_throws AssertionError backward(Y, rand(2, 3), rand(1), algo)

        # union
        W = hcat([1.0])
        b = [1.0]
        Y = UnionSetArray([Singleton([2.0]), Singleton([3.0])])
        @test UnionSetArray([Singleton([1.0]), Singleton([2.0])]) ⊆ backward(Y, W, b, algo)
    end

    # exact algorithms
    for algo in (PolyhedraBackward(),)
        # 1D affine map
        W = hcat([2.0])
        b = [1.0]
        x = [1.0]
        X = Singleton(x)
        Y = affine_map(W, X, b)
        y = element(Y)
        @test backward(y, W, b, algo) == x
        @test isequivalent(backward(Y, W, b, algo), X)
        # non-polytope
        Y = convert(HPolyhedron, Y)
        @test isequivalent(backward(Y, W, b, algo), X)
        # half-space
        Y = HalfSpace([1.0], 2.0)
        X = HalfSpace([1.0], 0.5)
        @test isequivalent(backward(Y, W, b, algo), X)
        # mixed numeric types
        @test isequivalent(backward(HalfSpace([1.0f0], 2.0f0), W, [1.0f0], algo), X)
        @test isequivalent(backward(Y, Float32.(W), b, algo), X)

        # 2D affine map
        W = hcat([2.0 3.0; -1.0 -2.0])
        b = [1.0, -2.0]
        X = Singleton([1.0, 2.0])
        Y = affine_map(W, X, b)
        @test isequivalent(backward(Y, W, b, algo), X)

        # 1D-2D affine map
        W = hcat([2.0; -1.0])
        b = [1.0, -2.0]
        X = Singleton([1.0])
        Y = affine_map(W, X, b)
        @test isequivalent(backward(Y, W, b, algo), X)

        # 2D-1D affine map
        W = hcat([2.0 3.0])
        b = [1.0]
        X = Singleton([1.0, 2.0])
        Y = affine_map(W, X, b)
        X2 = backward(Y, W, b, algo)
        @test X ⊆ X2
        # special case: Interval output
        W = hcat([2.0 3.0])
        b = [1.0]
        X = LineSegment([1.0, 2.0], [3.0, 4.0])
        Y = convert(Interval, affine_map(W, X, b))
        X2 = backward(Y, W, b, algo)
        H1 = HalfSpace([2.0, 3.0], 18.0)
        H2 = HalfSpace([-2.0, -3.0], -8.0)
        @test X ⊆ X2 && (X2.constraints == [H1, H2] || X2.constraints == [H2, H1])
        # special case: HalfSpace output
        Y = HalfSpace([2.0], 38.0)
        @test backward(Y, W, b, algo) == H1
        # special case: Universe output
        Y = Universe(1)
        @test backward(Y, W, b, algo) == Universe(2)
    end

    # approximate algorithms
    for algo in (BoxBackward(),)
        # 1D affine map
        W = hcat([2.0])
        b = [1.0]
        X = Singleton([1.0])
        Y = affine_map(W, X, b)
        @test backward(Y, W, b, PolyhedraBackward()) ⊆ backward(Y, W, b, algo)

        # 2D affine map
        W = hcat([2.0 3.0; -1.0 -2.0])
        b = [1.0, -2.0]
        X = Singleton([1.0, 2.0])
        Y = affine_map(W, X, b)
        @test backward(Y, W, b, PolyhedraBackward()) ⊆ backward(Y, W, b, algo)

        # 1D-2D affine map
        W = hcat([2.0; -1.0])
        b = [1.0, -2.0]
        X = Singleton([1.0])
        Y = affine_map(W, X, b)
        @test backward(Y, W, b, PolyhedraBackward()) ⊆ backward(Y, W, b, algo)
    end
end

@testset "Backward Id activation" begin
    y = [1.0, 2]
    Y = BallInf(y, 0.1)
    Yu = UnionSetArray([Y, Y])

    # exact algorithms
    for algo in (BoxBackward(), PolyhedraBackward())
        @test backward(y, Id(), algo) == y

        @test isequivalent(backward(Y, Id(), algo), Y)
        @test isequivalent(backward(Yu, Id(), algo), Yu)
    end
end

@testset "Backward ReLU activation" begin
    # exact results for all algorithms
    for algo in (BoxBackward(), PolyhedraBackward())
        # vector
        y = [1.0, 2]
        @test backward(y, ReLU(), algo) == y

        # 1D
        Y = HalfSpace([-2.0], 1.0)
        @test backward(Y, ReLU(), algo) == Universe(1)
        Y = Universe(1)
        @test backward(Y, ReLU(), algo) == Universe(1)

        # 2D, strictly negative
        Y = LineSegment([-1.0, -1.0], [-2.0, -2.0])
        @test backward(Y, ReLU(), algo) == EmptySet(2)
    end

    # exact algorithms
    for algo in (PolyhedraBackward(),)
        # 1D
        Y = BallInf([2.0], 1.0)
        @test backward(Y, ReLU(), algo) == Y
        Y = HalfSpace([2.0], 1.0)
        @test backward(Y, ReLU(), algo) == HalfSpace([1.0], 0.5)
        Y = HalfSpace([-2.0], -1.0)
        @test backward(Y, ReLU(), algo) == HalfSpace([-1.0], -0.5)
        Y = Interval(1.0, 2.0)
        @test backward(Y, ReLU(), algo) == Y
        Y = Interval(-1.0, 2.0)
        @test backward(Y, ReLU(), algo) == HalfSpace([1.0], 2.0)

        # 2D
        Pneg = HPolyhedron([HalfSpace([1.0, 0.0], 0.0), HalfSpace([0.0, 1.0], 0.0)])
        Px = HPolyhedron([HalfSpace([0.0, 1.0], 0.0), HalfSpace([1.0, 0.0], 2.0),
                          HalfSpace([-1.0, 0.0], -1.0)])
        Py = HPolyhedron([HalfSpace([1.0, 0.0], 0.0), HalfSpace([0.0, 1.0], 2.0),
                          HalfSpace([0.0, -1.0], -1.0)])
        Qx = HPolyhedron([HalfSpace([0.0, 1.0], 0.0), HalfSpace([1.0, 0.0], 2.0),
                          HalfSpace([-1.0, 0.0], 0.0)])
        Qy = HPolyhedron([HalfSpace([1.0, 0.0], 0.0), HalfSpace([0.0, 1.0], 2.0),
                          HalfSpace([0.0, -1.0], 0.0)])
        # strictly positive
        Y = LineSegment([1.0, 1.0], [2.0, 2.0])
        @test backward(Y, ReLU(), algo) == Y
        # only origin
        Y = Singleton([0.0, 0.0])
        @test backward(Y, ReLU(), algo) == Pneg
        # origin + positive
        Y = LineSegment([0.0, 0.0], [2.0, 2.0])
        @test backward(Y, ReLU(), algo) == UnionSetArray([Y, Pneg])
        # only x-axis
        Y = LineSegment([1.0, 0.0], [2.0, 0.0])
        @test backward(Y, ReLU(), algo) == Px
        # positive + x-axis
        Y = VPolygon([[1.0, 0.0], [2.0, 2.0], [2.0, 0.0]])
        @test backward(Y, ReLU(), algo) == UnionSetArray([Y, Px])
        # only y-axis
        Y = LineSegment([0.0, 1.0], [0.0, 2.0])
        @test backward(Y, ReLU(), algo) == Py
        # positive + y-axis
        Y = VPolygon([[0.0, 1.0], [2.0, 2.0], [0.0, 2.0]])
        @test backward(Y, ReLU(), algo) == UnionSetArray([Y, Py])
        # positive + both axes
        Y = VPolygon([[0.0, 1.0], [0.0, 2.0], [1.0, 0.0], [2.0, 0.0]])
        @test backward(Y, ReLU(), algo) == UnionSetArray([Y, Px, Py])
        # positive + x-axis + origin
        Y = VPolygon([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])
        @test backward(Y, ReLU(), algo) == UnionSetArray([Y, Qx, Pneg])
        # positive + y-axis + origin
        Y = VPolygon([[0.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
        @test backward(Y, ReLU(), algo) == UnionSetArray([Y, Qy, Pneg])
        # positive + both axes + origin
        Y = VPolygon([[0.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
        @test backward(Y, ReLU(), algo) == UnionSetArray([Y, Qx, Qy, Pneg])
        # origin + negative
        Y = LineSegment([0.0, 0.0], [-2.0, -2.0])
        @test backward(Y, ReLU(), algo) == Pneg
        # positive + negative + both axes + origin
        Y = VPolygon([[-1.0, -1.0], [-1.0, 3.0], [3.0, -1.0]])
        X = backward(Y, ReLU(), algo)
        @test X isa UnionSetArray && length(X) == 4 && X[2:4] == [Qx, Qy, Pneg] &&
              isequivalent(X[1], VPolygon([[0.0, 0.0], [0.0, 2.0], [2.0, 0.0]]))
        # unbounded
        Y = HalfSpace([-1.0, 0.0], -1.0)
        X = backward(Y, ReLU(), algo)
        ## union is too complex -> only perform partial tests
        @test X ⊆ Y && low(X) == [1.0, -Inf] && high(X) == [Inf, Inf]
        Y = HalfSpace([0.0, -1.0], -1.0)
        X = backward(Y, ReLU(), algo)
        ## union is too complex -> only perform partial tests
        @test X ⊆ Y && low(X) == [-Inf, 1.0] && high(X) == [Inf, Inf]
        # union
        Y = UnionSetArray([LineSegment([1.0, 1.0], [2.0, 2.0]), Singleton([0.0, 0.0])])
        @test backward(Y, ReLU(), algo) == UnionSetArray([Y[1], Pneg])

        # 3D
        # positive point
        Y = Singleton([1.0, 1.0, 1.0])
        @test backward(Y, ReLU(), algo) == Y
        # positive + negative + both axes + origin
        Y = BallInf(zeros(3), 1.0)
        X = backward(Y, ReLU(), algo)  # result: x <= 1 && y <= 1 && z <= 1
        # union is too complex -> only perform partial tests
        @test X isa UnionSetArray && length(X) == 8
        @test all(high(X, i) == 1.0 for i in 1:3)
        @test all(low(X, i) == -Inf for i in 1:3)

        # union
        Y = UnionSetArray([Singleton([2.0]), Singleton([3.0])])
        @test Y == backward(Y, ReLU(), algo)
    end

    # approximate algorithms
    for algo in (BoxBackward(),)
        # 1D
        Y = BallInf([2.0], 1.0)
        @test Y ⊆ backward(Y, ReLU(), algo)
        Y = HalfSpace([2.0], 1.0)
        @test HalfSpace([1.0], 0.5) ⊆ backward(Y, ReLU(), algo)
        Y = HalfSpace([-2.0], -1.0)
        @test HalfSpace([-1.0], -0.5) ⊆ backward(Y, ReLU(), algo)

        # 2D
        Pneg = HPolyhedron([HalfSpace([1.0, 0.0], 0.0), HalfSpace([0.0, 1.0], 0.0)])
        Px = HPolyhedron([HalfSpace([0.0, 1.0], 0.0), HalfSpace([1.0, 0.0], 2.0),
                          HalfSpace([-1.0, 0.0], -1.0)])
        Py = HPolyhedron([HalfSpace([1.0, 0.0], 0.0), HalfSpace([0.0, 1.0], 2.0),
                          HalfSpace([0.0, -1.0], -1.0)])
        Qx = HPolyhedron([HalfSpace([0.0, 1.0], 0.0), HalfSpace([1.0, 0.0], 2.0),
                          HalfSpace([-1.0, 0.0], 0.0)])
        Qy = HPolyhedron([HalfSpace([1.0, 0.0], 0.0), HalfSpace([0.0, 1.0], 2.0),
                          HalfSpace([0.0, -1.0], 0.0)])
        # strictly positive
        Y = LineSegment([1.0, 1.0], [2.0, 2.0])
        @test Y ⊆ backward(Y, ReLU(), algo)
        # only origin
        Y = Singleton([0.0, 0.0])
        @test Pneg ⊆ backward(Y, ReLU(), algo)
        # origin + positive
        Y = LineSegment([0.0, 0.0], [2.0, 2.0])
        @test UnionSetArray([Y, Pneg]) ⊆ backward(Y, ReLU(), algo)
        # only x-axis
        Y = LineSegment([1.0, 0.0], [2.0, 0.0])
        @test Px ⊆ backward(Y, ReLU(), algo)
        # positive + x-axis
        Y = VPolygon([[1.0, 0.0], [2.0, 2.0], [2.0, 0.0]])
        @test UnionSetArray([Y, Px]) ⊆ backward(Y, ReLU(), algo)
        # only y-axis
        Y = LineSegment([0.0, 1.0], [0.0, 2.0])
        @test Py ⊆ backward(Y, ReLU(), algo)
        # positive + y-axis
        Y = VPolygon([[0.0, 1.0], [2.0, 2.0], [0.0, 2.0]])
        @test UnionSetArray([Y, Py]) ⊆ backward(Y, ReLU(), algo)
        # positive + both axes
        Y = VPolygon([[0.0, 1.0], [0.0, 2.0], [1.0, 0.0], [2.0, 0.0]])
        @test UnionSetArray([Y, Px, Py]) ⊆ backward(Y, ReLU(), algo)
        # positive + x-axis + origin
        Y = VPolygon([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])
        @test UnionSetArray([Y, Qx, Pneg]) ⊆ backward(Y, ReLU(), algo)
        # positive + y-axis + origin
        Y = VPolygon([[0.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
        @test UnionSetArray([Y, Qy, Pneg]) ⊆ backward(Y, ReLU(), algo)
        # positive + both axes + origin
        Y = VPolygon([[0.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
        @test UnionSetArray([Y, Qx, Qy, Pneg]) ⊆ backward(Y, ReLU(), algo)
        # origin + negative
        Y = LineSegment([0.0, 0.0], [-2.0, -2.0])
        @test backward(Y, ReLU(), algo) == Pneg
        # positive + negative + both axes + origin
        Y = VPolygon([[-1.0, -1.0], [-1.0, 3.0], [3.0, -1.0]])
        X = backward(Y, ReLU(), algo)
        @test UnionSetArray([VPolygon([[0.0, 0.0], [0.0, 2.0], [2.0, 0.0]]), Qx, Qy, Pneg]) ⊆ X
        # unbounded
        Y = HalfSpace([-1.0, 0.0], -1.0)
        @test Y ⊆ backward(Y, ReLU(), algo)
        Y = HalfSpace([0.0, -1.0], -1.0)
        @test Y ⊆ backward(Y, ReLU(), algo)
        # # union
        Y = UnionSetArray([LineSegment([1.0, 1.0], [2.0, 2.0]), Singleton([0.0, 0.0])])
        @test UnionSetArray([Y[1], Pneg]) ⊆ backward(Y, ReLU(), algo)

        # 3D
        # positive point
        Y = Singleton([1.0, 1.0, 1.0])
        @test Y ⊆ backward(Y, ReLU(), algo)
        # positive + negative + both axes + origin
        Y = BallInf(zeros(3), 1.0)
        X = backward(Y, ReLU(), algo)  # result: x <= 1 && y <= 1 && z <= 1
        @test all(high(X, i) >= 1 for i in 1:3)
        @test all(low(X, i) == -Inf for i in 1:3)

        # union
        Y = UnionSetArray([Singleton([2.0]), Singleton([3.0])])
        @test Y ⊆ backward(Y, ReLU(), algo)
    end
end

@testset "Backward sigmoid activation" begin
    x = [0.0, 1]
    y = Sigmoid()(x)
    Y = BallInf(y, 0.0)

    # all algorithms
    for algo in (BoxBackward(), PolyhedraBackward(), DummyBackward())
        @test backward(y, Sigmoid(), algo) == x
    end

    # exact algorithms
    for algo in (BoxBackward(),)
        @test isequivalent(backward(Y, Sigmoid(), algo), Singleton(x))
    end

    # algorithms not supporting sigmoid activation
    for algo in (PolyhedraBackward(),)
        @test_throws ArgumentError backward(Y, Sigmoid(), algo)
    end
end

@testset "Backward leaky-ReLU activation" begin
    lr = LeakyReLU(0.1)
    x = [0.0, 1]
    X = Singleton(x)
    y = lr(x)
    Y = BallInf(y, 0.0)

    # all algorithms
    for algo in (BoxBackward(), PolyhedraBackward(), DummyBackward())
        @test backward(y, lr, algo) == x
    end

    # exact algorithms
    for algo in (PolyhedraBackward(),)
        @test isequivalent(backward(Y, lr, algo), X)

        Y2 = BallInf([0.0], 1.0)
        lr0 = LeakyReLU(0.0)
        X2 = backward(Y2, lr0, algo)  # equivalent to HalfSpace([1.0], 1.0)
        # union is too complex -> only perform partial tests
        @test high(X2) == [1.0] && low(X2) == [-Inf]
    end

    # exact algorithms (in this case)
    for algo in (BoxBackward(),)
        @test isequivalent(backward(Y, lr, algo), X)
    end

    # default algorithm for union
    for algo in (DummyBackward(),)
        y1 = Singleton([2.0])
        y2 = Singleton([3.0])
        x1 = backward(y1, lr, algo)
        x2 = backward(y2, lr, algo)
        Y2 = UnionSetArray([y1, y2])
        X2 = backward(Y2, lr, algo)
        @test X2 == UnionSetArray([x1, x2])
    end
end

@testset "Backward layer" begin
    W = [1.0 0; 0 1]
    b = [1.0, 0]
    L = DenseLayerOp(W, b, Id())
    y = [2.0, 1]
    Y = BallInf(y, 0.1)

    for algo in (BoxBackward(), PolyhedraBackward())
        @test backward(y, L, algo) == [1.0, 1]

        @test [1.0, 1] ∈ backward(Y, L, algo)
    end
end

@testset "Backward network" begin
    # 2D network
    N = example_network_222()
    x = [1.0, 2.0]
    Y = Singleton(N(x))
    for algo in (BoxBackward(), PolyhedraBackward())
        X = backward(Y, N, algo)
        @test x ∈ X
        x = [-4.0, 0.0]
        @test Singleton(N(x)) == Y && x ∈ X
    end

    # 1D/2D network
    N = example_network_1221()
    X = Interval(2.5, 5.0)
    Y = convert(Interval, forward(X, N, ConcreteForward()))

    # exact algorithms
    for algo in (PolyhedraBackward(),)
        @test isequivalent(X, backward(Y, N, algo))
    end

    # approximate algorithms
    for algo in (BoxBackward(),)
        @test X ⊆ backward(Y, N, algo)
    end
end
