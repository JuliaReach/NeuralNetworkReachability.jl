for dummy in [1]
    N = example_network_222()
    X = BallInf(zeros(2), 1.0)

    # Verisig
    if !isdefined(@__MODULE__, :ReachabilityAnalysis)
        @test_throws AssertionError Verisig()
        @test_throws AssertionError forward(X, N, Verisig(nothing))
    end

    # AI2Zonotope
    if !isdefined(@__MODULE__, :IntervalConstraintProgramming)
        @test_throws AssertionError forward(X, N, AI2Zonotope())
    end
end
