@testset "PartitioningLeakyReLU" begin
    using NeuralNetworkReachability.BackwardAlgorithms: pwa_partitioning, PartitioningLeakyReLU

    # pwa_partitioning
    @test pwa_partitioning(ReLU(), 3, Float32) == PartitioningLeakyReLU{Float32}(3, 0.0f0)
    @test pwa_partitioning(LeakyReLU(0.1), 3, Float32) == PartitioningLeakyReLU{Float32}(3, 0.1f0)

    # PartitioningLeakyReLU
    P = PartitioningLeakyReLU(2, 0.1)
    @test length(P) == 4

    ev = [(HPolyhedron([HalfSpace([-1.0, 0.0], 0.0), HalfSpace([0.0, -1.0], 0.0)]),
           ([1.0 0; 0 1], [0.0, 0])),
          (HPolyhedron([HalfSpace([-1.0, 0.0], 0.0), HalfSpace([0.0, 1.0], 0.0)]),
           ([1 0; 0 0.1], [0.0, 0])),
          (HPolyhedron([HalfSpace([1.0, 0.0], 0.0), HalfSpace([0.0, -1.0], 0.0)]),
           ([0.1 0; 0 1], [0.0, 0])),
          (HPolyhedron([HalfSpace([1.0, 0.0], 0.0), HalfSpace([0.0, 1.0], 0.0)]),
           ([0.1 0; 0 0.1], [0.0, 0]))]
    rv = collect(P)
    @test length(rv) == 4 && length(unique(rv)) == 4
    @test ev[1] ∈ rv && ev[2] ∈ rv && ev[3] ∈ rv && ev[4] ∈ rv
end
