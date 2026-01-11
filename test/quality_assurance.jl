using NeuralNetworkReachability, Test
import Aqua, ExplicitImports

@testset "ExplicitImports tests" begin
    ignores = (:affine_map_inverse, :TaylorModelN, :_preallocate_constraints)
    @test isnothing(ExplicitImports.check_all_explicit_imports_are_public(NeuralNetworkReachability;
                                                                          ignore=ignores))
    ignores = (:TaylorModelN,)
    @test isnothing(ExplicitImports.check_all_explicit_imports_via_owners(NeuralNetworkReachability;
                                                                          ignore=ignores))
    ignores = (:_preallocate_constraints,)
    @test isnothing(ExplicitImports.check_all_qualified_accesses_are_public(NeuralNetworkReachability;
                                                                            ignore=ignores))
    @test isnothing(ExplicitImports.check_all_qualified_accesses_via_owners(NeuralNetworkReachability))
    @test isnothing(ExplicitImports.check_no_implicit_imports(NeuralNetworkReachability))
    @test isnothing(ExplicitImports.check_no_self_qualified_accesses(NeuralNetworkReachability))
    # false positive due to meta-programming and external macro
    ignores = (:LeakyReLU, :TaylorIntegration, :Taylor1, :TaylorN)
    @test isnothing(ExplicitImports.check_no_stale_explicit_imports(NeuralNetworkReachability;
                                                                    ignore=ignores))
end

@testset "Aqua tests" begin
    Aqua.test_all(NeuralNetworkReachability)
end
