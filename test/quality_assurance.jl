using NeuralNetworkReachability, Test
import Aqua, ExplicitImports

@testset "ExplicitImports tests" begin
    ignores = (:affine_map_inverse, :_preallocate_constraints)
    @test isnothing(ExplicitImports.check_all_explicit_imports_are_public(NeuralNetworkReachability;
                                                                          ignore=ignores))
    @test isnothing(ExplicitImports.check_all_explicit_imports_via_owners(NeuralNetworkReachability))
    # false positive due to package extensions:
    ignores = (:_ext_constructor_Verisig, :_ext_forward_AI2Zonotope, :_ext_forward_Verisig)
    @test isnothing(ExplicitImports.check_all_qualified_accesses_are_public(NeuralNetworkReachability;
                                                                            ignore=ignores))
    @test isnothing(ExplicitImports.check_all_qualified_accesses_via_owners(NeuralNetworkReachability))
    @test isnothing(ExplicitImports.check_no_implicit_imports(NeuralNetworkReachability))
    @test isnothing(ExplicitImports.check_no_self_qualified_accesses(NeuralNetworkReachability))
    # false positive due to meta-programming and external macro
    ignores = (:LeakyReLU, :Taylor1, :TaylorN)
    @test isnothing(ExplicitImports.check_no_stale_explicit_imports(NeuralNetworkReachability;
                                                                    ignore=ignores))
end

@testset "Aqua tests" begin
    # Requires is only used in old versions
    @static if VERSION >= v"1.9"
        stale_deps = (ignore=[:Requires],)
    else
        stale_deps = true
    end

    Aqua.test_all(NeuralNetworkReachability; stale_deps=stale_deps)
end
