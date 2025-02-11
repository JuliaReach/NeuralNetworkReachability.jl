using NeuralNetworkReachability, Test
import Aqua

@testset "Aqua tests" begin
    # Requires is only used in old versions
    @static if VERSION >= v"1.9"
        stale_deps = (ignore=[:Requires],)
    else
        stale_deps = true
    end

    Aqua.test_all(NeuralNetworkReachability; stale_deps=stale_deps)
end
