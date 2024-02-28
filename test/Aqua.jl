using NeuralNetworkReachability, Test
import Aqua

@testset "Aqua tests" begin
    Aqua.test_all(NeuralNetworkReachability; ambiguities=false)

    # do not warn about ambiguities in dependencies
    Aqua.test_ambiguities(NeuralNetworkReachability)
end
