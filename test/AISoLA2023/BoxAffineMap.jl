using NeuralNetworkReachability.ForwardAlgorithms: ForwardAlgorithm
import NeuralNetworkReachability.ForwardAlgorithms: forward

struct BoxAffineMap <: ForwardAlgorithm end

function forward(X::LazySet, W::AbstractMatrix, b::AbstractVector, ::BoxAffineMap)
    return box_approximation(W * X + b)
end
