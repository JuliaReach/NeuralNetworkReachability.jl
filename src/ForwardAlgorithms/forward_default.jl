# propagate x through network
function forward(x, net::FeedforwardNetwork, algo::ForwardAlgorithm=DefaultForward())
    for L in layers(net)
        x = forward(x, L, algo)
    end
    return x
end

# propagate x through network and store all intermediate results
function _forward_store(x, net::FeedforwardNetwork, algo::ForwardAlgorithm)
    results = Vector{Tuple}()
    for L in layers(net)
        y, z = _forward_intermediate(x, L, algo)
        push!(results, (y, z))
        x = z
    end
    return results
end

# propagate x through layer
function forward(x, L::DenseLayerOp, algo::ForwardAlgorithm)
    y = forward(x, L.weights, L.bias, algo)  # apply affine map
    z = forward(y, L.activation, algo)  # apply activation function
    return z
end

# propagate x through layer and return all intermediate results
function _forward_intermediate(x, L::DenseLayerOp, algo::ForwardAlgorithm)
    y = forward(x, L.weights, L.bias, algo)  # apply affine map
    z = forward(y, L.activation, algo)  # apply activation function
    return y, z
end

# apply affine map
function forward(x, W::AbstractMatrix, b::AbstractVector, ::ForwardAlgorithm)
    return W * x + b
end

# apply activation function
function forward(x, act::ActivationFunction, ::ForwardAlgorithm)
    return act(x)
end

# activation functions must be explicitly supported for sets
function forward(X::LazySet, act::ActivationFunction, algo::ForwardAlgorithm)
    throw(ArgumentError("activation function $act not supported by algorithm " *
                        "$algo for set type $(typeof(X))"))
end

# identity activation is automatically supported for sets
function forward(X::LazySet, ::Id, ::ForwardAlgorithm)
    return X
end
