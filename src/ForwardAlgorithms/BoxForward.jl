"""
    BoxForward{AMA<:ForwardAlgorithm} <: ForwardAlgorithm

Forward algorithm that uses a box approximation for non-identity activations and
applies the affine map according to the specified algorithm.

### Fields

- `affine_map_algorithm` -- algorithm to apply for affine maps
"""
struct BoxForward{AMA<:ForwardAlgorithm} <: ForwardAlgorithm
    affine_map_algorithm::AMA
end

# default constructor: compute concrete affine map (which is a zonotope)
function BoxForward()
    return BoxForward(ConcreteForward())
end

# apply affine map according to the algorithm options
function forward(X::LazySet, W::AbstractMatrix, b::AbstractVector, algo::BoxForward)
    return forward(X, W, b, algo.affine_map_algorithm)
end

# apply ReLU activation function (exploits `Box(ReLU(x)) = ReLU(Box(X))`)
function forward(X::LazySet, ::ReLU, ::BoxForward)
    return rectify(box_approximation(X))
end

# apply monotonic activation function
for ACT in (:Sigmoid, :Tanh, :LeakyReLU)
    @eval function forward(X::LazySet, act::$ACT, ::BoxForward)
        @assert isbounded(X) "this algorithm requires a bounded input set"
        l, h = extrema(X)
        return Hyperrectangle(; low=act(l), high=act(h))
    end
end
