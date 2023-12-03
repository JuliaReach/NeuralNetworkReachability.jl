"""
    ConcreteForward <: ForwardAlgorithm

Forward algorithm that uses concrete set operations.
"""
struct ConcreteForward <: ForwardAlgorithm end

# apply concrete affine map
function forward(X::LazySet, W::AbstractMatrix, b::AbstractVector, ::ConcreteForward)
    return affine_map(W, X, b)
end

# apply ReLU activation function
function forward(X::LazySet, ::ReLU, ::ConcreteForward)
    return concretize(rectify(X))
end
