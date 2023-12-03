"""
    LazyForward <: ForwardAlgorithm

Forward algorithm that uses lazy set operations.
"""
struct LazyForward <: ForwardAlgorithm end

# apply lazy ReLU activation function
function forward(X::LazySet, ::ReLU, ::LazyForward)
    return Rectification(X)
end
