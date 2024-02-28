# backpropagate y through network
function backward(y, net::FeedforwardNetwork, algo::BackwardAlgorithm)
    @inbounds for L in reverse(layers(net))
        y = backward(y, L, algo)

        # early termination check
        if y isa EmptySet
            y = EmptySet(dim_in(net))
            break
        end
    end
    return y
end

# backpropagate y through layer
function backward(y, L::DenseLayerOp, algo::BackwardAlgorithm)
    x = backward(y, L.activation, algo)  # apply inverse activation function
    remove_constraints(algo, x) && remove_constraints!(x)
    if x isa EmptySet
        return x
    end
    w = backward(x, L.weights, L.bias, algo)  # apply inverse affine map
    remove_constraints(algo, x) && remove_constraints!(w)
    return w
end

remove_constraints!(::LazySet) = nothing

function remove_constraints!(P::LazySets.HPoly)
    m1 = length(P.constraints)
    remove_redundant_constraints!(P)
    m2 = length(P.constraints)
    # println("$(m1 - m2)/$m1 constraints removed")
    return nothing
end

# apply inverse affine map to Y
function backward(Y, W::AbstractMatrix, b::AbstractVector, ::BackwardAlgorithm)
    return _backward_affine_map(W, Y, b)
end

# try to invert the matrix
function _backward_affine_map(W::AbstractMatrix, y::AbstractVector, b::AbstractVector)
    return inv(W) * (y .- b)
end

function _backward_affine_map(W::AbstractMatrix, Y::LazySet, b::AbstractVector)
    X = affine_map_inverse(W, Y, b)
    return simplify_set(X)
end

# apply inverse affine map to a union of sets
function backward(Y::UnionSetArray, W::AbstractMatrix, b::AbstractVector,
                  algo::BackwardAlgorithm)
    return _backward_union(Y, W, b, algo)
end

function _backward_union(Y::UnionSetArray{N}, W::AbstractMatrix,
                         b::AbstractVector, algo::BackwardAlgorithm) where {N}
    @assert dim(Y) == size(W, 1) == length(b)
    out = []
    for Yi in array(Y)
        append_sets!(out, backward(Yi, W, b, algo))
    end
    filter!(!isempty, out)
    return simplify_union(out; n=size(W, 2), N=N)
end

append_sets!(Xs, X::LazySet) = push!(Xs, X)
append_sets!(Xs, X::UnionSetArray) = append!(Xs, array(X))

# apply inverse piecewise-affine activation function to a union of sets
for T in (:ReLU, :LeakyReLU)
    @eval begin
        function backward(Y::UnionSetArray, act::$T, algo::BackwardAlgorithm)
            return _backward_union(Y, act, algo)
        end
    end
end

function _backward_union(Y::LazySet{N}, act::ActivationFunction,
                         algo::BackwardAlgorithm) where {N}
    out = []
    for Yi in array(Y)
        Xs = backward(Yi, act, algo)
        if !(Xs isa EmptySet)
            append_sets!(out, Xs)
        end
    end
    return simplify_union(out; n=dim(Y), N=N)
end

function backward(y::AbstractVector, act::ActivationFunction, ::BackwardAlgorithm)
    return _inverse(y, act)
end

_inverse(x::AbstractVector, act::ActivationFunction) = [_inverse(xi, act) for xi in x]
_inverse(x::Number, ::ReLU) = x >= zero(x) ? x : zero(x)
_inverse(x::Number, ::Sigmoid) = @. -log(1 / x - 1)
_inverse(x::Number, act::LeakyReLU) = x >= zero(x) ? x : x / act.slope

# invertible activations defined for numbers can be defined for singletons
for T in (:Sigmoid, :LeakyReLU)
    @eval begin
        function backward(Y::Singleton, act::$T, algo::BackwardAlgorithm)
            return Singleton(backward(element(Y), act, algo))
        end
    end
end

# activation functions must be explicitly supported for sets
function backward(X::LazySet, act::ActivationFunction, algo::BackwardAlgorithm)
    throw(ArgumentError("activation function $act not supported by algorithm " *
                        "$algo for set type $(typeof(X))"))
end

# disambiguation: apply inverse identity activation function to Y
for T in (:AbstractVector, :LazySet, :UnionSetArray)
    @eval begin
        function backward(y::$T, ::Id, ::BackwardAlgorithm)
            return y
        end
    end
end
