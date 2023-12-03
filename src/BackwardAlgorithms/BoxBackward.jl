"""
    BoxBackward <: BackwardAlgorithm

Backward algorithm that uses a polyhedral approximation with axis-aligned
linear constraints.
"""
struct BoxBackward <: BackwardAlgorithm end

function backward(Y::LazySet, act::ReLU, ::BoxBackward)
    return _backward_box(Y, act)
end

function backward(Y::UnionSetArray, act::ReLU, ::BoxBackward)
    return _backward_box(Y, act)
end

function _backward_box(Y::LazySet{N}, ::ReLU) where {N}
    n = dim(Y)

    # intersect with nonnegative orthant
    Q₊ = HPolyhedron([HalfSpace(SingleEntryVector(i, n, -one(N)), zero(N)) for i in 1:n])
    Y₊ = intersection(Y, Q₊)
    if isempty(Y₊)  # pre-image is empty if image was not nonnegative
        return EmptySet{N}(dim(Y))
    end

    constraints = Vector{HalfSpace{N,SingleEntryVector{N}}}()
    @inbounds for i in 1:n
        e₊ = SingleEntryVector(i, n, one(N))
        upper = ρ(e₊, Y₊)
        if upper < N(Inf)
            push!(constraints, HalfSpace(e₊, upper))
        end

        e₋ = SingleEntryVector(i, n, -one(N))
        lower = -ρ(e₋, Y₊)
        if !_leq(lower, zero(N))
            push!(constraints, HalfSpace(e₋, lower))
        end
    end
    if isempty(constraints)
        return Universe{N}(n)
    end
    return HPolyhedron(constraints)
end

for T in (Sigmoid, LeakyReLU)
    @eval begin
        function backward(Y::AbstractHyperrectangle, act::$T, ::BoxBackward)
            l = _inverse(low(Y), act)
            h = _inverse(high(Y), act)
            return Hyperrectangle(; low=l, high=h)
        end
    end
end
