"""
    PolyhedraBackward <: BackwardAlgorithm

Backward algorithm for piecewise-affine activations; uses a union of polyhedra.
"""
struct PolyhedraBackward <: BackwardAlgorithm end

function remove_constraints(::PolyhedraBackward, X::LazySet)
    return true
end

# apply inverse affine map to Y
function backward(Y::LazySet, W::AbstractMatrix, b::AbstractVector,
                  algo::PolyhedraBackward)
    m = size(W, 1)
    @assert m == dim(Y)
    if m == 1
        X = _backward_1D(Y, W, b, algo)
    else
        X = _backward_nD(Y, W, b, algo)
    end
    return simplify_set(X)
end

# apply inverse affine map to one-dimensional Y
function _backward_1D(Y::LazySet, W::AbstractMatrix, b::AbstractVector,
                      algo::PolyhedraBackward)
    return _backward_nD(Y, W, b, algo)  # fall back to general method
end

# specialization for polytopes
function _backward_1D(Y::AbstractPolytope, W::AbstractMatrix, b::AbstractVector,
                      ::PolyhedraBackward)
    @assert dim(Y) == size(W, 1) == length(b) == 1
    # if Y = [l, h], then X should have two constraints:
    # ax + b <= h  <=>  ax <= h - b
    # ax + b >= l  <=>  -ax <= b - l
    l, h = low(Y, 1), high(Y, 1)
    a = vec(W)
    N = promote_type(eltype(l), eltype(b))
    if eltype(a) != N
        a = Vector{N}(a)
    end
    return HPolyhedron([HalfSpace(a, h - b[1]), HalfSpace(-a, b[1] - l)])
end

# specialization for HalfSpace
function _backward_1D(Y::HalfSpace, W::AbstractMatrix, b::AbstractVector,
                      ::PolyhedraBackward)
    @assert dim(Y) == size(W, 1) == length(b) == 1
    # if Y = cx <= d (normalized: x <= d/c), then X should have one constraint:
    # ax + b <= d/c  <=>  ax <= d/c - b
    a = vec(W)
    offset = Y.b / Y.a[1] - b[1]
    N = promote_type(eltype(a), typeof(offset))
    if eltype(a) != N
        a = Vector{N}(a)
    end
    if typeof(offset) != N
        offset = N(offset)
    end
    return HalfSpace(a, offset)
end

# apply inverse affine map to n-dimensional Y
function _backward_nD(Y::LazySet, W::AbstractMatrix, b::AbstractVector,
                      ::PolyhedraBackward)
    return _backward_affine_map(W, Y, b)
end

# apply inverse affine map to universe
function backward(Y::Universe{N}, W::AbstractMatrix, b::AbstractVector,
                  ::PolyhedraBackward) where {N}
    @assert dim(Y) == size(W, 1) == length(b)
    return Universe{N}(size(W, 2))
end

# apply inverse affine map to union of sets
function backward(Y::UnionSetArray, W::AbstractMatrix, b::AbstractVector,
                  algo::PolyhedraBackward)
    return _backward_union(Y, W, b, algo)
end

# apply inverse leaky ReLU activation function
function backward(Y::LazySet, act::LeakyReLU, algo::PolyhedraBackward)
    return _backward_pwa(Y, act, algo)
end

function _backward_pwa(Y::LazySet{N}, act::ActivationFunction,
                       ::PolyhedraBackward) where {N}
    @assert ispolyhedral(Y) "expected a polyhedron, got $(typeof(Y))"

    out = LazySet{N}[]
    n = dim(Y)

    for (Pj, αj) in pwa_partitioning(act, n, N)
        # inverse affine map
        αA, αb = αj
        if iszero(αA)
            # constant case
            if αb ∈ Y
                push!(out, Pj)
            end
        else
            # injective case
            R = affine_map_inverse(αA, Y, αb)
            X = intersection(Pj, R)
            push!(out, X)
        end
    end

    filter!(!isempty, out)
    return simplify_union(out; n=dim(Y), N=N)
end

# apply inverse ReLU activation function
function backward(Y::LazySet, act::ReLU, ::PolyhedraBackward)
    return _backward_PolyhedraBackward(Y, act)
end

function _backward_PolyhedraBackward(Y::LazySet, act::ReLU)
    n = dim(Y)
    if n == 1
        X = _backward_1D(Y, act)
    elseif n == 2
        X = _backward_2D(Y, act)
    else
        X = _backward_nD(Y, act)
    end
    return simplify_set(X)
end

function _backward_1D(Y::LazySet{N}, ::ReLU) where {N}
    l, h = extrema(Y, 1)
    if !_leq(l, zero(N))  # l > 0
        if isinf(h)
            # positive but unbounded from above
            return HalfSpace(N[-1], N(-l))
        else
            # positive and bounded from above, so ReLU⁻¹ = Id
            return Y
        end
    elseif isinf(h)
        # unbounded everywhere
        return Universe{N}(1)
    else
        # bounded from above
        return HalfSpace(N[1], N(h))
    end
end

# more efficient special case for Interval
function _backward_1D(Y::Interval{N}, ::ReLU) where {N}
    if !_leq(min(Y), zero(N))
        return Y
    else
        return HalfSpace(N[1], max(Y))
    end
end

# apply inverse ReLU activation function to 2D polyhedron
function _backward_2D(Y::LazySet{N}, ::ReLU) where {N}
    @assert ispolyhedral(Y) "expected a polyhedron, got $(typeof(Y))"

    out = LazySet{N}[]

    # intersect with nonnegative quadrant
    Q₊ = HPolyhedron([HalfSpace(N[-1, 0], zero(N)),
                      HalfSpace(N[0, -1], zero(N))])
    if Y ⊆ Q₊
        Y₊ = Y
    else
        Y₊ = intersection(Y, Q₊)
    end
    if isempty(Y₊)  # pre-image is empty if image was not nonnegative
        return EmptySet{N}(dim(Y))
    end
    if !_leq(high(Y₊, 1), zero(N)) && !_leq(high(Y₊, 2), zero(N))  # at least one positive point (assuming convexity)
        push!(out, Y₊)
    end

    # intersect with x-axis
    H₋x = HalfSpace(N[0, 1], zero(N))
    Rx = intersection(Y₊, H₋x)
    isempty_Rx = isempty(Rx)
    if !isempty_Rx
        _extend_relu_2d_xaxis!(out, Rx, H₋x, N)
    end

    # intersect with y-axis
    H₋y = HalfSpace(N[1, 0], zero(N))
    Ry = intersection(Y₊, H₋y)
    isempty_Ry = isempty(Ry)
    if !isempty_Ry
        _extend_relu_2d_yaxis!(out, Ry, H₋y, N)
    end

    # if the origin is contained, the nonpositive quadrant is part of the solution
    if !isempty_Rx && !isempty_Ry && N[0, 0] ∈ Y₊
        Tz = HPolyhedron([H₋y, H₋x])
        push!(out, Tz)
    end

    return simplify_union(out; n=dim(Y), N=N)
end

function _extend_relu_2d_xaxis!(out, R::AbstractPolyhedron, H₋, N)
    h = high(R, 1)
    if h <= zero(N)  # not relevant, case handled elsewhere
        return
    end
    l = low(R, 1)
    if isinf(h)  # upper-bound constraint redundant
        T = HPolyhedron([H₋, HalfSpace(N[-1, 0], -l)])
    else
        T = HPolyhedron([H₋, HalfSpace(N[1, 0], h), HalfSpace(N[-1, 0], -l)])
    end
    push!(out, T)
    return nothing
end

function _extend_relu_2d_yaxis!(out, R::AbstractPolyhedron, H₋, N)
    h = high(R, 2)
    if h <= zero(N)  # not relevant, case handled elsewhere
        return
    end
    l = low(R, 2)
    if isinf(h)  # upper-bound constraint redundant
        T = HPolyhedron([H₋, HalfSpace(N[0, -1], -l)])
    else
        T = HPolyhedron([H₋, HalfSpace(N[0, 1], h), HalfSpace(N[0, -1], -l)])
    end
    push!(out, T)
    return nothing
end

# apply inverse ReLU activation function to arbitrary polyhedron
#
# First compute the intersection Y₊ with the nonnegative orthant.
# Use a bitvector v, where entry 1 means "nonnegative" and entry 0 means "0".
# For instance, in 2D, v = (1, 1) stands for the positive orthant and v =(0, 1)
# stands for "x = 0". For each orthant, the corresponding preimage is the
# inverse linear map of Y₊ under Diagonal(v). Since this is a special case, it
# maps a linear constraint cx <= d to a new constraint whose normal vector is
# [c1v1, c2v2, ...], and since v is a bitvector, it acts as a filter.
function _backward_nD(Y::LazySet{N}, ::ReLU) where {N}
    @assert ispolyhedral(Y) "expected a polyhedron, got $(typeof(Y))"

    out = LazySet{N}[]
    n = dim(Y)

    # intersect with nonnegative orthant
    Q₊ = HPolyhedron([HalfSpace(SingleEntryVector(i, n, -one(N)), zero(N)) for i in 1:n])
    Y₊ = intersection(Y, Q₊)
    if isempty(Y₊)  # pre-image is empty if image was not nonnegative
        return EmptySet{N}(dim(Y))
    end

    # find dimensions in which the set is positive (to save case distinctions)
    skip = falses(n)
    @inbounds for i in 1:n
        if !_leq(low(Y₊, i), zero(N))
            skip[i] = true
        end
    end
    fix = trues(n)  # value at non-skip indices does not matter

    for v in BitvectorIterator(skip, fix, false)
        if !any(v)
            # nonpositive orthant: case needs special treatment
            if zeros(N, n) ∈ Y₊
                Tz = HPolyhedron([HalfSpace(SingleEntryVector(i, n, one(N)), zero(N)) for i in 1:n])
                push!(out, Tz)
            end
            continue
        elseif all(v)
            # nonnegative orthant: more efficient treatment of special case
            # add if set contains at least one positive point
            if all(!_leq(high(Y₊, i), zero(N)) for i in 1:n)
                push!(out, Y₊)
            end
            # last iteration
            break
        end

        # inverse linear map
        R = _linear_map_inverse(v, Y₊)

        # compute orthant corresponding to v
        constraints = HalfSpace{N,SingleEntryVector{N}}[]
        @inbounds for i in 1:n
            if v[i]
                push!(constraints, HalfSpace(SingleEntryVector(i, n, -one(N)), zero(N)))
            else
                push!(constraints, HalfSpace(SingleEntryVector(i, n, one(N)), zero(N)))
            end
        end
        O = HPolyhedron(constraints)

        # intersect preimage of Y with orthant
        X = intersection(R, O)

        push!(out, X)
    end

    filter!(!isempty, out)
    return simplify_union(out; n=dim(Y), N=N)
end

# d is a vector representing a diagonal matrix
function _linear_map_inverse(d::AbstractVector{<:Number}, P::LazySet)
    constraints_P = constraints_list(P)
    constraints_MP = LazySets._preallocate_constraints(constraints_P)
    has_undefs = false
    N = promote_type(eltype(d), eltype(P))
    @inbounds for (i, c) in enumerate(constraints_P)
        cinv = _linear_map_inverse_mult(d, c.a)
        if iszero(cinv)
            if zero(N) <= c.b
                # constraint is redundant
                has_undefs = true
            else
                # constraint is infeasible
                return EmptySet{N}(length(cinv))
            end
        else
            constraints_MP[i] = HalfSpace(cinv, c.b)
        end
    end
    if has_undefs  # there were redundant constraints, so remove them
        constraints_MP = [constraints_MP[i]
                          for i in eachindex(constraints_MP)
                          if isassigned(constraints_MP, i)]
    end
    if isempty(constraints_MP)
        # in the current usage, each dimension is constrained and d is nonzero, so
        # at least one constraint must remain
        # COV_EXCL_START
        return Universe{N}(size(A, 2))
        # COV_EXCL_STOP
    end
    return HPolyhedron(constraints_MP)
end

function _linear_map_inverse_mult(d::AbstractVector{<:Number}, a)
    return [d[i] * a[i] for i in eachindex(a)]
end

function _linear_map_inverse_mult(d::AbstractVector{Bool}, a)
    return [d[i] ? a[i] : zero(eltype(a)) for i in eachindex(a)]
end

# disambiguation
for T in (:ReLU, :LeakyReLU)
    @eval begin
        function backward(Y::Singleton, act::$T, algo::PolyhedraBackward)
            if all(>(0), element(Y))
                return Singleton(backward(element(Y), act, algo))
            else
                return _backward_PolyhedraBackward(Y, act)
            end
        end

        function backward(Y::UnionSetArray, act::$T, algo::PolyhedraBackward)
            return _backward_union(Y, act, algo)
        end
    end
end
