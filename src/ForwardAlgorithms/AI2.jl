"""
    AI2Box <: AI2

AI2 forward algorithm for ReLU activation functions based on abstract
interpretation with the interval domain from [GehrMDTCV18](@citet).

### Notes

This algorithm is less precise than [`BoxForward`](@ref) because it abstracts
after every step, including the affine map.
"""
struct AI2Box <: ForwardAlgorithm end

"""
    AI2Zonotope <: AI2

AI2 forward algorithm for ReLU activation functions based on abstract
interpretation with the zonotope domain from [GehrMDTCV18](@citet).

### Fields

- `join_algorithm` -- (optional; default: `"join"`) algorithm to compute the
                      join of two zonotopes
"""
struct AI2Zonotope{S} <: ForwardAlgorithm
    join_algorithm::S
end

# the default join algorithm is "join"
AI2Zonotope() = AI2Zonotope("join")

"""
    AI2Polytope <: AI2

AI2 forward algorithm for ReLU activation functions based on abstract
interpretation with the polytope domain from [GehrMDTCV18](@citet).
"""
struct AI2Polytope <: ForwardAlgorithm end

# meet and join algorithms for different abstract domains
const _meet_zonotope = (X, Y) -> overapproximate(X ∩ Y, Zonotope)
const _join_zonotope(algo) = (X, Y) -> overapproximate(X ∪ Y, Zonotope; algorithm=algo)
const _meet_polytope = intersection
const _join_polytope = convex_hull

# apply affine map

# box: box approximation of the affine map
function forward(H, W::AbstractMatrix, b::AbstractVector, ::AI2Box)
    return box_approximation(W * H + b)
end

# zonotope and polytope: closed under affine map
function forward(X, W::AbstractMatrix, b::AbstractVector, ::Union{AI2Zonotope,AI2Polytope})
    return affine_map(W, X, b)
end

# apply ReLU activation function
# for each dimension 1:n
# 1(a) if nonnegative: nothing
# 1(b) if negative: project
# 1(c) if both nonnegative and negative: intersect with half-spaces and project negative
# 2: take the domain element(s) corresponding to the previous set(s)
# 3(c): union of the two sets, then take the corresponding domain element

# box: exploits that Box(ReLU(H)) = ReLU(H)
function forward(H::AbstractHyperrectangle, ::ReLU, ::AI2Box)
    return rectify(H)
end

# zonotope: intersection the zonotope overapproximation of all pairwise projected intersections
function forward(Z::AbstractZonotope, ::ReLU, algo::AI2Zonotope)
    _load_IntervalConstraintProgramming(nothing)

    return _forward_AI2_ReLU(Z; meet=_meet_zonotope, join=_join_zonotope(algo.join_algorithm))
end

# defined in `IntervalConstraintProgrammingExt.jl`
function _load_IntervalConstraintProgramming(dummy)
    mod = isdefined(Base, :get_extension) ?
          Base.get_extension(@__MODULE__, :IntervalConstraintProgrammingExt) : @__MODULE__
    require(mod, :IntervalConstraintProgramming; fun_name="forward", explanation="with AI2Zonotope")
    return nothing
end

# polytope: the convex hull of all pairwise polytopes
function forward(P::AbstractPolytope, ::ReLU, ::AI2Polytope)
    return _forward_AI2_ReLU(P; meet=_meet_polytope, join=_join_polytope)
end

function _forward_AI2_ReLU(X::LazySet{N}; meet, join) where {N}
    n = dim(X)
    d = ones(N, n)  # reused vector for "almost" identity matrices
    for i in 1:n
        if low(X, i) >= 0  # nonnegative case
            continue
        elseif high(X, i) <= 0  # negative case
            d[i] = zero(N)
            D = Diagonal(d)
            X = linear_map(D, X)
            d[i] = one(N)
        else  # mixed case
            # nonnegative part
            H1 = HalfSpace(SingleEntryVector(i, n, -one(N)), zero(N))
            X1 = meet(X, H1)

            # negative part
            H2 = HalfSpace(SingleEntryVector(i, n, one(N)), zero(N))
            X2 = meet(X, H2)
            d[i] = zero(N)
            D = Diagonal(d)
            X2′ = linear_map(D, X2)
            d[i] = one(N)

            # join
            X = join(X1, X2′)
        end
    end
    return X
end
