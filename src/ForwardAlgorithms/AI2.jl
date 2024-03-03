"""
    AI2Box <: AI2

[`AI2`](@ref) forward algorithm for ReLU activation functions based on abstract
interpretation with the interval domain from [1].

### Notes

This algorithm is less precise than [`BoxForward`](@ref) because it abstracts
after every step, including the affine map.

[1]: Gehr et al.: *AI²: Safety and robustness certification of neural networks
with abstract interpretation*, SP 2018.
"""
struct AI2Box <: ForwardAlgorithm end

"""
    AI2Zonotope <: AI2

[`AI2`](@ref) forward algorithm for ReLU activation functions based on abstract
interpretation with the zonotope domain from [1].

### Fields

- `join_algorithm` -- (optional; default: `"join"`) algorithm to compute the
                      join of two zonotopes

[1]: Gehr et al.: *AI²: Safety and robustness certification of neural networks
with abstract interpretation*, SP 2018.
"""
struct AI2Zonotope{S} <: ForwardAlgorithm
    join_algorithm::S
end

# the default join algorithm is "join"
AI2Zonotope() = AI2Zonotope("join")

"""
    AI2Polytope <: AI2

[`AI2`](@ref) forward algorithm for ReLU activation functions based on abstract
interpretation with the polytope domain from [1].

[1]: Gehr et al.: *AI²: Safety and robustness certification of neural networks
with abstract interpretation*, SP 2018.
"""
struct AI2Polytope <: ForwardAlgorithm end

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
    meet = (X, Y) -> overapproximate(Intersection(X, Y), Zonotope)
    join = (X, Y) -> overapproximate(ConvexHull(X, Y), Zonotope; algorithm=algo.join_algorithm)
    return _forward_AI2_ReLU(Z; meet=meet, join=join)
end

# polytope: the convex hull of all pairwise polytopes
function forward(P::AbstractPolytope, ::ReLU, ::AI2Polytope)
    return _forward_AI2_ReLU(P; meet=intersection, join=convex_hull)
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
