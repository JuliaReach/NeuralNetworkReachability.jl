# polynomial approximation algorithms
# implementing types must implement:
# - `_order(::PolynomialApproximation)::Int`
# - `_hq(::PolynomialApproximation, h, q)`
abstract type PolynomialApproximation end

# quadratic approximation algorithms
abstract type QuadraticApproximation <: PolynomialApproximation end

# order of quadratic polynomials
_order(::QuadraticApproximation) = 2

# output size of the generator matrices
function _hq(::QuadraticApproximation, h, q)
    h′ = h + div(h * (h + 1), 2)
    q′ = (h + 1) * q + div(q * (q + 1), 2)
    return (h′, q′)
end

struct RegressionQuadratic <: QuadraticApproximation
    samples::Int  # number of samples

    function RegressionQuadratic(samples::Int)
        @assert samples >= 3 "need at least 3 samples for the regression"
        return new(samples)
    end
end

struct ClosedFormQuadratic <: QuadraticApproximation end

struct TaylorExpansionQuadratic <: QuadraticApproximation end

"""
    PolyZonoForward{A<:PolynomialApproximation,N,R} <: ForwardAlgorithm

Forward algorithm based on poynomial zonotopes via polynomial approximation from
[1].

### Fields

- `polynomial_approximation` -- method for polynomial approximation
- `reduced_order`          -- order to which the result will be reduced after
                                each layer
- `compact`                  -- predicate for compacting the result after each
                                layer

### Notes

The default constructor takes keyword arguments with the following defaults:

- `polynomial_approximation`: `RegressionQuadratic(10)`, i.e., quadratic
  regression with 10 samples
- `compact`: `() -> true`, i.e., compact after each layer

See the subtypes of `PolynomialApproximation` for available polynomial
approximation methods.

[1]: Kochdumper et al.: *Open- and closed-loop neural network verification using
polynomial zonotopes*, NFM 2023.
"""
struct PolyZonoForward{A<:PolynomialApproximation,O,R} <: ForwardAlgorithm
    polynomial_approximation::A
    reduced_order::O
    compact::R

    function PolyZonoForward(; reduced_order::O,
                             polynomial_approximation::A=RegressionQuadratic(10),
                             compact=() -> true) where {A<:PolynomialApproximation,O}
        return new{A,O,typeof(compact)}(polynomial_approximation, reduced_order, compact)
    end
end

# apply affine map
function forward(PZ, W::AbstractMatrix, b::AbstractVector, ::PolyZonoForward)
    return affine_map(W, PZ, b)
end

# apply activation function (Algorithm 1)
function forward(PZ::AbstractPolynomialZonotope{N}, act::ActivationFunction,
                 algo::PolyZonoForward) where {N}
    n = dim(PZ)
    c = center(PZ)
    G = genmat_dep(PZ)
    GI = genmat_indep(PZ)
    E = expmat(PZ)
    l, u = extrema(PZ)
    h = ngens_dep(PZ)
    q = ngens_indep(PZ)
    h′, q′ = _hq(algo.polynomial_approximation, h, q)
    c′ = zeros(N, n)
    G′ = Matrix{N}(undef, n, h′)
    GI′ = Matrix{N}(undef, n, q′)
    E′ = Matrix{Int}(undef, n, h′)
    dl = zeros(N, n)
    du = zeros(N, n)
    # compute polynomial approximation for each neuron
    @inbounds for i in 1:n
        c′[i], G′[i, :], GI′[i, :], E′[i, :], dl[i], du[i] = _forward_neuron(act, c[i],
                                                                             @view(G[i, :]),
                                                                             @view(GI[i, :]),
                                                                             @view(E[i, :]), l[i],
                                                                             u[i], algo)
    end
    PZ2 = SparsePolynomialZonotope(c′, G′, GI′, E′)
    PZ3 = minkowski_sum(PZ2, Hyperrectangle(; low=dl, high=du))
    PZ4 = reduce_order(PZ3, algo.reduced_order)
    PZ5 = algo.compact() ? remove_redundant_generators(PZ4) : PZ4
    return PZ5
end

# disambiguation for identity activation function
function forward(PZ::AbstractPolynomialZonotope, ::Id, algo::PolyZonoForward)
    return PZ
end

# ReLU neuron approximation: compute exact result if l >= 0 or u <= 0
function _forward_neuron(act::ReLU, c::N, G, GI, E, l, u, algo::PolyZonoForward) where {N}
    if u <= zero(N)
        # nonpositive -> 0
        c, G, GI, E = _polynomial_image_zero(c, G, GI, E, algo.polynomial_approximation)
        dl, du = zero(N), zero(N)
        return (c, G, GI, E, dl, du)
    end
    if l >= zero(N)
        # nonnegative -> identity
        c, G, GI, E = _polynomial_image_id(c, G, GI, E, algo.polynomial_approximation)
        dl, du = zero(N), zero(N)
        return (c, G, GI, E, dl, du)
    end
    return _forward_neuron_general(act, c, G, GI, E, l, u, algo)
end

# general neuron approximation
function _forward_neuron(act::ActivationFunction, c, G, GI, E, l, u, algo::PolyZonoForward)
    return _forward_neuron_general(act, c, G, GI, E, l, u, algo)
end

function _forward_neuron_general(act::ActivationFunction, c, G, GI, E, l, u, algo::PolyZonoForward)
    polynomial = _polynomial_approximation(act, l, u, algo.polynomial_approximation)
    c′, G′, GI′, E′ = _polynomial_image(c, G, GI, E, polynomial, algo.polynomial_approximation)
    dl, du = _approximation_error(act, l, u, polynomial, algo.polynomial_approximation)
    return (c′, G′, GI′, E′, dl, du)
end

#########################################
# Quadratic approximation (Section 3.1) #
#########################################

# any activation function: quadratic regression
function _polynomial_approximation(act::ActivationFunction, l::N, u::N,
                                   reg::RegressionQuadratic) where {N}
    xs = range(l, u; length=reg.samples)
    A = hcat(xs .^ 2, xs, ones(N, reg.samples))
    b = act.(xs)
    return A \ b
end

# ReLU activation function: closed-form expression
function _polynomial_approximation(::ReLU, l, u, ::ClosedFormQuadratic)
    throw(ArgumentError("not implemented yet"))
end

# sigmoid and tanh activation functions: Taylor-series expansion
function _polynomial_approximation(::Union{Sigmoid,Tanh}, l, u, ::TaylorExpansionQuadratic)
    throw(ArgumentError("not implemented yet"))
end

########################################################################
# Image of a polynomial zonotope under a polynomial function (Prop. 2) #
########################################################################

function _Eq(E, h)
    Ê2(i) = [E[i] + E[j] for j in (i + 1):h]
    Ê = vcat(2 .* E, vcat([Ê2(i) for i in 1:(h - 1)]...))
    return vcat(E, Ê)
end

_Ḡ(G, GI, h, q, a₁, N) = iszero(q) ? N[] : [2 * a₁ * G[i] * GI for i in 1:h]

# quadratic polynomial
function _polynomial_image(c::N, G, GI, E, polynomial, ::QuadraticApproximation) where {N}
    h = length(G)
    q = length(GI)
    a₁, a₂, a₃ = polynomial
    Ĝ2(i) = [2 * G[i] * G[j] for j in (i + 1):h]
    Ĝ = vcat(G .^ 2, vcat([Ĝ2(i) for i in 1:(h - 1)]...))
    Ḡ = _Ḡ(G, GI, h, q, a₁, N)
    Ǧ2(i) = [GI[i] * GI[j] for j in (i + 1):q]
    Ǧ = iszero(q) ? N[] :
         vcat((0.5 * a₁) .* GI .^ 2, vcat([2 * a₁ * Ǧ2(i) for i in 1:(q - 1)]...))

    cq = a₁ * c^2 + a₂ * c + a₃
    if !isempty(GI)
        cq += a₁ / 2 * sum(gj -> gj^2, GI)
    end
    Gq = vcat(((2 * a₁ * c + a₂) * G), (a₁ * Ĝ))
    GIq = vcat((2 * a₁ * c + a₂) * GI, Ḡ, Ǧ)
    Eq = _Eq(E, h)

    return (cq, Gq, GIq, Eq)
end

# default for zero polynomial: fall back to standard implementation
function _polynomial_image_zero(c, G, GI, E, approx)
    polynomial = zeros(N, _order(approx) + 1)
    return _polynomial_image(c, G, GI, E, polynomial, approx)
end

# image under quadratic zero polynomial
function _polynomial_image_zero(::N, G, GI, E, approx::QuadraticApproximation) where {N}
    h = length(G)
    q = length(GI)
    h′, q′ = _hq(approx, h, q)
    cq = zero(N)
    Gq = zeros(N, h′)
    GIq = zeros(N, q′)
    Eq = _Eq(E, h)
    return (cq, Gq, GIq, Eq)
end

# default for identity polynomial: fall back to standard implementation
function _polynomial_image_id(c::N, G, GI, E, approx) where {N}
    polynomial = zeros(N, _order(approx) + 1)
    polynomial[end - 1] = one(N)
    return _polynomial_image(c, G, GI, E, polynomial, approx)
end

# image under quadratic identity polynomial
function _polynomial_image_id(c::N, G, GI, E, approx::QuadraticApproximation) where {N}
    h = length(G)
    q = length(GI)
    h′, q′ = _hq(approx, h, q)
    Gq = vcat(G, zeros(N, h′ - h))
    if iszero(q)
        GIq = GI
    else
        Ḡ = _Ḡ(G, GI, h, q, zero(N), N)
        GIq = vcat(GI, Ḡ, zeros(N, q′ - q - h))
    end
    Eq = _Eq(E, h)
    return (c, Gq, GIq, Eq)
end

################################################################
# Error estimate of the polynomial approximation (Section 3.2) #
################################################################

# ReLU activation function for quadratic polynomial
function _approximation_error(::ReLU, l::N, u::N, polynomial, ::QuadraticApproximation) where {N}
    a₁, a₂, a₃ = polynomial
    d⁻(x) = -a₁ * x^2 - a₂ * x - a₃
    d⁺(x) = -a₁ * x^2 + (1 - a₂) * x - a₃

    # find x⁰⁻ s.t. d⁻'(x⁰⁻) = -2a₁ * x⁰⁻ - a₂ = 0 ⇔ x⁰⁻ = -a₂/(2a₁)
    # find x⁰⁺ s.t. d⁺'(x⁰⁺) = -2a₁ * x⁰⁺ + 1 - a₂ = 0 ⇔ x⁰⁺ = (1-a₂)/(2a₁)
    # in both cases, we need to check for division by zero: a₁ ≠ 0
    if isapproxzero(a₁)
        dl = min(d⁻(l), d⁻(zero(N)))
        du = max(d⁺(l), d⁺(zero(N)))
    else
        x⁰⁻ = -a₂ / (2a₁)
        x⁰⁺ = (1 - a₂) / (2a₁)
        if l <= x⁰⁻ <= zero(N)
            dl = min(d⁻(l), d⁻(x⁰⁻), d⁻(zero(N)))
            du = max(d⁻(l), d⁻(x⁰⁻), d⁻(zero(N)))
        else
            dl = min(d⁻(l), d⁻(zero(N)))
            du = max(d⁻(l), d⁻(zero(N)))
        end
        if zero(N) <= x⁰⁺ <= u
            dl = min(dl, d⁺(zero(N)), d⁺(x⁰⁺), d⁺(u))
            du = max(du, d⁺(zero(N)), d⁺(x⁰⁺), d⁺(u))
        else
            dl = min(dl, d⁺(zero(N)), d⁺(u))
            du = max(du, d⁺(zero(N)), d⁺(u))
        end
    end
    return dl, du
end

# sigmoid and tanh activation functions
function _approximation_error(::Union{Sigmoid,Tanh}, l::N, u::N, polynomial,
                              ::PolynomialApproximation) where {N}
    throw(ArgumentError("not implemented yet"))
end
