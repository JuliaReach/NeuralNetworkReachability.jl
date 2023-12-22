"""
    DeepZ <: ForwardAlgorithm

Forward algorithm based on zonotopes for ReLU, sigmoid, and tanh activation
functions from [1].

[1]: Singh et al.: *Fast and Effective Robustness Certification*, NeurIPS 2018.
"""
struct DeepZ <: ForwardAlgorithm end

# apply affine map
function forward(Z, W::AbstractMatrix, b::AbstractVector, ::DeepZ)
    return affine_map(W, Z, b)
end

# apply ReLU activation function (implemented in LazySets)
function forward(Z::AbstractZonotope, ::ReLU, ::DeepZ)
    return overapproximate(Rectification(Z), Zonotope)
end

# apply sigmoid activation function
function forward(Z::AbstractZonotope, ::Sigmoid, ::DeepZ)
    f(x) = _sigmoid_DeepZ(x)
    f′(x) = _sigmoid2_DeepZ(x)
    return _overapproximate_zonotope(Z, f, f′)
end

# apply tanh activation function
function forward(Z::AbstractZonotope, ::Tanh, ::DeepZ)
    f(x) = tanh(x)
    f′(x) = 1 - tanh(x)^2
    return _overapproximate_zonotope(Z, f, f′)
end

function _sigmoid_DeepZ(x::Number)
    ex = exp(x)
    return ex / (1 + ex)
end

function _sigmoid2_DeepZ(x::Number)
    ex = exp(x)
    return ex / (1 + ex)^2
end

function _overapproximate_zonotope(Z::AbstractZonotope{N}, f, f′) where {N}
    c = copy(center(Z))
    G = copy(genmat(Z))
    n, m = size(G)
    row_idx = Vector{Int}()
    μ_idx = Vector{N}()

    @inbounds for i in 1:n
        lx, ux = low(Z, i), high(Z, i)
        uy = f(ux)

        if _isapprox(lx, ux)
            c[i] = uy
            for j in 1:m
                G[i, j] = zero(N)
            end
        else
            ly = f(lx)
            λ = min(f′(lx), f′(ux))
            μ₁ = (uy + ly - λ * (ux + lx)) / 2
            # Note: there is a typo in the paper (missing parentheses)
            μ₂ = (uy - ly - λ * (ux - lx)) / 2
            c[i] = c[i] * λ + μ₁
            for j in 1:m
                G[i, j] = G[i, j] * λ
            end
            push!(row_idx, i)
            push!(μ_idx, μ₂)
        end
    end

    q = length(row_idx)
    if q >= 1
        Gnew = zeros(N, n, q)
        j = 1
        @inbounds for i in row_idx
            Gnew[i, j] = μ_idx[j]
            j += 1
        end
        Gout = hcat(G, Gnew)
    else
        Gout = G
    end

    return Zonotope(c, remove_zero_columns(Gout))
end
