module ReachabilityAnalysisExt

using NeuralNetworkReachability.ForwardAlgorithms
using LazySets, ControllerFormats

@static if isdefined(Base, :get_extension)
    using ReachabilityAnalysis: @taylorize, TaylorIntegration, TaylorN, Taylor1, TaylorModelN,
                                TMJets, IVP, BlackBoxContinuousSystem, solve, evaluate
else
    using ..ReachabilityAnalysis: @taylorize, TaylorIntegration, TaylorN, Taylor1, TaylorModelN,
                                  TMJets, IVP, BlackBoxContinuousSystem, solve, evaluate
end

# COV_EXCL_START

# Eq. (4)-(6)
# d(σ(x))/dx = σ(x) * (1 - σ(x))
# g(t, x) = σ(tx) = 1 / (1 + exp(-tx))
# dg(t, x)/dt = g'(t, x) = x * g(t, x) * (1 - g(t, x))
@taylorize function _Verisig_sigmoid!(dx, x, p, t)
    n = div(length(x), 2)
    for i in 1:n
        xᴶ = x[i]
        xᴾ = x[n + i]
        dx[i] = zero(xᴶ)
        dx[n + i] = xᴶ * (xᴾ - xᴾ^2)
    end
    return dx
end

# Footnote 3
# d(tanh(x))/dx = 1 - tanh(x)^2
# g(t, x) = tanh(tx)
# dg(t, x)/dt = g'(t, x) = x * (1 - g(t, x)^2)
@taylorize function _Verisig_tanh!(dx, x, p, t)
    n = div(length(x), 2)
    for i in 1:n
        xᴶ = x[i]
        xᴾ = x[n + i]
        dx[i] = zero(xᴶ)
        dx[n + i] = xᴶ * (1 - xᴾ^2)
    end
    return dx
end

# COV_EXCL_STOP

function ForwardAlgorithms._default_algorithm_Verisig(::Nothing)
    return TMJets(; abstol=1e-14, orderQ=2, orderT=6)
end

function ForwardAlgorithms._forward_Verisig(X::LazySet, net::FeedforwardNetwork,
                                            algo::Verisig)
    @assert algo.algo isa TMJets "reachability algorithm of type " *
                                 "$(typeof(algo.algo)) is not supported"
    xᴾ = X

    for layer in layers(net)
        W = layer.weights
        b = layer.bias
        m = length(layer)
        act = layer.activation

        xᴶ = W * xᴾ + b  # affine map
        if act isa Id
            xᴾ = xᴶ
            continue
        elseif act isa Sigmoid
            xᴾ = Singleton(fill(0.5, m))
            act! = _Verisig_sigmoid!
        elseif act isa Tanh
            xᴾ = Singleton(fill(0.0, m))
            act! = _Verisig_tanh!
        else
            throw(ArgumentError("unsupported activation function: $act"))
        end
        X0 = _cartesian_product(xᴶ, xᴾ)
        ivp = IVP(BlackBoxContinuousSystem(act!, 2 * m), X0)
        sol = solve(ivp; tspan=(0.0, 1.0), alg=algo.algo)
        # obtain final reach set (Vector{TaylorModelN})
        xᴾ = evaluate(sol.F.Xk[end], 1.0)[(m + 1):(2 * m)]
    end
    return xᴾ
end

function _cartesian_product(X::AffineMap, Y::Singleton)
    return X × Y
end

function _cartesian_product(X::Vector{<:TaylorModelN}, Y::Singleton)
    # not implemented
    return error("the Verisig algorithm for multiple hidden layers " *
                 "is not implemented yet")
end

end  # module
