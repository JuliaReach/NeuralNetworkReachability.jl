"""
    Verisig{R} <: ForwardAlgorithm

Forward algorithm for sigmoid and tanh activation functions from [1].

### Fields

- `algo` -- reachability algorithm of type `TMJets`

### Notes

The implementation is known to be unsound in some cases.

The implementation currently only supports neural networks with a single hidden
layer.

[1] Ivanov et al.: *Verisig: verifying safety properties of hybrid systems with
neural network controllers*, HSCC 2019.
"""
struct Verisig{R} <: ForwardAlgorithm
    algo::R
end

# default constructor
function Verisig()
    require(@__MODULE__, :ReachabilityAnalysis; fun_name="Verisig")

    return Verisig(_Verisig_default_algorithm())
end

function forward(X::LazySet, net::FeedforwardNetwork, algo::Verisig)
    require(@__MODULE__, :ReachabilityAnalysis; fun_name="forward")

    return _forward(X, net, algo)
end

# disambiguation for singleton
function forward(X::AbstractSingleton, net::FeedforwardNetwork, ::Verisig)
    return forward(X, net, DefaultForward())
end

function load_Verisig()
    return quote
        function _Verisig_default_algorithm()
            return TMJets(; abstol=1e-14, orderQ=2, orderT=6)
        end

        function _forward(X::LazySet, net::FeedforwardNetwork, algo::Verisig)
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
    end  # quote
end  # function load_Verisig()
