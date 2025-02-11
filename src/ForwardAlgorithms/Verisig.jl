"""
    Verisig{R} <: ForwardAlgorithm

Forward algorithm for sigmoid and tanh activation functions from [IvanovWAPL19](@citet).

### Fields

- `algo` -- reachability algorithm of type `TMJets`

### Notes

The implementation is known to be unsound in some cases.

The implementation currently only supports neural networks with a single hidden
layer.
"""
struct Verisig{R} <: ForwardAlgorithm
    algo::R
end

# default constructor
function Verisig()
    algo = _default_algorithm_Verisig(nothing)
    return Verisig(algo)
end

function _default_algorithm_Verisig(dummy)
    mod = isdefined(Base, :get_extension) ?
          Base.get_extension(@__MODULE__, :ReachabilityAnalysisExt) : @__MODULE__
    require(mod, :ReachabilityAnalysis; fun_name="Verisig")
end

function forward(X::LazySet, net::FeedforwardNetwork, algo::Verisig)
    return _forward_Verisig(X, net, algo)
end

function _forward_Verisig(X, net, algo)
    mod = isdefined(Base, :get_extension) ?
          Base.get_extension(@__MODULE__, :ReachabilityAnalysisExt) : @__MODULE__
    require(mod, :ReachabilityAnalysis; fun_name="forward")
end

# disambiguation for singleton
function forward(X::AbstractSingleton, net::FeedforwardNetwork, ::Verisig)
    return forward(X, net, DefaultForward())
end
