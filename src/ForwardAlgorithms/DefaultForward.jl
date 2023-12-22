"""
    DefaultForward <: ForwardAlgorithm

Default forward algorithm, which works for vector-like inputs.
"""
struct DefaultForward <: ForwardAlgorithm end

# propagating set through network not supported (exception below)
function forward(::LazySet, ::FeedforwardNetwork, algo::DefaultForward)
    throw(ArgumentError("cannot apply $(typeof(algo)) to a set input"))
end

# propagate singleton through network
function forward(X::AbstractSingleton, net::FeedforwardNetwork,
                 algo::DefaultForward=DefaultForward())
    x = element(X)
    y = forward(x, net, algo)
    return Singleton(y)
end

function forward(X::AbstractSingleton, net::FeedforwardNetwork, ::ForwardAlgorithm)
    return forward(X, net, DefaultForward())
end

# propagate singleton through network and store all intermediate results
function _forward_store(X::AbstractSingleton, net::FeedforwardNetwork,
                        algo::DefaultForward=DefaultForward())
    x = element(X)
    results = _forward_store(x, net, algo)
    return [(Singleton(y), Singleton(z)) for (y, z) in results]
end

function _forward_store(X::AbstractSingleton, net::FeedforwardNetwork, ::ForwardAlgorithm)
    return _forward_store(X, net, DefaultForward())
end
