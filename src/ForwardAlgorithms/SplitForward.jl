"""
    SplitForward{S<:ForwardAlgorithm,FS,FM} <: ForwardAlgorithm

Forward algorithm that splits a set, then computes the image under the neural
network, and finally merges the resulting sets again, all according to a policy.

### Fields

- `algo` -- algorithm to be applied between splitting and merging
- `split_function` -- function for splitting
- `merge_function` -- function for merging
"""
struct SplitForward{FA<:ForwardAlgorithm,FS,FM} <: ForwardAlgorithm
    algo::FA
    split_function::FS
    merge_function::FM
end

# default constructor
function SplitForward(algo::ForwardAlgorithm)
    # box approximation and split in two sets per dimension
    split_fun = X -> split(box_approximation(X), 2 * ones(Int, dim(X)))
    # box approximation of the union
    merge_fun = X -> box_approximation(X)
    return SplitForward(algo, split_fun, merge_fun)
end

# split X, propagate sets through network, and merge
function forward(X::LazySet, net::FeedforwardNetwork, algo::SplitForward)
    X_split = algo.split_function(X)
    Y_union = UnionSetArray()
    for X0 in X_split
        Y = forward(X0, net, algo.algo)
        push!(array(Y_union), Y)
    end
    return algo.merge_function(Y_union)
end

# disambiguation for singleton
function forward(X::AbstractSingleton, net::FeedforwardNetwork, ::SplitForward)
    return forward(X, net, DefaultForward())
end
