"""
    SimpleBidirectional{FA<:ForwardAlgorithm, BA<:BackwardAlgorithm} <: BidirectionalAlgorithm

Simple bidirectional algorithm parametric in a forward and backward algorithm.

### Fields

- `fwd_algo` -- forward algorithm
- `bwd_algo` -- backward algorithm
"""
struct SimpleBidirectional{FA<:ForwardAlgorithm,BA<:BackwardAlgorithm} <: BidirectionalAlgorithm
    fwd_algo::FA
    bwd_algo::BA
end

# getter of forward algorithm
function fwd(algo::SimpleBidirectional)
    return algo.fwd_algo
end

# getter of backward algorithm
function bwd(algo::SimpleBidirectional)
    return algo.bwd_algo
end

# intersection of forward and backward result with box algorithms
function _fwd_bwd_intersection(::SimpleBidirectional{<:BoxForward,<:BoxBackward},
                               X::LazySet, Y::LazySet)
    return box_approximation(intersection(X, Y))
end

# convenience constructor that uses the polyhedra algorithms
function PolyhedraBidirectional()
    return SimpleBidirectional(ConcreteForward(), PolyhedraBackward())
end

# convenience constructor that uses the box algorithms
function BoxBidirectional()
    return SimpleBidirectional(BoxForward(), BoxBackward())
end
