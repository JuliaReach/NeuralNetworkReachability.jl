module BidirectionalAlgorithms

using ..ForwardAlgorithms
using ..ForwardAlgorithms: ForwardAlgorithm, _forward_store
using ..BackwardAlgorithms
using ..BackwardAlgorithms: BackwardAlgorithm
using ControllerFormats: FeedforwardNetwork, layers
using LazySets: EmptySet, LazySet, box_approximation, dim, intersection

export bidirectional,
       SimpleBidirectional,
       PolyhedraBidirectional,
       BoxBidirectional

include("BidirectionalAlgorithm.jl")
include("bidirectional_default.jl")
include("SimpleBidirectional.jl")

end  # module
