module BackwardAlgorithms

using ControllerFormats: ActivationFunction, DenseLayerOp, FeedforwardNetwork,
                         Id, LeakyReLU, ReLU, Sigmoid, dim_in, layers
using LazySets: AbstractHyperrectangle, AbstractPolyhedron, AbstractPolytope,
                EmptySet, HPolyhedron, HPolytope, HalfSpace, Hyperrectangle,
                Interval, LazySet, Singleton, UnionSetArray, Universe, array,
                constraints_list, dim, element, high, intersection,
                ispolyhedral, low, remove_redundant_constraints!, ρ,
                affine_map_inverse, _preallocate_constraints  # NOTE: these are internal functions
using ReachabilityBase.Arrays: SingleEntryVector
using ReachabilityBase.Comparison: _leq
using ReachabilityBase.Iteration: BitvectorIterator
using LinearAlgebra: Diagonal

export backward,
       PolyhedraBackward,
       BoxBackward

include("simplify_sets.jl")
include("PartitioningLeakyReLU.jl")
include("BackwardAlgorithm.jl")
include("backward_default.jl")
include("PolyhedraBackward.jl")
include("BoxBackward.jl")

end  # module
