module ForwardAlgorithms

using ..Util
using LinearAlgebra: Diagonal
using ControllerFormats: ActivationFunction, DenseLayerOp, FeedforwardNetwork,
                         FlattenLayerOp, Id, LeakyReLU, ReLU, Sigmoid, Tanh,
                         layers
using LazySets: AbstractHyperrectangle, AbstractPolynomialZonotope,
                AbstractPolytope, AbstractSingleton, AbstractZonotope,
                AffineMap, Arrays, HalfSpace, Hyperrectangle, LazySet,
                Rectification, Singleton, SparsePolynomialZonotope,
                UnionSetArray, Zonotope, affine_map, array, box_approximation,
                center, concretize, convex_hull, dim, element, expmat, genmat,
                genmat_dep, genmat_indep, high, intersection, isbounded,
                linear_map, low, minkowski_sum, ngens_dep, ngens_indep,
                nparams, overapproximate, rectify, reduce_order,
                remove_redundant_generators, ×
using ReachabilityBase.Arrays: SingleEntryVector, remove_zero_columns
using ReachabilityBase.Comparison: _isapprox, isapproxzero
using ReachabilityBase.Require: require
using Requires: @require

export forward,
       DefaultForward,
       ConcreteForward,
       LazyForward,
       BoxForward,
       SplitForward,
       DeepZ,
       AI2Box, AI2Zonotope, AI2Polytope,
       Verisig,
       PolyZonoForward

include("ForwardAlgorithm.jl")
include("DefaultForward.jl")
include("forward_default.jl")
include("ConcreteForward.jl")
include("LazyForward.jl")
include("BoxForward.jl")
include("SplitForward.jl")
include("DeepZ.jl")
include("AI2.jl")
include("Verisig.jl")
include("PolyZonoForward.jl")

include("init.jl")

end  # module
