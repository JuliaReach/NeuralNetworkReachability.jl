module ForwardAlgorithms

using ..Util
using LinearAlgebra: Diagonal
using ControllerFormats
using LazySets
using LazySets: remove_zero_columns
using ReachabilityBase.Arrays: SingleEntryVector
using ReachabilityBase.Comparison: _isapprox, isapproxzero
using ReachabilityBase.Require: require

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
