"""
BackwardAlgorithm

Abstract supertype of backward algorithms.
"""
abstract type BackwardAlgorithm end

function remove_constraints(::BackwardAlgorithm, x)
    return false
end
