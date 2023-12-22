"""
BackwardAlgorithm

Abstract supertype of backward algorithms.
"""
abstract type BackwardAlgorithm end

remove_constraints(::BackwardAlgorithm, x) = false
