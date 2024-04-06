"""
    ConvSet{T<:LazySet{N}}

Wrapper of a set to represent a three-dimensional structure.

### Fields

- `set`  -- set of dimension `dims[1] * dims[2] * dims[3]`
- `dims` -- 3-tuple with the dimensions
"""
struct ConvSet{T<:LazySet}
    set::T
    dims::NTuple{3,Int}

    function ConvSet(set::T, dims::NTuple{3,Int}; validate=Val(true)) where {T}
        if validate isa Val{true} && (dim(set) != dims[1] * dims[2] * dims[3] ||
           dims[1] <= 0 || dims[2] <= 0 || dims[3] <= 0)
            throw(ArgumentError("invalid dimensions $(dim(set)) and $dims"))
        end
        return new{T}(set, dims)
    end
end
