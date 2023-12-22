"""
    PartitioningLeakyReLU{N<:Real}

Iterator over the partitions of a leaky ReLU activation.

### Fields

- `n`     -- dimension
- `slope` -- slope of the leaky ReLU activation
"""
struct PartitioningLeakyReLU{N<:Real}
    n::Int
    slope::N
end

function Base.length(it::PartitioningLeakyReLU)
    return 2^it.n
end

function Base.iterate(it::PartitioningLeakyReLU{N}, state=nothing) where {N}
    if isnothing(state)
        bv_it = BitvectorIterator(falses(it.n), trues(it.n), true)
        bv, bv_state = iterate(bv_it)
    else
        bv_it, bv_state = state
        bv_res = iterate(bv_it, bv_state)
        if isnothing(bv_res)
            return nothing
        end
        bv, bv_state = bv_res
    end
    state = isnothing(bv_state) ? nothing : (bv_it, bv_state)
    P = _pwa_partition_LeakyReLU(bv, N)
    v = [bv[i] ? one(N) : it.slope for i in eachindex(bv)]
    αA = Diagonal(v)
    αb = zeros(N, length(v))
    res = (P, (αA, αb))
    return (res, state)
end

function _pwa_partition_LeakyReLU(bv, N)
    return HPolyhedron([HalfSpace(SingleEntryVector(i, length(bv), bv[i] ? -one(N) : one(N)),
                                  zero(N)) for i in eachindex(bv)])
end

function pwa_partitioning(::ReLU, n::Int, N)
    return PartitioningLeakyReLU(n, zero(N))
end

function pwa_partitioning(act::LeakyReLU, n::Int, N)
    return PartitioningLeakyReLU(n, N(act.slope))
end
