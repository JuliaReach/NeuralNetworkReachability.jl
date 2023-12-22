simplify_set(X::EmptySet) = X
simplify_set(X::LazySet{N}) where {N} = isempty(X) ? EmptySet{N}(dim(X)) : X

function simplify_union(sets::AbstractVector; n=1, N=Float64)
    if length(sets) > 1
        return UnionSetArray([X for X in sets])  # allocate set-specific array
    elseif length(sets) == 1
        return sets[1]
    else
        return EmptySet{N}(n)
    end
end
