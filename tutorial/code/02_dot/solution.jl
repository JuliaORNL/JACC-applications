import JACC
JACC.@init_backend

function dot_jacc(x::AbstractVector, y::AbstractVector)
    return JACC.parallel_reduce(length(x), x, y) do i, x, y
        @inbounds return x[i] * y[i]
    end
end
