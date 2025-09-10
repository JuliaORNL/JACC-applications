import JACC
JACC.@init_backend

function axpy_jacc!(x, y, alpha = 2.5)
    JACC.parallel_for(length(x), alpha, x, y) do i, alpha, x, y
        @inbounds x[i] += alpha * y[i]
    end
end
