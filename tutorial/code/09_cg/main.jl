module Ex_cg

function axpy(i, alpha, x, y)
    @inbounds x[i] += alpha * y[i]
end

function dot(i, x, y)
    @inbounds return x[i] * y[i]
end

function init(SIZE; dtype = Float64)
    a0 = ones(dtype, SIZE)
    a1 = fill(dtype(4.0), SIZE)
    a2 = ones(dtype, SIZE)
    r = fill(dtype(0.5), SIZE)
    p = fill(dtype(0.5), SIZE)
    s = zeros(dtype, SIZE)
    x = zeros(dtype, SIZE)
    r_old = zeros(dtype, SIZE)
    r_aux = zeros(dtype, SIZE)

    return (; a0, a1, a2, r, p, s, x, r_old, r_aux)
end

end
