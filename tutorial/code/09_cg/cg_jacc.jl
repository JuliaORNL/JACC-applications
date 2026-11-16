using .Ex_cg: axpy, dot

function init_jacc(data)
    a0 = JACC.to_device(data.a0)
    a1 = JACC.to_device(data.a1)
    a2 = JACC.to_device(data.a2)
    r = JACC.to_device(data.r)
    p = JACC.to_device(data.p)
    s = JACC.to_device(data.s)
    x = JACC.to_device(data.x)
    r_old = JACC.to_device(data.r_old)
    r_aux = JACC.to_device(data.r_aux)

    return (; a0, a1, a2, r, p, s, x, r_old, r_aux)
end

function cg_jacc(data)
    function matvecmul(i, a1, a2, a3, x, y, SIZE)
        if i == 1
            y[i] = a2[i] * x[i] + a1[i] * x[i + 1]
        elseif i == SIZE
            y[i] = a3[i] * x[i - 1] + a2[i] * x[i]
        elseif i > 1 && i < SIZE
            y[i] = a3[i] * x[i - 1] + a2[i] * +x[i] + a1[i] * +x[i + 1]
        end
    end
    a0, a1, a2, r, p, s, x, r_old, r_aux = data
    SIZE = length(a0)
    cond = 1.0
    while cond[1, 1] >= 1e-14
        r_old = copy(r)

        JACC.parallel_for(SIZE, matvecmul, a0, a1, a2, p, s, SIZE)

        alpha0 = JACC.parallel_reduce(SIZE, dot, r, r)
        alpha1 = JACC.parallel_reduce(SIZE, dot, p, s)

        alpha = alpha0 / alpha1
        negative_alpha = alpha * -1.0

        JACC.parallel_for(SIZE, axpy, negative_alpha, r, s)
        JACC.parallel_for(SIZE, axpy, alpha, x, p)

        beta0 = JACC.parallel_reduce(SIZE, dot, r, r)
        beta1 = JACC.parallel_reduce(SIZE, dot, r_old, r_old)
        beta = beta0 / beta1

        r_aux = copy(r)

        JACC.parallel_for(SIZE, axpy, beta, r_aux, p)
        ccond = JACC.parallel_reduce(SIZE, dot, r, r)
        cond = ccond

        p = copy(r_aux)
    end
end
