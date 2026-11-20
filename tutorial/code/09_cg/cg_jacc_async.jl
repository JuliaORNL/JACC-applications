using .Ex_cg: axpy, dot

function init_jacc_async(data)
    a0 = JACC.Async.array(1, data.a0)
    a1 = JACC.Async.array(1, data.a1)
    a2 = JACC.Async.array(1, data.a2)
    r = JACC.Async.array(2, data.r)
    p = JACC.Async.array(1, data.p)
    s1 = JACC.Async.array(1, data.s)
    s2 = JACC.Async.array(2, data.s)
    x = JACC.Async.array(1, data.x)
    r_old = JACC.Async.array(1, data.r_old)
    r_aux = JACC.Async.array(1, data.r_aux)

    return (; a0, a1, a2, r, p, s1, s2, x, r_old, r_aux)
end

function cg_jacc_async(data)
    function matvecmul(i, a1, a2, a3, x, y, SIZE)
        if i == 1
            y[i] = a2[i] * x[i] + a1[i] * x[i + 1]
        elseif i == SIZE
            y[i] = a3[i] * x[i - 1] + a2[i] * x[i]
        elseif i > 1 && i < SIZE
            y[i] = a3[i] * x[i - 1] + a2[i] * +x[i] + a1[i] * +x[i + 1]
        end
    end

    a0, a1, a2, r, p, s1, s2, x, r_old, r_aux = data

    SIZE = length(a0)
    cond = 1.0

    while cond[1, 1] >= 1e-14
        copyto!(r_old, r)

        JACC.Async.parallel_for(1, SIZE, matvecmul, a0, a1, a2, p, s1, SIZE)

        alpha1 = JACC.Async.parallel_reduce(1, SIZE, dot, p, s1)
        alpha0 = JACC.Async.parallel_reduce(2, SIZE, dot, r, r)
        JACC.Async.synchronize()

        alpha = JACC.to_host(alpha0)[] / JACC.to_host(alpha1)[]
        negative_alpha = -alpha

        copyto!(s2, s1)
        JACC.Async.parallel_for(1, SIZE, axpy, alpha, x, p)
        JACC.Async.parallel_for(2, SIZE, axpy, negative_alpha, r, s2)
        JACC.Async.synchronize()

        beta1 = JACC.Async.parallel_reduce(1, SIZE, dot, r_old, r_old)
        beta0 = JACC.Async.parallel_reduce(2, SIZE, dot, r, r)
        JACC.Async.synchronize()
        beta = JACC.to_host(beta0)[] / JACC.to_host(beta1)[]

        copyto!(r_aux, r)

        JACC.Async.parallel_for(1, SIZE, axpy, beta, r_aux, p)
        ccond = JACC.Async.parallel_reduce(2, SIZE, dot, r, r)
        JACC.Async.synchronize()
        cond = JACC.to_host(ccond)[]

        copyto!(p, r_aux)
    end
end
