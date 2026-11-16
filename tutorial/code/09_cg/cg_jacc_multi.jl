using .Ex_cg: axpy, dot

function init_jacc_multi(data)
    gja0 = JACC.Multi.array(data.a0; ghost_dims = 1)
    gja1 = JACC.Multi.array(data.a1; ghost_dims = 1)
    gja2 = JACC.Multi.array(data.a2; ghost_dims = 1)
    jr = JACC.Multi.array(data.r)
    jp = JACC.Multi.array(data.p)
    gjp = JACC.Multi.array(data.p; ghost_dims = 1)
    js = JACC.Multi.array(data.s)
    gjs = JACC.Multi.array(data.s; ghost_dims = 1)
    jx = JACC.Multi.array(data.x)
    jr_old = JACC.Multi.array(data.r_old)
    jr_aux = JACC.Multi.array(data.r_aux)

    return (; gja0, gja1, gja2, jr, jp, gjp, js, gjs, jx, jr_old, jr_aux)
end

function cg_jacc_multi(data)
    function matvecmul(i, a1, a2, a3, x, y, ndev)
        ind = JACC.Multi.ghost_shift(i, a1)
        dev_id = JACC.Multi.device_id(a1)
        if dev_id == 1 && ind == 1
            y[ind] = a2[ind] * x[ind] + a1[ind] * x[ind + 1]
        elseif dev_id == ndev && ind == length(y)
            y[ind] = a3[ind] * x[ind - 1] + a2[ind] * x[ind]
        else
            y[ind] = a3[ind] * x[ind - 1] + a2[ind] * x[ind] +
                     a1[ind] * x[ind + 1]
        end
    end

    gja0, gja1, gja2, jr, jp, gjp, js, gjs, jx, jr_old, jr_aux = data
    ndev = JACC.Multi.ndev()
    SIZE = gja0.orig_size
    FloatType = eltype(gja0.a2[1])

    cond = 1.0
    while cond >= 1e-14
        JACC.Multi.copy!(jr_old, jr)

        JACC.Multi.parallel_for(
            SIZE, matvecmul, gja0, gja1, gja2, gjp, gjs, ndev)
        JACC.Multi.sync_ghost_elems!(gjs)
        JACC.Multi.copy!(js, gjs) #js = gjs

        alpha0 = JACC.Multi.parallel_reduce(SIZE, dot, jr, jr)
        alpha1 = JACC.Multi.parallel_reduce(SIZE, dot, jp, js)

        alpha = alpha0 / alpha1
        m_alpha = -alpha

        JACC.Multi.parallel_for(SIZE, axpy, m_alpha, jr, js)
        JACC.Multi.parallel_for(SIZE, axpy, alpha, jx, jp)

        beta0 = JACC.Multi.parallel_reduce(SIZE, dot, jr, jr)
        beta1 = JACC.Multi.parallel_reduce(SIZE, dot, jr_old, jr_old)
        beta = beta0 / beta1

        JACC.Multi.copy!(jr_aux, jr)

        JACC.Multi.parallel_for(SIZE, axpy, beta, jr_aux, jp)
        ccond = JACC.Multi.parallel_reduce(SIZE, dot, jr, jr)
        cond = ccond

        JACC.Multi.copy!(jp, jr_aux)
        JACC.Multi.copy!(gjp, jp) #gjp = jp
        JACC.Multi.sync_ghost_elems!(gjp)
    end
end
