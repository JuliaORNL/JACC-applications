import JACC
JACC.@init_backend

const JF = JACC.default_float()

function est_pi_jacc(N::Integer)
    halfpi = JACC.parallel_reduce(N; op = *) do n
        nf = JF(n)
        return 4.0 * nf^2 / (4.0 * nf^2 - 1.0)
    end
    return 2.0 * halfpi
end

function est_pi_jacc_2(N::Integer)
    halfpi = JACC.parallel_reduce(JACC.launch_spec(), N; op = *) do n
        nf = JF(n)
        return 4.0 * nf^2 / (4.0 * nf^2 - 1.0)
    end
    return 2.0 * Base.Array(halfpi)[]
end
