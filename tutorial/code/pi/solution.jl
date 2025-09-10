import JACC
JACC.@init_backend

function est_pi_jacc(N::Integer)
    halfpi = JACC.parallel_reduce(N, *; init=1.0) do n
        return 4.0 * n^2 / (4.0 * n^2 - 1.0)
    end
    return 2.0 * halfpi
end
