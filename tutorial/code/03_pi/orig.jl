"""
    est_pi(N::Integer)

This code uses the
["Wallis product"](https://en.wikipedia.org/wiki/Wallis_product) to estimate pi.
N is the number of terms, which in this case is the number of loop iterations.
"""
function est_pi(N::Integer)
    halfpi = 1.0
    for n in 1:N
        # 4*n^2 overflows Int64 at the values of N where this is interesting
        nf = Float64(n)
        halfpi *= 4.0nf^2 / (4.0 * nf^2 - 1.0)
    end
    return 2.0 * halfpi
end
