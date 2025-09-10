function est_pi(N::Integer)
    halfpi = 0.0
    for n in 1:N
        halfpi += 4.0 * n^2 / (4.0 * n^2 - 1.0)
    end
    return 2.0 * halfpi
end
