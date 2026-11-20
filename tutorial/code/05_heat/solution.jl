using JACC
JACC.@init_backend

function evolve_jacc!(curr, prev, a, dt)
    JACC.parallel_for((curr.nx, curr.ny), curr.data, prev.data, curr.dx, curr.dy, a, dt) do i, j, curr, prev, dx, dy, a, dt
        xderiv = (prev[i-1, j] - 2.0 * prev[i, j] + prev[i+1, j]) / dx^2
        yderiv = (prev[i, j-1] - 2.0 * prev[i, j] + prev[i, j+1]) / dy^2
        curr[i, j] = prev[i, j] + a * dt * (xderiv + yderiv)
    end
end

using Adapt
Adapt.@adapt_structure Ex_heat.Field
function evolve_jacc_adapt!(curr, prev, a, dt)
    JACC.parallel_for((curr.nx, curr.ny), curr, prev, a, dt) do i, j, curr, prev, a, dt
        xderiv = (prev.data[i-1, j] - 2.0 * prev.data[i, j] + prev.data[i+1, j]) / curr.dx^2
        yderiv = (prev.data[i, j-1] - 2.0 * prev.data[i, j] + prev.data[i, j+1]) / curr.dy^2
        curr.data[i, j] = prev.data[i, j] + a * dt * (xderiv + yderiv)
    end
end
