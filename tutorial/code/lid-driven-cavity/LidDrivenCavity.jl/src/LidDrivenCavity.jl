module LidDrivenCavity

export run

import YAML
import JACC

include("structs.jl")

function _read_input_file(filename::String)::Settings
    data = YAML.load_file(filename)
    type = get(data, "precision", "Float32")
    T = type == "Float64" ? Float64 : Float32

    return Settings(get(data, "x_grid", 32),
        get(data, "y_grid", 32),
        get(data, "length_x", 1.0),
        get(data, "length_y", 1.0),
        get(data, "max_steps", 500),
        get(data, "viscosity", 0.1),
        get(data, "density", 1.0),
        get(data, "max_iterations", 100),
        get(data, "beta", 1.5),
        get(data, "max_error", 0.001),
        get(data, "u_north", 1.0),
        get(data, "u_south", 0.0),
        get(data, "v_east", 0.0),
        get(data, "v_west", 0.0),
        get(data, "initial_time", 0.0),
        get(data, "time_step", 0.002),
        T)
end

function run(config_file::String)
    settings = _read_input_file(config_file)

    # Array allocations
    x_grid = settings.x_grid
    y_grid = settings.y_grid
    T = settings.T

    u = JACC.zeros(T, x_grid + 1, y_grid + 2) # x-velocity
    v = JACC.zeros(T, x_grid + 2, y_grid + 1) # y-velocity
    uu = JACC.zeros(T, x_grid + 1, y_grid + 1)
    vv = JACC.zeros(T, x_grid + 1, y_grid + 1)
    p = JACC.zeros(T, x_grid + 2, y_grid + 2) # pressure
    pold = JACC.zeros(T, x_grid + 2, y_grid + 2) # old pressure

    ut = JACC.zeros(T, x_grid + 1, y_grid + 2) # temporary x-velocity
    vt = JACC.zeros(T, x_grid + 2, y_grid + 1) # temporary y-velocity

    # initial_condition for pressure (needs some kernel work)
    c = JACC.zeros(T, x_grid + 2, y_grid + 2)
    Δx = settings.length_x / x_grid
    Δy = settings.length_y / y_grid

    # kernel to initialize pressure coefficient 2D field
    c = JACC.zeros(T, x_grid + 2, y_grid + 2)
    JACC.parallel_for(
        (x_grid + 2, y_grid + 2), _kernel_pressure_coeff_initial!,
        c, Δx, Δy, x_grid, y_grid)

    # start the simulation to advance u, v, p in time
    for time_step in 1:(settings.max_steps)

        # assign boundary conditions
        JACC.parallel_for((x_grid + 1, y_grid + 2),
            _kernel_assign_boundary_conditions!,
            u, v,
            settings.u_south, settings.u_north,
            settings.v_west, settings.v_east,
            x_grid, y_grid)

        # compute temporary velocities ut and vt
        JACC.parallel_for((x_grid + 1, y_grid + 2), _kernel_temp_velocities!,
            u, v, ut, vt, x_grid, y_grid)

        # compute pressure field p
        # update velocities u and v

        # (to be implemented)

    end
end

"""
Kernel to initialize the 2D pressure field "c" coefficient field.
Think what each i,j "would do" , aka fine-granularity
"""
function _kernel_assign_boundary_conditions!(i, j,
        u, v, u_south, u_north, v_west, v_east, x_grid, y_grid)
    if i <= x_grid + 1
        if j == 1
            u[i, j] = 2 * u_south - u[i, 2] # south wall
        elseif j == y_grid + 2
            u[i, j] = 2 * u_north - u[i, y_grid + 1] # north wall
        end
    end

    if j <= y_grid + 1
        if i == 1
            v[i, j] = 2 * v_west - v[2, j] # west wall
        elseif i == x_grid + 2
            v[i, j] = 2 * v_east - v[x_grid + 1, j] # east wall
        end
    end
end

"""
Kernel to initialize the 2D pressure field "c" coefficient field.
Think what each i,j "would do" , aka fine-granularity
"""
function _kernel_pressure_coeff_initial!(c, i, j, Δx, Δy, x_grid, y_grid)
    if i <= x_grid + 1 && j <= y_grid + 1
        c[i, j] = 1 / (2 / Δx^2 + 2 / Δy^2)
    elseif i == 2 && j >= 3 && j <= y_grid
        c[i, j] = 1 / (1 / Δx^2 + 2 / Δy^2)
    elseif i == x_grid + 1 && j >= 3 && j <= y_grid
        c[i, j] = 1 / (1 / Δx^2 + 2 / Δy^2)
    elseif i >= 3 && i <= x_grid && j == 2
        c[i, j] = 1 / (1 / Δx^2 + 2 / Δy^2)
    elseif i >= 3 && i <= x_grid && j == y_grid + 1
        c[i, j] = 1 / (1 / Δx^2 + 2 / Δy^2)
    elseif i == 2 && j == 2
        c[i, j] = 1 / (1 / Δx^2 + 1 / Δy^2)
    elseif i == 2 && j == y_grid + 1
        c[i, j] = 1 / (1 / Δx^2 + 1 / Δy^2)
    elseif i == x_grid + 1 && j == 2
        c[i, j] = 1 / (1 / Δx^2 + 1 / Δy^2)
    elseif i == x_grid + 1 && j == y_grid + 1
        c[i, j] = 1 / (1 / Δx^2 + 1 / Δy^2)
    end
end

end # module LidDrivenCavity
