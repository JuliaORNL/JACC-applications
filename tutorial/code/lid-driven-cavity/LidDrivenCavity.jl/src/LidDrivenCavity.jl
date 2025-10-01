module LidDrivenCavity

export run

import YAML
import JACC
import Plots

include("structs.jl")

function _read_input_file(filename::String)::Settings
    data = YAML.load_file(filename)
    type = get(data, "precision", "Float64")
    T = type == "Float32" ? Float32 : Float64

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
        get(data, "plot_interval", 10),
        T)
end

function run(config_file::String)
    settings = _read_input_file(config_file)
    println("Simulation settings: ")
    println(settings)

    # Array allocations
    x_grid = settings.x_grid
    y_grid = settings.y_grid
    T = settings.T

    u = JACC.zeros(T, x_grid + 1, y_grid + 2) # x-velocity
    v = JACC.zeros(T, x_grid + 2, y_grid + 1) # y-velocity
    uu = JACC.zeros(T, x_grid + 1, y_grid + 1)
    vv = JACC.zeros(T, x_grid + 1, y_grid + 1)
    p = JACC.zeros(T, x_grid + 2, y_grid + 2) # pressure

    pt = JACC.zeros(T, x_grid + 2, y_grid + 2) # temporary pressure
    ut = JACC.zeros(T, x_grid + 1, y_grid + 2) # temporary x-velocity
    vt = JACC.zeros(T, x_grid + 2, y_grid + 1) # temporary y-velocity

    # initial_condition for pressure (needs some kernel work)
    c = JACC.zeros(T, x_grid + 2, y_grid + 2)
    Δx = settings.length_x / x_grid
    Δy = settings.length_y / y_grid
    Δt = settings.time_step
    viscosity = settings.viscosity

    # Fill in the grid points, always on CPU for plotting only
    x = zeros(Float64, x_grid + 1, y_grid + 1)
    y = zeros(Float64, x_grid + 1, y_grid + 1)

    for i in 1:(x_grid + 1)
        for j in 1:(y_grid + 1)
            x[i, j] = Δx * (i - 1)
            y[i, j] = Δy * (j - 1)
        end
    end

    # kernel to initialize pressure coefficient 2D field
    c = JACC.zeros(T, x_grid + 2, y_grid + 2)
    JACC.parallel_for((x_grid + 2, y_grid + 2), _kernel_pressure_coeff_init!,
        c, Δx, Δy, x_grid, y_grid)

    # start the simulation to advance u, v, p in time
    time = settings.initial_time
    fnames = String[]

    for time_step in 1:(settings.max_steps)

        # assign boundary conditions
        JACC.parallel_for((x_grid + 2, y_grid + 2),
            _kernel_assign_boundary_conditions!,
            u, v,
            settings.u_south, settings.u_north,
            settings.v_west, settings.v_east,
            x_grid, y_grid)

        # compute temporary velocities ut and vt
        JACC.parallel_for((x_grid + 1, y_grid + 1), _kernel_tmp_velocities!,
            ut, vt, u, v, Δx, Δy, Δt, viscosity, x_grid, y_grid)

        # compute pressure field p
        for it_p in 1:(settings.max_iterations) # solve for pressure
            JACC.parallel_for((x_grid + 2, y_grid + 2),
                _kernel_pressure!,
                pt, c, p, ut, vt, Δx, Δy, Δt,
                settings.density, settings.beta,
                x_grid, y_grid)

            Err = JACC.parallel_reduce((x_grid, y_grid), pt, p) do i, j, pt, p
                return abs(pt[i, j] - p[i, j])
            end

            if Err <= settings.max_error
                break # stop if converged 
                println("Converged at iteration: $it_p with error: $Err")
            end
        end

        # update velocities u and v
        JACC.parallel_for((x_grid + 1, y_grid + 1), _kernel_update_velocities!,
            u, v, ut, vt, p, Δx, Δy, Δt,
            settings.density, x_grid, y_grid)

        time += Δt
        if time_step % settings.plot_interval == 0
            # update uu and vv for plotting
            JACC.parallel_for((x_grid + 1, y_grid + 1)) do i, j
                uu[i, j] = 0.5 * (u[i, j] + u[i, j + 1])
                vv[i, j] = 0.5 * (v[i, j] + v[i + 1, j])
            end

            println("Saving time step: $time_step, time: $(time)")

            plt = Plots.quiver!(x, y; quiver = (Array(uu), Array(vv)))
            push!(fnames, "$(lpad(time_step, 6, "0")).png")
            Plots.savefig(
                plt, "$(lpad(time_step, 6, "0")).png")
        end
    end
    println(fnames)
    anim = Plots.Animation(".", fnames)
    Plots.buildanimation(
        anim, "lid_driven_cavity.gif"; fps = 1, show_msg = false) #set a suitable fps
end

function _kernel_update_velocities!(
        i, j, u, v, ut, vt, p, Δx, Δy, Δt, rho, x_grid, y_grid)
    if i >= 2 && i <= x_grid && j >= 2 && j <= y_grid + 1
        u[i, j] = ut[i, j] - (Δt / rho) * (p[i + 1, j] - p[i, j]) / Δx
    end

    if i >= 2 && i <= x_grid + 1 && j >= 2 && j <= y_grid
        v[i, j] = vt[i, j] - (Δt / rho) * (p[i, j + 1] - p[i, j]) / Δy
    end
end

function _kernel_pressure!(i, j,
        pt, c, p, ut, vt, Δx, Δy, Δt, rho, beta, x_grid, y_grid)
    pt[i, j] = p[i, j] # default to old value
    if i >= 2 && i <= x_grid + 1 && j >= 2 && j <= y_grid + 1
        p[i, j] = beta * c[i, j] *
                  ((p[i + 1, j] + p[i - 1, j]) / Δx^2 +
                   (p[i, j + 1] + p[i, j - 1]) / Δy^2 -
                   (rho / Δt) * ((ut[i, j] - ut[i - 1, j]) / Δx +
                    (vt[i, j] - vt[i, j - 1]) / Δy)) + (1 - beta) * p[i, j]
    end
end

function _kernel_tmp_velocities!(
        i, j, ut, vt, u, v, Δx, Δy, Δt, viscosity, x_grid, y_grid)
    if i >= 2 && i <= x_grid && j >= 2 && j <= y_grid + 1
        ut[i, j] = u[i, j] +
                   Δt * (-0.25 * (
            ((u[i + 1, j] + u[i, j])^2 - (u[i, j] + u[i - 1, j])^2) / Δx +
            ((u[i, j + 1] + u[i, j]) * (v[i + 1, j] + v[i, j]) -
             (u[i, j] + u[i, j - 1]) * (v[i + 1, j - 1] + v[i, j - 1])) / Δy) +
                    viscosity *
                    ((u[i + 1, j] + u[i - 1, j] - 2 * u[i, j]) / Δx^2 +
                     (u[i, j + 1] + u[i, j - 1] - 2 * u[i, j]) / Δy^2))
    end

    if i >= 2 && i <= x_grid + 1 && j >= 2 && j <= y_grid
        vt[i, j] = v[i, j] +
                   Δt * (-0.25 * (
            ((u[i, j + 1] + u[i, j]) * (v[i + 1, j] + v[i, j]) -
             (u[i - 1, j + 1] + u[i - 1, j]) * (v[i, j] + v[i - 1, j])) / Δx +
            ((v[i, j + 1] + v[i, j])^2 - (v[i, j] + v[i, j - 1])^2) / Δy) +
                    viscosity *
                    ((v[i + 1, j] + v[i - 1, j] - 2 * v[i, j]) / Δx^2 +
                     (v[i, j + 1] + v[i, j - 1] - 2 * v[i, j]) / Δy^2))
    end
end

"""
Kernel to assign boundary conditions.
Think what each (i,j) "would do", aka fine-granularity
"""
function _kernel_assign_boundary_conditions!(i, j,
        u, v, u_south, u_north, v_west, v_east, x_grid, y_grid)
    if i >= 1 && i <= x_grid + 1 && j == 1
        u[i, j] = 2 * u_south - u[i, 2] # south wall
    end

    if i >= 1 && i <= x_grid + 1 && j == y_grid + 2
        u[i, j] = 2 * u_north - u[i, y_grid + 1] # north wall
    end

    if i == 1 && j >= 1 && j <= y_grid + 1
        v[i, j] = 2 * v_west - v[2, j] # west wall
    end

    if i == x_grid + 2 && j >= 1 && j <= y_grid + 1
        v[i, j] = 2 * v_east - v[x_grid + 1, j] # east wall
    end
end

"""
Kernel to initialize the 2D pressure field "c" coefficient field.
Think what each (i,j) "would do", aka fine-granularity
"""
function _kernel_pressure_coeff_init!(i, j, c, Δx, Δy, x_grid, y_grid)
    if i <= x_grid + 1 && j <= y_grid + 1
        c[i, j] = 1 / (2 / Δx^2 + 2 / Δy^2)
    end

    if i == 2 && j >= 3 && j <= y_grid
        c[i, j] = 1 / (1 / Δx^2 + 2 / Δy^2)
    end

    if i == x_grid + 1 && j >= 3 && j <= y_grid
        c[i, j] = 1 / (1 / Δx^2 + 2 / Δy^2)
    end

    if i >= 3 && i <= x_grid && j == 2
        c[i, j] = 1 / (1 / Δx^2 + 2 / Δy^2)
    end

    if i >= 3 && i <= x_grid && j == y_grid + 1
        c[i, j] = 1 / (1 / Δx^2 + 2 / Δy^2)
    end

    if i == 2 && j == 2
        c[i, j] = 1 / (1 / Δx^2 + 1 / Δy^2)
    end

    if i == 2 && j == y_grid + 1
        c[i, j] = 1 / (1 / Δx^2 + 1 / Δy^2)
    end

    if i == x_grid + 1 && j == 2
        c[i, j] = 1 / (1 / Δx^2 + 1 / Δy^2)
    end

    if i == x_grid + 1 && j == y_grid + 1
        c[i, j] = 1 / (1 / Δx^2 + 1 / Δy^2)
    end
end

end # module LidDrivenCavity
