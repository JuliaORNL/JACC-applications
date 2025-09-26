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

    u = JACC.zeros(T, x_grid+1, y_grid+2) # x-velocity
    v = JACC.zeros(T, x_grid+2, y_grid+1) # y-velocity
    uu = JACC.zeros(T, x_grid+1, y_grid+1)
    vv = JACC.zeros(T, x_grid+1, y_grid+1) 
    p = JACC.zeros(T, x_grid+2, y_grid+2) # pressure
    pold = JACC.zeros(T, x_grid+2, y_grid+2) # old pressure
    
    ut = JACC.zeros(T, x_grid+1, y_grid+2) # temporary x-velocity
    vt = JACC.zeros(T, x_grid+2, y_grid+1) # temporary y-velocity
    
    # initial_condition for pressure (needs some kernel work)
    c = JACC.zeros(T, x_grid+2, y_grid+2)
    Δx  = settings.length_x / x_grid
    Δy  = settings.length_y / y_grid

    # kernel to initialize pressure field


end







end # module LidDrivenCavity
