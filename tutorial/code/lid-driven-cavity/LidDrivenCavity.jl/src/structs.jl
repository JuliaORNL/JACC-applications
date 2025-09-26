


# Create a Julia struct to hold YAML entries

"""
    Settings

Configuration struct for lid-driven cavity simulation using MAC (Marker-and-Cell) method.
Contains all parameters needed for the CFD simulation.
"""
struct Settings{ T <: AbstractFloat}
    # Grid Configuration
    x_grid::Int
    y_grid::Int
    
    # Computational Domain
    length_x::Float64
    length_y::Float64
    
    # Simulation Parameters
    max_steps::Int
    viscosity::Float64
    density::Float64
    
    # SOR (Successive Over-Relaxation) Parameters
    max_iterations::Int
    beta::Float64
    max_error::Float64
    
    # Boundary Conditions
    u_north::Float64 # Top wall velocity (lid velocity)
    u_south::Float64 # Bottom wall velocity
    v_east::Float64 # Right wall velocity
    v_west::Float64 # Left wall velocity
    
    # Time Integration
    initial_time::Float64
    time_step::Float64

    # Precision
    T::Type{T}
end
