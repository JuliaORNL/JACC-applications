""" 
    average_temperature(f::Field)

Calculate average temperature of a temperature field.        
"""
average_temperature(f::Field) = sum(f.data[2:f.nx+1, 2:f.ny+1]) / (f.nx * f.ny)

"""
    simulate!(current, previous, nsteps)

Run the heat equation solver on fields curr and prev for nsteps.
"""
function simulate!(curr::Field, prev::Field, nsteps, evolve_fn)
    # Diffusion constant
    a = 0.5
    # Largest stable time step
    dt = curr.dx^2 * curr.dy^2 / (2.0 * a * (curr.dx^2 + curr.dy^2))
    
    for i = 1:nsteps
        # calculate new state based on previous state
        evolve_fn(curr, prev, a, dt)

        # swap current and previous fields
        curr, prev = swap_fields(curr, prev)
    end 
end
