include("main.jl")
include("solution.jl")

using .Ex_heat: simulate!

function check_heat()
    ncols = 8192
    nrows = 4096
    nsteps = 20
    
    A1_h, A2_h = Ex_heat.initialize(nrows, ncols)
    @time simulate!(A1_h, A2_h, nsteps, Ex_heat.evolve!)
    @time simulate!(A1_h, A2_h, nsteps, Ex_heat.evolve!)
    
    println()
    
    A1_d, A2_d = Ex_heat.initialize(nrows, ncols, JACC.array_type())
    @time simulate!(A1_d, A2_d, nsteps, evolve_jacc!)
    @time simulate!(A1_d, A2_d, nsteps, evolve_jacc!)
    
    println()
    
    @time simulate!(A1_d, A2_d, nsteps, evolve_jacc_adapt!)
    @time simulate!(A1_d, A2_d, nsteps, evolve_jacc_adapt!)
end

check_heat()
