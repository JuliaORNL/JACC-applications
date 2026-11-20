module Ex_heat

include("heat.jl")
include("simulate.jl")
include("orig.jl")

function init(ncols = 2048, nrows = 2048; array_type=Matrix)
    return initialize(ncols, nrows, array_type)
end

function run(ncols = 2048, nrows = 2048; nsteps = 10, vis = false,
    evolve_fn=evolve!, array_type=Matrix)
    # initialize current and previous states to the same state
    curr, prev = initialize(ncols, nrows, array_type)

    # visualize initial field, requires Plots.jl
    if vis
        visualize(curr, "initial.png")
    end

    println("Initial average temperature: ", average_temperature(curr))

    # simulate temperature evolution for nsteps
    @time simulate!(curr, prev, nsteps, evolve_fn)

    # print final average temperature
    println("Final average temperature: ", average_temperature(curr))

    # visualize final field, requires Plots.jl
    if vis
        visualize(curr, "final.png")
    end
end

end
