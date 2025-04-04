module LoadData
# import Simulation

export NuclideGridPoint, default_input, Input, immutableSimulationData, immutableNuclideGridPoint


struct Input
    nthreads::Int
    n_isotopes::Int
    n_gridpoints::Int
    lookups::Int
    # HM::String
    # HM::Vector{UInt8}
    HM::Char
    grid_type::Int
    hash_bins::Int
    particles::Int
    simulation_method::Int
    binary_mode::Int
    kernel_id::Int
end

struct immutableNuclideGridPoint
    energy::Float64
    total_xs::Float64
    elastic_xs::Float64
    absorbtion_xs::Float64
    fission_xs::Float64
    nu_fission_xs::Float64
end

struct immutableSimulationData
    num_nucs::Vector{Int64}
    concs::Vector{Float64}
    mats::Vector{Int64}
    unionized_energy_array::Vector{Float64}
    index_grid::Vector{Int64}
    nuclide_grid::Vector{immutableNuclideGridPoint}
    length_num_nucs::Int32
    length_concs::Int32
    length_mats::Int32
    length_unionized_energy_array::Int32
    length_index_grid::Int64
    length_nuclide_grid::Int32
    max_num_nucs::Int64
    p_energy_samples::Vector{Float64}
    length_p_energy_samples::Int32
    mat_samples::Vector{Int32}
    length_mat_samples::Int32
end


function default_input()
    return Input(
        192,          # nthreads
        355,        # n_isotopes
        11303,      # n_gridpoints
        17000000,   # lookups
        'l',        # HM originally "large"
        0,          # grid_type
        10000,      # hash_bins
        0,          # particles
        2,          # simulation_method
        0,          # binary_mode
        0           # kernel_id
    )
end

function load_sim_data_hash(fname::AbstractString)
    # Attempt to open the file in read mode
    file = open(fname, "r")

    # Check if the file was successfully opened
    if file === nothing
        error("Failed to open file: $fname")
    end

    println("Reading Data from file $(basename(fname))...")

    num_nucs = Vector{Int32}(undef, 12)
    concs = Vector{Float64}(undef, 3852)
    mats = Vector{Int32}(undef, 3852)
    unionized_energy_array = Vector{Float64}(undef, 0)
    index_grid = Vector{Int32}(undef, 3550000)
    nuclide_grid = Vector{immutableNuclideGridPoint}(undef, 4012565)
    p_energy_samples = Vector{Float64}(undef, 0)
    mat_samples = Vector{Int32}(undef, 0)

    length_num_nucs = 12
    length_concs = 3852
    length_mats = 3852
    length_unionized_energy_array = 0
    length_index_grid = 3550000
    length_nuclide_grid = 4012565
    length_p_energy_samples = 0
    length_mat_samples = 0
    max_num_nucs = 321

    # Read the num_nucs value amount of integers from the file
    for i in 1:length_num_nucs
        num_nucs[i] = parse(Int32, readline(file))
    end

    # Print all data inside num_nucs array
    # println("num_nucs: ", num_nucs)


    # Read the concs value amount of floats from the file
    for i in 1:length_concs
        concs[i] = parse(Float64, readline(file))
    end

    # Read the mats value amount of integers from the file
    for i in 1:length_mats
        mats[i] = parse(Int32, readline(file))
    end

    # Read the unionized_energy_array value amount of floats from the file
    for i in 1:length_unionized_energy_array
        unionized_energy_array[i] = parse(Float64, readline(file))
    end

    # Read the index_grid value amount of integers from the file
    for i in 1:length_index_grid
        index_grid[i] = parse(Int32, readline(file))
    end

    # Read the nuclide_grid value amount of NuclideGridPoint structs from the file
    for i in 1:length_nuclide_grid
        line = readline(file)
        energy = parse(Float64, line)
        line = readline(file)
        total_xs = parse(Float64, line)
        line = readline(file)
        elastic_xs = parse(Float64, line)
        line = readline(file)
        absorbtion_xs = parse(Float64, line)
        line = readline(file)
        fission_xs = parse(Float64, line)
        line = readline(file)
        nu_fission_xs = parse(Float64, line)
        nuclide_grid[i] = immutableNuclideGridPoint(energy, total_xs, elastic_xs, absorbtion_xs, fission_xs, nu_fission_xs)
    end

    # Close the file
    close(file)
    # println("File closed successfully.")

    return immutableSimulationData(num_nucs, concs, mats, unionized_energy_array, index_grid, nuclide_grid, length_num_nucs, length_concs, length_mats, length_unionized_energy_array, length_index_grid, length_nuclide_grid, max_num_nucs, p_energy_samples, length_p_energy_samples, mat_samples, length_mat_samples)

end


function load_sim_data(fname::AbstractString)
    # Attempt to open the file in read mode
    file = open(fname, "r")

    # Check if the file was successfully opened
    if file === nothing
        error("Failed to open file: $fname")
    end

    println("Reading Data from file $(basename(fname))...")
    num_nucs = Vector{Int32}(undef, 12)
    concs = Vector{Float64}(undef, 3852)
    mats = Vector{Int32}(undef, 3852)
    unionized_energy_array = Vector{Float64}(undef, 4012565)
    index_grid = Vector{Int32}(undef, 1424460575)
    nuclide_grid = Vector{immutableNuclideGridPoint}(undef, 4012565)
    p_energy_samples = Vector{Float64}(undef, 0)
    mat_samples = Vector{Int32}(undef, 0)
    length_num_nucs = 12
    length_concs = 3852
    length_mats = 3852
    length_unionized_energy_array = 4012565
    length_index_grid = 1424460575
    length_nuclide_grid = 4012565
    length_p_energy_samples = 0
    length_mat_samples = 0
    max_num_nucs = 321

    # Read the num_nucs value amount of integers from the file
    for i in 1:length_num_nucs
        num_nucs[i] = parse(Int32, readline(file))
    end

    # Print all data inside num_nucs array
    # println("num_nucs: ", num_nucs)


    # Read the concs value amount of floats from the file
    for i in 1:length_concs
        concs[i] = parse(Float64, readline(file))
    end

    # Read the mats value amount of integers from the file
    for i in 1:length_mats
        mats[i] = parse(Int32, readline(file))
    end

    # Read the unionized_energy_array value amount of floats from the file
    for i in 1:length_unionized_energy_array
        unionized_energy_array[i] = parse(Float64, readline(file))
    end

    # Read the index_grid value amount of integers from the file
    for i in 1:length_index_grid
        index_grid[i] = parse(Int32, readline(file))
    end

    # Read the nuclide_grid value amount of NuclideGridPoint structs from the file
    for i in 1:length_nuclide_grid
        line = readline(file)
        energy = parse(Float64, line)
        line = readline(file)
        total_xs = parse(Float64, line)
        line = readline(file)
        elastic_xs = parse(Float64, line)
        line = readline(file)
        absorbtion_xs = parse(Float64, line)
        line = readline(file)
        fission_xs = parse(Float64, line)
        line = readline(file)
        nu_fission_xs = parse(Float64, line)
        nuclide_grid[i] = immutableNuclideGridPoint(energy, total_xs, elastic_xs, absorbtion_xs, fission_xs, nu_fission_xs)
    end

    # Close the file
    close(file)
    # println("File closed successfully.")

    return immutableSimulationData(num_nucs, concs, mats, unionized_energy_array, index_grid, nuclide_grid, length_num_nucs, length_concs, length_mats, length_unionized_energy_array, length_index_grid, length_nuclide_grid, max_num_nucs, p_energy_samples, length_p_energy_samples, mat_samples, length_mat_samples)

end





end # module julia
