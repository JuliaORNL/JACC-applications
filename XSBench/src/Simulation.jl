module Simulation
export run_event_based_simulation, hello

import ..LoadData

import JACC
JACC.@init_backend
const JACCArray = JACC.array_type()

# Grid types
const UNIONIZED = 0
const NUCLIDE = 1
const HASH = 2
const STARTING_SEED = 1070
using DelimitedFiles
using Printf
using Base.MathConstants
using Decimals
# import numpy as np
using Dates


function test_simulation()
    dist = [
        0.140,  # fuel
        0.052,  # cladding
        0.275,  # cold, borated water
        0.134,  # hot, borated water
        0.154,  # RPV
        0.064,  # Lower, radial reflector
        0.066,  # Upper reflector / top plate
        0.055,  # bottom plate
        0.008,  # bottom nozzle
        0.015,  # top nozzle
        0.025,  # top of fuel assemblies
        0.013   # bottom of fuel assemblies
    ]

    num_nucs = [321, 5, 4, 4, 27, 21, 21, 21, 21, 21, 9, 9]
    num_nucs = JACCArray(num_nucs)

    sizeofa = 10
    a = zeros(Float64, sizeofa)
    lookups = 1

    dist_d = JACCArray(dist)
    a = JACCArray(a)

    JACC.parallel_for(lookups, test_kernel, dist_d, a, num_nucs)

    a_h = Array(a)
    for i in 1:sizeofa
        println("a[", i, "] is: ", a_h[i])
    end

end

function test_kernel(lookups, dist_d, a, num_nucs)

    seed = UInt64(STARTING_SEED)
    n = UInt64((lookups - 1) * 2)

    # Forward seed to lookup index (we need 2 samples per lookup)
    seed = fast_forward_LCG(seed, n, a)  # Functional !!

    # Randomly pick an energy and material for the particle
    p_energy, seed = LCG_random_double_test(seed) # Functional !!
    mat, seed = pick_mat_test(seed, a, lookups, dist_d) # Functional !!!

    hash_bins = 10000
    du = 1.0 / hash_bins
    idx = floor(Int, p_energy / du)
    a[1] = idx





end

function LCG_random_double_test(seed::UInt64)
    # LCG parameters
    m::UInt64 = 9223372036854775808  # 2^63
    a::UInt64 = 2806196910506780709
    c::UInt64 = 0x0000000000000001

    # Calculate new seed
    seed = (a * seed + c) % m
    
    # Calculate and format the result to match C's precision
    result = Float64(seed) / Float64(m)
    formatted_result = round(result, digits=40)
    
    # Return the formatted result as a Float64
    return formatted_result, seed
end


function pick_mat_test(seed::UInt64, a, lookups, dist_d)
    # Generate random number and update seed
    roll, new_seed = LCG_random_double_test(seed)

    for i in 0:11  # Changed to 0-based indexing
        running = 0.0
        for j in i:-1:1  # Note: This still starts from i because Julia arrays are 1-indexed
            running += dist_d[j+1]  # Add 1 to index into dist array
        end
        if roll < running
            return i, new_seed  # Return i, not i+1
        end
    end
    return 0, new_seed  # Return 0, not 12
end


function run_event_based_simulation(in:: LoadData.Input, SD:: LoadData.immutableSimulationData)

    println("Running baseline event-based simulation...")

    println("Lookups: ", in.lookups)
    

    macro_xs_vector = JACCArray(zeros(Float64, 5))
    grid_type = 2
    n_isotopes = 355
    n_gridpoints = 11303
    p_energy = 0.626011
    hash_bins = 10000
    mat = 5
    max_num_nucs = 321

    dist = [
        0.140,  # fuel
        0.052,  # cladding
        0.275,  # cold, borated water
        0.134,  # hot, borated water
        0.154,  # RPV
        0.064,  # Lower, radial reflector
        0.066,  # Upper reflector / top plate
        0.055,  # bottom plate
        0.008,  # bottom nozzle
        0.015,  # top nozzle
        0.025,  # top of fuel assemblies
        0.013   # bottom of fuel assemblies
    ]

    dist_d = JACCArray(dist)
    unionized_energy_array = JACCArray(SD.unionized_energy_array) # e_grid
    num_nucs = JACCArray(SD.num_nucs)
    xs_vector = JACCArray(zeros(Float64, 5))
    mats = JACCArray(SD.mats)
    concs = JACCArray(SD.concs)
    index_grid = JACCArray(SD.index_grid) # index_data
    nu_grids = Vector{immutableNuclideGridPoint}()
    for i in 1:355
        push!(nu_grids, immutableNuclideGridPoint(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    end
    nuclide_grid = JACCArray(SD.nuclide_grid)
    verification = JACCArray(zeros(UInt64, in.lookups))
    sizeofa = 17000000
    na = zeros(Float64, sizeofa)
    a = JACCArray(na)

    ba = zeros(Float64, 5)
    b = JACCArray(ba)

    test_macro_xs_vector = JACCArray(zeros(Float64, 5, in.lookups))
    test_xs_vector = JACCArray(zeros(Float64, 5, in.lookups))



    if grid_type == UNIONIZED
        println("Unionized Energy Grid")
    elseif grid_type == NUCLIDE
        println("Nuclide Grid")
    elseif grid_type == HASH
        println("Hash Grid")
    end
    
    
    verification_hash = 0
    # println("Number of threads before kernel: ", Threads.nthreads())

    # Warmup
    JACC.parallel_for(in.lookups, kernel, a, macro_xs_vector, grid_type, n_isotopes, n_gridpoints, p_energy, unionized_energy_array, hash_bins,
    mat, num_nucs, xs_vector, mats, max_num_nucs, concs, index_grid, nuclide_grid, dist_d, verification, b, test_macro_xs_vector, test_xs_vector)

    verification_hash_two = JACC.parallel_reduce(in.lookups, reduce, verification)

    start_time = time_ns()  # Start time in nanoseconds


    # println("Number of blocks: ", num_blocks)
    # println("Number of threads: ", num_threads)
    for i in 1:10   

        JACC.parallel_for(in.lookups, kernel, a, macro_xs_vector, grid_type, n_isotopes, n_gridpoints, p_energy, unionized_energy_array, hash_bins,
        mat, num_nucs, xs_vector, mats, max_num_nucs, concs, index_grid, nuclide_grid, dist_d, verification, b, test_macro_xs_vector, test_xs_vector)

        verification_hash_two = JACC.parallel_reduce(in.lookups, reduce, verification)
    end
    end_time = time_ns()    

    total_time_ms = (end_time - start_time) / 1_000_000 

    println("Total Time taken: ", total_time_ms / 1000.0, " seconds")

    # Convert the average to seconds
    average_time_seconds = total_time_ms / 10.0 / 1000.0
    println("Average Time per iteration: ", average_time_seconds, " seconds")

    # println("Simulation Complete.")
    na = Array(a)

    two = Array(verification_hash_two)

    # println("two is: ", two)
    # # Print data in na
    for i in 1:sizeofa
        if na[i] != 0
            println("a[", i, "] is: ", na[i])
        end
    end

    
    verification_h = Array(verification)

    # Save verification_h array to a file
    # verification_h_array = Array(verification_h)
    writedlm("verification_h_data_tests.txt", verification_h)

    calc = two[1] % 999983
    
    if calc == 952131
        # Print the following "Verification checksum: *insert verificationF here* (Valid)"
        println("Verification: Checksum: ", calc, " (Valid)")

    else
        println("Verification: Failed, Expected: 952131, Actual: ", calc)
    end

end

function reduce(lookups, verification)
    verification[lookups]
end


function grid_search(n::Int64, quarry::Float64, A)::Int64
    lowerLimit = 1
    upperLimit = n
    length = upperLimit - lowerLimit

    while length > 1
        examinationPoint = lowerLimit + div(length, 2)
        
        if A[examinationPoint] > quarry
            upperLimit = examinationPoint
        else
            lowerLimit = examinationPoint
        end
        
        length = upperLimit - lowerLimit
    end
    
    return lowerLimit
end


function fast_forward_LCG(seed::UInt64, n::UInt64, b):: UInt64 # Appears to be correct
    m = UInt64(9223372036854775808) # 2^63
    a = UInt64(2806196910506780709)
    c = UInt64(1)

    n = n % m
    a_new = UInt64(1)
    c_new = UInt64(0)

    while n > 0
        if n & 1 != 0
            a_new *= a
            c_new = c_new * a + c
        end
        c *= (a + 1)
        a *= a
        n >>= 1
    end
    return (a_new * seed + c_new) % m
end




function kernel(lookups, a, macro_xs_vector, grid_type, n_isotopes, n_gridpoints, p_energy, unionized_energy_array, hash_bins,
    mat, num_nucs, xs_vector, mats, max_num_nucs, concs, index_grid, nuclide_grid, dist_d, verification, b, test_macro_xs_vector, test_xs_vector)

    # unionized_energy_array = JACC.shared(unionized_energy_array)
    # Set the initial seed value
    seed = UInt64(STARTING_SEED)
    n = UInt64((lookups - 1) * 2)

    # Forward seed to lookup index (we need 2 samples per lookup)
    seed = fast_forward_LCG(seed, n, a)  # Functional !!

    # Randomly pick an energy and material for the particle
    p_energy, seed = LCG_random_double_test(seed) # Functional !!
    mat, seed = pick_mat_test(seed, a, lookups, dist_d) # Functional !!!

	# Perform macroscopic Cross Section Lookup
    macro_xs_1, macro_xs_2, macro_xs_3, macro_xs_4, macro_xs_5 = macro_xs(macro_xs_vector, a, grid_type, n_isotopes, n_gridpoints, p_energy, unionized_energy_array, hash_bins,
    mat, num_nucs, xs_vector, mats, max_num_nucs, concs, index_grid, nuclide_grid, lookups, b, test_macro_xs_vector, test_xs_vector)

    # # For verification, and to prevent the compiler from optimizing
	# # all work out, we interrogate the returned macro_xs_vector array
	# # to find its maximum value index, then increment the verification
	# # value by that index. In this implementation, we have each thread
	# # write to its thread_id index in an array, which we will reduce
	# # with a thrust reduction kernel after the main simulation kernel.

    max = macro_xs_1
    max_idx = 1

    if macro_xs_2 > max
        max = macro_xs_2
        max_idx = 2
    end

    if macro_xs_3 > max
        max = macro_xs_3
        max_idx = 3
    end

    if macro_xs_4 > max
        max = macro_xs_4
        max_idx = 4
    end

    if macro_xs_5 > max
        max = macro_xs_5
        max_idx = 5
    end

    # #  Look for max function in Julia

    # Use the following lines for AMD due to iteration out of bounds
    if lookups < 17000001
        verification[lookups] = max_idx
    end

    # # Use the following for CUDA and Threads
    # verification[lookups] = max_idx

end

@inline function macro_xs(macro_xs_vector, a, grid_type, n_isotopes, n_gridpoints, p_energy, egrid, hash_bins,
    mat, num_nucs, xs_vector, mats, max_num_nucs, concs, index_data, nuclide_grids, lookups, b, test_macro_xs_vector, test_xs_vector)

    p_nuc = 0 # the nuclide we are looking up
    idx = -1
    conc = 0.0 # the concentration of the nuclide in the material

    # If we are using the unionized energy grid (UEG), we only
	# need to perform 1 binary search per macroscopic lookup.
	# If we are using the nuclide grid search, it will have to be
	# done inside of the "calculate_micro_xs" function for each different
	# nuclide in the material.
    if grid_type == UNIONIZED
        idx = grid_search(n_isotopes * n_gridpoints, p_energy, egrid)
        idx = idx - 1

    elseif grid_type == HASH
        du = 1.0 / hash_bins
        idx = floor(Int, p_energy / du)
    end

    # Once we find the pointer array on the UEG, we can pull the data
	# from the respective nuclide grids, as well as the nuclide
	# concentration data for the material
	# Each nuclide from the material needs to have its micro-XS array
	# looked up & interpolatied (via calculate_micro_xs). Then, thex
	# micro XS is multiplied by the concentration of that nuclide
	# in the material, and added to the total macro XS array.
	# (Independent -- though if parallelizing, must use atomic operations
	#  or otherwise control access to the xs_vector and macro_xs_vector to
	#  avoid simulataneous writing to the same data structure)
    macro_xs_1 = 0.0
    macro_xs_2 = 0.0
    macro_xs_3 = 0.0
    macro_xs_4 = 0.0
    macro_xs_5 = 0.0

    @inbounds for j in 1:num_nucs[mat + 1]
        p_nuc = mats[mat * max_num_nucs + j]
        conc = concs[mat * max_num_nucs + j]

        total_xs, elastic_xs, absorbtion_xs, fission_xs, nu_fission_xs = calculate_micro_xs(p_energy, a, p_nuc, n_isotopes, n_gridpoints, egrid, index_data, nuclide_grids, idx, xs_vector, grid_type, hash_bins, lookups, j, conc, test_xs_vector)

        macro_xs_1 += total_xs * conc
        macro_xs_2 += elastic_xs * conc
        macro_xs_3 += absorbtion_xs * conc
        macro_xs_4 += fission_xs * conc
        macro_xs_5 += nu_fission_xs * conc
    end
    return macro_xs_1, macro_xs_2, macro_xs_3, macro_xs_4, macro_xs_5
end

@inline function calculate_micro_xs(p_energy::Float64, a, nuc::Int, n_isotopes::Int64,
    n_gridpoints::Int64, egrid,
    index_data, nuclide_grids,
    idx::Int64, xs_vector, grid_type::Int, hash_bins::Int, lookups, j, conc, test_xs_vector)

    # Variables
    low = immutableNuclideGridPoint(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    high = immutableNuclideGridPoint(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    low_idx = 1

    # If using only the nuclide grid
    if grid_type == NUCLIDE

        # Perform binary search on the Nuclide Grid
        idx = grid_search_nuclide(n_gridpoints, p_energy,
                                    nuclide_grids,
                                    0, n_gridpoints - 1, nuc*n_gridpoints)
        # Ensure we're not reading off the end
        if idx == n_gridpoints - 1
            low = nuclide_grids[nuc*n_gridpoints + idx] 
            low_idx = nuc*n_gridpoints + idx 
        else
            low = nuclide_grids[nuc*n_gridpoints + idx + 1]
            low_idx = nuc*n_gridpoints + idx + 1
        end


    # Unionized Energy Grid
    elseif grid_type == UNIONIZED 
        
        if index_data[idx * n_isotopes + nuc + 1] == n_gridpoints - 1
            low = nuclide_grids[nuc*n_gridpoints + index_data[idx * n_isotopes + nuc + 1]]
            low_idx = nuc*n_gridpoints + index_data[idx * n_isotopes + nuc + 1]
        else
            low = nuclide_grids[nuc*n_gridpoints + index_data[idx * n_isotopes + nuc + 1] + 1]
            low_idx = nuc*n_gridpoints + index_data[idx * n_isotopes + nuc + 1] + 1
        end

    # Hash grid
    else
        # Load lower bounding index
        u_low = index_data[idx * n_isotopes + nuc + 1]

        # Determine higher bounding index
        u_high = if idx == hash_bins - 1
            n_gridpoints - 1
        else
            index_data[(idx+1)*n_isotopes + nuc + 1] + 1 
        end

        # Check edge cases and search for gridpoint
        e_low  = nuclide_grids[nuc*n_gridpoints + u_low + 1].energy
        e_high = nuclide_grids[nuc*n_gridpoints + u_high + 1].energy

        lower = if p_energy <= e_low
            0
        elseif p_energy >= e_high
            n_gridpoints - 1
        else
            grid_search_nuclide(n_gridpoints, p_energy, nuclide_grids, u_low, u_high, nuc*n_gridpoints)
        end

        if lower == n_gridpoints - 1
            low = nuclide_grids[nuc*n_gridpoints + lower]
            low_idx = nuc*n_gridpoints + lower
        else
            low = nuclide_grids[nuc*n_gridpoints + lower + 1]
            low_idx = nuc*n_gridpoints + lower + 1
        end
    end

    # Get the higher energy grid point
    high = nuclide_grids[low_idx + 1]
    
    # Interpolation factor
    f = Float64(Float64((high.energy - p_energy)) / Float64((high.energy - low.energy)))

    # // Total XS
    total_xs = Float64(high.total_xs) - Float64(f * (high.total_xs - low.total_xs))

    # Elastic XS
    elastic_xs = Float64(high.elastic_xs) - Float64(f * (high.elastic_xs - low.elastic_xs))

    # Absorbtion XS
    absorbtion_xs = Float64(high.absorbtion_xs) - Float64(f * (high.absorbtion_xs - low.absorbtion_xs))

    # Fission XS
    fission_xs = Float64(high.fission_xs) - Float64(f * (high.fission_xs - low.fission_xs))

    # Nu Fission XS
    nu_fission_xs = Float64(high.nu_fission_xs) - Float64(f * (high.nu_fission_xs - low.nu_fission_xs))

    return total_xs, elastic_xs, absorbtion_xs, fission_xs, nu_fission_xs
end

function grid_search_nuclide(n::Int64, quarry::Float64, A, low::Int64, high::Int64, offset)
    lowerLimit = low
    upperLimit = high
    length = upperLimit - lowerLimit
    
    examinationPoint = lowerLimit + (length ÷ 2)
    while length > 1
        examinationPoint = lowerLimit + (length ÷ 2)
        
        if A[offset + examinationPoint + 1].energy > quarry
            upperLimit = examinationPoint
        else
            lowerLimit = examinationPoint
        end
        
        length = upperLimit - lowerLimit
    end
    
    return lowerLimit
end


end # module
