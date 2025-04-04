using HartreeFock

# Test parameters and benchmark code
#const natom = 1024
const natom = 1024
const ngauss = 3
const txpnt = [6.3624214, 1.1589230, 0.3136498]
const tcoef = [0.154328967295, 0.535328142282, 0.444634542185]

function generate_geometry(natom)
    tgeom = zeros(Float64, 3, natom)
    seed = 12345
    for i in 1:natom
        for j in 1:3
            seed = mod(1103515245 * seed + 12345, Int64(2)^31)
            n = mod(seed, 181)
            tgeom[j, i] = (n / 10.0) * HartreeFock.tobohrs
        end
    end
    return tgeom
end

function run_benchmark(func, input)
    println("Performing warmup run...")
    E, t = func(input)
    println("Warmup E = ", E)
    println("Warmup section time: ", round(t; digits=5), " seconds")
    
    println("\nPerforming 10 timed runs:")
    sum_total = 0.0
    
    for i in 1:10
        E, t = func(input)
        println("Run ", i, ": Time = ", round(t; digits=5), " seconds, 2e- energy = ", E)
        sum_total += t
        flush(stdout)
    end
    
    avg_time = sum_total / 10
    println("\nAverage time of 10 calls: ", round(avg_time; digits=5), " seconds")
end

# Create input data and run benchmark
tgeom = generate_geometry(natom)
input = (; ngauss, natom, xpnt=txpnt, coef=tcoef, geom=tgeom)

printstyled("Running JACC version:\n"; bold=true, color=:cyan)
run_benchmark(HartreeFock.bhfp_jacc, input)
