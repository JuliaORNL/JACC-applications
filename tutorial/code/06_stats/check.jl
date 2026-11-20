include("../ck_time.jl")
include("main.jl")
include("solution.jl")

const DFloat = JACC.default_float()

function check_stats()
    N = 2_000_000_000
    
    x_h = Ex_stats.init(N; dtype = DFloat)
    
    mn::DFloat = 0.0
    mx::DFloat = 0.0
    av::DFloat = 0.0
    
    println("min_max (host)")
    ck_time() do
        mn, mx = Ex_stats.min_max(x_h)
    end
    @show mn, mx

    println("\nmin_max_avg (host)")
    ck_time() do
        mn, mx, av = Ex_stats.min_max_avg(x_h)
    end
    @show mn, mx, av
    
    println()
    
    x_d = JACC.to_device(x_h)
    println("min_max (jacc)")
    ck_time() do
        mn, mx = min_max_jacc(x_d)
    end
    @show mn, mx

    println("\nmin_max (jacc)")
    ck_time() do
        mn, mx, av = min_max_avg_jacc(x_d)
    end
    @show mn, mx, av
end

check_stats()
