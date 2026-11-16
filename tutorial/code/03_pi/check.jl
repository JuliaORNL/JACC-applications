include("../ck_time.jl")
include("main.jl")
include("solution.jl")

N = 1_000_000_000

function check_pi()
    res = 0.0
    
    ck_time(() -> begin
        res = Ex_pi.est_pi(N)
    end)
    @show res
    println()
    
    ck_time(() -> res = est_pi_jacc(N))
    @show res
    println()
    
    ck_time(() -> res = est_pi_jacc_2(N))
    @show res
end

check_pi()
