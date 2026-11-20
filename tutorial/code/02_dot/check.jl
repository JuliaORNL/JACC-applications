include("../ck_time.jl")
include("main.jl")
include("solution.jl")

function check_dot()
    N = 100_000_000
    res = 0.0
    
    x_h, y_h = Ex_dot.init(N; dtype = JACC.default_float())
    ck_time() do
        res = Ex_dot.dot(x_h, y_h)
    end
    @show res
    
    println()
    
    x_d, y_d = JACC.to_device.((x_h, y_h))
    ck_time() do
        res = dot_jacc(x_d, y_d)
    end
    @show res
end

check_dot()
