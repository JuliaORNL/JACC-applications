include("../ck_time.jl")
include("main.jl")
include("solution.jl")

function check_axpy()
    N = 200_000_000
    
    x_h, y_h = Ex_axpy.init(N; dtype = JACC.default_float())
    ck_time(() -> Ex_axpy.axpy!(x_h, y_h))
    
    println()
    
    x_d, y_d = JACC.to_device.((x_h, y_h))
    ck_time() do
        axpy_jacc!(x_d, y_d)
    end
end

check_axpy()
