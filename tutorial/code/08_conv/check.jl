include("../ck_time.jl")
include("main.jl")
include("solution.jl")

function check_conv()
    n_inputs = 256
    n_x = 1024
    n_y = 1024
    n_filter = 5
    
    out_h, in_h, filter_h = Ex_conv.init(
        n_inputs, n_x, n_y, n_filter; dtype = JACC.default_float())
    
    ck_time() do
	Ex_conv.conv!(out_h, in_h, filter_h)
    end
    
    println()
    
    out, in, filter = JACC.to_device.((out_h, in_h, filter_h))
    
    ck_time() do
	conv_jacc!(out, in, filter)
    end
    
    println()
    
    ck_time() do
	conv_jacc_shared!(out, in, filter)
    end
end

check_conv()
