include("../ck_time.jl")
include("main.jl")
include("solution.jl")

function check_cg()
    SIZE = 200_000_000
    data = Ex_cg.init(SIZE; dtype = JACC.default_float())
    
    data_jacc = init_jacc(data)
    ck_time() do
        cg_jacc(data_jacc)
    end
    
    println()
    
    data_multi = init_jacc_multi(data)
    ck_time() do
	cg_jacc_multi(data_multi)
    end
    
    println()
    
    data_async = init_jacc_async(data)
    ck_time() do
	cg_jacc_async(data_async)
    end
end

check_cg()
