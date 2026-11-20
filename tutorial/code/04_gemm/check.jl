include("../ck_time.jl")
include("main.jl")
include("solution.jl")

function check_gemm()
    Arows = 2_000
    Acols = 2_000
    Bcols = 2_000
    
    A_h, B_h, C_h = Ex_gemm.init(Arows, Acols, Bcols; dtype = JACC.default_float())
    ck_time(() -> Ex_gemm.gemm!(A_h, B_h, C_h))
    
    println()
    
    A_d = JACC.to_device(A_h)
    B_d = JACC.to_device(B_h)
    C_d = JACC.to_device(C_h)
    ck_time(() -> gemm_jacc!(A_d, B_d, C_d))
end

check_gemm()
