module Ex_gemm

include("orig.jl")

function init(Arows = 100, Acols = 100, Bcols = 100; array_type=Matrix)
    a = round.(rand(Arows, Acols) * 100)
    b = round.(rand(Acols, Bcols) * 100)
    c = zeros(Arows, Bcols)

    A = array_type(a)
    B = array_type(b)
    C = array_type(c)

    return A, B, C
end

end
