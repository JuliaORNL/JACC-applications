import JACC
JACC.@init_backend

function gemm_jacc!(A, B, C)
    A_rows = size(A)[1]
    A_cols = size(A)[2]
    B_cols = size(B)[2]

    JACC.parallel_for((A_rows, B_cols), A, B, C) do i, j, A, B, C
        @inbounds begin
            sum = zero(eltype(C))
            for l = 1:size(A, 2)
                sum += A[i, l] * B[l, j]
            end
            C[i, j] = sum
        end
    end
    return nothing
end
