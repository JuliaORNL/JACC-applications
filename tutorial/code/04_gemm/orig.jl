function gemm!(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix)
    A_rows = size(A)[1]
    A_cols = size(A)[2]
    B_cols = size(B)[2]

    for j = 1:B_cols
        for l = 1:A_cols
            @inbounds temp = B[l, j]
            for i = 1:A_rows
                @inbounds C[i, j] += temp * A[i, l]
            end
        end
    end
end
