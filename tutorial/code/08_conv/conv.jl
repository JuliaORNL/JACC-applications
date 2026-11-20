
function compute_conv(n, i, j, input, filter)
    conv = 0.0
    f_ni, f_nj = size(filter)
    i_s = i - f_ni
    j_s = j - f_nj
    for fj in 1:f_nj
        for fi in 1:f_ni
            if (i_s + fi > 0) && (j_s + fj > 0)
                @inbounds conv += filter[fi, fj] * input[n, i_s + fi, j_s + fj]
            end
        end
    end
    return conv
end
