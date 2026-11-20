
function conv!(out, in, filter)
    (n_inputs, n_x, n_y) = size(in)
    for j in 1:n_y
        for i in 1:n_x
            for n in 1:n_inputs
                @inbounds out[n, i, j] = compute_conv(n, i, j, in, filter)
            end
        end
    end
end
