using .Ex_conv: compute_conv

using JACC
JACC.@init_backend

function conv_jacc!(out, in, filter)
    (n_inputs, n_x, n_y) = size(in)
    n_f = length(filter)
    JACC.parallel_for((n_x, n_y), (; out, in, filter, n_inputs)) do i, j, t
        for n in 1:t.n_inputs
            @inbounds t.out[n, i, j] = compute_conv(n, i, j, t.in, t.filter)
        end
    end
end

function conv_jacc_shared!(out, in, filter)
    (n_inputs, n_x, n_y) = size(in)
    n_f = length(filter)
    JACC.parallel_for((n_x, n_y), (; out, in, filter, n_inputs)) do i, j, t
        filter_sh = JACC.shared(t.filter)
        for n in 1:t.n_inputs
            @inbounds t.out[n, i, j] = compute_conv(n, i, j, t.in, filter_sh)
        end
    end
end
