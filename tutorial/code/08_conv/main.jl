module Ex_conv

include("conv.jl")
include("orig.jl")

function init(n_inputs = 256, n_x = 1024, n_y = 1024, n_filter = 5;
        array_type = Array, dtype = Float64)
    output = zeros(dtype, (n_inputs, n_x, n_y))
    input = round.(rand(dtype, (n_inputs, n_x, n_y)) * 100)
    filter = round.(rand(dtype, (n_filter, n_filter)) * 100)
    return array_type(output), array_type(input), array_type(filter)
end

end
