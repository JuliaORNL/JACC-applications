module Ex_spectral

include("orig.jl")

function init(num_bands = 60, num_voxel = 1024, size_voxel = 4096;
        array_type = Array, dtype = Float64)
    image = round.(rand(dtype, (num_bands, num_voxel, size_voxel)) * 100)
    filter = round.(rand(dtype, size_voxel) * 100)

    return array_type(image), array_type(filter)
end

end
