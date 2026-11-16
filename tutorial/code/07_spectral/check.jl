include("../ck_time.jl")
include("main.jl")
include("solution.jl")

function check_spectral()
    num_bands = 60
    num_voxel = 4096
    size_voxel = 64*64
    
    img_h, filter_h = Ex_spectral.init(
        num_bands, num_voxel, size_voxel; dtype = JACC.default_float())
    
    ck_time() do
        Ex_spectral.spectral!(num_voxel, size_voxel, img_h, filter_h, num_bands)
    end
    
    println()
    
    image = JACC.to_device(img_h)
    filter = JACC.to_device(filter_h)
    
    ck_time() do
        spectral_jacc!(num_voxel, size_voxel, image, filter, num_bands)
    end
    
    println()
    
    ck_time() do
	spectral_jacc_shared!(num_voxel, size_voxel, image, filter, num_bands)
    end
end

check_spectral()
