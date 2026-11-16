using JACC
JACC.@init_backend

function sp(i, j, image, filter, num_bands)
    for b in 1:num_bands
        @inbounds image[b, i, j] *= filter[j]
    end
end

function spectral_jacc!(num_voxel, size_voxel, image, filter, num_bands)
    JACC.parallel_for((num_voxel,size_voxel), sp, image, filter, num_bands)
end

function sp_sh(i, j, image, filter, num_bands)
    filter_shared = JACC.shared(filter) # init shared memory
    for b in 1:num_bands
        @inbounds image[b, i, j] *= filter_shared[j]
    end
end

function spectral_jacc_shared!(num_voxel, size_voxel, image, filter, num_bands)
    JACC.parallel_for((num_voxel,size_voxel), sp_sh, image, filter, num_bands)
end
