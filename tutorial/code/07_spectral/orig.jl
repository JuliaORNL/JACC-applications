function spectral!(num_voxel, size_voxel, image, filter, num_bands)
    for j = 1:size_voxel
        for i = 1:num_voxel
            for b = 1:num_bands
                @inbounds image[b, i, j] *= filter[j]
            end
        end
    end
end
