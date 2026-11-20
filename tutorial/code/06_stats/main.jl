module Ex_stats

export min_max
export min_max_avg

include("orig.jl")

function init(N; array_type=Vector, dtype = Float64)
    return array_type(round.(rand(dtype, N) * 100))
end

end
