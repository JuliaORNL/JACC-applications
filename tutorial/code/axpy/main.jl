module Ex_axpy

include("orig.jl")

function init(N = 100; array_type=Vector)
    a = round.(rand(N) * 100)
    b = round.(rand(N) * 100)

    x = array_type(a)
    y = array_type(b)

    return x, y
end

end
