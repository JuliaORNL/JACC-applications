function dot(x, y)
    r = 0.0
    for i in 1:length(x)
        @inbounds r += x[i] * y[i]
    end
    return r
end
