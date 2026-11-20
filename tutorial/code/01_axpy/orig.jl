function axpy!(x, y, alpha = 2.5)
    for i in 1:length(x)
        @inbounds x[i] += alpha * y[i]
    end
end
