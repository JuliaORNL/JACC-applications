
function min_max(x::AbstractArray)
    mn = eltype(x)(Inf)
    mx = eltype(x)(-Inf)

    for v in x
        mn = min(mn, v)
        mx = max(mx, v)
    end

    return (mn, mx)
end

function min_max_avg(x::AbstractArray)
    mn = eltype(x)(Inf)
    mx = eltype(x)(-Inf)
    av = eltype(x)(0.0)

    for v in x
        mn = min(mn, v)
        mx = max(mx, v)
        av += v
    end

    av /= length(x)

    return (mn, mx, av)
end
