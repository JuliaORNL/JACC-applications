import JACC
JACC.@init_backend

const JF = JACC.default_float()
const MnMx = NTuple{2, JF}
mn_mx(a::MnMx, b::MnMx) = (min(a[1], b[1]), max(a[2], b[2]))

Base.convert(::Type{MnMx}, x::JF) = (x, x)
JACC.default_init(::Type{JF}, ::typeof(mn_mx)) = (Inf, -Inf)

function min_max_jacc(x::JACC.array_type())
    mn, mx = JACC.parallel_reduce(mn_mx, x; init = (Inf, -Inf))
    return mn, mx
end

const MnMxAv = NTuple{3, JF}
mn_mx_av(a::MnMxAv, b::MnMxAv) = (min(a[1], b[1]), max(a[2], b[2]), a[3] + b[3])

Base.convert(::Type{MnMxAv}, x::JF) = (x, x, x)
JACC.default_init(::Type{JF}, ::typeof(mn_mx_av)) = (Inf, -Inf, 0.0)

function min_max_avg_jacc(x::JACC.array_type())
    mn, mx, av = JACC.parallel_reduce(mn_mx_av, x; init = (Inf, -Inf, 0.0))
    return mn, mx, av / length(x)
end
