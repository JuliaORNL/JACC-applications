using JACC
JACC.@init_backend

using StaticArrays

a = JACC.to_device([Float64(i) for i in 1:10])

const JACCFloat = JACC.default_float()
const MinMax = SVector{3, JACCFloat}
MinMax() = MinMax(Inf, -Inf, 0.0)
# const MinMax = NTuple{3, JACCFloat}

Base.convert(::Type{MinMax}, x::JACCFloat) = MinMax(x, x, x)
my_minmax(a::MinMax, b::MinMax) = MinMax(min(a[1], b[1]), max(a[2], b[2]), +(a[3], b[3]))
my_minmax(a::MinMax, b::JACCFloat) = MinMax(min(a[1], b), max(a[2], b), +(a[3], b))

@show JACC.parallel_reduce(my_minmax, a; init = MinMax(Inf, -Inf, 0.0))
