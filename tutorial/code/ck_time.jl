using BenchmarkTools

function ck_time(f)
    println("@time 1")
    @time f()
    println("@time 2")
    @time f()
    println("@btime")
    @btime $f()
    println()
end
