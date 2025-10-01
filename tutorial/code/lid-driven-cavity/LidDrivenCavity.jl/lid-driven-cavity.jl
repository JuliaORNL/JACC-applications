
import LidDrivenCavity

# using StatProfilerHTML
# using Profile
# using PProf

function julia_main()::Cint
    if length(ARGS) != 1
        println("Usage: julia --project=. lid-driven-cavity.jl <config_file.yaml>")
        println(ARGS)
        return 1
    end

    LidDrivenCavity.run(ARGS[1])
    return 0
end

if !isdefined(Base, :active_repl)
    @time julia_main()
    # @profilehtml julia_main()
    # @profile julia_main()
    # pprof()
end