


using Test

import LidDrivenCavity


@testset "LidDrivenCavity Input" begin

    settings = LidDrivenCavity._read_input_file("data/config.yaml")
    @test settings.x_grid == 64
    @test settings.y_grid == 64
    @test settings.length_x == 2.0
    @test settings.length_y == 2.0
    @test settings.max_steps == 1000
    @test settings.viscosity == 0.04
    @test settings.density == 2.0
    @test settings.max_iterations == 200
    @test settings.beta == 3.0
    @test settings.max_error == 0.002
    @test settings.u_north == 1.0
    @test settings.u_south == 0.0
    @test settings.v_east == 0.0
    @test settings.v_west == 0.0
    @test settings.initial_time == 0.0
    @test settings.time_step == 0.004
    @test settings.T == Float32

end
