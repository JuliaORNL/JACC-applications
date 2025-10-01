
# LID-DRIVEN CAVITY SIMULATION using JACC.jl

Originally from: [github:aturanb/lid-driven-cavity](https://github.com/aturanb/lid-driven-cavity)

The following code is used to create a simple 2D lid-driven cavity simulation using the JACC.jl library. 
JACC.jl enables the same code to be executed on both CPU and GPU hardware by setting a `JACC.set_backend("CUDA")` or `JACC.set_backend("AMDGPU")` calls.

The simulation solves the incompressible Navier-Stokes equations using a finite difference method on a staggered grid. 