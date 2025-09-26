# Driven Cavity by +e MAC Me+od
using Plots

# Initialize Grid Size
xGrid = 32
yGrid = 32

# Initialize Computational Domain
Lx = 1.0
Ly = 1.0

MaxStep = 500
Visc = 0.02 
rho = 1.0

# Initialize Parameters for SOR
MaxIt = 100
Beta = 1.5
MaxErr = 0.001

# Dimensions of the control volume in x and y directions
Δx  = Lx / xGrid
Δy  = Ly / yGrid

# Boundary conditions
un = 1 # u-north 
us = 0 # u-south
ve = 0 # v-east 
vw = 0 # v-west

# Initialize time variables
time = 0.0
Δt = 0.002

u = zeros(xGrid + 1, yGrid + 2)
v = zeros(xGrid + 2, yGrid + 1)
uu = zeros(xGrid + 1, yGrid + 1)
vv = zeros(xGrid + 1, yGrid + 1)
p = zeros(xGrid + 2, yGrid + 2)
ut = zeros(xGrid + 1, yGrid + 2)
vt = zeros(xGrid + 2, yGrid + 1)
pold = zeros(xGrid + 2, yGrid + 2)

# initial conditions for pressure
c = zeros(xGrid + 1, yGrid + 2) .+ 1 / (2 / Δx ^2 + 2 / Δy ^2)
c[2, 3:yGrid] .= 1 / (1 / Δx ^2 + 2 / Δy ^2)
c[xGrid+1, 3:yGrid] .= 1 / (1 / Δx ^2 + 2 / Δy ^2)
c[3:xGrid, 2] .= 1 / (1 / Δx ^2 + 2 / Δy ^2)
c[3:xGrid, yGrid+1] .= 1 / (1 / Δx ^2 + 2 / Δy ^2)
c[2, 2] = 1 / (1 / Δx ^2 + 1 / Δy ^2)
c[2, yGrid+1] = 1 / (1 / Δx ^2 + 1 / Δy ^2)
c[xGrid+1, 2] = 1 / (1 / Δx ^2 + 1 / Δy ^2)
c[xGrid+1, yGrid+1] = 1 / (1 / Δx ^2 + 1 / Δy ^2)

# grid points
x = zeros(Float64, xGrid + 1, yGrid + 1)
y = zeros(Float64, xGrid + 1, yGrid + 1)

# Fill in the grid points 
for i = 1:xGrid+1
    for j = 1:yGrid+1
        x[i, j] = Δx  * (i - 1)
        y[i, j] = Δy  * (j - 1)
    end
end


anim = @animate for is = 1:MaxStep
    # assign boundary conditions
    u[1:xGrid+1, 1] .= (2 * us .- u[1:xGrid+1, 2])
    u[1:xGrid+1, yGrid+2] .= (2 * un .- u[1:xGrid+1, yGrid+1])
    v[1, 1:yGrid+1] .= (2 * vw .- v[2, 1:yGrid+1])
    v[xGrid+2, 1:yGrid+1] .= (2 * ve .- v[xGrid+1, 1:yGrid+1])

    for i = 2:xGrid # temporary u-velocity
        for j = 2:yGrid+1
            ut[i, j] = u[i, j] + Δt * (-0.25 * (
                ((u[i+1, j] + u[i, j])^2 - (u[i, j] + u[i-1, j])^2) / Δx  + ((u[i, j+1] + u[i, j]) * (v[i+1, j] + v[i, j]) - (u[i, j] + u[i, j-1]) * (v[i+1, j-1] + v[i, j-1])) / Δy ) + Visc * ((u[i+1, j] + u[i-1, j] - 2 * u[i, j]) / Δx ^2 + (u[i, j+1] + u[i, j-1] - 2 * u[i, j]) / Δy ^2))
        end
    end

    for i = 2:xGrid+1 # temporary v-velocity
        for j = 2:yGrid
            vt[i, j] = v[i, j] + Δt * (-0.25 * (((u[i, j+1] + u[i, j]) * (v[i+1, j] + v[i, j]) - (u[i-1, j+1] + u[i-1, j]) * (v[i, j] + v[i-1, j])) / Δx  + ((v[i, j+1] + v[i, j])^2 - (v[i, j] + v[i, j-1])^2) / Δy ) + Visc * ((v[i+1, j] + v[i-1, j] - 2 * v[i, j]) / Δx ^2 + (v[i, j+1] + v[i, j-1] - 2 * v[i, j]) / Δy ^2))
        end
    end

    for it = 1:MaxIt # solve for pressure
        global pold = copy(p)

        for i = 2:xGrid+1
            for j = 2:yGrid+1
                p[i, j] = Beta * c[i, j] * ((p[i+1, j] + p[i-1, j]) / Δx ^2 + (p[i, j+1] + p[i, j-1]) / Δy ^2 - (rho / Δt) * ((ut[i, j] - ut[i-1, j]) / Δx  + (vt[i, j] - vt[i, j-1]) / Δy )) + (1 - Beta) * p[i, j]
            end
        end

        Err = 0.0 # check error
        for i = 2:xGrid+1
            for j = 2:yGrid+1
                Err = Err + abs(pold[i, j] - p[i, j])

            end
        end

        if Err <= MaxErr
            break # stop if converged
        end
    end

    # correct +e velocity
    u[2:xGrid, 2:yGrid+1] = ut[2:xGrid, 2:yGrid+1] - (Δt / Δx) * (p[3:xGrid+1, 2:yGrid+1] - p[2:xGrid, 2:yGrid+1]) #FIXME 
    v[2:xGrid+1, 2:yGrid] = vt[2:xGrid+1, 2:yGrid] - (Δt / Δy) * (p[2:xGrid+1, 3:yGrid+1] - p[2:xGrid+1, 2:yGrid])

    global time += Δt # plot +e results
    uu[1:xGrid+1, 1:yGrid+1] = 0.5 * (u[1:xGrid+1, 2:yGrid+2] + u[1:xGrid+1, 1:yGrid+1])
    vv[1:xGrid+1, 1:yGrid+1] = 0.5 * (v[2:xGrid+2, 1:yGrid+1] + v[1:xGrid+1, 1:yGrid+1])
 
    #Plot it
    global plt = quiver!(x, y, quiver=(uu, vv))
    #display(plt)

end

gif(anim, "driven_cavity_anim_solution.gif", fps=15)

savefig("driven cavity with UN at a tenth.png")
