# Attempt to perform a bifurcation analysis of the Hodgkin–Huxley ODEs with temperature as the
# free parameter

using DifferentialEquations
using Plots
using BifurcationKit
using ForwardDiff




# Alpha, beta functions
alpha_n(v) = (0.02 * (v - 25.0)) / (1.0 - exp((-1.0 * (v - 25.0)) / 9.0))
beta_n(v) = (-0.002 * (v - 25.0)) / (1.0 - exp((v - 25.0) / 9.0))

# Sodium ion-channel rate functions
alpha_m(v) = (0.182 * (v + 35.0)) / (1.0 - exp((-1.0 * (v + 35.0)) / 9.0))
beta_m(v) = (-0.124 * (v + 35.0)) / (1.0 - exp((v + 35.0) / 9.0))

alpha_h(v) = 0.25 * exp((-1.0 * (v + 90.0)) / 12.0)
beta_h(v) = (0.25 * exp((v + 62.0) / 6.0)) / exp((v + 90.0) / 12.0)

# steady state values
n_inf(v) = alpha_n(v) / (alpha_n(v) + beta_n(v))
m_inf(v) = alpha_m(v) / (alpha_m(v) + beta_m(v))
h_inf(v) = alpha_h(v) / (alpha_h(v) + beta_h(v))

# temperature
temperature_factor(t) = 3^((t - 6.3) / 10)


# Define the Hodgkin-Huxley model ODE
function hh!(du, u, p, t=0)
    # parameters
    (;gK, gNa, gL, EK, ENa, EL, T, C, I) = p
	T_factor = temperature_factor(T)

    # state variables
    v, n, m, h = u

    # Actual ODE
    du[1] = (-(gK * (n^4.0) * (v - EK)) - (gNa * (m^3.0) * h * (v - ENa)) - (gL * (v - EL)) + I) / C
    du[2] =  T_factor * (alpha_n(v) * (1.0 - n)) - (beta_n(v) * n)
    du[3] =  T_factor * (alpha_m(v) * (1.0 - m)) - (beta_m(v) * m)
    du[4] =  T_factor * (alpha_h(v) * (1.0 - h)) - (beta_h(v) * h)
	du
end


# Choice of parameters
gK = 36.0
gNa = 40.0
gL = 0.3
EK = -77.0
ENa = 55.0
EL = -65.
C = 1.

V_rest = -65.

temperature = -0.2 # Starting point of temperature
I=1.

p = (
gK=gK, 
gNa=gNa, 
gL=gL, 
EK=EK, 
ENa=ENa, 
EL=EL, 
T=temperature,
C=C, 
I=I
)


# Initial conditions
u0 = [V_rest, n_inf(0), m_inf(0), h_inf(0)]

# Time span
tspan = (0.0, 200.0)

# Jacobian of H.H. Model by AD

# Wrap hh! into non-in-place hh
function hh(u, p, t=0)
    du = similar(u)
    hh!(du, u, p, t)
    return du
end
D_hh(u0, p) = ForwardDiff.jacobian(u -> hh(u, p), u0)



# function to record information from a solution (boilerplate)
recordFromSolution(x, p; k...) = (u1 = x[1], u2 = x[2], u3=x[3], u4=x[4])#, u3 = x[3], u4 = x[4])


# setup bifurcation problem
prob = BifurcationProblem(hh!, u0, p, (@optic _[7]), record_from_solution = recordFromSolution)

# continuation options
opts_br = ContinuationPar(
	p_min = -5.0, 
	p_max = 30., 
	ds=0.00002, # positive direction
	dsmax = 0.02,
	dsmin=0.00001,
	n_inversion = 8, 
	detect_bifurcation = 3,
	# number of eigenvalues
	nev = 8,
	# maximum number of continuation steps
	max_steps = 1000
	)


@info "All setup done; preparing to run continuation for steady state finding"
# run the continuation
br = continuation(prob, PALC(), opts_br, bothside = true)

@info br "Continuation done; plotting the bifurcation diagram"
# plot
diagram = bifurcationdiagram(
	prob, 
	PALC(),
	3,
	opts_br
	)
scene = plot(diagram; code = (), title="$(size(diagram)) branches", legend = false)

display(scene)
@info "Plotting done; continuation for periodic orbits"

# Finding periodic orbits based from Hopf bifurcation point at I = 2.8
br_po1 = continuation(
		br, 
		2, 
		opts_br,
		δp= 0.0002,
		#ampfactor=0.1,
        PeriodicOrbitOCollProblem(25, 5),
		bothside=true
        )


# Find periodic orbits starting at other Hopf bifurcation
br_po2 = continuation(
		br, 
		5, 
		opts_br,
		δp= 0.00002,
        PeriodicOrbitOCollProblem(25, 5),
		bothside=true,
        )

p_po1 = plot(br, br_po1, branchlabel = ["equib" "periodic orbits"])
p_po2 = plot(br, br_po2, branchlabel = ["equib" "periodic orbits"])

display(p_po1)
display(p_po2)
@info "Periodic Orbits finding finished" br_po1 br_po2

"""
Get a periodic orbit:
sol1 = get_periodic_orbit(br_po2, 4)
plot(sol1.t, sol1[1,:], label = "V", xlabel = "time")
plot(sol1.t, sol1[2,:], label = "n", xlabel = "time")
plot!(sol1.t, sol1[3,:], label = "m", xlabel = "time")
plot!(sol1.t, sol1[4,:], label = "h", xlabel = "time")
"""

"Paper:
With fixed temperature,

Single Spike Explanation? At below Hopf condition

Fix I and vary T as bifurcation curve
"