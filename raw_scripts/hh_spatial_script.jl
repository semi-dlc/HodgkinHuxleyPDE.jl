using DifferentialEquations
using LinearAlgebra
using SparseArrays
using Plots
using NLsolve
using ForwardDiff
using PlutoUI

# Hodgkin-Huxley parameter struct
mutable struct HHParams
    gK::Float64
    gNa::Float64
    gL::Float64
    EK::Float64
    ENa::Float64
    EL::Float64 #
    Tfac::Float64 # temperature factor (dimensionless)
    C::Float64 # membrane capacitance [F/mm^2]
    I_stim::Float64 # stimulation current functional (maps time to I_stim(x) over space)
end

# Hodgkin-Huxley parameter struct for the PDE case
mutable struct HH_PDE_Params
    gK::Float64
    gNa::Float64
    gL::Float64
    EK::Float64
    ENa::Float64
    EL::Float64 #
	V_rest::Float64
    Tfac::Float64 # temperature factor (dimensionless)
    C::Float64 # membrane capacitance [F/mm^2]
	I_base::Float64
    I_stim::Float64 # stimulation current functional (maps time to I_stim(x) over space)
	D::Float64 # Constant of the diffusivity matrix (~alpha)
	L::Float64 # Length of the cable
	N::Int64 # Discretization of the cable
end

# Q10 temperature scaling. We might want to check that the parameters of HHParams are really temperature-independent. Especially the reversal potential seems to come from a temperature-dependent equation, mathematically.
temperature_factor(t) = 3^((t - 6.3) / 10)


L = 10.0                # cm, cable length
N = 201                # number of grid points
dx = L/(N-1)
x = range(0, L, length=N)
D = 1.0                # seems to be a diffusivity constant.

# Hodgkin-Huxley parameters
gNa, gK, gL = 120.0, 36.0, 0.3
ENa, EK, EL = 50.0, -77.0, -54.4
C = 1.0
V_rest = -65.
T = 10.

I_stim = 200.
p = HH_PDE_Params(gK, gNa, gL, EK, ENa, EL, V_rest, temperature_factor(T), C, 0., I_stim, D, L, N)



begin
	function alpha_n(v)
        vred = v + 65
        0.01 * (10.0 - vred) / (exp((10.0 - vred)/10.0) - 1.0)
    end
    function beta_n(v)
        vred = v + 65
        0.125 * exp(-vred / 80.0)
    end
    function alpha_m(v)
        vred = v + 65
        0.1 * (25.0 - vred) / (exp((25.0 - vred)/10.0) - 1.0)
    end
    function beta_m(v)
        vred = v + 65
        4.0 * exp(-vred / 18.0)
    end
    function alpha_h(v)
        vred = v + 65
        0.07 * exp(-vred / 20.0)
    end
    function beta_h(v)
        vred = v + 65
        1.0 / (exp((30.0 - vred)/10.0) + 1.0)
    end

    n_inf(v) = alpha_n(v)/(alpha_n(v) + beta_n(v))
    m_inf(v) = alpha_m(v)/(alpha_m(v) + beta_m(v))
    h_inf(v) = alpha_h(v)/(alpha_h(v) + beta_h(v))
end

DIRICHLET = false
NEUMANN = true
ROBIN = false

function laplacian_matrix(N, dx)
        main = -2 * ones(N)
        off = ones(N-1)
        A = spdiagm(-1 => off, 0 => main, 1 => off) 
		if DIRICHLET
        	A[1,:] .= 0;  A[end,:] .= 0    # Dirichlet: boundary rows are zero (V fixed)
		elseif NEUMANN
			A[1,1] = -1; A[end, end] = -1
		end
		A /= dx^2
        return A
end

Δ = laplacian_matrix(N, dx) # find a way to make this be computed by hh_cable! once on initial step.

# Vector ODE system: [V₁..V_N, n₁..n_N, m₁..m_N, h₁..h_N]
function hh_cable!(du, u, p, t)

	# Extract quantities from the state vector
	V = @view u[1:p.N]
	n = @view u[p.N+1:2p.N]
	m = @view u[2p.N+1:3p.N]
	h = @view u[3p.N+1:4p.N]
	dV = @view du[1:p.N]
	dn = @view du[p.N+1:2p.N]
	dm = @view du[2p.N+1:3p.N]
	dh = @view du[3p.N+1:4p.N]
	
	# Laplacian for V, zero at boundaries
	diffV = p.D * (Δ * V)
	
	I_ext = zeros(N)
	stim = convert(Int, round(p.N/2)) : convert(Int, round(p.N/2 + 10))
	if 0.0 < t < 8.0
		I_ext[stim] .= p.I_stim    # μA/cm²
	end

	for i in 1:p.N
		I_Na = p.gNa * m[i]^3 * h[i] * (V[i] - p.ENa)
		I_K  = p.gK * n[i]^4 * (V[i] - p.EK)
		I_L  = p.gL * (V[i] - p.EL)
		dV[i] = (diffV[i] - I_Na - I_K - I_L + I_ext[i]) / C
		dn[i] = p.Tfac * (alpha_n(V[i]) * (1.0 - n[i]) - beta_n(V[i]) * n[i])
		dm[i] = p.Tfac * (alpha_m(V[i]) * (1.0 - m[i]) - beta_m(V[i]) * m[i])
		dh[i] = p.Tfac * (alpha_h(V[i]) * (1.0 - h[i]) - beta_h(V[i]) * h[i])
	end
	nothing
end

# For NLsolve: Stationary system (no stimulation)
function hh_stat!(u::AbstractVector{T}, p::HH_PDE_Params) where T
    v, n, m, h = u
    dv = (-(p.gK * n^4 * (v - p.EK)) -
             (p.gNa * m^3 * h * (v - p.ENa)) -
             (p.gL * (v - p.EL)) +
             p.I_stim*0 + p.I_base) / p.C
    dn = p.Tfac * (alpha_n(v) * (1.0 - n) - beta_n(v) * n)
    dm = p.Tfac * (alpha_m(v) * (1.0 - m) - beta_m(v) * m)
    dh = p.Tfac * (alpha_h(v) * (1.0 - h) - beta_h(v) * h)
    return [dv, dn, dm, dh]
end

function find_fixpoint(p)
	u0_guess = [V_rest*1., n_inf(V_rest), m_inf(V_rest), h_inf(V_rest)]
	sol_stat = nlsolve(u -> hh_stat!(u, p), u0_guess, autodiff=:forward, xtol=1e-3)
	u0 = sol_stat.zero
	return u0
end

Vrest = -65.0
u0_id = find_fixpoint(p)
V0 = u0_id[1] * ones(N)
n0 = u0_id[2] * ones(N)
m0 = u0_id[3] * ones(N)
h0 = u0_id[4] * ones(N)

u0 = vcat(V0, n0, m0, h0)
tspan = (0.0, 10.0)

prob = ODEProblem(hh_cable!, u0, tspan, p)

# TRBDF2 is generally fine, as it is a (moderately) stiff problem
# Also, we are interested in the time evolution of the trajectory.
sol = solve(prob, TRBDF2(), saveat=0.01) 



# Plotting

Nt = length(sol.t)
Nspace = N

Vmat = zeros(Nt, Nspace)
nmat = zeros(Nt, Nspace)
mmat = zeros(Nt, Nspace)
hmat = zeros(Nt, Nspace)

INa_mat = zeros(Nt, Nspace)
IK_mat  = zeros(Nt, Nspace)
IL_mat  = zeros(Nt, Nspace)

for i in 1:Nt
    u = sol[i]
    V = @view u[1:Nspace]
    n = @view u[Nspace+1 : 2Nspace]
    m = @view u[2Nspace+1 : 3Nspace]
    h = @view u[3Nspace+1 : 4Nspace]

    Vmat[i, :] = V
    nmat[i, :] = n
    mmat[i, :] = m
    hmat[i, :] = h

    @inbounds for j in 1:Nspace
        INa_mat[i,j] = gNa * m[j]^3 * h[j] * (V[j] - ENa)
        IK_mat[i,j]  = gK  * n[j]^4       * (V[j] - EK)
        IL_mat[i,j]  = gL  * (V[j] - EL)
    end
end

# Potential and gating variable heatmaps
pV = heatmap(x, sol.t, Vmat, xlabel="x (cm)", ylabel="t (ms)", colorbar_title="V (mV)",
                title="Membrane Potential V(x,t)")
pn = heatmap(x, sol.t, nmat, xlabel="x (cm)", ylabel="t (ms)", colorbar_title="n",
                title="n-Gating Variable")
pm = heatmap(x, sol.t, mmat, xlabel="x (cm)", ylabel="t (ms)", colorbar_title="m",
                title="m-Gating Variable")
ph = heatmap(x, sol.t, hmat, xlabel="x (cm)", ylabel="t (ms)", colorbar_title="h",
                title="h-Gating Variable")

# Current heatmaps
pNa = heatmap(x, sol.t, INa_mat, xlabel="x (cm)", ylabel="t (ms)", colorbar_title="I_Na (μA/cm²)",
                title="Sodium Current I_Na")
pK  = heatmap(x, sol.t, IK_mat,  xlabel="x (cm)", ylabel="t (ms)", colorbar_title="I_K (μA/cm²)",
                title="Potassium Current I_K")
pL  = heatmap(x, sol.t, IL_mat,  xlabel="x (cm)", ylabel="t (ms)", colorbar_title="I_L (μA/cm²)",
                title="Leak Current I_L")

# Arrange into 3x2 layout
plot(pV, pn, pm, ph, pNa, pK, pL, layout=(3,3), size=(1800, 1800))

# Plot as animation
@gif for i in 1:Nt
    plot(x, Vmat[i, :],
         xlabel = "x (cm)", ylabel = "V (mV)",
         title = "Time: $(round(sol.t[i], digits=2)) ms",
         ylim = extrema(Vmat),  
         label = false)
end every 4  