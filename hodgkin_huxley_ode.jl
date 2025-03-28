# Define the Hodgkin-Huxley model

# global parameters
giant_axon = true

I = 0.1 # Current injection 
T = 6.3 # Temperature factor

if giant_axon
    function alpha_n(v)
        vred = v - V_rest
        A_n = exp((-vred+10.0)/10.0)
        return 0.01 * (-vred + 10.0) / (A_n - 1.0)
    end

    function beta_n(v)
        vred = v - V_rest
        return 0.125 * exp(-vred/80.0)
    end
    
    function alpha_m(v)
        vred = v - V_rest
        A_m = exp((-vred+25.0)/10.0)
        return 0.1 * (-vred + 25.0) / (A_m - 1.0)
    end

    function beta_m(v)
        vred = v - V_rest
        return 4.0 * exp(-vred/18.0)
    end

    function alpha_h(v)
        vred = v - V_rest
        return 0.07 * exp(-vred/20.0)
    end
    
    function beta_h(v)
        vred = v - V_rest
        B_h = exp((-vred+30.0)/10.0)
        return 1.0 / (B_h + 1.0)
    end



    temperature_factor(t) = 3^((t - 6.3) / 10)

    # Define parameters
    gK = 36.0
    gNa = 120.0
    gL = 0.3
    EK = -77.0
    ENa = 55.0
    EL = -65.
    C = 1.

    V_rest = -69.7  # See to which value it converges in steady state

else
    alpha_n(v) = (0.02 * (v - 25.0)) / (1.0 - exp((-1.0 * (v - 25.0)) / 9.0))
    beta_n(v) = (-0.002 * (v - 25.0)) / (1.0 - exp((v - 25.0) / 9.0))
    
    # Sodium ion-channel rate functions
    alpha_m(v) = (0.182 * (v + 35.0)) / (1.0 - exp((-1.0 * (v + 35.0)) / 9.0))
    beta_m(v) = (-0.124 * (v + 35.0)) / (1.0 - exp((v + 35.0) / 9.0))
    
    alpha_h(v) = 0.25 * exp((-1.0 * (v + 90.0)) / 12.0)
    beta_h(v) = (0.25 * exp((v + 62.0) / 6.0)) / exp((v + 90.0) / 12.0)

    gK = 36.0
    gNa = 40.0
    gL = 0.3
    EK = -77.0
    ENa = 55.0
    EL = -65.
    C = 1.
    V_rest = -63.  # See to which value it converges in steady state
end


temperature_factor(t) = 3^((t - 6.3) / 10)
n_inf(v) = alpha_n(v) / (alpha_n(v) + beta_n(v))
m_inf(v) = alpha_m(v) / (alpha_m(v) + beta_m(v))
h_inf(v) = alpha_h(v) / (alpha_h(v) + beta_h(v))

function hh!(du, u, p, t)
    # Extract parameters
    (;gK, gNa, gL, EK, ENa, EL, T, C, I) = p

    # Extract state variables
    v, n, m, h = u

    # Define the ODEs
    du[1] = (-(gK * (n^4.0) * (v - EK)) - (gNa * (m^3.0) * h * (v - ENa)) - (gL * (v - EL)) + I) / C
    du[2] =  T * (alpha_n(v) * (1.0 - n)) - (beta_n(v) * n)
    du[3] =  T * (alpha_m(v) * (1.0 - m)) - (beta_m(v) * m)
    du[4] =  T * (alpha_h(v) * (1.0 - h)) - (beta_h(v) * h)
	du
end

# Initial conditions
u0 = [V_rest, n_inf(V_rest), m_inf(V_rest), h_inf(V_rest)]
	
# Time span
tspan = (0.0, 100.0)

function hh(u, p, t=0)
    du = similar(u)
    hh!(du, u, p, t)
    return du
end
D_hh(u0, p) = ForwardDiff.jacobian(u -> hh(u, p), u0)

temp_factor = temperature_factor(T)
p = (
gK=gK, 
gNa=gNa, 
gL=gL, 
EK=EK, 
ENa=ENa, 
EL=EL, 
T=temp_factor,
C=C, 
I=I
)

recordFromSolution(x, p; k...) = (u1 = x[1], u2 = x[2], u3=x[3], u4=x[4])#, u3 = x[3], u4 = x[4])

probODE = ODEProblem(hh!, u0, tspan, p)
	
# Solve the ODE
sol = solve(probODE, Rosenbrock23())

# Extract solution components
t_vals = sol.t                     # Time values
V_vals = [sol[i][1] for i in eachindex(t_vals)]  # Membrane Voltage V
m_vals = [sol[i][2] for i in eachindex(t_vals)]  # Gating variable m
h_vals = [sol[i][3] for i in eachindex(t_vals)]  # Gating variable h
n_vals = [sol[i][4] for i in eachindex(t_vals)]  # Gating variable n

plot(sol, vars=(0, 1), xlabel="Time (ms)", ylabel="Membrane Potential (mV)", label="V(t)")#, ylims=[-0., 30.])


plot(sol, vars=(2:4), xlabel="Time (ms)", ylabel="Gating variables", label=["n" "m" "h"])


# Plot phase diagrams
plot(V_vals, m_vals, label="V vs m", xlabel="Voltage (mV)", ylabel="Gating Variable", title="Phase Plot V-m")
plot!(V_vals, h_vals, label="V vs h")
plot!(V_vals, n_vals, label="V vs n")

	