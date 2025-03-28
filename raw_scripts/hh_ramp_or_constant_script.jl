using DifferentialEquations
using Plots
using CSV 
using DataFrames

# Ion-channel rate functions
alpha_n(v) = (0.02 * (v - 25.0)) / (1.0 - exp((-1.0 * (v - 25.0)) / 9.0))
beta_n(v) = (-0.002 * (v - 25.0)) / (1.0 - exp((v - 25.0) / 9.0))

alpha_m(v) = (0.182 * (v + 35.0)) / (1.0 - exp((-1.0 * (v + 35.0)) / 9.0))
beta_m(v) = (-0.124 * (v + 35.0)) / (1.0 - exp((v + 35.0) / 9.0))

alpha_h(v) = 0.25 * exp((-1.0 * (v + 90.0)) / 12.0)
beta_h(v) = (0.25 * exp((v + 62.0) / 6.0)) / exp((v + 90.0) / 12.0)

# Hodgkin-Huxley ODE system
function hh!(du, u, p, t)
    (;gK, gNa, gL, EK, ENa, EL, C, I_func) = p
    I_stim = I_func(t)
    v, n, m, h = u
    du[1] = (-(gK * n^4 * (v - EK)) - (gNa * m^3 * h * (v - ENa)) - (gL * (v - EL)) + I_stim) / C
    du[2] = alpha_n(v) * (1 - n) - beta_n(v) * n
    du[3] = alpha_m(v) * (1 - m) - beta_m(v) * m
    du[4] = alpha_h(v) * (1 - h) - beta_h(v) * h
    return du
end

n_inf(v) = alpha_n(v) / (alpha_n(v) + beta_n(v))
m_inf(v) = alpha_m(v) / (alpha_m(v) + beta_m(v))
h_inf(v) = alpha_h(v) / (alpha_h(v) + beta_h(v))
temperature_factor(t) = 3^((t - 6.3) / 10)

function ramp(minimum, maximum, start, ending, t; steady=false)
    if t <= start
        return steady ? minimum : 0
    elseif t >= ending
        return steady ? maximum : 0
    else
        return minimum + (t - start) / (ending - start) * (maximum - minimum)
    end
end

function I_trig(t)
    """
    Hard coded value I=4. which is not great. Change to making I_trig accept another parameter in a later version
    """
    t_start = 0.0
    t_end = 4000.0
    t_mid = (t_end - t_start) / 2
    if t <= t_mid
        return ramp(0, 4.0, t_start, t_mid, t)
    else
        return ramp(4.0, 0, t_mid, t_end, t)
    end
end

# Parameters
gK = 36.0
gNa = 40.0
gL = 0.3
EK = -77.0
ENa = 55.0
EL = -65.0
C = 1.0
T = 29.0
I_max = 0.4 # Current stimulation
temp_factor = temperature_factor(T)
V_rest = -65.0

p = (
    gK = gK,
    gNa = gNa,
    gL = gL,
    EK = EK,
    ENa = ENa,
    EL = EL,
    C = C,
    T = temp_factor,
    I_func = I_trig
)

constant_stim = true
if constant_stim
    p = (
    gK = gK,
    gNa = gNa,
    gL = gL,
    EK = EK,
    ENa = ENa,
    EL = EL,
    C = C,
    T = temp_factor,
    I_func = t -> I_max # constant stimulation with I_max
)
end
    

# Initial conditions (Note: evaluates at v=0, not at rest potential)
u0 = [V_rest, n_inf(0), m_inf(0), h_inf(0)]
tspan = (0.0, 5000.0)

@info "Setup finished. Solving ODE..."
probODE = ODEProblem(hh!, u0, tspan, p)
sol = solve(probODE, Tsit5())

@info "Solved ODE (finished) "
t_vals = sol.t
V_vals = [sol[i][1] for i in eachindex(t_vals)]
m_vals = [sol[i][2] for i in eachindex(t_vals)]
h_vals = [sol[i][3] for i in eachindex(t_vals)]
n_vals = [sol[i][4] for i in eachindex(t_vals)]

@info "Plotting"
# Plotting

display(plot(sol, vars=(0, 1), xlabel="Time (ms)", ylabel="Membrane Potential (mV)", label="V(t)"))
gui()
display(plot(sol.t, I_trig.(sol.t), xlabel="Time (ms)", ylabel="Stimulation current", label="I(t)"))
gui()
display(plot(sol, vars=(2:4), xlabel="Time (ms)", ylabel="Gating variables", label=["n" "m" "h"]))
gui()
p_phase = plot(V_vals, m_vals, label="V vs m", xlabel="Voltage (mV)", ylabel="m", title="Phase Plot V-m")
plot!(p_phase, V_vals, h_vals, label="V vs h")
plot!(p_phase, V_vals, n_vals, label="V vs n")
display(p_phase)
gui()

