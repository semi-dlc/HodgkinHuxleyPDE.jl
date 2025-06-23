### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

begin
    using DifferentialEquations
    using Plots
    using PlutoUI
    using ForwardDiff
    using NLsolve
    using NPZ
end

# Select neuron type
const GIANT_AXON = true

# Hodgkin-Huxley gating kinetics
if GIANT_AXON
    const V_rest = -65.0  # mV

    function alpha_n(v)
        vred = v - V_rest
        x = (-vred + 10.0) / 10.0
        denom = expm1(x)
        abs(denom) < 1e-6 && (denom = x + 0.5x^2) # Taylor expansion for small x
        return 0.01 * (-vred + 10.0) / denom
    end

    function beta_n(v)
        vred = v - V_rest
        return 0.125 * exp(-vred / 80.0)
    end

    function alpha_m(v)
        vred = v - V_rest
        x = (-vred + 25.0) / 10.0
        denom = expm1(x)
        abs(denom) < 1e-6 && (denom = x + 0.5x^2)
        return 0.1 * (-vred + 25.0) / denom
    end

    function beta_m(v)
        vred = v - V_rest
        return 4.0 * exp(-vred / 18.0)
    end

    function alpha_h(v)
        vred = v - V_rest
        return 0.07 * exp(-vred / 20.0)
    end

    function beta_h(v)
        vred = v - V_rest
        return 1.0 / (exp((-vred + 30.0) / 10.0) + 1.0)
    end

    temperature_factor(T) = 3^((T - 6.3) / 10)
    # Standard parameters
    gK, gNa, gL = 36.0, 120.0, 0.3
    EK, ENa, EL = -77.0, 50.0, -54.4
    C = 1.0
else
    # Define your cortical neuron or other variant here if needed
    error("Only giant squid axon implemented in this notebook.")
end

# Steady-state gating variables
n_inf(v) = alpha_n(v) / (alpha_n(v) + beta_n(v))
m_inf(v) = alpha_m(v) / (alpha_m(v) + beta_m(v))
h_inf(v) = alpha_h(v) / (alpha_h(v) + beta_h(v))

# Hodgkin-Huxley parameter struct
mutable struct HHParams
    gK::Float64
    gNa::Float64
    gL::Float64
    EK::Float64
    ENa::Float64
    EL::Float64
    Tfac::Float64     # temperature factor (dimensionless)
    C::Float64
    I_stim::Float64   # stimulation current
    I_base::Float64   # base (holding) current
end

# Main HH ODE system (in-place, fully parameterized)
function hh!(du, u, p::HHParams, t)
    v, n, m, h = u
    du[1] = (-(p.gK * n^4 * (v - p.EK)) -
             (p.gNa * m^3 * h * (v - p.ENa)) -
             (p.gL * (v - p.EL)) +
             p.I_stim + p.I_base) / p.C
    du[2] = p.Tfac * (alpha_n(v) * (1.0 - n) - beta_n(v) * n)
    du[3] = p.Tfac * (alpha_m(v) * (1.0 - m) - beta_m(v) * m)
    du[4] = p.Tfac * (alpha_h(v) * (1.0 - h) - beta_h(v) * h)
    nothing
end

# For NLsolve: Stationary system (no stimulation)
function hh_stat!(u, p::HHParams)
    v, n, m, h = u
    dv = (-(p.gK * n^4 * (v - p.EK)) -
          (p.gNa * m^3 * h * (v - p.ENa)) -
          (p.gL * (v - p.EL)) +
          p.I_base) / p.C
    dn = p.Tfac * (alpha_n(v) * (1.0 - n) - beta_n(v) * n)
    dm = p.Tfac * (alpha_m(v) * (1.0 - m) - beta_m(v) * m)
    dh = p.Tfac * (alpha_h(v) * (1.0 - h) - beta_h(v) * h)
    return [dv, dn, dm, dh]
end

# Setup parameters
T = 6.3  # °C (baseline)
I_stim = 0.0
I_base = 0.0
p = HHParams(gK, gNa, gL, EK, ENa, EL, temperature_factor(T), C, I_stim, I_base)

# Find stationary point for initial condition
u0_guess = [V_rest, n_inf(V_rest), m_inf(V_rest), h_inf(V_rest)]
sol_stat = nlsolve(u -> hh_stat!(u, p), u0_guess, autodiff=:forward)
u0 = sol_stat.zero

@show u0

# ODE Problem and solution (no stimulation)
tspan = (0.0, 100.0)
probODE = ODEProblem(hh!, u0, tspan, p)
sol = solve(probODE, Tsit5(), save_everystep=true, reltol=1e-6)

# Plot membrane voltage and gating variables
plot(sol, vars=(0, 1), xlabel="Time (ms)", ylabel="Membrane Potential (mV)",
    label="V(t)", title="Giant Axon at T=$T°C, I_stim=$I_stim μA/cm²", size=(800, 800))

plot(sol, vars=(2, 3, 4), xlabel="Time (ms)", ylabel="Gating Variables",
    label=["n" "m" "h"], title="Giant Axon at T=$T°C, I_stim=$I_stim μA/cm²", size=(800, 600))

# Extract solution components for further analysis
t_vals = sol.t
V_vals = [u[1] for u in sol.u]
n_vals = [u[2] for u in sol.u]
m_vals = [u[3] for u in sol.u]
h_vals = [u[4] for u in sol.u]

# Phase plots (V vs gating variables)
p_phase = plot(layout=(1, 1), size=(800, 600), title="Phase Plots at T=$T°C")
plot!(p_phase, V_vals, m_vals, label="V vs m", xlabel="Voltage (mV)", ylabel="m")
plot!(p_phase, V_vals, h_vals, label="V vs h")
plot!(p_phase, V_vals, n_vals, label="V vs n")
display(p_phase)

# Utility: Test if a spike occurs (m > 0.8 = spike)
function test_spike(model_pars::HHParams; I_stim=0.0, I_base=0.0)
    ptest = deepcopy(model_pars)
    ptest.I_stim = I_stim
    ptest.I_base = I_base
    # Recompute steady-state for this current as initial condition
    u0_test = nlsolve(u -> hh_stat!(u, ptest), u0_guess, autodiff=:forward).zero
    prob = ODEProblem(hh!, u0_test, (0.0, 200.0), ptest)
    sol = solve(prob, Tsit5(), save_everystep=true, reltol=1e-6)
    m_vals = [u[3] for u in sol.u]
    return any(m -> m > 0.8, m_vals)
end

# Find minimal I_stim for a spike at a given temperature
function single_spike_boundary(model_pars::HHParams; temperature::Union{Nothing, Float64}=nothing,
    I_max::Float64=800.0, atol::Float64=0.1, iter_max::Int=1000, I_base::Float64=0.0)
    # Set temperature factor if given
    ptest = deepcopy(model_pars)
    if temperature !== nothing
        ptest.Tfac = temperature_factor(temperature)
    end
    ptest.I_base = I_base

    # Find bracketing interval
    I_lo, I_hi = 0.0, 1.0
    while !test_spike(ptest, I_stim=I_hi, I_base=I_base) && I_hi < I_max
        I_lo = I_hi
        I_hi = min(2 * I_hi, I_max)
    end
    if !test_spike(ptest, I_stim=I_hi, I_base=I_base)
        error("No spike detected up to I_max=$I_max")
    end

    # Bisection
    for it in 1:iter_max
        I_mid = (I_lo + I_hi) / 2
        if test_spike(ptest, I_stim=I_mid, I_base=I_base)
            I_hi = I_mid
        else
            I_lo = I_mid
        end
        if abs(I_hi - I_lo) <= atol
            return I_hi
        end
    end
    @warn "No convergence after $iter_max iterations"
    return I_hi
end

# Example: Compute single-spike threshold over temperature range
t_range = range(0., 30., 31)
single_spike_range = zeros(length(t_range))
for (i, Tcurr) in enumerate(t_range)
    single_spike_range[i] = single_spike_boundary(p; temperature=Tcurr, atol=1e-2, I_base=0.0)
end

plot(t_range, single_spike_range, xlabel="Temperature (°C)", ylabel="Threshold I_stim (μA/cm²)",
    label="Threshold for Spike", title="Single-Spike Threshold vs Temperature", size=(800, 600))

# Save threshold curve if needed
# npzwrite("outputs/single_spiking_range.npz", t_range=t_range, single_spike=single_spike_range)

# Compute derivatives (first and second) for analysis
N = length(t_vals)
du_log = [zeros(4) for _ in 1:N]
d2u_log = [zeros(4) for _ in 1:N]
for i in 1:N
    u = sol.u[i]
    du = similar(u)
    hh!(du, u, p, t_vals[i])
    du_log[i] = copy(du)
    J = ForwardDiff.jacobian(u -> begin d = zeros(4); hh!(d, u, p, t_vals[i]); d end, u)
    d2u_log[i] = J * du
end

du_mat = reduce(hcat, du_log)'
d2u_mat = reduce(hcat, d2u_log)'

# Plot derivatives
pV = plot(t_vals, du_mat[:, 1], label="dV/dt", xlabel="Time (ms)", ylabel="dV/dt",
    title="dV/dt and d²V/dt²", size=(800, 600))
plot!(pV, t_vals, d2u_mat[:, 1], label="d²V/dt²")
display(pV)

pgate = plot(layout=(2, 1), size=(800, 700), xlabel="Time (ms)", title="Gating Derivatives")
gating_names = ["n", "m", "h"]
for i in 2:4
    plot!(pgate[1], t_vals, du_mat[:, i], label="d$(gating_names[i-1])/dt")
    plot!(pgate[2], t_vals, d2u_mat[:, i], label="d²$(gating_names[i-1])/dt²")
end
ylabel!(pgate[1], "1st Derivative")
ylabel!(pgate[2], "2nd Derivative")
display(pgate)
