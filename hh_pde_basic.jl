module NeuronalSimulation

module Stimulations

export CoilParams,
       step_envelope, ramp_envelope, gaussian_envelope, sinusoidal_envelope,
       rectangular_spatial, gaussian_spatial,
       combine,
       rectangular_pulse, gaussian_pulse, sine_stim,
       coil_stim, electrode_single

# Coil-based stimulation parameters
x0_def = 0.
distance_def = 1e-3
radius_def = 1e-2
I_coil_def = -1.5e-3 
R_coil_def = 3.
struct CoilParams
    x0
    distance   # distance from coil to tissue center (cm)
    radius     # coil radius (cm)
    I_coil      # coil current amplitude (A)
    R_coil
end

CoilParams() = CoilParams(
    x0_def,
    distance_def,
    radius_def,
    I_coil_def,
    R_coil_def
)

# Temporal envelopes
step_envelope(t_start, t_end) = t -> (t >= t_start && t <= t_end) ? 1.0 : 0.0

function ramp_envelope(t_start, t_end) 
    function ramp(t)
        if (t < t_start) 
            return 0.0
        elseif (t > t_end)
            return 1.0
        else
            return (t - t_start)/(t_end - t_start)
        end
    end
    return ramp
end

gaussian_envelope(t0, sigma) = t ->
    exp(-0.5*((t - t0)/sigma)^2)
sinusoidal_envelope(freq, phase=0.0) = t ->
    sin(2π * freq * t + phase)

"""
    tms_coil_envelope(; 
        L=10e-6, R=0.05, C=100e-6,
        t0=0.0, duration=1e-3, 
        I0=1.0, phase=0.0, 
        output::Symbol=:current)

Return a function f(t) giving a TMS-like coil temporal profile.

- L,R,C: series RLC parameters (Henries, Ohms, Farads).
- t0: pulse onset (s).
- duration: window length after t0 (s).
- I0: amplitude scale (A for :current; arbitrary for :dIdt).
- phase: initial phase (rad) to shift monophasic/biphasic shape.
- output: :current for I(t), :dIdt for time-derivative (E-field-like).
- window: :hard uses rectangular window [t0, t0+duration], 
          :gauss uses Gaussian edge with sigma=duration/6.

If L,C imply overdamped case, falls back to single-exponential cosine with real frequency=0.
"""
function tms_coil_envelope(; 
    L=10e-6, R=0.05, C=100e-6,
    t0=0.0, duration=1., 
    I0=1.0, phase=0.0, 
    output::Symbol=:current)
    @warn "Unfinished"
    α = R/(2L)
    ω0_sq = 1/(L*C)
    Δ = ω0_sq - α^2
    underdamped = Δ > 0
    ωd = underdamped ? sqrt(Δ) : 0.0

    # Core waveform (unwindowed, causal from t0)
    I_core = underdamped ?     
    (t -> I0 * exp(-α*(t - t0)) * sin(ωd*(t - t0) + phase)) :
        (t -> I0 * exp(-α*(t - t0)) * sin(phase))  # degenerate fallback

    # Time derivative (for E-field-like profile)
    dI_core = underdamped ?
        (t -> I0 * exp(-α*(t - t0)) * (ωd*cos(ωd*(t - t0) + phase) - α*sin(ωd*(t - t0) + phase))) :
        (t -> -α * I0 * exp(-α*(t - t0)) * sin(phase))

    # window [t0, t0 + duration] from beginning until end of discharge
    win = t -> (t >= t0 && t <= t0 + duration) ? 1.0 : 0.0
    if output === :current
        return t -> (t < t0 ? 0.0 : I_core(t)) * win(t)
    elseif output === :dIdt
        return t -> (t < t0 ? 0.0 : dI_core(t)) * win(t)
    else
        error("output must be :current or :dIdt")
    end
end

# Spatial profiles
rectangular_spatial(x0, width) = x -> abs(x - x0) <= width/2 ? 1.0 : 0.0
gaussian_spatial(x0, sigma) = x -> exp(-0.5*((x - x0)/sigma)^2)

# combine spatial and temporal profile
combine(spatial_fn::Function, temporal_fn::Function, amp=1.0) = (x,t) -> amp * spatial_fn(x) * temporal_fn(t)

"""
Combines multiple stimulation profiles by summing them up.
function_array: stimulation functions of the type (x, t)
"""
function combine(function_array)
    fvec = collect(function_array)
    return (x, t) -> begin
        s = 0.0
        @inbounds @simd for f in fvec
            s += f(x, t)
        end
        s
    end
end
# Predefined stimulation functions


"""
Rectangular pulse around mid x0 with width width from t_start to t_end with amplitude amp.
"""
rectangular_pulse(;x0=0., width=1.,
                  t_start=0., t_end=1.,
                  amp=50.) = combine(
    rectangular_spatial(x0, width),
    step_envelope(t_start, t_end),
    amp
)

gaussian_pulse(x0, sigma_x,
               t0, sigma_t,
               amp) = combine(
    gaussian_spatial(x0, sigma_x),
    gaussian_envelope(t0, sigma_t),
    amp
)

sine_stim(x0, width,
          t_start,
          freq, amp) = combine(
    rectangular_spatial(x0, width),
    sinusoidal_envelope(freq, -2π*freq*t_start),
    amp
)

# Coil-induced stimulation using a dipole approximation
function coil_stim(p::CoilParams; env::Function=t->1.)
    spatial = x -> p.I_amp * 4π * 1e-7 * p.radius^2 / (2 * (p.radius^2 + ((x- p.x0) - p.distance)^2)^(3/2))
    return (x,t) -> spatial(x) * env(t)
end

#
function electrode_single(p::CoilParams; env::Function=t->1.)
    spatial = x -> p.R_coil * p.I_coil / (4 * π * sqrt(p.distance^2 + (x - p.x0)^2))
    return (x,t) -> spatial(x) * env(t)
end

end

module HHCable

using DifferentialEquations
using LinearAlgebra
using SparseArrays
using NonlinearSolve
using ForwardDiff

using PyPlot
using Dates

using StatsBase
using LaTeXStrings
using ..Stimulations
# using GLMakie

export HHParams, HH_PDE_Params, temperature_factor,
       laplacian_matrix, hh_cable!, hh_stat, find_fixpoint,
       initialize, run_hh_cable, compute_wave_speed, get_state_matrices, plot_state, plot_state_1D, ProgressLogging, save_plot, is_spike

############# Parameter Structs #############

mutable struct HHParams
    gK
    gNa
    gL
    EK
    ENa
    EL
    Tfac
    C
    I_ext
end

N_def = 201
L_def = 10.0
D_def = 1.0
gNa_def = 120.
gK_def = 36.
gL_def = 0.3
ENa_def = 50.0
EK_def = -77.0
EL_def = -54.4
C_def = 1.0
V_rest_def = -65.0
temperature_def = 10.0
I_base_def = 0.0

r_term_def = 1000.0
T_end_def = 20.

I_ext_def = rectangular_pulse(x0=L_def/2, width=L_def/10, t_start=0., t_end=T_end_def, amp=200.)

mutable struct HH_PDE_Params
    gK
    gNa
    gL
    EK
    ENa
    EL
    V_rest
    Tfac
    C
    I_base
    I_ext::Function
    D
    L
    N::Int64
    bctype::Symbol
    r_term
end

HH_PDE_Params() = HH_PDE_Params(
    gK_def,
    gNa_def,
    gL_def,
    EK_def,
    ENa_def,
    EL_def,
    V_rest_def,
    temperature_factor(temperature_def),
    C_def,
    I_base_def,
    I_ext_def,
    D_def,
    L_def,
    N_def,
    :neumann,
    r_term_def
)

temperature_factor(t) = 3^((t - 6.3) / 10)

function get_x_range(p::HH_PDE_Params)
    """
    Creates a uniform grid along x axis of length L with N nodes.
    """
    return range(0, p.L, length=p.N)    
end

############# Gating Dynamics #############

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

############# Laplacian Assembly #############

"""
    laplacian_matrix(N, dx; bctype=:robin, resistance=1e3)

Return N×N sparse Laplacian matrix with boundary condition:
- :dirichlet — fixed V at ends
- :neumann   — zero-flux at ends
- :robin     — termination resistance (Robin) at ends
"""
function laplacian_matrix(N, dx; bctype=:robin, resistance=1e3)
    main = -2 * ones(N)
    off  = ones(N-1)
    A = spdiagm(-1 => off, 0 => main, 1 => off)
    if bctype == :dirichlet
        A[1,:] .= 0;  A[end,:] .= 0
    elseif bctype == :robin
        A[1,1] = -2 + dx/resistance;   A[1,2] = 2
        A[N,N] = -2 + dx/resistance;   A[N,N-1] = 2
    elseif bctype == :neumann
        A[1,1] = -1; A[N,N] = -1
    else
        error("Unknown boundary type $bctype")
    end
    return A / dx^2
end

############# HH Cable PDE Vector Field #############

function hh_cable!(du, u, p::HH_PDE_Params, t)
    Δ = laplacian_matrix(p.N, p.L/(p.N-1); bctype=p.bctype, resistance=p.r_term)
    V = @view u[1:p.N]
    n = @view u[p.N+1:2p.N]
    m = @view u[2p.N+1:3p.N]
    h = @view u[3p.N+1:4p.N]
    dV = @view du[1:p.N]
    dn = @view du[p.N+1:2p.N]
    dm = @view du[2p.N+1:3p.N]
    dh = @view du[3p.N+1:4p.N]
    diffV = p.D * (Δ * V)

    I_ext = zeros(p.N)
    xrange = get_x_range(p)
    for i in 1:p.N
        I_ext[i] = p.I_ext(xrange[i], t)
    end

    for i in 1:p.N
        I_Na = p.gNa * m[i]^3 * h[i] * (V[i] - p.ENa)
        I_K  = p.gK * n[i]^4 * (V[i] - p.EK)
        I_L  = p.gL * (V[i] - p.EL)
        dV[i] = (diffV[i] - I_Na - I_K - I_L + I_ext[i]) / p.C
        dn[i] = p.Tfac * (alpha_n(V[i]) * (1.0 - n[i]) - beta_n(V[i]) * n[i])
        dm[i] = p.Tfac * (alpha_m(V[i]) * (1.0 - m[i]) - beta_m(V[i]) * m[i])
        dh[i] = p.Tfac * (alpha_h(V[i]) * (1.0 - h[i]) - beta_h(V[i]) * h[i])
    end
    nothing
end

############# Steady State (Fixed Point) #############

function hh_stat(u::AbstractVector{T}, p::HH_PDE_Params) where T
    v, n, m, h = u
    dv = (-(p.gK * n^4 * (v - p.EK))
          - (p.gNa * m^3 * h * (v - p.ENa))
          - (p.gL * (v - p.EL))
          + p.I_base) / p.C
    dn = p.Tfac * (alpha_n(v) * (1.0 - n) - beta_n(v) * n)
    dm = p.Tfac * (alpha_m(v) * (1.0 - m) - beta_m(v) * m)
    dh = p.Tfac * (alpha_h(v) * (1.0 - h) - beta_h(v) * h)
    [dv, dn, dm, dh]
end

function find_fixpoint(p::HH_PDE_Params)
    V_rest = p.V_rest
    u0_guess = [V_rest, n_inf(V_rest), m_inf(V_rest), h_inf(V_rest)]
    prob = NonlinearProblem(hh_stat, u0_guess, p)
    sol_stat = solve(prob)
    return sol_stat.u
end

function initialize(p::HH_PDE_Params)
    u0_id = find_fixpoint(p)
    V0 = u0_id[1] * ones(p.N)
    n0 = u0_id[2] * ones(p.N)
    m0 = u0_id[3] * ones(p.N)
    h0 = u0_id[4] * ones(p.N)
    return vcat(V0, n0, m0, h0)
end

############# Solver and Postprocessing #############

# Callback that stops simulation when membrane potential crosses threshold
function spike_termination_callback(p; threshold::Float64 = 0.0)
    # Condition: max(V) - threshold crosses zero upward
    g(u, t, integrator) = maximum(@view u[1:p.N]) - threshold
    # Effect: terminate integration
    function affect!(integrator)
        @info "Spike detected. Stopping simulation"
        terminate!(integrator)
    end
    ContinuousCallback(g, affect!; rootfind=true, direction=+1)  # upward crossing only
end

function run_hh_cable(p::HH_PDE_Params; tspan=(0.0, 10.0), saveat=0.01, solver=TRBDF2(), threshold=0.0, halt_on_spike=false)
    u0 = initialize(p)
    prob = ODEProblem(hh_cable!, u0, tspan, p)
    
    if halt_on_spike
        cb_spike = spike_termination_callback(p; threshold=threshold)
        # safety callback to stop at tspan[end] without spike (no-op affect)
        cbs = CallbackSet(cb_spike)
        return solve(prob, solver; saveat=saveat, abstol=1e-3, reltol=1e-3,
                 progress=true, adaptive=true, callback=cbs)
    else
        return solve(prob, solver; saveat=saveat, abstol=1e-3, reltol=1e-3,
                 progress=true, adaptive=true)
    end
end


function get_state_matrices(sol, p::HH_PDE_Params)
    Nt = length(sol.t)
    N = p.N
    Vmat = zeros(Nt, N)
    nmat = zeros(Nt, N)
    mmat = zeros(Nt, N)
    hmat = zeros(Nt, N)
    INa_mat = zeros(Nt, N)
    IK_mat  = zeros(Nt, N)
    IL_mat  = zeros(Nt, N)
    for i in 1:Nt
        u = sol[i]
        V = @view u[1:N]
        n = @view u[N+1:2N]
        m = @view u[2N+1:3N]
        h = @view u[3N+1:4N]
        Vmat[i, :] = V
        nmat[i, :] = n
        mmat[i, :] = m
        hmat[i, :] = h
        @inbounds for j in 1:N
            INa_mat[i,j] = p.gNa * m[j]^3 * h[j] * (V[j] - p.ENa)
            IK_mat[i,j]  = p.gK  * n[j]^4 * (V[j] - p.EK)
            IL_mat[i,j]  = p.gL  * (V[j] - p.EL)
        end
    end
    return Vmat, nmat, mmat, hmat, INa_mat, IK_mat, IL_mat
end

function is_spike(sol, p; x_percentile = -1.)
    Vmat, nmat, mmat, hmat, _, _, _ = get_state_matrices(sol, p)
    spike = false
    v_threshold = -10. # a spike can be expected at this membrane potential, through phase plane analysis.
    if !(0. <= x_percentile <= 1.)
        if (maximum(Vmat) > v_threshold)
            spike = true
        end
    else
        i = (p.N * x_percentile)
        if (maximum(Vmat[:, i]) > v_threshold)
            spike = true
        end
    end
    return spike
end



    


function compute_wave_speed(sol, p; plot=false, x1=0.2, x2=0.8)
    """
    Computes the travelling wave speed by taking the profile at x=0.4L, x=0.6L and calculating the delay via autocorrelation. 
    Does work ONLY if the wave does not start between the two measuring points, for obvious reasons.
    """
    Vmat, nmat, mmat, hmat, INa_mat, IK_mat, IL_mat = get_state_matrices(sol, p)

    if !(maximum(Vmat) > -20.)
        @info "no spike, no membrane potential above -20 mV, maximum membrane potential is" maximum(Vmat)
        if plot
            return 0., 0.
        else
            return 0.
        end
    end

    N = p.N
    L = p.L
    dx = L / (N - 1)

    # Spatial indices for x1 and x2 position
    x1_idx = round(Int, x1 * (N - 1)) + 1
    x2_idx = round(Int, x2 * (N - 1)) + 1
    Δx = abs((x2_idx - x1_idx) * dx)

    V1 = Vmat[:, x1_idx]
    V2 = Vmat[:, x2_idx]

    #I1 = INa_mat[:, x1_idx]
    #I2 = INa_mat[:, x2_idx]
    
    # constant step size dt
    dt = sol.t[2] - sol.t[1]

    # Compute normalized cross-correlation
    delays = -(length(sol.t)-1):(length(sol.t)-1)
    crosscor_v = crosscor(V1, V2, delays)
    #crosscor_i = crosscor(I1, I2, delays)

    # max correlation
    max_idx_v = argmax(crosscor_v)
    #max_idx_i = argmax(crosscor_i)

    # most likely delay between the two peaks 
    delay_v = delays[max_idx_v] 
    #delay_i = delays[max_idx_i]

    τ_v = delay_v * dt               # delay in milliseconds
    #τ_i = delay_i * dt               # delay in milliseconds

    # Compute wave speed (distance [cm] / Time t [ms])
    speed_v = Δx / abs(τ_v)
    #speed_i = Δx / abs(τ_i)

    # less than 20 % difference
    """
    if abs((speed_v - speed_i)/speed_i) < 0.1
        speed = (speed_i + speed_v) / 2
        τ = (τ_v + τ_i) / 2
    else 
        speed = 0
        τ = 0
        @info "no wave detected" speed_v speed_i
    end
    """
    speed = speed_v

    if plot
        time_delays = (-(length(V1)-1):(length(V1)-1)) .* dt

        plt = plot(time_delays, crosscor_v,
            xlabel = "delay [ms]",
            ylabel = "Cross-correlation",
            label = "Cross-correlation of membrane potential",
            title = "Cross-correlation between x=20% vs x=40% \n delay = $(round(τ, digits=4)) ms, speed ≈ $(round(speed, digits=3)) cm/ms",
            legend =:topleft,
            size=(800, 600)
    )
        plot!(plt, time_delays, crosscor_i, label="Cross-correlation of potassium current")
        return speed, plt
    else 
        return speed
    end
end

"""
membrane potential is compared against the threshold
"""
function has_spike(sol, p::HH_PDE_Params; threshold=0.)
    Vmat, nmat, mmat, hmat, INa_mat, IK_mat, IL_mat = get_state_matrices(sol, p)
    if any(Vmat) > threshold
        return true
    else
        return false
    end
end

function plot_state_1D(sol, p::HH_PDE_Params, xidx::Int)
    # Extract data
    Vmat, nmat, mmat, hmat, INa_mat, IK_mat, IL_mat = get_state_matrices(sol, p)
    tvec = Array(sol.t)

    V  = Vmat[:, xidx]
    n  = nmat[:, xidx]
    m  = mmat[:, xidx]
    h  = hmat[:, xidx]
    INa = INa_mat[:, xidx]
    IK  = IK_mat[:, xidx]
    IL  = IL_mat[:, xidx]
    Iion = INa .+ IK .+ IL

    fig, axs = subplots(3, 1, figsize=(7.5, 8.5), sharex=true)

    # V trace
    axs[1].plot(tvec, V, color="C0", lw=1.5)
    axs[1].set_ylabel("V (mV)")
    axs[1].set_title("V(t) at x[$xidx]")
    axs[1].grid(alpha=0.3)

    # gating
    axs[2].plot(tvec, n, label="n", color="C1", lw=1.2)
    axs[2].plot(tvec, m, label="m", color="C2", lw=1.2)
    axs[2].plot(tvec, h, label="h", color="C3", lw=1.2)
    axs[2].set_ylabel("Gates")
    axs[2].set_title("n, m, h at x[$xidx]")
    axs[2].grid(alpha=0.3)
    axs[2].legend(frameon=false, fontsize=9)

    # currents
    axs[3].plot(tvec, INa, label=L"I_{Na}", color="C4", lw=1.2)
    axs[3].plot(tvec, IK,  label=L"I_{K}",  color="C5", lw=1.2)
    axs[3].plot(tvec, IL,  label=L"I_{L}",  color="C6", lw=1.2)
    axs[3].plot(tvec, Iion, label=L"I_{ion}", color="k", lw=1.0, ls="--")
    axs[3].set_xlabel("Time [ms]")
    axs[3].set_ylabel("Current")
    axs[3].set_title("Ionic currents at x[$xidx]")
    axs[3].grid(alpha=0.3)
    axs[3].legend(frameon=false, fontsize=9)

    fig.tight_layout()
    return fig
end


function plot_state(sol, p::HH_PDE_Params; show=true, save=false, backend::Symbol=:pyplot, I_ext=nothing)
    x = collect(get_x_range(p))
    t = Array(sol.t)
    Vmat, nmat, mmat, hmat, INa_mat, IK_mat, IL_mat = get_state_matrices(sol, p)

    titles = ["V [mV]", "n", "m", "h", L"I_{Na}", L"I_{K}", L"I_{L}"]
    data   = [Vmat,      nmat, mmat, hmat, INa_mat,  IK_mat,  IL_mat]

    # set clims for gates; others auto
    clims_map = Dict("n" => (0.0, 1.0), "m" => (0.0, 1.0), "h" => (0.0, 1.0))

    nrows, ncols = 4, 2
    fig, axs = subplots(nrows, ncols, figsize=(10, 10), sharex=true, sharey=true)
    axs = reshape(axs, :,)  # vectorize

    for (i, (ttl, M)) in enumerate(zip(titles, data))
        ax = axs[i]
        im = ax.imshow(M; origin="lower",
                       extent=(minimum(x), maximum(x), minimum(t), maximum(t)),
                       aspect="auto", cmap="viridis",
                       vmin=get(clims_map, ttl, nothing) === nothing ? nothing : clims_map[ttl][1],
                       vmax=get(clims_map, ttl, nothing) === nothing ? nothing : clims_map[ttl][2])
        ax.set_title(string(ttl))
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("t [ms]")
        cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
    end

    # optional I_ext panel
    if I_ext isa Function
        I_mat = [I_ext(xi, ti) for ti in t, xi in x]
        ax = axs[length(titles)+1]
        im = ax.imshow(I_mat; origin="lower",
                       extent=(minimum(x), maximum(x), minimum(t), maximum(t)),
                       aspect="auto", cmap="coolwarm")
        ax.set_title("I_ext")
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("t [ms]")
        fig.colorbar(im, ax=ax, pad=0.01, fraction=0.046)
    end

    fig.tight_layout()
    if save
        fig.savefig("hh_plot_pyplot.pdf", bbox_inches="tight")
    end
    if show
        display(fig)
    end
    return fig
end


minspan_clims(mat; min_span=1.0, symmetric=false) = begin
    mn = minimum(mat); mx = maximum(mat)
    if symmetric
        M = max(abs(mn), abs(mx))
        M = max(M, min_span/2)
        return (-M, M)
    else
        span = mx - mn
        if span < min_span
            mid = (mx + mn)/2
            half = min_span/2
            return (mid - half, mid + half)
        else
            return (mn, mx)
        end
    end
end


function save_plot(sol, p::HH_PDE_Params;
                   outdir::AbstractString="plots",
                   backend::Symbol=:pyplot,
                   I_ext=nothing,
                   cmap="viridis",
                   dpi=300,
                   min_span_currents=1.0,
                   symmetric_currents=false)

    isdir(outdir) || mkpath(outdir)

    x = collect(get_x_range(p))
    t = Array(sol.t)
    Vmat, nmat, mmat, hmat, INa_mat, IK_mat, IL_mat = get_state_matrices(sol, p)

    clims_nmh = (0.0, 1.0)
    clims_INa = minspan_clims(INa_mat; min_span=min_span_currents, symmetric=symmetric_currents)
    clims_IK  = minspan_clims(IK_mat;  min_span=min_span_currents, symmetric=symmetric_currents)
    clims_IL  = minspan_clims(IL_mat;  min_span=min_span_currents, symmetric=symmetric_currents)

    function save_heatmap_pdf(fname, X, Y, Z; title="", cbar="", clims=nothing,
                              xlabel="Position x [cm]", ylabel="Time t [ms]", cmap="viridis", dpi=300)
        fig, ax = subplots(figsize=(6, 5))
        im = ax.imshow(Z; origin="lower",
                       extent=(minimum(X), maximum(X), minimum(Y), maximum(Y)),
                       aspect="auto", cmap=cmap,
                       vmin = clims === nothing ? nothing : clims[1],
                       vmax = clims === nothing ? nothing : clims[2])
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
        cb = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02)
        cb.set_label(cbar)
        fig.savefig(joinpath(outdir, fname), bbox_inches="tight", dpi=dpi)
        close(fig)
        return nothing
    end

    save_heatmap_pdf("V_membrane.pdf", x, t, Vmat; title="Membrane potential V [mV]", cbar="V [mV]", cmap=cmap)
    save_heatmap_pdf("n_gate.pdf", x, t, nmat; title="Potassium gate n", cbar="n", clims=clims_nmh, cmap=cmap)
    save_heatmap_pdf("m_gate.pdf", x, t, mmat; title="Sodium gate m", cbar="m", clims=clims_nmh, cmap=cmap)
    save_heatmap_pdf("h_gate.pdf", x, t, hmat; title="Sodium gate h", cbar="h", clims=clims_nmh, cmap=cmap)
    save_heatmap_pdf("I_Na.pdf", x, t, INa_mat; title="Sodium current " * L"I_{Na}", cbar=L"I_{Na}", clims=clims_INa, cmap=cmap)
    save_heatmap_pdf("I_K.pdf",  x, t, IK_mat;  title="Potassium current " * L"I_{K}", cbar=L"I_{K}", clims=clims_IK, cmap=cmap)
    save_heatmap_pdf("I_L.pdf",  x, t, IL_mat;  title="Leak current "* L"I_{L}", cbar=L"I_{L}", clims=clims_IL, cmap=cmap)

    if I_ext isa Function
        I_mat = [I_ext(xi, ti) for ti in t, xi in x]
        cl_Iext = minspan_clims(I_mat; min_span=min_span_currents, symmetric=symmetric_currents)
        save_heatmap_pdf("I_ext.pdf", x, t, I_mat; title="External current " * L"I_{ext}", cbar= L"I_{ext}",
                         clims=cl_Iext, cmap="coolwarm", dpi=dpi)
    end

    return nothing
end

end 

end
