# This script uses hh_pde.jl to solve the Hodgkin-Huxley partial differential equation system 
# under stimulation of a coil, detecting the minimum threshold of required stimulation to 
# generate a propagating spike (mathematically: a travelling wave in the solution of the 
# respective reaction-diffusion-system)
using Revise
include("../hh_pde.jl")
using .NeuronalSimulation.HHCable
using .NeuronalSimulation.Stimulations

using Dates
using JSON3
using NPZ

time = Dates.format(now(), "yyyymmdd-HHMMSS")
folder = "outputs/$time"
mkpath(folder)

parameters_file = "$folder/hh_sim_parameters.json"

# Parameters
N = 201
L = 5.0
D = 2.0
gNa, gK, gL = 120.0, 36.0, 0.3
ENa, EK, EL = 50.0, -77.0, -54.4
V_rest = -65.0
temperature = 6.3
C = 1.0
I_base = 0.0
r_term = 200.0
T_end = 20.0

# Define coil parameters
coil_center = 2.5
coil_distance = 1e-2
coil_radius = 1e1
R_coil = 3.0


"""
    triggers_spike(I_coil; temperature=6.3, T_end=20., electrode=electrode_single)

Test if the fiber triggers a spike under given coil current and temperature.
Returns true if spike detected.
"""
function triggers_spike(I_coil; temperature=6.3, T_end=20., electrode=electrode_single)
    coil_params = CoilParams(
        coil_center,
        coil_distance,
        coil_radius,
        I_coil,
        R_coil
    )

    I_ext = electrode(coil_params; env = t -> t < T_end)

    p = HH_PDE_Params(gK, gNa, gL, EK, ENa, EL, V_rest, temperature_factor(temperature),
                      C, I_base, I_ext, D, L, N, :robin, r_term)

    sol = run_hh_cable(p; tspan=(0, T_end))

    return is_spike(sol, p; x_percentile=25)
end

"""
    find_threshold_Icoil(temperature; I_lo=0.0, I_hi=500.0, atol=1e-3, maxiter=40)

Find minimum coil current required to trigger spike using bisection method.
"""
function find_threshold_Icoil(temperature; I_lo=0.0, I_hi=500.0, atol=1e-3, maxiter=40)
    # Ensure lower bound doesn't spike
    if triggers_spike(I_lo; temperature=temperature)
        while I_lo > 0 && triggers_spike(I_lo; temperature=temperature)
            I_lo = max(0.0, 0.5 * I_lo)
        end
    end
    
    # Ensure upper bound does spike
    while !triggers_spike(I_hi; temperature=temperature)
        I_hi *= 2.0
        if I_hi > 1e6
            error("Failed to bracket threshold: no spike up to I_coil=$(I_hi).")
        end
    end

    # Bisection loop
    iter = 0
    while (I_hi - I_lo) > atol && iter < maxiter
        Imid = 0.5 * (I_lo + I_hi)
        @info "Now trying I= $Imid"
        if triggers_spike(Imid; temperature=temperature)
            I_hi = Imid   # still spikes; lower the upper bound
        else
            I_lo = Imid   # no spike; raise the lower bound
        end
        iter += 1
    end

    return I_hi  # minimal I that spikes within tolerance
end

# Main loop over temperatures
temperatures = 0:30
thresholds = Float64[]
speeds = Float64[]

for temperature in temperatures
    time_temp = Dates.format(now(), "yyyymmdd-HHMMSS")
    folder_temp = "outputs/$time/$time_temp"
    mkpath(folder_temp)

    parameters_file = "$folder_temp/hh_sim_parameters.json"
    @info "Searching I_coil threshold at T=$(temperature)..."
    I_thr = find_threshold_Icoil(temperature; I_lo=0.0, I_hi=30.0, atol=1e-2, maxiter=20)
    push!(thresholds, I_thr)

    @info "Threshold current at T=$(temperature) is $(I_thr), computing wave speed..."
    
    # Run simulation at threshold to compute wave speed
    coil_params = CoilParams(coil_center, coil_distance, coil_radius, I_thr*1.02, R_coil)
    I_ext = electrode_single(coil_params; env = t -> t < T_end)
    p = HH_PDE_Params(gK, gNa, gL, EK, ENa, EL, V_rest, temperature_factor(temperature),
                      C, I_base, I_ext, D, L, N, :robin, r_term)
    sol = run_hh_cable(p; tspan=(0, T_end), halt_on_spike=false)
    
    # Compute wave speed
    wave_speed = compute_wave_speed(sol, p; plot=false, x1=0.2, x2=0.4)
    push!(speeds, wave_speed)

    # Save parameters at this temperature
    params_dict = Dict(
        "gK" => gK, "gNa" => gNa, "gL" => gL, 
        "EK" => EK, "ENa" => ENa, "EL" => EL, 
        "V_rest" => V_rest,
        "temperature" => temperature,
        "temperature_factor" => temperature_factor(temperature),
        "C" => C, "I_base" => I_base, 
        "D" => D, "L" => L, "N" => N, 
        "bctype" => :robin, "r_term" => r_term,
        "coil_center" => coil_center, 
        "coil_distance" => coil_distance, 
        "coil_radius" => coil_radius,
        "R_coil" => R_coil, 
        "I_coil_threshold" => I_thr, 
        "T_end" => T_end,
        "wave_speed" => wave_speed
    )
    save_plot(sol, p; outdir = folder_temp, I_ext = I_ext)
    
    open(parameters_file, "w") do io
        JSON3.write(io, params_dict)
    end

    @info "Wave speed at T=$(temperature): $(wave_speed) cm/ms"
end

# Create speed plot
"""speed_plot = plot(temperatures, speeds,
    xlabel="Temperature T [°C]",
    ylabel="Signal speed c [cm/ms]",
    title="Propagation speed of axon signal",
    legend=false,
    marker=:circle,
    linewidth=2,
    markersize=4)"""

#savefig(speed_plot, "outputs/hh_speed_$time.pdf")

# Save data to NPZ file
npzwrite("outputs/hh_simulation_temp_speed_$time.npz",
    Dict("temp" => Vector{Float64}(temperatures),
         "speed" => Vector{Float64}(speeds),
         "thresholds" => Vector{Float64}(thresholds)))

@info "Simulation completed successfully!"
