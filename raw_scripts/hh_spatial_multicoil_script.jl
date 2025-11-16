# This script uses hh_pde.jl to solve the Hodgkin-Huxley partial differential equation system 
# under stimulation of multiple coils.

using Revise
include("../hh_pde.jl")
using .NeuronalSimulation.HHCable
using .NeuronalSimulation.Stimulations
using Dates
using PyPlot

using JSON3
using NPZ

time = Dates.format(now(), "yyyymmdd-HHMMSS")
folder = "outputs/$time"
mkpath(folder)

parameters_file = "$folder/hh_sim_parameters.json"

# ---------------------------
# Hodgkin–Huxley cable setup AI GENERATED COMMENTS
# ---------------------------

# Geometry and discretization
N = 501          # Number of spatial grid points along the cable; higher N = finer spatial resolution 
L = 10.0         # Cable length (arbitrary units used consistently across the code) 
D = 2.0          # Axial coupling (diffusion) coefficient controlling how voltage spreads in space 

# Maximal conductances (mS/cm^2) and reversal potentials (mV) for HH channels
gNa, gK, gL = 120.0, 36.0, 0.3   # Sodium, potassium, and leak maximal conductances 
ENa, EK, EL = 50.0, -77.0, -54.4 # Reversal potentials for Na+, K+, and leak current 

# Resting state, temperature, and membrane capacitance
V_rest = -65.0     # Reference resting potential (mV) used by the HH formulation 
temperature = 27 # Simulation temperature (°C) used to scale gating rates via a Q10-like factor 
C = 1.0            # Membrane capacitance per unit area (µF/cm^2), standard HH value 

# Baseline external current (applied everywhere)
I_base = 0.0       # No DC bias current; all stimulation comes from coils below 

# Optional termination parameter (e.g., end resistance) for record-keeping
r_term = 200.0     # Stored for provenance; not passed into HH_PDE_Params in this snippet 

# Simulation time window
T_end = 5.0        # End time of the simulation (same time units used by solver) 

# Duration during which the coils are active
T_stim = 1.0 * T_end       # Coils apply current only for t < T_stim, then turn off 

# ---------------------------
# Coil definitions (stimuli)
# ---------------------------

# A grid search may be required to find the right point where blocking is possible

coil_center = 0.0     # Not used directly in these two-coil definitions; placeholder for single-coil setups 
coil_distance = 0.2  # Distance from coil to cable (affects induced current strength/shape) 
coil_radius = 1e1     # Coil radius (affects spatial spread of induced field/current) 
I_coil = 4000.0       # Coil drive current amplitude 

I_coil_block = -300.0 
R_coil = 3.0          # Coil series resistance (used by stimulus model, if applicable) 

# Pack parameters for the left coil centered at 20% of the cable
coil_params_left = CoilParams(
    L * 0.2,        # center position along the cable (x = 0.2*L) 
    coil_distance,  # vertical/horizontal distance to cable 
    coil_radius,    # coil size 
    I_coil,         # current amplitude 
    R_coil          # coil resistance 
)

# Pack parameters for the right coil centered at 80% of the cable
coil_params_right = CoilParams(
    L * 0.8,        # center position along the cable (x = 0.8*L) 
    coil_distance,  # distance to cable 
    coil_radius,    # coil size 
    I_coil_block,         # current amplitude 
    R_coil          # coil resistance 
)

# Build time-gated coil current sources: active only while t < T_stim
coil_left = electrode_single(coil_params_left; env = t -> t < T_stim)   # Left coil stimulus function I_left(x,t) 
coil_right = electrode_single(coil_params_right; env = t -> t < T_stim) # Right coil stimulus function I_right(x,t) 

# Save coil parameter objects for provenance and later reuse
coil_data = [coil_params_left, coil_params_right]  # For JSON export to reconstruct stimuli 

# Combine the two coil sources into one external current density I_ext(x,t)
I_ext = combine([coil_left, coil_right])           # Linear superposition of the two coil currents 

# ---------------------------
# Assemble PDE parameters
# ---------------------------

p = HH_PDE_Params(
    gK, gNa, gL,                # Channel maximal conductances 
    EK, ENa, EL,                # Reversal potentials 
    V_rest,                     # Reference resting potential 
    temperature_factor(temperature), # Kinetic scaling factor from temperature (Q10-like) 
    C,                          # Membrane capacitance 
    I_base,                     # Baseline (DC) current 
    I_ext,                      # External space- and time-dependent current from coils 
    D,                          # Axial coupling (diffusion) coefficient 
    L,                          # Cable length 
    N,                          # Number of spatial grid points 
    :neumann,                   # Boundary condition: sealed ends (zero-flux) 
    r_term                      # Termination parameter stored here or used by downstream routines 
)

@info "Parameters set up"       # Log message for progress tracking 

# ---------------------------
# Run simulation
# ---------------------------

# Integrate the HH cable PDE from t=0 to T_end, saving snapshots every 0.005 time units
@time sol = run_hh_cable(p; tspan = (0, T_end), saveat = 5e-3)  # Returns time/space solution object 

@info "Simulation finished"     # Log message after solver completes 

# ---------------------------
# Save parameters to JSON
# ---------------------------

params_dict = Dict(
    "gK" => gK,
    "gNa" => gNa,
    "gL" => gL,
    "EK" => EK,
    "ENa" => ENa,
    "EL" => EL,
    "V_rest" => V_rest,
    "temperature_factor" => temperature_factor(temperature),
    "C" => C,
    "I_base" => I_base,
    "D" => D,
    "L" => L,
    "N" => N,
    "bctype" => p.bctype,
    "r_term" => r_term,
    "coil_data" => coil_data    # Include coil configs so stimuli are reproducible 
)

# Write parameter dictionary to a JSON file for reproducibility
open(parameters_file, "w") do io
    JSON3.write(io, params_dict)  # Persist all key simulation settings 
end

# ---------------------------
# Plotting and visualization
# ---------------------------

plot_percentile = 0.3   # Choose spatial index near 30% along the cable for 1D trace plotting 

# Convert percentile into integer grid index safely: div(N,100)*Int(30) ≈ N*0.3 with integer arithmetic
plot_sol_1D = plot_state_1D(sol, p, div(N, 100) * Int(plot_percentile * 100))
display(plot_sol_1D)     # Show the plot interactively (voltage vs time at selected location) 
savefig("$folder/hh_plots_x=$(plot_percentile)_$(time).pdf");  # Save figure to disk 

# Save standard plots (e.g., space–time maps, stimuli) to the output directory
save_plot(sol, p; outdir = folder, I_ext = I_ext)  # Generates multiple summary figures 

@info "Plotting finished"       # Log message after plotting 

# ---------------------------
# Extract and save state data
# ---------------------------

# Extract full time×space matrices for voltage, gates, and channel currents
Vmat, nmat, mmat, hmat, INa_mat, IK_mat, IL_mat = get_state_matrices(sol, p)  # Dense arrays for offline analysis 

#has_spike = is_spike(sol, p)
#@info has_spike


# NPZ NUMPY EXPORT
npzwrite("$folder/hh_simulation_data_$time.npz";
    V = Vmat,     # Membrane voltage over time and space 
    n = nmat,     # Potassium activation gate n 
    m = mmat,     # Sodium activation gate m 
    h = hmat,     # Sodium inactivation gate h 
    INa = INa_mat,# Sodium current density 
    IK = IK_mat,  # Potassium current density 
    IL = IL_mat   # Leak current density 
)

# ---------------------------
# Optional: wave speed (disabled)
# ---------------------------

# Compute approximate conduction velocity by tracking threshold crossing times across positions
# speed = compute_wave_speed(sol, p)  # Typical approach: fit slope of x vs t for V crossing (e.g., -20 mV) 

# @info "Speed is $speed [cm ms^-1]"  # Units depend on L and time units in the model 