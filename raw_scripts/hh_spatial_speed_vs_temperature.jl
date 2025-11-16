# This script uses hh_pde.jl to solve the Hodgkin-Huxley partial differential equation system 
# under stimulation of a coil to analyze the speed of transmission over varying temperatures.
using Revise
include("../hh_pde.jl")
using .NeuronalSimulation.HHCable
using .NeuronalSimulation.Stimulations
using Plots
using Dates

using JSON3
using NPZ

time = Dates.format(now(), "yyyymmdd-HHMMSS")
parameters_file = "outputs/hh_sim_parameters_$time.json"

# Parameters
N = 401
L = 6.0
D = 2.0
gNa, gK, gL = 120.0, 36.0, 0.3
ENa, EK, EL = 50.0, -77.0, -54.4

V_rest = -65.0
temperature = 6.
C = 1.0

I_base = 0.

r_term = 200.0
T_end = 10.


speeds = []
temperatures = -40:5:0

for temperature in temperatures
    coil_center = 0.
    coil_distance = 1e-2
    coil_radius = 1e1
    I_coil = 30.
    R_coil = 3.

    coil_params = CoilParams(
        coil_center, # center
        coil_distance, # distance
        coil_radius, # radius
        I_coil, # coil current
        R_coil # coil resistance
    )

    I_ext = electrode_single(coil_params;env=t->t<T_end)

    p = HH_PDE_Params(gK, gNa, gL, EK, ENa, EL, V_rest, temperature_factor(temperature), C, I_base, I_ext, D, L, N, :robin, r_term)


    @info "Parameters set up"

    @time sol = run_hh_cable(p; tspan=(0, T_end))

    @info "Simulation finished"

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

        "x0" => coil_params.x0,
        "distance" => coil_params.distance,
        "radius" => coil_params.radius,
        "I_coil" => coil_params.I_coil,
        "R_coil" => coil_params.R_coil,
    )

    # Write to JSON
    open(parameters_file, "w") do io
        JSON3.write(io, params_dict)
    end

    #plot_sol_1D = plot_state_1D(sol, p, div(N, 10) * 2)
    #display(plot_sol_1D)
    #savefig(plot_sol, "outputs/hh_plots_1D_$time.pdf");

    #plot_sol_1D_2 = plot_state_1D(sol, p, div(N, 10) * 3)
    #display(plot_sol_1D_2)

    plot_sol = plot_state(sol, p; show=false, I_ext=I_ext)
    #display(plot_sol)
    savefig(plot_sol, "outputs/hh_plots_$time _T=$temperature .pdf");
    @info "Plotting finished"


    #Vmat, nmat, mmat, hmat, INa_mat, IK_mat, IL_mat = get_state_matrices(sol, p)

    """npzwrite("outputs/hh_simulation_data_$time.npz"; 
        V=Vmat, 
        n=nmat, 
        m=mmat, 
        h=hmat, 
        INa=INa_mat, 
        IK=IK_mat, 
        IL=IL_mat
    )"""


    # autocorrelate
    speed = compute_wave_speed(sol, p; x1=0.3, x2=0.7)

    @info "Speed is $speed [cm ms^-1]"
    push!(speeds, speed)
end

speed_plot = plot(temperatures, speeds)
savefig(speed_plot, "outputs/hh_speed_$time .pdf");
""";
xaxis="Temperature T [°C]",
yaxis="Signal speed c [cm * ms^-1]",
"Propagation speed of axon signal",
legend=true)"""

npzwrite("outputs/hh_simulation_temp_speed_$time.npz",
    temp=Vector{Float64}(temperatures),
    speed=Vector{Float64}(speeds)
    )