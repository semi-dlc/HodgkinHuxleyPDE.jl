# Hodgkin-Huxley simulator
A range of scripts and Pluto notebooks based on the file hh_pde.jl and DifferentialEquations.jl to demonstrate the behavior of the Hodgkin-Huxley PDE and ODE, including finding its fixed points, spiking frequency, and stimulation thresholds.

The module implemented in hh_pde.jl solves the spatially extended Hodgkin-Huxley partial differential equation in the form of a reaction-diffusion system via finite differences.
The "classical" Hodgkin-Huxley ODE is also implemented (and has been implemented before by many others)
