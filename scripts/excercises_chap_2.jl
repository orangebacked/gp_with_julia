
using DrWatson
@quickactivate "gaussian_processes"

using LinearAlgebra
using Plots

include(srcdir("gp_prediction.jl"))


xₜᵣ = [-4, -3, -1, 0, 2]

xₜₑ =  collect(range(-5, stop=5, length = 102))

yₜᵣ = [-2, 0, 1, 2, -1]

k(x) =  exp(-(0.2)*norm(x[1] - x[2]))

σₙ²=  0.01

fₔ, V, log_pyx =  gp_prediction(xₜᵣ, xₜₑ, yₜᵣ, σₙ²,k)
