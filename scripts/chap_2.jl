using DrWatson
@quickactivate "gaussian_processes"


using LinearAlgebra
using Plots

## data simulation 

xₜᵣ = [-4, -3, -1, 0, 2]

xₜₑ =  collect(range(-5, stop=5, length = 102))

yₜᵣ = [-2, 0, 1, 2, -1]


n = size(xₜᵣ)[1]


### plot the data (always a good idea)
plot(xₜᵣ, yₜᵣ)

scatter(xₜᵣ, yₜᵣ)

## define kernel function
Mₓ =  collect(Iterators.product(xₜᵣ, xₜᵣ))

k(x) =  exp(-(0.5)*norm(x[1] - x[2]))

K = k.(Mₓ)

#L := Cholesky(K+σₙ*I)

σₙ²=  0.01

mm = K+σₙ²*I

L = cholesky(mm)

U = L.U

# α :=  L'\(L\y)

α = U\(U'\yₜᵣ)

# fₔ := Kₔ'α
## we have to define kₔ

Mₔ =  collect(Iterators.product(xₜᵣ, xₜₑ))

Kₔ = k.(Mₔ)

## noe we define fₔ
fₔ = Kₔ'α

# v:= L\K*

v = U'\Kₔ

size(Kₔ)

# V[fə] := k(x*, x*) - v'v 

## we define Kₔₔ
Mₔₔ =  collect(Iterators.product(xₜₑ, xₜₑ))

Kₔₔ = k.(Mₔₔ)  

## now we define the variance
V = Kₔₔ  - v'v
### lets plot the min function #### first lets calculate the square root of the variance 
σ =  V.^(1/2) 
scatter(xₜᵣ, yₜᵣ, size=(800,700)) 

σ₊ =  fₔ + 2*diag(σ) 

σ₋ =  fₔ - 2*diag(σ)

plot!(xₜₑ, fₔ)

plot!(xₜₑ, σ₊)

plot!(xₜₑ, σ₋)

# logp(P|X) :=  -1/2ytα - ΣᵢlogLᵢᵢ = n/2log2π

logp = 1/2*yₜᵣ'*α - sum(log.(diag(U))) - (n/2)*log(2*pi)

# return \f*, V, logp 


### defining the functions for future uses


k(x) =  exp(-(0.5)*norm(x[1] - x[2]))

function compute_kernel(X, y)

    M =  collect(Iterators.product(X, y)) ## compute combinations of both sets into a matrix of tuples
    K = k.(M) # compute the kernel function 

    return K

end



function gp_prediction(xₜᵣ, xₜₑ, yₜᵣ, σₙ²)

    # step 1 L := Cholesky(K+σₙ*I)
    Kᵣ = compute_kernel(xₜᵣ, xₜᵣ) ## compute the kernel
    mmᵣ = Kᵣ+σₙ²*I ## compute the matrix for the cholesky decomposition
    Lᵣ = cholesky(mm)
    

    # step 2 α :=  L'\(L\y)
    U = Lᵣ.U
    α = U\(U'\yₜᵣ)

    # step 3 fₔ := Kₔ'α
    Kₔ = compute_kernel(xₜᵣ, xₜₑ)
    fₔ = Kₔ'α

    #step 4 # v:= L\K*
    v = U'\Kₔ

    #step 5 V[fə] := k(x*, x*) - v'v 
    Kₔₔ = compute_kernel(xₜₑ, xₜₑ)
    V = Kₔₔ  - v'v

    #step 6 logp(P|X) :=  -1/2ytα - ΣᵢlogLᵢᵢ = n/2log2π
    logp = 1/2*yₜᵣ'*α - sum(log.(diag(U))) - (n/2)*log(2*pi)

    return fₔ, V, logp
end


gp_prediction(xₜᵣ, xₜₑ, yₜᵣ, σₙ²)