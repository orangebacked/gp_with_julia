using LinearAlgebra
using Plots

function compute_kernel(X, y, k)

    M =  collect(Iterators.product(X, y)) ## compute combinations of both sets into a matrix of tuples
    K = k.(M) # compute the kernel function 

    return K

end


function gp_prediction(xₜᵣ, xₜₑ, yₜᵣ, σₙ², k)

    # step 1 L := Cholesky(K+σₙ*I)
    Kᵣ = compute_kernel(xₜᵣ, xₜᵣ, k) ## compute the kernel
    mmᵣ = Kᵣ+σₙ²*I ## compute the matrix for the cholesky decomposition
    Lᵣ = cholesky(mmᵣ)
    

    # step 2 α :=  L'\(L\y)
    U = Lᵣ.U
    α = U\(U'\yₜᵣ)

    # step 3 fₔ := Kₔ'α
    Kₔ = compute_kernel(xₜᵣ, xₜₑ, k)
    fₔ = Kₔ'α

    #step 4 # v:= L\K*
    v = U'\Kₔ

    #step 5 V[fə] := k(x*, x*) - v'v 
    Kₔₔ = compute_kernel(xₜₑ, xₜₑ, k)
    V = Kₔₔ  - v'v

    #step 6 logp(P|X) :=  -1/2ytα - ΣᵢlogLᵢᵢ = n/2log2π

    n = size(xₜᵣ)[1]
    logp = 1/2*yₜᵣ'*α - sum(log.(diag(U))) - (n/2)*log(2*pi)

    return fₔ, V, logp
end