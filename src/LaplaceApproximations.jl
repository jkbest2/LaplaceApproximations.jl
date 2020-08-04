module LaplaceApproximations

using Zygote
using LinearAlgebra

export 
    laplace_approx,
    GPML

include("GPML.jl")

"""
    mnll, u = laplace_approx(nll, u0; grtol = sqrt(eps(eltype(u0))))

NOT ROBUST! Uses standard Newton updates with no checking or scaling.

For negative log-likelihood function `nll`, calculate the approximate marginal
likelihood after integrating out the parameters in `u` using the Laplace
approximation. Returns the marginal negative log-likelihood and the values of
`u` at the minimum. The function `nll` should accept only the argument `u`.
Newton steps are taken until the maximum gradient component is less than `grtol`
or `maxit`, whichever comes first.
"""
function laplace_approx(nll::Function, u0;
                        grtol = sqrt(eps(eltype(u0))),
                        maxit = 100)
    u = copy(u0)
    gr = Zygote.gradient(nll, u)[1]
    he = Zygote.hessian(nll, u)
    nit = 0
    while(true)
        nit += 1
        u .-= he \ gr
        gr .= Zygote.gradient(nll, u)[1]
        he .= Zygote.hessian(nll, u)
        maximum(gr) < grtol && break
        if nit ≥ maxit
            @warn "Gradient tolerance not reached in $nit Newton steps" gr = gr
            break
        end
    end
    mnll = -length(u) / 2 * log(2π) + logdet(he) / 2 + nll(u)
    return (mnll, u)
end

end