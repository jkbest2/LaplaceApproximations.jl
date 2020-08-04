module LaplaceApproximations

using Zygote
using LinearAlgebra

export 
    laplace_approx,
    GPML

include("GPML.jl")

"""
    laplace_approx(nll, u; niter = 10)

NOT ROBUST! Uses standard Newton updates with no checking or scaling.

For negative log-likelihood function `nll`, calculate the approximate marginal
likelihood after integrating out the parameters in `u` using the Laplace
approximation. Returns the marginal negative log-likelihood and the values of
`u` at the minimum. The function `nll` should accept only the argument `u`.
Number of Newton steps is controlled with `niter`. For quadratic functions (e.g.
a normal-normal hierarchical model) this can be set to 1. Some tuning may be
required to ensure that `u` reaches an optimum.
"""
function laplace_approx(nll::Function, u; niter::Integer = 10)
    for i in 1:niter
        g = Zygote.gradient(nll, u)[1]
        H = Zygote.hessian(nll, u)
        u = u - H \ g
    end
    H = Zygote.hessian(nll, u)
    mnll = 1//2 * logdet(H) + nll(u) - length(u) / 2 * log(2π)
    return (mnll, u)
end

end
