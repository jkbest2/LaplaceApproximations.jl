module LaplaceApproximations

using LinearAlgebra
using LineSearches
using Zygote

export 
    laplace_approx,
    GPML

include("inner_control.jl")
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
function laplace_approx(fn::Function, u0, innerprob::AbstractInnerProblem = NonConvexInnerProblem();
                        grtol = sqrt(eps(eltype(u0))), maxit = 100)
    u = copy(u0)
    s = similar(u0)

    function gr!(grad, u)
        grad .= Zygote.gradient(fn, u)[1]
    end
    function he!(hess, u)
        hess .= Zygote.hessian(fn, u)
    end

    grad = similar(u0)
    hess = Matrix{eltype(u0)}(undef, length(u0), length(u0))

    gr!(grad, u0)
    he!(hess, u0)

    nit = 0
    while(true)
        nit += 1
        s .= -hess \ grad
        α, obj = linesearch(innerprob, fn, gr!, u, s)
        u .+= α .* s

        gr!(grad, u)
        he!(hess, u)
        maximum(grad) < grtol && break
        if nit ≥ maxiter(innerprob)
            @warn "Gradient tolerance not reached in $nit Newton steps" mgc = maximum(grad)
            break
        end
    end
    mnll = -length(u) / 2 * log(2π) + logdet(hess) / 2 + fn(u)
    return (mnll, u)
end

# mutable struct LaplaceApproximation{Tf, Tgr, The, Tu, Ta, Tq}
#     f::Tf
#     gr::Tgr
#     he::The
#     u::Tu
#     a::Ta
#     logq::Tq
#     function LaplaceApproximation(u, a, logq)
#         new(u, a, logq)
#     end
# end

# loglikelihood(la::LaplaceApproximation) = la.logq

end
