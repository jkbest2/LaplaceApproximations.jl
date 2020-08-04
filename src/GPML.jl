module GPML
using QuadGK
using StatsFuns
using LinearAlgebra

function post_funs(datalik, Σ)
    function f(u)
        datalik.f(u) - 1 / 2 * u' * (Σ \ u) - 1 / 2 * logdet(Σ) - length(u) / 2 * log(2π)
    end
    function gr(u)
        datalik.gr(u) - Σ \ u
    end
    function he(u)
        datalik.he(u) - inv(Σ)
    end
    (f = f, gr = gr, he = he)
end

function normal_funs(y, σobs)
    n = length(y)
    function f(u)
        sum(normlogpdf.(u, σobs, y))
    end
    function gr(u)
        @. (y - u) / σobs^2
    end
    function he(u)
        Diagonal(-ones(n) / σobs^2)
    end
    function ∇³(u)
        zeros(n) 
    end

    (f = f, gr = gr, he = he, ∇³ = ∇³)
end

function logistic_funs(y)
    # Convert data to [-1, 1] GPML format
    y2 = 2y .- 1
    # Data likelihood
    function f(u)
        -sum(log.(1 .+ exp.(-y2 .* u)))
    end
    # Gradient
    function gr(u)
        p = logistic.(u)
        y .- p
    end
    # Diagonal Hessian
    function he(u)
        p = logistic.(u)
        Diagonal(-p .* (1 .- p))
    end

    (f = f, gr = gr, he = he)
end

function probit_funs(y)
    y2 = 2y .- 1
    # Data likelihood
    function f(u)
        sum(normlogcdf.(y2 .* u))
    end
    # Gradient
    function gr(u)
        @. y2 * normpdf(u) / normcdf(y2 * u)
    end
    # Hessian
    function he(u)
        u_pdf = normpdf.(u)
        yu_cdf = normcdf.(y2 .* u)
        d = @. -(u_pdf / yu_cdf)^2 - y2 * u * u_pdf / yu_cdf
        Diagonal(d)
    end

    (f = f, gr = gr, he = he)
end
"""
    gpml_laplace(datalik, Σ, u0; utol = 1e-8, maxit = 10)

- `datalik` Named tuple with observation likelihood (`f`), gradient (`gr`), and
    Hessian (`he`).
- `Σ` Covariance matrix of `u`.
- `u` Initial value for Newton iteration of `u`.
- `utol` Difference between successive `u` values that ends Newton steps.
- `maxit` Maximum number of Newton iterations.

Approximate a marginal likelihood using the Laplace approximation. Adapted from
Algorithm 3.1 of Gaussian Processes for Machine Learning (p. 46).
"""
function gpml_laplace(datalik, Σ, u0; utol = 1e-8, maxit = 10)
    u = copy(u0)
    it = 0
    while true
        it += 1
        W = -datalik.he(u)
        Wsqrt = sqrt(W)
        global L = cholesky(Hermitian(I + Wsqrt * Σ * Wsqrt))
        b = W * u + datalik.gr(u)
        global a = b - Wsqrt * (L \ (Wsqrt * Σ * b))
        unew = Σ * a
        if maximum(unew .- u) < utol
            u .= unew
            break
        elseif it ≥ maxit
            break
        end
        u .= unew
    end
    logq = -1 / 2 * a' * u + datalik.f(u) - logdet(L) / 2
    logq, u, a
end

"""
    gpml_pred(xpred, datalik, uhat, xobs, Σ, covfun, sigmoidfun)

Provide classification predictions at locations `xpred` with observation
likelihood `datalik` and `uhat` estimated by `gpml_laplace` with observations at
`xobs`, covariance matrix `Σ`, covariance function `covfun`, and sigmoid (e.g.
`logistic`) `signmoidfun`. Algorithm 3.2 of Gaussian Processes for Machine
Learning (p. 47).
"""
function gpml_pred(xpred, datalik, uhat, xobs, Σ, covfun, sigmoidfun)
    W = -datalik.he(uhat)
    Wsqrt = sqrt(W)
    L = cholesky(Hermitian(I + Wsqrt * Σ * Wsqrt))
    kcross = covfun.(xobs, xpred')
    upred = kcross' * datalik.gr(uhat)
    v = L.L \ (Wsqrt * kcross)
    Vpred = covfun.(xpred, xpred') - v' * v
    ppred = quadgk(z -> sigmoidfun(z) * normpdf(upred, sqrt(Vpred), z), -Inf, Inf)[1]
    ppred, upred, sqrt(Vpred)
end

function gpml_grad(covpars, covfun, datalik, u0)
    K = covfun.f(covpars)
    logq, uhat, a = gpml_laplace(datalik, K, u0)
    W = -datalik.he(uhat)
    Wsqrt = sqrt(W)
    L = cholesky(Hermitian(I + Wsqrt * K * Wsqrt))
    R = Wsqrt * (L \ Wsqrt)
    C = L.L \ (Wsqrt * K)
    s2 = -Diagonal(diag(K) - diag(C'C)) / 2 * datalik.∇³(uhat)
    grad_covpars = similar(covpars)
    C = covfun.gr(covpars)
    for j in 1:length(covpars)
        s1 = (a' * C[j] * a - tr(R * C[j])) / 2
        b = C[j] * datalik.gr(uhat)
        s3 = b - K * R * b
        grad_covpars[j] = s1 + s2's3
    end
    logq, grad_covpars, uhat
end

end # module GPML
