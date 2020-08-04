using LaplaceApproximations
using LaplaceApproximations.GPML
using Distributions
using LinearAlgebra
using StatsFuns
# using Plots
# using Optim

"Matérn (ν = 3/2) correlation kernel"
function mat32(x1, x2, ρ = 1)
    d = norm(x1 .- x2)
    _mat32(d, ρ)
end
function _mat32(d, ρ = 1)
    (1 + sqrt(3) * d / ρ) * exp(-sqrt(3) * d / ρ)
end
function ddρ_mat32(d, σ, ρ)
    3 * d^2 * σ^2 * exp(-sqrt(3) * d / ρ) / ρ^3
end

function mat32cov_funs(x)
    d = norm.(x .- x')
    function f(p)
        p[1] * _mat32.(d, p[2])
    end
    function gr(p)
        ddσ = 2 * p[1] * _mat32.(d, p[2])
        ddρ = ddρ_mat32.(d, p[1], p[2])
        (ddσ, ddρ)
    end
    (f = f, gr = gr)
end


n = 100
xobs = 5rand(n)
sort!(xobs)
xpred = range(0, 5, length = 101)

mat32cov = mat32cov_funs(xobs)
Σ = Matrix{Float64}(undef, n, n)
for j in 1:n, i in 1:n
    Σ[i, j] = mat32(xobs[i], xobs[j])
    Σ[j, i] = Σ[i, j]
end

utrue = rand(MvNormal(Σ))

# Normal example
σobs = 0.05
normy = utrue .+ σobs * randn(n)
normdatalik = GPML.normal_funs(normy, σobs)
normlogq, normyhat = GPML.gpml_laplace(normdatalik, Σ, zeros(n); maxit = 2)

# normpost = GPML.post_funs(normdatalik, Σ)
# normlogq_la, normyhat_la = laplace_approx(normpost.f, zeros(n); maxit = 10)

# normlogq_la - normlogq
# isapprox(normuhat_la, normyhat, normdatalik, zeros(n))

# plot(xpred, getindex.(normpreds, 2))
# scatter!(xobs, normy)
# scatter!(xobs, yobs)

GPML.gpml_grad([1.0, 1.0], mat32cov, normdatalik, zeros(n))

# normval(covpar) = GPML.gpml_grad(covpar, mat32cov, normdatalik, zeros(n))[1]
# normgr(G, covpar) = G .= GPML.gpml_grad(covpar, mat32cov, normdatalik, zeros(n))[2]
# Not currently working; end up with non-PD B matrix at some point
# maximize(normval, normgr, ones(2), ConjugateGradient())
# maximize(normval, ones(2))

# Logistic example
pobs = logistic.(utrue)
yobs = rand.(Bernoulli.(pobs))
ldatalik = GPML.logistic_funs(yobs)
llogq, luhat = GPML.gpml_laplace(ldatalik, Σ, zeros(n))
lpreds = map(xpred) do x
    GPML.gpml_pred(x, ldatalik, luhat, xobs, Σ, mat32, logistic)
end

# plot(xpred, getindex.(lpreds, 2))
# plot!(xobs, utrue)
# scatter!(xobs, yobs)
# plot!(xpred, getindex.(lpreds, 1))

# lres = maximize(p -> GPML.gpml_laplace(
#     ldatalik,
#     p[1]^2 * mat32.(xobs, xobs', p[2]),
#     ones(n))[1], ones(2))
# lcovpars = Optim.maximizer(lres)
# lq, luhat = GPML.gpml_laplace(ldatalik, lcovpars[1] * mat32.(xobs, xobs', covpars[2]), ones(n))

# Probit example
pyobs = @. rand(Bernoulli(normcdf(utrue)))
pdatalik = GPML.probit_funs(pyobs)
plogq, puhat = GPML.gpml_laplace(pdatalik, Σ, zeros(n))
ppreds = map(xpred) do x
    GPML.gpml_pred(x, pdatalik, puhat, xobs, Σ, mat32, normcdf)
end

# plot(xpred, getindex.(ppreds, 2))
# plot!(xobs, utrue)
# plot!(xpred, getindex.(ppreds, 1))
# scatter!(xpred, pyobs)

ppost = GPML.post_funs(pdatalik, Σ)
u = zeros(n)
for _ in 1:10
    global u -= ppost.he(u) \ ppost.gr(u)
end