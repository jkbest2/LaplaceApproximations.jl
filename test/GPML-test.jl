using LaplaceApproximations
using LaplaceApproximations.GPML
using Distributions
using LinearAlgebra
using StatsFuns
# using Plots

"Matérn (ν = 3/2) correlation kernel"
function mat32(x1, x2)
    d = norm(x1 .- x2)
    (1 + (sqrt(3) * d)) * exp(-sqrt(3) * d)
end

n = 100
xobs = 5rand(n)
sort!(xobs)
xpred = range(0, 5, length = 101)

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

normpost = GPML.post_funs(normdatalik, Σ)
normlogq_la, normyhat_la = laplace_approx(normpost.f, zeros(n); maxit = 10)

normlogq_la - normlogq
isapprox(normuhat_la, normyhat)

# plot(xpred, getindex.(normpreds, 2))
# scatter!(xobs, normy)
# scatter!(xobs, yobs)



# Logistic example
pobs = logistic.(utrue)
yobs = rand.(Bernoulli.(pobs))
ldatalik = GPML.logistic_funs(yobs)
llogq, luhat = GPML.gpml_laplace(ldatalik, Σ, zeros(n))
lpreds = map(xpred) do x
    GPML.gpml_pred(x, ldatalik, luhat, xobs, Σ, mat32, logistic)
end

# plot(xpred, getindex.(lpreds, 2))
# scatter!(xobs, utrue)
# scatter!(xobs, yobs)

# plot(xpred, getindex.(lpreds, 1))
# scatter!(xobs, yobs)

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