using LaplaceApproximations
using Test
using Distributions
using DistributionsAD
using LinearAlgebra
using QuadGK
using StatsFuns

include("normal-normal-test-setup.jl")
include("logistic-normal-test-setup.jl")
include("GPML-test.jl")

## Simulation setup
n = 1000
X = hcat(ones(n), randn(n))
β = [0.0, 1.0]
Z = reduce(vcat,
        [circshift([1. 0 0 0 0], [0, r ÷ (n ÷ 5)]) for r in 0:(n - 1)])
σgrp = 1.0
u = σgrp * randn(5)
σobs = 0.2

## Normal-normal hierarchical model
## y ~ Normal(Xβ + Zu, σobs)
## u ~ Normal(0, σgrp)
y = X * β + Z * u + σobs * randn(n)

nld = NormLLData(y, X, Z)
nll = NormLL(β, zeros(5), 0.2, 1.0, nld)

norm_la, _ = laplace_approx(nll, zeros(5))

norm_qu = 0.0
for g in 1:5
    nld_tmp = NormLLData(nld, g)
    nll_tmp = NormLL(β, zeros(1), σobs, σgrp, nld_tmp)
    l_qu, _ = quadgk(u -> exp(-nll_tmp(u)), -Inf, Inf)
    global norm_qu -= log(l_qu)
end

## Logistic regression model
## y ~ Bernoulli(p)
## p = logistic(ol)
## ol = Xβ + Zu
## u ~ Normal(0, σgrp)
lo = X * β + Z * u
p = logistic.(lo)
y = rand.(Bernoulli.(p))

bld = BinomLLData(y, X, Z)
bll = BinomLL(β, u, σgrp, bld)

binom_la, _ = laplace_approx(bll, zeros(5))

binom_qu = 0.0
for g in 1:5
    bld_tmp = BinomLLData(bld, g)
    bll_tmp = BinomLL(β, zeros(1), σgrp, bld_tmp)
    l_qu, _ = quadgk(u -> exp(-bll_tmp(u)), -Inf, Inf)
    global binom_qu -= log(l_qu)
end

@testset "LaplaceApproximations.jl" begin

# Normal-normal Laplace approximation should be exact
@test isapprox(norm_qu, norm_la)
# Expect some error for the logistic example
@test isapprox(binom_qu, binom_la, rtol = 1e-2)
# Make sure you get a warning when grtol not reached in maxit steps
@test_logs (:warn, "Gradient tolerance not reached in 1 Newton steps")
    laplace_approx(bll, zeros(5); grtol = 1e-20, maxit = 1)

end