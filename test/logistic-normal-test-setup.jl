"""
Holds data for hierarchical logistic model test example.
"""
struct BinomLLData{Ty, T}
    y::Ty
    X::T
    Z::T
end
"""
Outer constructor for subsetting by random effects group
"""
function BinomLLData(bld::BinomLLData, grp::Integer)
    idx = Z[:, grp] .== 1
    BinomLLData(bld.y[idx], bld.X[idx, :], bld.Z[idx, grp:grp])
end
"""
Functor that holds parameter values and data for logistic hierarchical
regression model.
"""
struct BinomLL{Tβ, Tu, Tσgrp, Tbld<:BinomLLData} <: Function
    β::Tβ
    u::Tu
    σgrp::Tσgrp
    bld::Tbld
end
"""
Convenient outer constructor to make swapping random effects vectors easier.
"""
function BinomLL(BLL::BinomLL, u)
    BinomLL(BLL.β, u, BLL.σgrp, BLL.bld)
end
"""
Hierarchical logistic regression negative log-likelihood.
"""
function (BLL::BinomLL)()
    lo = BLL.bld.X * BLL.β + BLL.bld.Z * BLL.u
    p = logistic.(lo)
    datalik = -sum(logpdf.(Bernoulli.(p), BLL.bld.y))
    randlik = -sum(logpdf.(Normal(0.0, BLL.σgrp), BLL.u))
    datalik + randlik
end
function (BLL::BinomLL)(u)
    BinomLL(BLL, u)()
end
