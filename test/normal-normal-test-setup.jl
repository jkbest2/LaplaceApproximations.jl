"""
    NormLLData
        y::Vector
        X::Matrix
        Z::Matrix

Holds data for hierarchical linear model, with response `y`, fixed effects
design matrix `X`, and random effects design matrix `Z`.
"""
struct NormLLData{T}
    y::Vector{T}
    X::Matrix{T}
    Z::Matrix{T}
end
"""
    NormLLData(::NormLLData, grp::Integer)

Outer constructor that returns data subset by random effects membership. Used
when integrating out random effect parameters one at a time using quadrature.
"""
function NormLLData(nld::NormLLData, grp::Integer)
    idx = Z[:, grp] .== 1
    NormLLData(nld.y[idx], nld.X[idx, :], nld.Z[idx, grp:grp])    
end

"""
    NormLL
        β
        u
        σobs
        σgrps
        nld::NormLLData

Holds parameter values and data for hierarchical linear model. Also a functor to
return the complete data likelihood. Fixed effects parameters are in the vector
`β`, random effects in `u`, observation standard deviation in σobs, and group
standard deviation σgrp. `nld` holds the associated data object.
"""
struct NormLL{Tβ, Tu, Tσobs, Tσgrp, Tnld<:NormLLData} <: Function
    β::Tβ
    u::Tu
    σobs::Tσobs
    σgrp::Tσgrp
    nld::Tnld
end
# Outer constructor to make changing `u` easy
"""
    NormLL(::NormLL, u)

Outer constructor to make swapping random effects vector easy. Makes both
Laplace approximation and quadrature.
"""
function NormLL(NLL::NormLL, u)
    NormLL(NLL.β, u, NLL.σobs, NLL.σgrp, NLL.nld)
end
"""
Negative log-likelihood function. Full data likelihood.
"""
function (NLL::NormLL)()
    pred = NLL.nld.X * NLL.β + NLL.nld.Z * NLL.u
    datalik = -sum(logpdf.(Normal.(pred, NLL.σobs), NLL.nld.y))
    randlik = -sum(logpdf.(Normal(0.0, NLL.σgrp), NLL.u))
    datalik + randlik
end
"""
Single argument version to make Laplace an quadrature simple.
"""
function (NLL::NormLL)(u)
    NormLL(NLL, u)()
end