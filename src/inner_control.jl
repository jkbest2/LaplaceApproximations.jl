# FIXME PR to LineSearches.jl instead of this ugly hack
const AbstractLineSearch = Union{HagerZhang, MoreThuente, BackTracking,
                                 StrongWolfe, Static}
const AbstractInitialStepLength = Union{InitialPrevious, InitialStatic,
                                        InitialHagerZhang, InitialQuadratic,
                                        InitialConstantChange}

mutable struct StepLengthState{T}
    alpha::T
    alphamin::T
    alphamax::T

    function StepLengthState(alpha::T, alphamin::T, alphamax::T) where T<:Number
        new{T}(alpha, alphamin, alphamax)
    end
end
StepLengthState(α::T) where T = StepLengthState(α, zero(T), convert(T, Inf))
StepLengthState() = StepLengthState(1.0)

abstract type AbstractInnerProblem end

struct QuadraticInnerProblem <: AbstractInnerProblem end

struct ConvexInnerProblem{T} <: AbstractInnerProblem
    tol::T
    maxiter::Integer

    function ConvexInnerProblem(tol::T, maxiter) where T<:Number
        new{T}(tol, maxiter)
    end
end
struct NonConvexInnerProblem{Tls, Tinit, T} <: AbstractInnerProblem
    linesearch::Tls
    alpha::Tinit
    maxiter::Integer
    grtol::T

    function NonConvexInnerProblem(linesearch::Tls, α0::Tinit, maxiter::Integer, grtol::T) where
        {Tls<:AbstractLineSearch, Tinit<:StepLengthState, T<:Number}
        new{Tls, Tinit, T}(linesearch, α0, maxiter, grtol)
    end
end
function NonConvexInnerProblem(ls = BackTracking(),
                               α0 = StepLengthState(),
                               maxiter = 100,
                               grtol = sqrt(eps()))
    NonConvexInnerProblem(ls, α0, maxiter, grtol)
end

maxiter(prob::AbstractInnerProblem) = prob.maxiter
maxiter(::QuadraticInnerProblem) = 1

initstep(ip::AbstractInnerProblem) = ip.alpha.alpha


# Line search
# All line searches use the signature:
# - φ(α) : returns objective at x .+ α s (closure over x and s)
# - dφ : returns φ'(α); ∇f(x + αs) ⋅ s
# - φdφ : function that returns the tuple of (φ, dφ)
# - α : initial step length
# - φ_0 : φ(x)
# - dφ_0 :

function linesearch(innerprob::AbstractInnerProblem, fn, gr!, u, s)
    α0 = initstep(innerprob)
    ls_funs = make_ls_funs(fn, gr!, u, s)
    α, val = innerprob.linesearch(ls_funs.φ, ls_funs.dφ, ls_funs.φdφ,
                                  α0, ls_funs.φ(0), ls_funs.dφ(0))
    innerprob.alpha.alpha = α
    (α, val)
end


function make_ls_funs(fn, gr!, u0, s)
    grad = similar(u0)
    function φ(α)
        fn(u0 .+ α .* s)
    end
    function dφ(α)
        gr!(grad, u0 .+ α .* s)
        dot(grad, s)
    end
    function φdφ(α)
        (φ(α), dφ(α))
    end
    (φ = φ, dφ = dφ, φdφ = φdφ)
end

