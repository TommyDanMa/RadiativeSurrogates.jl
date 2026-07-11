# DeepONet helpers for scripts/28_deeponet_baseline.jl — kept OUT of the core
# module on purpose, mirroring models_kan.jl (the script includes this file
# directly). NeuralOperators.jl v0.6.4 provides the DeepONet; training mirrors
# src/models.jl train! exactly (full-batch Adam, geometric lr anneal, keep-best)
# so the comparison budget is matched by construction.

using NeuralOperators
using Lux, Optimisers, Zygote

"""
    build_deeponet(; hidden, p = 8, rng) -> (model, ps, st)

DeepONet with symmetric tanh branch/trunk `(2, hidden, hidden, p)`: branch takes
the standardized parameters μ̃ ∈ ℝ², trunk takes node coordinates x ∈ ℝ², output
is the field on all trunk points (n × M). `hidden = 60` gives 8,656 parameters
(POD-MLP: 8,642, +0.16%); `hidden = 108` gives 25,936 (3× budget: 25,926, +0.04%).
"""
function build_deeponet(; hidden::Integer, p::Integer = 8,
                        rng::AbstractRNG = Random.default_rng())
    model = DeepONet(; branch = (2, hidden, hidden, p),
                     trunk = (2, hidden, hidden, p),
                     branch_activation = tanh, trunk_activation = tanh)
    ps, st = Lux.setup(rng, model)
    return model, Lux.f64(ps), Lux.f64(st)
end

"""
    train_deeponet!(model, ps, st, μX, xy, Y; epochs, lr, lr_final, …) -> (ps, losses)

Full-batch Adam on `mean((model((μX, xy)) − Y)²)`; byte-identical schedule to
`train!` in src/models.jl (geometric lr anneal, keep-best snapshot). `μX` is
2 × M, `xy` is 2 × n, `Y` is n × M (scaled fields).
"""
function train_deeponet!(model, ps, st, μX::AbstractMatrix, xy::AbstractMatrix,
                         Y::AbstractMatrix;
                         epochs::Integer = 6000, lr::Real = 1e-3, lr_final::Real = 1e-5,
                         keep_best::Bool = true, verbose::Bool = true,
                         logevery::Integer = 2000)
    opt = Optimisers.setup(Optimisers.Adam(lr), ps)
    γ = epochs > 1 ? (lr_final / lr)^(1 / (epochs - 1)) : 1.0
    lr_t = float(lr)
    losses = Float64[]
    best_loss, best_ps = Inf, ps
    function lossfn(p)
        Ŷ, _ = model((μX, xy), p, st)
        return mean(abs2, Ŷ .- Y)
    end
    for epoch in 1:epochs
        res = Zygote.withgradient(lossfn, ps)
        push!(losses, res.val)
        if keep_best && res.val < best_loss
            best_loss, best_ps = res.val, deepcopy(ps)
        end
        opt, ps = Optimisers.update(opt, ps, res.grad[1])
        if γ != 1.0
            lr_t *= γ
            Optimisers.adjust!(opt, lr_t)
        end
        if verbose && (epoch == 1 || epoch % logevery == 0)
            @printf("    epoch %5d / %d   loss %.4e   (lr %.1e)\n",
                    epoch, epochs, res.val, lr_t)
        end
    end
    return (keep_best ? best_ps : ps), losses
end

"Deterministic DeepONet forward pass in evaluation mode; returns n × M fields."
function predict_deeponet(model, ps, st, μX::AbstractMatrix, xy::AbstractMatrix)
    Ŷ, _ = model((μX, xy), ps, Lux.testmode(st))
    return Ŷ
end

"""
    train_frozen_trunk!(branch, ps, st, μX, Φ, Y; …) -> (ps, losses)

Arm-3 trainer: the trunk is the fixed matrix `Φ` (n × r, the POD basis), only the
branch `μ̃ ↦ c ∈ ℝ^r` learns; loss is the same scaled-field MSE as the learned
arms, `mean((Φ · branch(μX) − Y)²)`. Isolates learned-basis vs fixed-basis.
"""
function train_frozen_trunk!(branch, ps, st, μX::AbstractMatrix, Φ::AbstractMatrix,
                             Y::AbstractMatrix;
                             epochs::Integer = 6000, lr::Real = 1e-3,
                             lr_final::Real = 1e-5, keep_best::Bool = true,
                             verbose::Bool = true, logevery::Integer = 2000)
    opt = Optimisers.setup(Optimisers.Adam(lr), ps)
    γ = epochs > 1 ? (lr_final / lr)^(1 / (epochs - 1)) : 1.0
    lr_t = float(lr)
    losses = Float64[]
    best_loss, best_ps = Inf, ps
    function lossfn(p)
        Ĉ, _ = branch(μX, p, st)
        return mean(abs2, Φ * Ĉ .- Y)
    end
    for epoch in 1:epochs
        res = Zygote.withgradient(lossfn, ps)
        push!(losses, res.val)
        if keep_best && res.val < best_loss
            best_loss, best_ps = res.val, deepcopy(ps)
        end
        opt, ps = Optimisers.update(opt, ps, res.grad[1])
        if γ != 1.0
            lr_t *= γ
            Optimisers.adjust!(opt, lr_t)
        end
        if verbose && (epoch == 1 || epoch % logevery == 0)
            @printf("    epoch %5d / %d   loss %.4e   (lr %.1e)\n",
                    epoch, epochs, res.val, lr_t)
        end
    end
    return (keep_best ? best_ps : ps), losses
end
