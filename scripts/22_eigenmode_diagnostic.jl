# Exploratory (NOT pre-registered; suggested by external review of the rank note):
# what sets the hard cutoff σ_{d+1}/σ₁ if not a local Taylor remainder (P3 miss)?
# Candidate: spectral-gap suppression. Expand the centered snapshots in the
# eigenbasis of the linearized operator  L = A_cond + 4ε₀σ|u₀|³ (boundary mass),
# generalized against the L² mass matrix, and measure the RMS coefficient α_k per
# eigenmode. If α_k ~ 1/λ_k the cliff is an eigenvalue-gap effect; if α_k falls
# faster (e.g. exponentially in k), neither mechanism suffices and that is the
# honest finding.

include(joinpath(@__DIR__, "config.jl"))
using Gridap
using CairoMakie
CairoMakie.activate!()

println("=== 22_eigenmode_diagnostic ===")

fom = build_steady_fom(CFG)
ε0, Q0 = 0.525, 450.0
s0 = solve_steady(fom; eps_r = ε0, Q = Q0, abstol = 1e-10)
@assert s0.converged
u0h = FEFunction(fom.U, s0.x)
cfg = fom.cfg

a_cond(du, v) = ∫(cfg.k * (∇(v) ⋅ ∇(du))) * fom.dΩ
b_rad(du, v) = ∫(v * du * ((4 * ε0 * cfg.sigma) * (abs3 ∘ u0h))) * fom.dΓ
a_mass(du, v) = ∫(du * v) * fom.dΩ
L = assemble_matrix(a_cond, fom.U, fom.V) + assemble_matrix(b_rad, fom.U, fom.V)
Mm = assemble_matrix(a_mass, fom.U, fom.V)

println("dense generalized eigensolve, n = $(fom.n) ...")
t = @elapsed F = eigen(Symmetric(Matrix(L)), Symmetric(Matrix(Mm)))
K = 20
λ = F.values[1:K]
V = F.vectors[:, 1:K]                     # M-orthonormal columns
@printf("done in %.1f s;  λ₁ %.3e, λ₂ %.3e, λ₂₀ %.3e\n", t, λ[1], λ[2], λ[K])

d = load_snapshots()
Sc = d["U"][:, d["train_idx"]] .- load_pod().P.mean
C = V' * (Mm * Sc)                        # eigenbasis coefficients of the responses
α = vec(sqrt.(mean(abs2, C; dims = 2)))

println("\n k      λ_k        α_k (RMS)   α_k/α_1     (λ_1/λ_k)")
for i in 1:K
    @printf("%2d   %.3e   %.3e   %.3e   %.3e\n",
            i, λ[i], α[i], α[i] / α[1], λ[1] / λ[i])
end
# power-law fit α ~ λ^(-p) over modes 2..K, and exponential fit α ~ exp(-c k)
lx, ly = log.(λ[2:K]), log.(α[2:K])
p = -(length(lx) * sum(lx .* ly) - sum(lx) * sum(ly)) /
    (length(lx) * sum(lx .^ 2) - sum(lx)^2)
ke = collect(2:K)
ce = -(length(ke) * sum(ke .* ly) - sum(ke) * sum(ly)) /
     (length(ke) * sum(ke .^ 2) - sum(ke)^2)
@printf("\nfit over k = 2..%d:  α ~ λ^-%.2f   |   α ~ exp(-%.2f·k)\n", K, p, ce)

# Collinearity check: the excited eigen-coefficients, viewed as functions of μ,
# may span far fewer dimensions than the number of excited modes. Row-normalize
# the excited block of C and look at its singular values: that is the dimension
# of the parameter-function space, which is what POD rank actually measures.
excited = findall(>(1e-6), α ./ α[1])
Cn = C[excited, :] ./ sqrt.(sum(abs2, C[excited, :]; dims = 2))
sv = svd(Cn).S
@printf("\n%d excited eigenmodes (parity-allowed): %s\n", length(excited),
        join(string.(excited), ","))
@printf("singular values of their row-normalized μ-functions: %s\n",
        join((@sprintf("%.1e", s) for s in sv[1:min(6, end)]), "  "))

jldsave(joinpath(DATA_DIR, "eigenmode_diagnostic.jld2");
        λ, α, p_powerlaw = p, c_exp = ce, eps0 = ε0, Q0,
        excited = collect(excited), func_svals = sv)

fig = Figure(size = (900, 380))
ax1 = Axis(fig[1, 1]; xscale = log10, yscale = log10, xlabel = "eigenvalue λ_k",
           ylabel = "RMS response coefficient α_k",
           title = "Response energy vs linearized-operator eigenvalue")
scatterlines!(ax1, λ, max.(α, 1e-17))
lines!(ax1, λ[2:K], α[2] .* (λ[2:K] ./ λ[2]) .^ (-1);
       linestyle = :dash, color = :gray, label = "1/λ (spectral-gap hypothesis)")
lines!(ax1, λ[2:K], α[2] .* (λ[2:K] ./ λ[2]) .^ (-p);
       linestyle = :dot, color = :tomato, label = @sprintf("fit λ^-%.1f", p))
axislegend(ax1; position = :lb)
ax2 = Axis(fig[1, 2]; yscale = log10, xlabel = "mode index k", ylabel = "α_k/α_1",
           title = "Same data vs mode index")
scatterlines!(ax2, 1:K, max.(α ./ α[1], 1e-17))
save(joinpath(FIG_DIR, "21_eigenmode_diagnostic.png"), fig)

println("saved data/eigenmode_diagnostic.jld2 and figures/21_eigenmode_diagnostic.png")
