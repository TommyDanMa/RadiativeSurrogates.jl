# Exploratory (NOT pre-registered; follows the §4b collinearity finding): does the
# factorized cliff formula generalize?
#
#   σ_{r*+1}/σ₁  ≈  ℓ_{r*+1} × w_{r*+1} / s₁
#
# where ℓ_{r*+1} is the (r*+1)-th singular value of the ROW-NORMALIZED excited
# eigen-coefficient functions (equal-weight parametric leakage), and w_{r*+1} is the
# amplitude the leaking singular direction actually carries (‖D·u_{r*+1}‖ with D the
# per-mode amplitude weights). The two factors are measured separately; the product is
# compared against the POD cliff measured independently by full-space SVD of the
# snapshots. Configs: base (ε,Q), P1 (ε,Q₁,Q₂ second heater), P2 (ε,Q,k), and the four
# P3 shrink boxes. A "ceiling" column (svals of the weighted excited block) shows how
# much of the cliff the top-K eigenspace can express at all.
#
# Caveat kept from scripts/22: POD is Euclidean on raw DOF vectors while the eigenbasis
# is M-orthonormal; on this uniform grid the ratio-level comparison is unaffected.

include(joinpath(@__DIR__, "config.jl"))
using QuasiMonteCarlo
using Gridap
using CairoMakie
CairoMakie.activate!()

println("=== 23_cliff_reconstruction ===")

fom = build_steady_fom(CFG)
cfg = fom.cfg
K = 20

const PATCH2 = (0.60, 0.75, 0.15, 0.35)      # identical to scripts/19
area2 = (PATCH2[2] - PATCH2[1]) * (PATCH2[4] - PATCH2[3])
source2(Q2) = x -> (PATCH2[1] <= x[1] <= PATCH2[2] &&
                    PATCH2[3] <= x[2] <= PATCH2[4]) ? Q2 / area2 : 0.0

"Generalized eigenbasis (K modes) of A_cond(k0) + 4ε0σ|u0|³ boundary mass vs L² mass."
function eigenbasis(u0h, eps0, k0)
    a_cond(du, v) = ∫(k0 * (∇(v) ⋅ ∇(du))) * fom.dΩ
    b_rad(du, v) = ∫(v * du * ((4 * eps0 * cfg.sigma) * (abs3 ∘ u0h))) * fom.dΓ
    a_mass(du, v) = ∫(du * v) * fom.dΩ
    L = assemble_matrix(a_cond, fom.U, fom.V) + assemble_matrix(b_rad, fom.U, fom.V)
    Mm = assemble_matrix(a_mass, fom.U, fom.V)
    F = eigen(Symmetric(Matrix(L)), Symmetric(Matrix(Mm)))
    return F.values[1:K], F.vectors[:, 1:K], Mm
end

"Sobol snapshot matrix (deterministic draw, reproduces scripts/19 spectra)."
function snapshots(NS, lb, ub; solvekw)
    mus = QuasiMonteCarlo.sample(NS, lb, ub, SobolSample())
    U = Matrix{Float64}(undef, fom.n, NS)
    for i in 1:NS
        s = solve_steady(fom; solvekw(mus[:, i])..., abstol = 1e-9)
        @assert s.converged
        U[:, i] = s.x
    end
    return U
end

"""
Measured POD cliff vs factorized prediction for one config.
`U` snapshots, `ref` NamedTuple of solver kwargs for the box-center reference,
`eps0`/`k0` the linearization parameters. Returns a NamedTuple for the table.
"""
function reconstruct_cliff(name, U, ref, eps0, k0; V = nothing, Mm = nothing)
    if V === nothing
        s0 = solve_steady(fom; ref..., abstol = 1e-10)
        @assert s0.converged
        _, V, Mm = eigenbasis(FEFunction(fom.U, s0.x), eps0, k0)
    end
    Sc = U .- mean(U; dims = 2)
    C = V' * (Mm * Sc)
    α = vec(sqrt.(mean(abs2, C; dims = 2)))
    excited = findall(>(1e-6), α ./ maximum(α))
    CE = C[excited, :]
    ρ = vec(sqrt.(sum(abs2, CE; dims = 2)))
    F = svd(CE ./ ρ)
    ℓ = F.S
    rstar = count(>=(0.1 * ℓ[1]), ℓ)
    w = norm(ρ .* F.U[:, rstar + 1])
    sw = svdvals(CE)
    naive = ℓ[rstar + 1] * w / sw[1]          # leakage × TOTAL amplitude: upper bound
    recon = sw[rstar + 1] / sw[1]             # amplitude-weighted collinearity: the claim
    σ = fit_pod(U).svals
    measured = σ[rstar + 1] / σ[1]
    @printf("%-12s r*=%d  excited %2d  ℓ_{r*+1} %.1e  →  recon %.1e   meas %.1e   (recon/meas %.2f;  naive product %.1e)\n",
            name, rstar, length(excited), ℓ[rstar + 1], recon, measured,
            recon / measured, naive)
    return (; name, rstar, n_excited = length(excited),
            leak = ℓ[rstar + 1], naive, recon, measured)
end

results = NamedTuple[]

# base (ε,Q): stored train snapshots, shared reference/eigenbasis with the shrink boxes
d = load_snapshots()
U_base = d["U"][:, d["train_idx"]]
εc, Qc = 0.525, 450.0
s0 = solve_steady(fom; eps_r = εc, Q = Qc, abstol = 1e-10)
@assert s0.converged
λb, Vb, Mmb = eigenbasis(FEFunction(fom.U, s0.x), εc, cfg.k)
push!(results, reconstruct_cliff("base", U_base, nothing, εc, cfg.k; V = Vb, Mm = Mmb))

# P1: second heater, μ = (ε, Q₁, Q₂); own reference (both heaters at box center)
U_p1 = snapshots(384, [EPS_RANGE[1], Q_RANGE[1], Q_RANGE[1]],
                 [EPS_RANGE[2], Q_RANGE[2], Q_RANGE[2]];
                 solvekw = m -> (eps_r = m[1], Q = m[2], f = source2(m[3])))
push!(results, reconstruct_cliff("P1 (ε,Q1,Q2)", U_p1,
                                 (eps_r = εc, Q = Qc, f = source2(Qc)), εc, cfg.k))

# P2: conductivity, μ = (ε, Q, k); reference at k₀ = 6 (box center of [2, 10])
U_p2 = snapshots(384, [EPS_RANGE[1], Q_RANGE[1], 2.0],
                 [EPS_RANGE[2], Q_RANGE[2], 10.0];
                 solvekw = m -> (eps_r = m[1], Q = m[2], k = m[3]))
push!(results, reconstruct_cliff("P2 (ε,Q,k)", U_p2,
                                 (eps_r = εc, Q = Qc, k = 6.0), εc, 6.0))

# P3 shrink boxes: same reference and eigenbasis as base
εh, Qh = 0.425, 350.0
for fr in (0.5, 0.25)
    Uε = snapshots(256, [εc - fr * εh, Q_RANGE[1]], [εc + fr * εh, Q_RANGE[2]];
                   solvekw = m -> (eps_r = m[1], Q = m[2]))
    push!(results, reconstruct_cliff("ε-box ×$(fr)", Uε, nothing, εc, cfg.k;
                                     V = Vb, Mm = Mmb))
    UQ = snapshots(256, [EPS_RANGE[1], Qc - fr * Qh], [EPS_RANGE[2], Qc + fr * Qh];
                   solvekw = m -> (eps_r = m[1], Q = m[2]))
    push!(results, reconstruct_cliff("Q-box ×$(fr)", UQ, nothing, εc, cfg.k;
                                     V = Vb, Mm = Mmb))
end

jldsave(joinpath(DATA_DIR, "cliff_reconstruction.jld2");
        names = [r.name for r in results], rstar = [r.rstar for r in results],
        n_excited = [r.n_excited for r in results],
        naive = [r.naive for r in results],
        recon = [r.recon for r in results],
        measured = [r.measured for r in results])

fig = Figure(size = (560, 480))
ax = Axis(fig[1, 1]; xscale = log10, yscale = log10,
          xlabel = "reconstructed from excited coefficient functions",
          ylabel = "measured POD  σ_{r*+1}/σ₁",
          title = "Cliff reconstruction across configurations")
lo = 10.0^floor(log10(minimum(r -> min(r.recon, r.measured), results)))
hi = 10.0^ceil(log10(maximum(r -> max(r.naive, r.measured), results)))
lines!(ax, [lo, hi], [lo, hi]; linestyle = :dash, color = :gray, label = "y = x")
scatter!(ax, [r.naive for r in results], [r.measured for r in results];
         markersize = 10, color = :transparent, strokecolor = :gray,
         strokewidth = 1.5, label = "naive: leakage × total amplitude")
scatter!(ax, [r.recon for r in results], [r.measured for r in results];
         markersize = 12, color = :steelblue,
         label = "weighted collinearity sval")
for r in results
    text!(ax, r.recon, r.measured; text = r.name, fontsize = 10, offset = (6, 4))
end
axislegend(ax; position = :rb)
save(joinpath(FIG_DIR, "22_cliff_reconstruction.png"), fig)

println("saved data/cliff_reconstruction.jld2 and figures/22_cliff_reconstruction.png")
