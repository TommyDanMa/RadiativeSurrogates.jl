# Operator inference (non-intrusive, structured) on the transient coefficients.
#
# ============================ PRE-REGISTERED (2026-07-12) ============================
# Written BEFORE this script's first run. FROZEN references: mapper traj rel L²
# 1.737e-2 at r = 8 (data/eval_transient.jld2); exact-Galerkin dynamics floor
# 3.719e-6 and trajectory-metric projection floor 2.949e-6 at r = 8 (scripts/30,
# data/galerkin_transient.jld2); deployed whitening cscales loaded from
# data/model_pod_mlp_t_r8.jld2, never recomputed.
#
# QUESTION: how far do SNAPSHOTS ALONE go toward the latent dynamics? Fit a
# structured reduced model by least squares on finite-difference coefficient
# derivatives (classic operator inference: no assembled operators, no solver
# access; the non-intrusive counterpart of scripts/30), then roll it out on the
# held-out trajectories.
#
# STRUCTURE (registered): the reduced dynamics are μ-dependent (radiation scales
# linearly in ε; the load channel scales with Qp), so a single global fit with an
# on/off indicator would confound both. Instead:
#     c̃̇ ≈ (A₀ + ε A₁) c̃ + (H₀ + ε H₁)(c̃ ⊗ c̃)_sym + (b₀ + ε b₁) + B q̃(t),
# with q̃(t) the range-normalized physical load (structurally exact channel: q
# enters the FOM linearly) and ε-affine operators. 91 features × 8 outputs = 728
# unknowns; ridge with fixed λ = 1e-8 · tr(GᵀG)/ncols, registered, not tuned.
# Degree-2-vs-quartic truncation is the named structural gap: u|u|³ projects to a
# non-polynomial nonlinearity in c; quadratic is the standard OpInf ansatz, not a
# claim of sufficiency.
#
# DERIVATIVES (registered): central differences on the 61-point phase grid, using
# only interior points whose two-sided stencil crosses no load switch, and
# excluding j = 2 (its stencil touches the t = 0 load-on discontinuity). The
# usable-row count is printed and saved. Exact-RHS derivatives would break the
# non-intrusive claim and are not used.
#
# PRIOR (registered, LOW confidence, stated as such): mean held-out rollout traj
# rel L² lands between 10× the Galerkin floor and the mapper plateau, i.e. inside
# (3.7e-5, 1.74e-2); assessed near a coin flip whether the quadratic ansatz
# captures the quartic radiation over the visited region. Registered failure
# modes, reported as results either way: (a) finite-time escape of the quadratic
# rollout on some test μ (count reported; quadratic ODEs can blow up where the
# true dynamics do not); (b) landing ABOVE the mapper (structure worse than
# black-box regression).
#
# PROTOCOL: fit on the 77 training trajectories (whitened coefficients, deployed
# cscales); evaluate by rollout on the 19 held-out trajectories, FBDF, tstops at
# the switches, saveat = the phase grid, abstol = reltol = 1e-9, retcode-guarded.
# Deterministic: no RNG. Artifact: data/opinf_transient.jld2; report §8.5 carries
# these rows when the learned-dynamics section lands (with scripts/32); until then
# this header and the artifact are the record.
#
# RESULT (run 2026-07-12, appended post-run; registered content above unchanged):
#   Fit: 3,850 usable rows × 91 features, λ = 9.93e-6. Rollout on the 19 held-out
#   trajectories: 18 escape in finite time (dt → 0 aborts, mid-trajectory,
#   t ≈ 0.4–6.4 ks); 1 survives, at 9.20e-4. Registered failure mode (a) is the
#   headline result. The survivor sits inside the prior band, but N = 1 makes the
#   band statement vacuous and nothing is claimed from it. Candidate mechanisms,
#   consistent with the signature and not distinguished here: (i) quadratic
#   extrapolation outside the fitted neighborhood without the contraction of the
#   truncated u|u|³ term; (ii) the registered stencil exclusions leave post-switch
#   fast content under-constrained, exactly where rollouts must cross; (iii) the
#   known rollout-stability pathology of inferred quadratic models. Consequence:
#   the non-intrusive degree-2 route needs stability-constrained variants to be
#   usable here (future work, not re-fit post hoc); the learned-f arm (scripts/32)
#   carries the exit. §8.5 reports this row as 18/19 escapes + the survivor value.
# =====================================================================================

include(joinpath(@__DIR__, "config.jl"))
using OrdinaryDiffEq

println("=== 31_operator_inference ===")

dt_ = load(snapshots_t_path())
pod = load(pod_t_path())
P = PODBasis(pod["mean"], pod["modes"], pod["svals"])
U3, mus = dt_["U"], dt_["mus"]
tr, te = dt_["train_idx"], dt_["test_idx"]
n, nt, _ = size(U3)
r = DEFAULT_R_T
cs = load_mlp(model_path("pod_mlp_t_r$(r)")).cscales
qmax = Q_RANGE[2]

pairs = [(i, j) for i in 1:r for j in i:r]   # 36 symmetric quadratic features

function feat(c̃, ε, q̃)
    quad = [c̃[i] * c̃[j] for (i, j) in pairs]
    return vcat(c̃, ε .* c̃, quad, ε .* quad, one(eltype(c̃)), ε, q̃)
end
nfeat = 4r + 2 * length(pairs) + 3 - 2r      # 2r + 2*36 + 3 = 91 for r = 8

# ----- training rows from FD derivatives ----------------------------------------------
rows_G = Vector{Vector{Float64}}()
rows_Y = Vector{Vector{Float64}}()
for k in tr
    ε, Qp, duty, period = mus[:, k]
    T_end = NCYCLES * period
    ts = range(0.0, T_end; length = nt)
    Δt = step(ts)
    sw = load_switch_times(period, duty, 0.0, T_end)
    C̃ = Matrix(project(P, U3[:, :, k], r)) ./ cs
    q = square_wave(Qp, period, duty)
    for j in 3:(nt - 1)
        any(s -> ts[j - 1] < s < ts[j + 1], sw) && continue
        push!(rows_G, feat(C̃[:, j], ε, q(ts[j]) / qmax))
        push!(rows_Y, (C̃[:, j + 1] .- C̃[:, j - 1]) ./ (2Δt))
    end
end
Gm = Matrix(reduce(hcat, rows_G)')            # N × 91
Ym = Matrix(reduce(hcat, rows_Y)')            # N × 8
lam = 1e-8 * LinearAlgebra.tr(Gm' * Gm) / size(Gm, 2)
Θ = (Gm' * Gm + lam * I) \ (Gm' * Ym)         # 91 × 8
@printf("fit: %d usable rows × %d features, ridge λ = %.3e\n", size(Gm, 1), size(Gm, 2), lam)

# ----- held-out rollout -----------------------------------------------------------------
traj_rel(Û, Uk) = norm(Û - Uk) / norm(Uk)

function opinf_rollout(k)
    ε, Qp, duty, period = mus[:, k]
    T_end = NCYCLES * period
    ts = collect(range(0.0, T_end; length = nt))
    q = square_wave(Qp, period, duty)
    f! = (dc, c, p, t) -> (dc .= Θ' * feat(c, ε, q(t) / qmax); nothing)
    c̃0 = vec(project(P, U3[:, 1, k], r)) ./ cs
    prob = ODEProblem(f!, c̃0, (0.0, T_end))
    sol = OrdinaryDiffEq.solve(prob, FBDF(); saveat = ts,
                               tstops = load_switch_times(period, duty, 0.0, T_end),
                               abstol = 1e-9, reltol = 1e-9)
    ok = OrdinaryDiffEq.SciMLBase.successful_retcode(sol) && size(Array(sol), 2) == nt
    return ok ? Matrix(reconstruct(P, cs .* Array(sol), r)) : nothing
end

rels = Float64[]
nblow = 0
for k in te
    Û = opinf_rollout(k)
    if Û === nothing
        global nblow += 1
        @printf("  k=%d: rollout failed (recorded as failure)\n", k)
        continue
    end
    push!(rels, traj_rel(Û, U3[:, :, k]))
end

mean_rel = isempty(rels) ? NaN : mean(rels)
@printf("\nOpInf rollout (N = %d held-out, %d failures): mean traj rel L² %.3e ",
        length(te), nblow, mean_rel)
@printf("[min %.2e, max %.2e]\n", isempty(rels) ? NaN : minimum(rels),
        isempty(rels) ? NaN : maximum(rels))
@printf("context: Galerkin floor 3.719e-6 | mapper 1.737e-2 | prior band (3.7e-5, 1.74e-2)\n")
in_band = 3.7e-5 < mean_rel < 1.74e-2
println("registered prior (low confidence): " *
        (nblow > 0 ? "failure mode (a) observed; " : "") *
        (in_band ? "CONSISTENT (inside the band)" : "INCONSISTENT (outside the band)"))

jldsave(joinpath(DATA_DIR, "opinf_transient.jld2");
        theta = Θ, lambda = lam, nrows = size(Gm, 1), nfeat = size(Gm, 2),
        pairs = collect.(pairs), rels, mean_rel, nblow, ntest = length(te),
        qmax, abstol = 1e-9, reltol = 1e-9)
println("\nsaved data/opinf_transient.jld2")
