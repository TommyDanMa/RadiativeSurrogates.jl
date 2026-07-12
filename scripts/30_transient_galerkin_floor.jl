# POD-Galerkin transient ROM, nonlinearity assembled exactly (no hyper-reduction):
# the floor under the latent-ODE exit.
#
# ============================ PRE-REGISTERED (2026-07-12) ============================
# Written BEFORE this script's first run. Reference values below are FROZEN numbers:
# the mapper's traj rel L² sweep (data/eval_transient.jld2; 1.74e-2 at the deployed
# r = 8), the per-column projection floors 2.5e-4 / 2.1e-6 / 1.1e-8 / 1.2e-10 at
# r = 4/8/16/32 (data/pod_transient.jld2), and the fact that the FOM dataset was
# integrated at abstol = reltol = 1e-7 (scripts/08). Quoted as thresholds, not
# recomputed.
#
# QUESTION: §8.3 licenses latent dynamics as the transient exit. Before anything is
# learned: how well do the Galerkin reduced dynamics track the FOM? Integrate the
# POD-Galerkin ROM
#     Mr ċ = −Φᵀ R_space(ū + Φc) + q(t) · Φᵀ F_unit,   Mr = Φᵀ M Φ,
# per held-out trajectory from the projected data IC, and compare to the FOM
# trajectory in the mapper's own metric (per-trajectory Frobenius rel L²).
# Terminology, to keep the claim scoped: "exactly" qualifies the ASSEMBLY (projected
# residual evaluated exactly at ū + Φc, no DEIM), not the dynamics. Evaluating R at
# ū + Φc instead of at the true full state IS the closure approximation under test;
# this is plain POD-Galerkin on span(Φ), not an inertial-manifold construction. The
# projection floor is recomputed IN THE TRAJECTORY METRIC here; the §8.2 floor
# column is per-column and is carried only as context (like compares with like).
#
# PREDICTIONS (scored after the run; misses are results):
#   P-G1 (closure adequacy): mean Galerkin traj rel L² at r = 8 lands within 10× of
#        the trajectory-metric projection floor at r = 8.
#   P-G2 (rank tracking): the Galerkin error decreases monotonically over r = 4 → 8
#        → 16 and stays within 30× of the trajectory-metric projection floor at
#        r ∈ {4, 8}. FENCE: at r = 16 and 32 the readout is expected to saturate at
#        the dataset's own integration error (dataset tol 1e-7); saturation there is
#        NOT evidence that the basis binds.
#   P-G3 (headroom): Galerkin error at r = 8 ≤ 1.74e-4 (mapper 1.74e-2 / 100).
#        Threshold rationale, priced before running: 100× separation means closure
#        contributes ≤ 1% of the mapper's error budget, the margin at which "the
#        plateau is map error" is a safe attribution. Known non-closure contributors
#        sit well below the threshold: projection floor ~2e-6 in this metric (~90×
#        under), dataset integration error ≤ ~1e-6 at its tol 1e-7 (~170× under),
#        this script's own tol 1e-9 (negligible), IC projection exact by
#        construction. A miss can therefore only be produced by closure itself.
#        P-G3 is deliberately weaker than P-G1: closure error in (~2e-5, 1.74e-4)
#        misses P-G1 while hitting P-G3, and that split is itself informative.
#   Registered failure reading: a Galerkin-to-projection gap that persists or grows
#   with r is EVIDENCE AGAINST the Markovian-closure hypothesis in the resolved
#   coordinates, not proof of memory. Candidate mechanisms this experiment does not
#   distinguish: truncated-mode coupling (memory; the leading candidate), the
#   Euclidean-POD vs M-inner-product projection mismatch (§11's mass-weighted POD),
#   and the reference data's own integration error (same signature: the projection
#   floor drops with r while a fixed noise floor does not; hence the r = 16/32
#   fence). §8.4 reports the signature and the candidates, not a mechanism verdict;
#   if the evidence goes against Markovian closure, the latent-ODE target changes
#   (augmented state / memory terms).
#
# Also measured, registered as measurements (no bets): a stiffness audit
# (eigenvalues of −Mr⁻¹ ΦᵀJΦ at the 61 saveat states of one test trajectory per
# rank; decides the scripts/32 training integrator and gates its no-adjoint
# fallback); solver receipts (sol.stats per rank); a tolerance spot check
# (2 trajectories, 1e-9 vs 1e-10).
#
# PROTOCOL: 19 held-out trajectories × r ∈ (4, 8, 16, 32); FBDF, tstops at the load
# switches, saveat = the 61-point phase grid, abstol = reltol = 1e-9; analytic
# reduced Jacobian −Mr⁻¹(ΦᵀJΦ) supplied (dense r×r prototype; ForwardDiff Duals must
# never reach Gridap assembly); Mr Cholesky-factorized once per rank; QoI tracking
# (peak-T, P_rad) at the deployed r = 8. Deterministic: no RNG anywhere.
# Artifacts: data/galerkin_transient.jld2, figures/29_transient_galerkin_floor.png;
# printed by the §8.4 block of scripts/18_report_numbers.jl.
#
# RESULT (run 2026-07-12, appended post-run; registered content above unchanged):
#   r= 4: Galerkin 3.902e-4   proj floor (traj) 3.307e-4   mapper 1.686e-2
#   r= 8: Galerkin 3.719e-6   proj floor (traj) 2.949e-6   mapper 1.737e-2
#   r=16: Galerkin 3.471e-7   proj floor (traj) 1.487e-8   (saturated; the fence)
#   r=32: Galerkin 3.449e-7   proj floor (traj) 2.659e-10  (saturated; the fence)
#   P-G1 HIT: 1.26× the projection floor at r = 8. P-G2 HIT: monotone 4→8→16,
#   1.2×/1.3× of floor at r = 4/8. P-G3 HIT: headroom 4,670×. The r = 16/32
#   readouts saturate at 3.5e-7, identical at both ranks: the dataset's own
#   integration error, exactly the registered fence; not closure, not the basis.
#   QoI floor at r = 8: peak-T err median 0.005 K, max 0.021 K; P_rad ≤ 1.9e-4.
#   Stiffness (r = 8 audit): |Re λ| ∈ [4.31e-5, 8.15e-2] 1/s, ratio 1.9e3, ~3.9e2
#   per period; scripts/32 trains with an implicit integrator (the explicit
#   unrolled-RK4 contingency would need ~8-16 substeps per phase step: viable,
#   second choice). Tolerance check 1e-9 vs 1e-10: max Δ 2.3e-12. 19/19
#   trajectories converged at every rank. All three predictions hit; the §8.2
#   plateau is 99.98% coefficient-map error at the deployed rank.
# =====================================================================================

include(joinpath(@__DIR__, "config.jl"))
using Gridap
using OrdinaryDiffEq
using CairoMakie
CairoMakie.activate!()

println("=== 30_transient_galerkin_floor ===")

fom = build_steady_fom(CFG)
dt_ = load(snapshots_t_path())
pod = load(pod_t_path())
P = PODBasis(pod["mean"], pod["modes"], pod["svals"])
U3, mus, te = dt_["U"], dt_["mus"], dt_["test_idx"]
n, nt, _ = size(U3)
ntest = length(te)
et = load(eval_t_path())

ranks = collect(RANKS_T)
ū = P.mean
M = mass_matrix(fom)
Funit = load_vector(fom, 1.0)
Φs = [Matrix(modes_r(P, r)) for r in ranks]
Mrchs = [cholesky(Symmetric(Φ' * (M * Φ))) for Φ in Φs]
Frs = [Φ' * Funit for Φ in Φs]

traj_rel(Û, Uk) = norm(Û - Uk) / norm(Uk)

"Integrate the exact reduced dynamics for test trajectory k at rank index i."
function galerkin_traj(i, k; abstol = 1e-9, reltol = 1e-9)
    Φ, Mrch, Fr = Φs[i], Mrchs[i], Frs[i]
    r = ranks[i]
    ε, Qp, duty, period = mus[:, k]
    q = square_wave(Qp, period, duty)
    T_end = NCYCLES * period
    ts = collect(range(0.0, T_end; length = nt))
    aop = Gridap.FESpaces.get_algebraic_operator(
        FEOperator(steady_form(fom; eps_r = ε, Q = 0.0), fom.U, fom.V))
    x = similar(ū)
    rb = similar(ū)
    Jfull = Gridap.Algebra.allocate_jacobian(aop, collect(U3[:, 1, k]))
    f! = (dc, c, p, t) -> begin
        mul!(x, Φ, c)
        x .+= ū
        Gridap.Algebra.residual!(rb, aop, x)
        dc .= Mrch \ (q(t) .* Fr .- Φ' * rb)
        return nothing
    end
    jac! = (Jr, c, p, t) -> begin
        mul!(x, Φ, c)
        x .+= ū
        Gridap.Algebra.jacobian!(Jfull, aop, x)
        Jr .= .-(Mrch \ (Φ' * (Jfull * Φ)))
        return nothing
    end
    c0 = Φ' * (U3[:, 1, k] .- ū)
    fode = ODEFunction(f!; jac = jac!, jac_prototype = zeros(r, r))
    prob = ODEProblem(fode, c0, (0.0, T_end))
    return OrdinaryDiffEq.solve(prob, FBDF(); saveat = ts,
                                tstops = load_switch_times(period, duty, 0.0, T_end),
                                abstol, reltol)
end

# ----- rank sweep over the held-out trajectories -------------------------------------
gal_rel = Float64[]
proj_floor_traj = Float64[]
nfail = zeros(Int, length(ranks))
stats_nf = zeros(Int, length(ranks))
stats_njacs = zeros(Int, length(ranks))
stats_naccept = zeros(Int, length(ranks))
Ûr8 = Dict{Int, Matrix{Float64}}()
for (i, r) in enumerate(ranks)
    rels = Float64[]
    pfs = Float64[]
    t0 = time()
    for k in te
        Uk = U3[:, :, k]
        sol = galerkin_traj(i, k)
        stats_nf[i] += sol.stats.nf
        stats_njacs[i] += sol.stats.njacs
        stats_naccept[i] += sol.stats.naccept
        if !OrdinaryDiffEq.SciMLBase.successful_retcode(sol) || size(Array(sol), 2) != nt
            nfail[i] += 1
            @printf("  r=%d k=%d: retcode %s (recorded as failure)\n",
                    r, k, string(sol.retcode))
            continue
        end
        Ĉ = Array(sol)
        Û = Matrix(reconstruct(P, Ĉ, r))
        r == DEFAULT_R_T && (Ûr8[k] = Û)
        push!(rels, traj_rel(Û, Uk))
        push!(pfs, traj_rel(Matrix(reconstruct(P, Matrix(project(P, Uk, r)), r)), Uk))
    end
    push!(gal_rel, isempty(rels) ? NaN : mean(rels))
    push!(proj_floor_traj, isempty(pfs) ? NaN : mean(pfs))
    @printf("r=%2d: Galerkin %.3e   proj floor (traj) %.3e   mapper %.3e   (%d/%d ok, %.0f s)\n",
            r, gal_rel[i], proj_floor_traj[i], et["sweep_rel"][i],
            ntest - nfail[i], ntest, time() - t0)
end

# ----- QoI tracking at the deployed rank ----------------------------------------------
dTpk = Float64[]
dPq = Float64[]
for k in te
    haskey(Ûr8, k) || continue
    Û = Ûr8[k]
    Uk = U3[:, :, k]
    ε = mus[1, k]
    pk̂ = vec(maximum(Û; dims = 1))
    pk = vec(maximum(Uk; dims = 1))
    push!(dTpk, maximum(abs.(pk̂ .- pk)))
    P̂r = [radiated_power(fom, FEFunction(fom.Un, Û[:, j]), ε) for j in 1:nt]
    Pr = [radiated_power(fom, FEFunction(fom.Un, Uk[:, j]), ε) for j in 1:nt]
    push!(dPq, maximum(abs.(P̂r .- Pr) ./ max.(abs.(Pr), 1.0)))
end
@printf("r=%d QoI: peak-T err median %.3f K max %.3f K; P_rad median %.2e max %.2e\n",
        DEFAULT_R_T, median(dTpk), maximum(dTpk), median(dPq), maximum(dPq))

# ----- stiffness audit along one test trajectory --------------------------------------
kaud = te[1]
εa, Qpa, dutya, perioda = mus[:, kaud]
aopa = Gridap.FESpaces.get_algebraic_operator(
    FEOperator(steady_form(fom; eps_r = εa, Q = 0.0), fom.U, fom.V))
stiff_max = zeros(length(ranks))
stiff_min = zeros(length(ranks))
stiff_r8_re = zeros(DEFAULT_R_T, nt)
stiff_r8_im = zeros(DEFAULT_R_T, nt)
for (i, r) in enumerate(ranks)
    Φ, Mrch = Φs[i], Mrchs[i]
    Jfull = Gridap.Algebra.allocate_jacobian(aopa, collect(U3[:, 1, kaud]))
    mx = -Inf
    mn = Inf
    for j in 1:nt
        Gridap.Algebra.jacobian!(Jfull, aopa, collect(U3[:, j, kaud]))
        A = -(Mrch \ (Φ' * (Jfull * Φ)))
        λ = eigvals(Matrix(A))
        re = abs.(real.(λ))
        mx = max(mx, maximum(re))
        mn = min(mn, minimum(re))
        if r == DEFAULT_R_T
            stiff_r8_re[:, j] = real.(λ)
            stiff_r8_im[:, j] = imag.(λ)
        end
    end
    stiff_max[i] = mx
    stiff_min[i] = mn
    @printf("stiffness r=%2d: |Re λ| ∈ [%.3e, %.3e] 1/s, ratio %.1e, per-period %.2e\n",
            r, mn, mx, mx / max(mn, 1e-300), mx * perioda)
end

# ----- tolerance spot check ------------------------------------------------------------
i8 = findfirst(==(DEFAULT_R_T), ranks)
tolcheck = Float64[]
for k in te[1:2]
    sol10 = galerkin_traj(i8, k; abstol = 1e-10, reltol = 1e-10)
    Û10 = Matrix(reconstruct(P, Array(sol10), DEFAULT_R_T))
    push!(tolcheck, abs(traj_rel(Û10, U3[:, :, k]) - traj_rel(Ûr8[k], U3[:, :, k])))
end
@printf("tolerance check (1e-9 vs 1e-10, 2 trajectories): max Δ traj rel %.2e\n",
        maximum(tolcheck))

# ----- registered predictions, scored --------------------------------------------------
i4 = findfirst(==(4), ranks)
i16 = findfirst(==(16), ranks)
pg1 = gal_rel[i8] <= 10 * proj_floor_traj[i8]
pg2 = (gal_rel[i4] > gal_rel[i8] > gal_rel[i16]) &&
      gal_rel[i4] <= 30 * proj_floor_traj[i4] &&
      gal_rel[i8] <= 30 * proj_floor_traj[i8]
pg3 = gal_rel[i8] <= 1.74e-2 / 100
println("\n=== registered predictions, scored ===")
@printf("P-G1 %s: Galerkin(r=8) %.3e vs 10× proj floor %.3e\n",
        pg1 ? "HIT " : "MISS", gal_rel[i8], 10 * proj_floor_traj[i8])
@printf("P-G2 %s: monotone 4→8→16 %s; within 30× at r=4 (%.1f×), r=8 (%.1f×)\n",
        pg2 ? "HIT " : "MISS",
        (gal_rel[i4] > gal_rel[i8] > gal_rel[i16]) ? "yes" : "no",
        gal_rel[i4] / proj_floor_traj[i4], gal_rel[i8] / proj_floor_traj[i8])
@printf("P-G3 %s: Galerkin(r=8) %.3e vs mapper/100 = 1.74e-4 (headroom %.0f×)\n",
        pg3 ? "HIT " : "MISS", gal_rel[i8], et["sweep_rel"][i8] / gal_rel[i8])

jldsave(joinpath(DATA_DIR, "galerkin_transient.jld2");
        ranks, gal_rel, proj_floor_traj, mapper_rel = collect(et["sweep_rel"]),
        recon_test_ctx = [pod["recon_test"][r] for r in ranks],
        dTpk, dPq, nfail, ntest,
        stats_nf, stats_njacs, stats_naccept,
        stiff_max, stiff_min, stiff_r8_re, stiff_r8_im,
        audit_traj = kaud, audit_period = perioda,
        stiff_per_period = stiff_max[i8] * perioda,
        tolcheck_max = maximum(tolcheck), abstol = 1e-9, reltol = 1e-9,
        scores = Dict("P-G1" => pg1, "P-G2" => pg2, "P-G3" => pg3))

# ----- figure ---------------------------------------------------------------------------
fig = Figure(size = (940, 400))
ax = Axis(fig[1, 1]; xscale = log2, yscale = log10, xlabel = "rank r",
          ylabel = "traj rel L² (mean, N = $(ntest))",
          title = "Exact reduced dynamics vs mapper vs projection",
          xticks = (ranks, string.(ranks)))
scatterlines!(ax, ranks, collect(et["sweep_rel"]); label = "(μ, t) mapper")
scatterlines!(ax, ranks, gal_rel; label = "Galerkin (exact dynamics)")
scatterlines!(ax, ranks, proj_floor_traj; label = "projection floor (traj metric)")
axislegend(ax; position = :rt)
ax2 = Axis(fig[1, 2]; yscale = log10, xlabel = "saved state index",
           ylabel = "|Re λ| (1/s)",
           title = "Reduced spectrum along audit trajectory (r = 8)")
for m in 1:DEFAULT_R_T
    scatter!(ax2, 1:nt, max.(abs.(stiff_r8_re[m, :]), 1e-12); markersize = 3)
end
save(joinpath(FIG_DIR, "29_transient_galerkin_floor.png"), fig)

println("\nsaved data/galerkin_transient.jld2 and figures/29_transient_galerkin_floor.png")
