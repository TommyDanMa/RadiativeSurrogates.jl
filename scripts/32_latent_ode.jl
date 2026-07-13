# Latent dynamics through the solver: the learned-f arm of the transient exit.
#
# ============================ PRE-REGISTERED (2026-07-12) ============================
# Written BEFORE this script's first run. FROZEN references: mapper traj rel L²
# 1.737e-2 at r = 8 and 5.2e-2 at 24 training trajectories (data/eval_transient.jld2,
# data/mapper_diagnostics.jld2); exact-Galerkin floor 3.719e-6, projection floor
# 2.949e-6, reduced spectrum |Re λ| ∈ [4.31e-5, 8.15e-2] 1/s ≈ 3.9e2 per period
# (scripts/30); OpInf arm 18/19 rollout escapes (scripts/31); frozen FOM references
# reused, not re-run: attractor traces (data/attractor_check.jld2, 20 cycles, 2
# points) and design-study QoI grids (data/design_study.jld2, 2 operating points).
# Deployed whitening cscales from data/model_pod_mlp_t_r8.jld2.
#
# MODEL (registered): f_θ = make_mlp(13, 8; hidden = 64, depth = 2), 13→64→64→8,
# 5,576 params. Inputs [c̃; μ̃; w]: whitened coefficients, the 4 standardized
# parameters, and w ∈ {0, 1} the load indicator, constant per solve call. Time is
# phase time s = t/period (tspan (0, NCYCLES), switches at integers k and k + duty);
# f_θ outputs dc̃/ds. Rationale: one tspan across the 6× period spread; period stays
# an input through μ̃.
#
# TRAINING (registered):
#   Stage A (collocation): targets are EXACT projected derivatives at the snapshot
#   states, ċ = Φᵀ M⁻¹(−R(u_j) + q(t_j) F_unit), whitened and converted to phase
#   time; full-batch train! (3,000 epochs, lr 1e-3 → 1e-5, keep-best). The Stage-A
#   model is checkpointed and evaluated as its own baseline, reported regardless.
#   Stage B (through-solver): multiple shooting, segment boundaries on the saveat
#   grid, teacher-forced segment ICs from projected data, growing horizon
#   G = 5 → 11 → 21 → 61 points with 60/40/30/20 epochs and Adam 3e-4 → 1e-5
#   geometric over the 150 Stage-B epochs, keep-best on the epoch loss. A segment
#   crossing a switch chains sub-solves with w constant per call (switch times are
#   θ-independent: no event sensitivity). Training solver FBDF at tol 1e-6 (the
#   scripts/30 stiffness audit rules out cheap explicit stepping); eval solver FBDF
#   at tol 1e-9. Sensitivity: SciMLSensitivity GaussAdjoint(autojacvec = ZygoteVJP())
#   primary, InterpolatingAdjoint(ZygoteVJP()) the registered fallback if the
#   primary errors on a probe gradient (whichever runs is recorded in the artifact).
#   Seeds: deployed StableRNG(SEED+3100); ensemble SEED+3200+s varies INIT ONLY.
#   ENSEMBLE RULE (registered, decided by clock, not results): N = 20 if the
#   deployed member's total train time ≤ 30 min wall, else N = 5.
#
# PREDICTIONS (scored post-run; misses are results):
#   L1 (beats the regression): deployed rollout traj rel L² on the 19 held-out
#       trajectories < 1.737e-2.
#   L2 (data efficiency, the mechanism bet): retrained on ttr[1:24] with cscales
#       recomputed from that subset, mirroring the scripts/17 recipe exactly
#       (shared basis, subset scales, same seed and budgets as the deployed
#       member): held-out traj rel L² < 5.2e-2, the mapper's measured 24-trajectory
#       point. Rationale: 61 states per trajectory constrain f locally, so the
#       dynamics factorization converts the same data better than trajectory-level
#       regression.
#   L3 (rollout stability, what scripts/31 lacked): 20-cycle rollout at the two
#       frozen attractor points, ICs = projected steady states at cycle-mean load;
#       settled-cycle deviation against the frozen FOM traces flat within 2× dev(3)
#       for cycles ≥ 3, mirroring A.4's registered criteria. τ > 1 is in-scope for
#       a latent ODE, unlike the mapper's phase features.
#   L4 (deployed use case): drop-in for the design study at both frozen operating
#       points against the frozen FOM QoI grids: candidate ICs from the steady
#       POD-MLP field projected onto the transient basis (cycle-mean loads sit
#       inside the steady training box, so the arm stays FOM-free): Spearman
#       ≥ 0.998, top-5 ≥ 4/5, same optimum, at both points. Disclosed sensitivity
#       check: the same grids with FOM steady-solve ICs.
#
# Deterministic: StableRNGs only. Artifacts: data/latent_ode.jld2 and
# data/model_latent_ode_r8.jld2 (ps stored as a plain NamedTuple; extras record the
# time convention, G schedule, solver, sensealg, seeds);
# figures/31_latent_ode.png. Report §8.5 and its printer block land with these
# results (commit C2).
#
# AMENDMENT (2026-07-12, appended before any results existed; registered content
#   above unchanged). Rounds 1-2 aborted on training infrastructure: with the
#   registered Adam 3e-4, the first Stage-B steps push the learned vector field into
#   mass segment-solver failure, the loss degenerates to the failure penalty, and
#   the gradient is correctly `nothing` (round 1 froze silently on this; round 2's
#   guard stopped loudly, first-epoch |g| = 17.7 healthy). Amended: Stage-B
#   optimiser is ClipNorm(1.0) ∘ Adam(3e-5 → 1e-6 geometric), same epoch counts;
#   the per-epoch failed-segment count is logged. Round 3: a NaN gradient reached
#   the optimiser through segments with finite-but-exploding states; amended
#   further: segment outputs that are non-finite or exceed |c̃| = 1e3 count as
#   failed (flat penalty), and epochs with non-finite gradient norm skip their
#   update (counted, logged). No prediction, threshold, or evaluation rule is
#   touched.
# =====================================================================================

include(joinpath(@__DIR__, "config.jl"))
using Gridap
using OrdinaryDiffEq
using SciMLSensitivity
using ComponentArrays
using Optimisers
using Zygote
using Lux
using CairoMakie
CairoMakie.activate!()

println("=== 32_latent_ode ===")

fom = build_steady_fom(CFG)
dt_ = load(snapshots_t_path())
podt = load(pod_t_path())
P = PODBasis(podt["mean"], podt["modes"], podt["svals"])
U3, mus = dt_["U"], dt_["mus"]
ttr, tte = dt_["train_idx"], dt_["test_idx"]
n, nt, _ = size(U3)
r = DEFAULT_R_T
cs_dep = load_mlp(model_path("pod_mlp_t_r$(r)")).cscales
Φ = Matrix(modes_r(P, r))
ū = P.mean
sgrid = collect(range(0.0, Float64(NCYCLES); length = nt))

GS = (5, 11, 21, 61)
EPOCHS_B = (60, 40, 30, 20)
TOL_TRAIN = 1e-6
TOL_EVAL = 1e-9

# ----- Stage-A data: exact projected derivatives --------------------------------------
M = mass_matrix(fom)
Z = cholesky(Symmetric(M)) \ Φ                      # M⁻¹Φ, n × r dense
Funit = load_vector(fom, 1.0)

"Features X (13×nt), phase-time whitened derivative targets Y (r×nt), whitened
coefficients C̃ (r×nt) for trajectory k under whitening scales csv."
function stage_a_data(k, csv)
    ε, Qp, duty, period = mus[:, k]
    μst = collect(standardize_mu_t(ε, Qp, duty, period))
    q = square_wave(Qp, period, duty)
    aop = Gridap.FESpaces.get_algebraic_operator(
        FEOperator(steady_form(fom; eps_r = ε, Q = 0.0), fom.U, fom.V))
    rb = similar(ū)
    X = zeros(13, nt)
    Y = zeros(r, nt)
    C̃ = Matrix(project(P, U3[:, :, k], r)) ./ csv
    for j in 1:nt
        tj = sgrid[j] * period
        Gridap.Algebra.residual!(rb, aop, collect(U3[:, j, k]))
        ċ = Z' * (q(tj) .* Funit .- rb)
        X[:, j] = vcat(C̃[:, j], μst, q(tj) > 0 ? 1.0 : 0.0)
        Y[:, j] = (ċ ./ csv) .* period
    end
    return (; X, Y, C̃, μst, duty, period)
end

println("assembling exact Stage-A derivative targets...")
dat_dep = Dict(k => stage_a_data(k, cs_dep) for k in vcat(ttr, tte))

# ----- shooting plumbing ---------------------------------------------------------------
load_on(s, duty) = mod(s, 1.0) < duty ? 1.0 : 0.0

struct SubPlan
    a::Float64
    b::Float64
    w::Float64
    sv::Vector{Float64}      # saveat for this sub-solve (always ends at b)
    keep::Vector{Int}        # indices of sv that are target points
end

"w-constant sub-solve plan covering (s0, s1) with the given target points."
function build_chain(s0, s1, duty, targets)
    bs = [s0]
    for k0 in floor(Int, s0):ceil(Int, s1)
        for sw in (Float64(k0), k0 + duty)
            s0 < sw < s1 && push!(bs, sw)
        end
    end
    push!(bs, s1)
    bs = sort(unique(bs))
    subs = SubPlan[]
    for i in 1:(length(bs) - 1)
        a, b = bs[i], bs[i + 1]
        inwin = [s for s in targets if a < s <= b]
        sv = sort(unique(vcat(inwin, [b])))
        keep = findall(s -> any(t -> t == s, inwin), sv)
        push!(subs, SubPlan(a, b, load_on((a + b) / 2, duty), sv, keep))
    end
    return subs
end

seg_f(model, st, μst, w) =
    (c, p, s) -> first(Lux.apply(model, vcat(c, μst, w), p, st))

"Execute a sub-solve chain; returns r × n_targets matrix (AD-safe, no mutation)."
function exec_chain(model, st, p, μst, c0, subs; abstol, reltol, sensealg = nothing)
    c = c0
    blocks = ()
    for sub in subs
        f = seg_f(model, st, μst, sub.w)
        prob = ODEProblem(f, c, (sub.a, sub.b), p)
        sol = sensealg === nothing ?
              OrdinaryDiffEq.solve(prob, FBDF(); saveat = sub.sv, abstol, reltol,
                                   save_start = false) :
              OrdinaryDiffEq.solve(prob, FBDF(); saveat = sub.sv, abstol, reltol,
                                   save_start = false, sensealg)
        A = Array(sol)
        if !OrdinaryDiffEq.SciMLBase.successful_retcode(sol) ||
           size(A, 2) != length(sub.sv)
            return nothing
        end
        c = A[:, end]
        isempty(sub.keep) || (blocks = (blocks..., A[:, sub.keep]))
    end
    return reduce(hcat, blocks)
end

struct SegPlan
    k::Int
    j0::Int
    j1::Int
    subs::Vector{SubPlan}
end

build_plans(dat, ks, G) = [SegPlan(k, j0, j0 + G - 1,
                                   build_chain(sgrid[j0], sgrid[j0 + G - 1],
                                               dat[k].duty,
                                               sgrid[(j0 + 1):(j0 + G - 1)]))
                           for k in ks for j0 in 1:(G - 1):(nt - G + 1)]

function epoch_loss(model, st, θ, dat, plans, sensealg, failct = Ref(0))
    tot = 0.0
    npt = 0
    for pl in plans
        d = dat[pl.k]
        Ŷ = exec_chain(model, st, θ, d.μst, d.C̃[:, pl.j0], pl.subs;
                       abstol = TOL_TRAIN, reltol = TOL_TRAIN, sensealg)
        bad = Ŷ === nothing || any(x -> !isfinite(x) || abs(x) > 1.0e3, Ŷ)
        if bad
            Zygote.ignore() do
                failct[] += 1
            end
            tot = tot + 1.0e3        # failed/escaped segment: flat penalty, no gradient
        else
            tot = tot + sum(abs2, Ŷ .- d.C̃[:, (pl.j0 + 1):pl.j1])
        end
        npt += pl.j1 - pl.j0
    end
    return tot / npt
end

# ----- training pipeline ---------------------------------------------------------------
function stage_b(model, st, psA, dat, ks, sensealg; verbose = true)
    psc = ComponentArray(psA)
    psc0 = copy(psc)
    opt = Optimisers.setup(Optimisers.OptimiserChain(Optimisers.ClipNorm(1.0),
                                                     Optimisers.Adam(3e-5)), psc)
    total = sum(EPOCHS_B)
    γ = (1e-6 / 3e-5)^(1 / (total - 1))
    lr = 3e-5
    best_loss, best_ps = Inf, copy(psc)
    losses = Float64[]
    nonfinite = 0
    for (round, G) in enumerate(GS)
        plans = build_plans(dat, ks, G)
        for ep in 1:EPOCHS_B[round]
            failct = Ref(0)
            res = Zygote.withgradient(θ -> epoch_loss(model, st, θ, dat, plans,
                                                      sensealg, failct), psc)
            gr = res.grad[1]
            gr === nothing &&
                error("no gradient: $(failct[])/$(length(plans)) segments failed; " *
                      "loss is degenerate (lower lr or tighten TOL_TRAIN)")
            round == 1 && ep == 1 &&
                @printf("  first gradient: |g| = %.3e\n", norm(gr))
            push!(losses, res.val)
            if res.val < best_loss
                best_loss, best_ps = res.val, copy(psc)
            end
            if !isfinite(norm(gr))
                nonfinite += 1
                continue                 # skip the update, keep the best params
            end
            opt, psc = Optimisers.update(opt, psc, gr)
            lr *= γ
            Optimisers.adjust!(opt, lr)
            verbose && ep % 10 == 0 &&
                @printf("  G=%2d epoch %3d: loss %.3e  |g| %.2e  |Δθ| %.2e  fails %d/%d\n",
                        G, ep, res.val, norm(gr), norm(psc - psc0), failct[],
                        length(plans))
        end
    end
    nonfinite > 0 &&
        @printf("  (%d Stage-B epochs skipped on non-finite gradient)\n", nonfinite)
    return NamedTuple(best_ps), losses
end

function train_member(seed, dat, ks, sensealg)
    t0 = time()
    model, ps0, st = build_pod_mlp(r; indim = 13, hidden = 64, depth = 2,
                                   rng = StableRNG(seed))
    XA = reduce(hcat, (dat[k].X for k in ks))
    YA = reduce(hcat, (dat[k].Y for k in ks))
    psA, lossesA = train!(model, ps0, st, XA, YA; epochs = 3000, verbose = false)
    psB, lossesB = stage_b(model, st, psA, dat, ks, sensealg; verbose = seed == SEED + 3100)
    return (; model, st, psA, psB, lossesA, lossesB, wall = time() - t0)
end

"Held-out rollout eval: mean traj rel L² reconstructing with scales csv."
function eval_rollouts(model, st, ps, dat, csv; ks = tte)
    p = ComponentArray(ps)
    rels = Float64[]
    nfail = 0
    for k in ks
        d = dat[k]
        subs = build_chain(sgrid[1], sgrid[end], d.duty, sgrid[2:end])
        Ĉ = exec_chain(model, st, p, d.μst, d.C̃[:, 1], subs;
                       abstol = TOL_EVAL, reltol = TOL_EVAL)
        if Ĉ === nothing
            nfail += 1
            continue
        end
        Û = Matrix(reconstruct(P, csv .* hcat(d.C̃[:, 1], Ĉ), r))
        push!(rels, norm(Û - U3[:, :, k]) / norm(U3[:, :, k]))
    end
    return rels, nfail
end

# ----- sensealg probe (registered fallback path) ---------------------------------------
probe_model, probe_ps, probe_st = build_pod_mlp(r; indim = 13, hidden = 64, depth = 2,
                                                rng = StableRNG(SEED + 3100))
probe_plan = build_plans(dat_dep, ttr[1:1], 5)[1:1]
SENSE, SENSE_NAME = try
    Zygote.gradient(θ -> epoch_loss(probe_model, probe_st, θ, dat_dep, probe_plan,
                                    GaussAdjoint(autojacvec = ZygoteVJP())),
                    ComponentArray(probe_ps))
    GaussAdjoint(autojacvec = ZygoteVJP()), "GaussAdjoint(ZygoteVJP)"
catch err
    @warn "primary sensealg failed on the probe; using registered fallback" err
    InterpolatingAdjoint(autojacvec = ZygoteVJP()), "InterpolatingAdjoint(ZygoteVJP)"
end
println("sensealg: $SENSE_NAME")

# ----- deployed member -----------------------------------------------------------------
dep = train_member(SEED + 3100, dat_dep, collect(ttr), SENSE)
safemean(x) = isempty(x) ? NaN : mean(x)
relsA, failA = eval_rollouts(dep.model, dep.st, dep.psA, dat_dep, cs_dep)
rels_dep, fail_dep = eval_rollouts(dep.model, dep.st, dep.psB, dat_dep, cs_dep)
@printf("\nStage-A baseline: mean traj rel L² %.3e (%d/%d rollouts failed)\n",
        safemean(relsA), failA, length(tte))
@printf("deployed (Stage B): mean traj rel L² %.3e (%d/%d failed, train %.0f s)\n",
        safemean(rels_dep), fail_dep, length(tte), dep.wall)

# ----- ensemble (registered clock rule) -------------------------------------------------
NENS = dep.wall <= 1800 ? 20 : 5
@printf("ensemble rule: deployed wall %.0f s → N = %d (ETA ≈ %.1f h)\n",
        dep.wall, NENS, NENS * dep.wall / 3600)
ens_rels = Float64[]
ens_fails = Int[]
for s in 1:NENS
    mem = train_member(SEED + 3200 + s, dat_dep, collect(ttr), SENSE)
    mr, mf = eval_rollouts(mem.model, mem.st, mem.psB, dat_dep, cs_dep)
    push!(ens_rels, safemean(mr))
    push!(ens_fails, mf)
    @printf("  member %2d: %.3e (%d failed, %.0f s)\n", s, ens_rels[end], mf, mem.wall)
end

# ----- L2: 24-trajectory retrain, scripts/17 recipe -------------------------------------
ks24 = ttr[1:24]
C24 = Matrix(project(P, reshape(U3[:, :, ks24], n, :), r))
s24 = vec(std(C24; dims = 2))
cs24 = max.(s24, 0.01 * s24[1])
dat_24 = Dict(k => stage_a_data(k, cs24) for k in vcat(ks24, tte))
mem24 = train_member(SEED + 3100, dat_24, collect(ks24), SENSE)
rels24, fail24 = eval_rollouts(mem24.model, mem24.st, mem24.psB, dat_24, cs24)
@printf("L2 retrain (24 trajectories, subset cscales): mean traj rel L² %.3e (%d failed)\n",
        safemean(rels24), fail24)

# ----- L3: 20-cycle attractor rollout vs frozen FOM traces ------------------------------
at = load(joinpath(DATA_DIR, "attractor_check.jld2"))
NCYC_LONG = at["ncycles"]
sg20 = collect(range(0.0, Float64(NCYC_LONG); length = NCYC_LONG * NPHASE + 1))
p_dep = ComponentArray(dep.psB)
dev_lat = Vector{Vector{Float64}}()
for (i, pt) in enumerate(at["points"])
    s0 = solve_steady(fom; eps_r = pt[:eps_r], Q = pt[:duty] * pt[:Qp])
    c̃0 = (Φ' * (s0.x .- ū)) ./ cs_dep
    μst = collect(standardize_mu_t(pt[:eps_r], pt[:Qp], pt[:duty], pt[:period]))
    subs = build_chain(sg20[1], sg20[end], pt[:duty], sg20[2:end])
    Ĉ = exec_chain(dep.model, dep.st, p_dep, μst, c̃0, subs;
                   abstol = TOL_EVAL, reltol = TOL_EVAL)
    if Ĉ === nothing
        push!(dev_lat, fill(Inf, NCYC_LONG))
        @printf("L3 point %d: 20-cycle rollout FAILED\n", i)
        continue
    end
    Û = Matrix(reconstruct(P, cs_dep .* hcat(c̃0, Ĉ), r))
    pk̂ = vec(maximum(Û; dims = 1))
    lat_settled = pk̂[((NCYC_LONG - 1) * NPHASE + 1):(NCYC_LONG * NPHASE)]
    pk = at["pk"][i]
    trace(c) = pk[(c - 1) * NPHASE .+ (1:NPHASE)]
    push!(dev_lat, [maximum(abs.(trace(c) .- lat_settled)) for c in 1:NCYC_LONG])
    @printf("L3 point %d: dev(3) %.3f K, dev c3–%d ∈ [%.3f, %.3f] K\n", i,
            dev_lat[i][3], NCYC_LONG, minimum(dev_lat[i][3:end]),
            maximum(dev_lat[i][3:end]))
end
l3 = all(all(isfinite, d) && maximum(d[3:end]) <= 2 * d[3] for d in dev_lat)

# ----- L4: design-study drop-in vs frozen FOM grids -------------------------------------
ds = load(joinpath(DATA_DIR, "design_study.jld2"))
pods = load_pod()
m2 = load_mlp(model_path("pod_mlp_r$(DEFAULT_R)"))
eps_grid, duty_grid = ds["eps_grid"], ds["duty_grid"]
lastcycle = (2 * NPHASE + 1):nt
qoi(traj) = maximum(maximum(view(traj, :, j)) for j in lastcycle)
rank_of(v) = invperm(sortperm(v))
spearman(a, b) = cor(float.(rank_of(vec(a))), float.(rank_of(vec(b))))

"Latent-ODE QoI grid at one operating point; ICs from `mode` ∈ (:mlp, :fom)."
function latent_grid(op, mode)
    Q̂ = zeros(length(eps_grid), length(duty_grid))
    for (i, ε) in enumerate(eps_grid), (j, dy) in enumerate(duty_grid)
        Q̄ = dy * op[:Qp]
        û0 = mode === :mlp ?
             vec(reconstruct(pods.P,
                             m2.cscales .* predict(m2.model, m2.ps, m2.st,
                                                   collect(standardize_mu(ε, Q̄))),
                             DEFAULT_R)) :
             solve_steady(fom; eps_r = ε, Q = Q̄).x
        c̃0 = (Φ' * (û0 .- ū)) ./ cs_dep
        μst = collect(standardize_mu_t(ε, op[:Qp], dy, op[:period]))
        subs = build_chain(sgrid[1], sgrid[end], dy, sgrid[2:end])
        Ĉ = exec_chain(dep.model, dep.st, p_dep, μst, c̃0, subs;
                       abstol = TOL_EVAL, reltol = TOL_EVAL)
        Q̂[i, j] = Ĉ === nothing ? NaN :
                   qoi(Matrix(reconstruct(P, cs_dep .* hcat(c̃0, Ĉ), r)))
    end
    return Q̂
end

l4res = []
for op in ds["ops"]
    Q̂ = latent_grid(op, :mlp)
    Q̂f = latent_grid(op, :fom)
    fomQ = op[:fomQ]
    sp = any(isnan, Q̂) ? NaN : spearman(Q̂, fomQ)
    spf = any(isnan, Q̂f) ? NaN : spearman(Q̂f, fomQ)
    t5 = length(intersect(sortperm(vec(Q̂))[1:5], sortperm(vec(fomQ))[1:5]))
    same = argmin(vec(Q̂)) == argmin(vec(fomQ))
    push!(l4res, (; Qp = op[:Qp], sp, spf, t5, same, Q̂, fomQ))
    @printf("L4 Qp=%.0f: Spearman %.4f (FOM-IC check %.4f), top-5 %d/5, same optimum %s, failed cells %d\n",
            op[:Qp], sp, spf, t5, string(same), count(isnan, Q̂))
end
l4 = all(x -> x.sp >= 0.998 && x.t5 >= 4 && x.same, l4res)

# ----- scorecard ------------------------------------------------------------------------
println("\nscoring convention (fixed before final results): any rollout failure in a")
println("prediction's own evaluation means that prediction cannot HIT.")
l1 = fail_dep == 0 && safemean(rels_dep) < 1.737e-2
l2 = fail24 == 0 && safemean(rels24) < 5.2e-2
println("\n=== registered predictions, scored ===")
@printf("L1 %s: deployed %.3e (%d failed) vs mapper 1.737e-2\n",
        l1 ? "HIT " : "MISS", safemean(rels_dep), fail_dep)
@printf("L2 %s: 24-traj %.3e (%d failed) vs mapper-at-24 5.2e-2\n",
        l2 ? "HIT " : "MISS", safemean(rels24), fail24)
@printf("L3 %s: settled-cycle deviation flat within 2× dev(3) at both points\n",
        l3 ? "HIT " : "MISS")
@printf("L4 %s: Spearman ≥ 0.998, top-5 ≥ 4/5, same optimum at both points\n",
        l4 ? "HIT " : "MISS")

jldsave(joinpath(DATA_DIR, "latent_ode.jld2");
        rels_dep, rel_dep = safemean(rels_dep), fail_dep, relsA,
        rel_stageA = safemean(relsA), failA,
        ens_rels, ens_fails, nens = NENS, rels24, rel24 = safemean(rels24), fail24,
        dev_lat, l4 = [Dict(pairs(x)) for x in l4res],
        sense = SENSE_NAME, wall_dep = dep.wall,
        gs = collect(GS), epochs_b = collect(EPOCHS_B),
        tol_train = TOL_TRAIN, tol_eval = TOL_EVAL,
        scores = Dict("L1" => l1, "L2" => l2, "L3" => l3, "L4" => l4))

jldsave(model_path("latent_ode_r$(r)");
        kind = "latent_ode", indim = 13, outdim = r, hidden = 64, depth = 2,
        ps = dep.psB, st = dep.st, losses = vcat(dep.lossesA, dep.lossesB),
        train_time = dep.wall, cscales = cs_dep,
        time_convention = "phase time s = t/period; f outputs dc̃/ds",
        w_convention = "w ∈ {0,1} load indicator, constant per sub-solve",
        gs = collect(GS), solver = "FBDF", sensealg = SENSE_NAME,
        seed = SEED + 3100)

# ----- figure ----------------------------------------------------------------------------
fig = Figure(size = (1240, 400))
ax1 = Axis(fig[1, 1]; yscale = log10, ylabel = "traj rel L² (N = $(length(tte)))",
           title = "The transient ladder at r = 8",
           xticks = (1:5, ["mapper", "Stage A", "latent ODE", "Galerkin", "projection"]))
ladder_v = [1.737e-2, safemean(relsA), safemean(rels_dep), 3.719e-6, 2.949e-6]
barplot!(ax1, 1:5, [isnan(x) ? 1.0 : x for x in ladder_v])
ax2 = Axis(fig[1, 2]; xlabel = "FOM cycle", ylabel = "peak-T deviation (K)",
           yscale = log10, title = "20-cycle attractor check (frozen FOM traces)")
for (i, d) in enumerate(dev_lat)
    scatterlines!(ax2, 1:length(d), max.(d, 1e-6); label = "point $i")
end
axislegend(ax2; position = :rt)
ax3 = Axis(fig[1, 3]; xlabel = "FOM QoI (K)", ylabel = "latent-ODE QoI (K)",
           title = "Design grids, both operating points")
for x in l4res
    scatter!(ax3, vec(x.fomQ), vec(x.Q̂); markersize = 5,
             label = @sprintf("Qp=%.0f", x.Qp))
end
ablines!(ax3, 0, 1; color = (:gray, 0.6))
axislegend(ax3; position = :rb)
save(joinpath(FIG_DIR, "31_latent_ode.png"), fig)

println("\nsaved data/latent_ode.jld2, data/model_latent_ode_r8.jld2, figures/31_latent_ode.png")
