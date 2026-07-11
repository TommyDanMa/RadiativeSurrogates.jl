# DeepONet baseline against the deployed POD-MLP (steady problem).
#
# ============================ PRE-REGISTERED (2026-07-11) ============================
# Written BEFORE this script's first run. Reference
# values below are FROZEN numbers from data/eval.jld2 and data/robustness.jld2
# (POD-MLP r=2 deployed: rel L² 9.18e-4, residual 6.53e-3, 8,642 params; N = 20
# ensemble 1.35e-3 ± 0.18e-3; direct-MLP residual 38.7; POD floor at r=2: rel L²
# 1.1e-9, residual 5.1e-7). They are quoted as thresholds, not recomputed.
#
# PROTOCOL (registered in report §7.4 and note §7 before this run):
#   Arms, all trained on the same 410 training snapshots, targets scaled fields
#   (u − ū)/s exactly as the direct-MLP baseline, all with the POD-MLP's training
#   budget byte-for-byte (full-batch Adam, 6,000 epochs, lr 1e-3 → 1e-5 geometric,
#   keep-best; matched gradient steps × batch size), Float64, no tuning on any arm:
#     arm 1 "equal": DeepONet, branch/trunk (2,60,60,8) tanh, p = 8 latent
#            functions, 8,656 params (+0.16% vs POD-MLP's 8,642).
#     arm 2 "3x":    DeepONet (2,108,108,8), 25,936 params (+0.04% vs 3× budget).
#     arm 3 "frozen-trunk": trunk fixed to the POD basis Φ_2, branch = the exact
#            POD-MLP architecture (2→64³→2, 8,642 params), same scaled-field MSE.
#            Isolates learned-basis vs fixed-basis at identical loss and budget.
#   p = 8 is deliberately generous to DeepONet (manifold rank is 3; deployed POD
#   rank is 2); the frozen arm needs only r = 2 by construction.
#   Ensembles: N = 20 fresh init-only seeds per arm (StableRNG SEED+7000/7100/7200+s);
#   a deployed-analog member per arm (SEED+400/410/420) for the table row and QoIs.
#   Sample efficiency: M ∈ {102, 205, 410} nested Sobol-order prefixes of the
#   training set; N = 5 seeds per (method, M); DeepONet arm-1 architecture vs
#   POD-MLP retrained exactly as scripts/04 (whitened coefficients, cscales from
#   the M-subset); held-out rel L².
#   Metrics on the 102 held-out parameters: rel L² and the assembled nonlinear
#   residual ‖R(û)‖/‖F‖ (PRIMARY, per the registered wedge); QoIs for the
#   deployed-analog members. Loss curves of arm 1 and the frozen POD-MLP overlaid
#   normalized to their initial value (different losses: scaled-field MSE vs
#   whitened-coefficient MSE; the overlay shows optimization dynamics only).
#
# PREDICTIONS (hit/miss scored after the run; misses are results):
#   D1 (PRIMARY, residual): arm-1 ensemble-mean relative residual ≥ 10× the
#       deployed POD-MLP's 6.53e-3 (i.e. ≥ 6.5e-2), and arm 2 (3× params) does
#       not close it below 3× (≥ 2.0e-2). Rationale: an L²-fit learned basis has
#       no mechanism to satisfy the discrete PDE; the POD basis is L²-optimal
#       and the Galerkin floor says the map is the only error left there.
#   D2 (accuracy): arm-1 ensemble-mean rel L² ≤ 5× the POD-MLP ensemble mean
#       1.35e-3 (i.e. ≤ 6.75e-3): "competitive on L²".
#   D3 (mechanism): arm-3 ensemble-mean residual ≤ 3× the deployed POD-MLP
#       (≤ 2.0e-2) AND arm-1/arm-3 ensemble-mean residual ratio ≥ 3: the D1 gap
#       is attributable to the learned basis, not branch capacity or the loss.
#   D4 (sample efficiency): the degradation factor rel L²(M=102)/rel L²(M=410)
#       (ensemble means, N = 5) is strictly larger for DeepONet arm 1 than for
#       the identically retrained POD-MLP.
#
# ROUND-1 SCORECARD (run 2026-07-11, appended post-run; registered content unchanged):
#   D1 HIT.  Arm-1 ensemble residual 1.851e+0, 283× the POD-MLP's 6.53e-3 (registered
#            threshold 6.5e-2); the 3× arm sits at 1.541e+0, gap not closed.
#   D2 HIT.  Arm-1 ensemble rel L² 3.304e-3, 2.4× the POD-MLP ensemble mean
#            (threshold ≤ 6.75e-3): competitive on L², ~280× worse on physics.
#   D3 MISS on the absolute clause, instructive. Frozen-trunk residual 1.246e-1 is
#            19× the POD-MLP (registered bound 3×): the shared scaled-field MSE
#            under-weights mode 2 by (σ₁/σ₂)² ≈ 76, the same loss-weighting effect
#            the A.1 whitening ablation measured; the registered bound assumed it
#            away. The relative clause held: arm-1/arm-3 residual ratio 14.9 (≥ 3),
#            so the learned basis, not the branch, carries the arm-1 gap.
#   D4 MISS, instructive. Degradation 410→102 snapshots: DeepONet 1.19× vs POD-MLP
#            1.46×. DeepONet is not data-hungry here; it is saturated: its error is
#            flat in M (3.3–4.4e-3 at every M, an architecture-set floor) while the
#            POD-MLP still converts data into accuracy. "Worse sample efficiency"
#            was the wrong frame; the measured phenomenon is a capacity ceiling
#            that more data cannot buy through.
#
# Environment: NeuralOperators.jl v0.6.4 added to Project.toml 2026-07-11.
# Artifacts: data/deeponet.jld2, figures/27_deeponet.png; printer section §7.4 in
# scripts/18_report_numbers.jl; report §7.4 gains the results and the scorecard.
# =====================================================================================

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(PROJECT_DIR, "src", "models_deeponet.jl"))
using Gridap
using CairoMakie
CairoMakie.activate!()

println("=== 28_deeponet_baseline ===")

fom = build_steady_fom(CFG)
d = load_snapshots()
pod = load_pod()
P = pod.P
tr, te = d["train_idx"], d["test_idx"]
Xtr, Xte = d["X"][:, tr], d["X"][:, te]
Utr, Ute = d["U"][:, tr], d["U"][:, te]
mus_te = [(d["mus"][1, k], d["mus"][2, k]) for k in te]
ntest = length(te)

# Trunk input: node coordinates standardized to [-1,1]², fixed for all queries.
xs, ys = node_axes(fom)
XY = Matrix(reduce(hcat, [[standardize(x, extrema(xs)...),
                           standardize(y, extrema(ys)...)]
                          for y in ys for x in xs]))

# Scaled-field targets, exactly the direct-MLP treatment (scripts/04).
ū = P.mean
s = std(Utr .- ū)
Ytr = (Utr .- ū) ./ s
Φ2 = Matrix(P.modes[:, 1:DEFAULT_R])

"Held-out rel L² and mean relative residual of an unscaled field matrix."
function member_metrics(Û; with_resid::Bool = true)
    rel = mean_relative_l2(Û, Ute)
    resid = with_resid ? mean_relative_residual(fom, mus_te, Û) : NaN
    return (; rel, resid)
end

unscale(Ŷ) = ū .+ s .* Ŷ

# ----- arms 1–2: DeepONet, learned trunk -------------------------------------------
EPOCHS = 6000
NSEEDS = 20
arms = Dict{String,Any}()
for (name, hidden, seed0, seed_dep) in
    (("equal", 60, 7000, 400), ("3x", 108, 7100, 410))
    println("\n--- arm $(name): DeepONet (2,$(hidden),$(hidden),8) ---")
    rels, resids = Float64[], Float64[]
    local dep_losses, dep_metrics, dep_qoi, nparams
    for member in 0:NSEEDS
        seed = member == 0 ? SEED + seed_dep : SEED + seed0 + member
        model, ps, st = build_deeponet(; hidden, rng = StableRNG(seed))
        member == 0 && (nparams = Lux.parameterlength(model))
        ps, losses = train_deeponet!(model, ps, st, Xtr, XY, Ytr;
                                     epochs = EPOCHS, verbose = member == 0)
        Û = Matrix(unscale(predict_deeponet(model, ps, st, Xte, XY)))
        mm = member_metrics(Û)
        if member == 0
            dep_losses, dep_metrics = losses, mm
            dep_qoi = (; dP = mean(qoi_errors(fom, Û[:, k], Ute[:, k];
                                              eps_r = mus_te[k][1]).dP_rel
                                   for k in 1:ntest),
                       dT = maximum(abs.(vec(maximum(Û; dims = 1)) .-
                                         vec(maximum(Ute; dims = 1)))))
            @printf("  deployed-analog: rel %.3e  resid %.3e  (%d params)\n",
                    mm.rel, mm.resid, nparams)
        else
            push!(rels, mm.rel); push!(resids, mm.resid)
            @printf("  seed %2d/%d: rel %.3e  resid %.3e\n",
                    member, NSEEDS, mm.rel, mm.resid)
        end
    end
    arms[name] = Dict("rels" => rels, "resids" => resids, "params" => nparams,
                      "dep_rel" => dep_metrics.rel, "dep_resid" => dep_metrics.resid,
                      "dep_dP" => dep_qoi.dP, "dep_dT" => dep_qoi.dT,
                      "losses" => dep_losses)
end

# ----- arm 3: trunk frozen to the POD basis ----------------------------------------
println("\n--- arm frozen: trunk = Φ_$(DEFAULT_R), branch = POD-MLP architecture ---")
let
    rels, resids = Float64[], Float64[]
    local dep_losses, dep_metrics, dep_qoi, nparams
    Φs = Φ2 ./ s   # so the branch output lives on the same scaled-target scale
    for member in 0:NSEEDS
        seed = member == 0 ? SEED + 420 : SEED + 7200 + member
        branch, ps, st = build_pod_mlp(DEFAULT_R; rng = StableRNG(seed))
        member == 0 && (nparams = Lux.parameterlength(branch))
        ps, losses = train_frozen_trunk!(branch, ps, st, Xtr, s .* Φs, Ytr;
                                         epochs = EPOCHS, verbose = member == 0)
        Ĉ = predict(branch, ps, st, Xte)
        Û = Matrix(ū .+ s .* ((s .* Φs) * Ĉ))
        mm = member_metrics(Û)
        if member == 0
            dep_losses, dep_metrics = losses, mm
            dep_qoi = (; dP = mean(qoi_errors(fom, Û[:, k], Ute[:, k];
                                              eps_r = mus_te[k][1]).dP_rel
                                   for k in 1:ntest),
                       dT = maximum(abs.(vec(maximum(Û; dims = 1)) .-
                                         vec(maximum(Ute; dims = 1)))))
            @printf("  deployed-analog: rel %.3e  resid %.3e  (%d params)\n",
                    mm.rel, mm.resid, nparams)
        else
            push!(rels, mm.rel); push!(resids, mm.resid)
            @printf("  seed %2d/%d: rel %.3e  resid %.3e\n",
                    member, NSEEDS, mm.rel, mm.resid)
        end
    end
    arms["frozen"] = Dict("rels" => rels, "resids" => resids, "params" => nparams,
                          "dep_rel" => dep_metrics.rel,
                          "dep_resid" => dep_metrics.resid,
                          "dep_dP" => dep_qoi.dP, "dep_dT" => dep_qoi.dT,
                          "losses" => dep_losses)
end

# ----- sample-efficiency curve ------------------------------------------------------
println("\n--- sample efficiency: M ∈ {102, 205, 410}, N = 5 seeds each ---")
MS = (102, 205, 410)
NCURVE = 5
curve = Dict("Ms" => collect(MS),
             "don" => zeros(NCURVE, length(MS)), "mlp" => zeros(NCURVE, length(MS)))
for (j, M) in enumerate(MS)
    XtrM, UtrM = Xtr[:, 1:M], Utr[:, 1:M]
    YtrM = (UtrM .- ū) ./ s
    CtrM = Matrix(project(P, UtrM, DEFAULT_R))
    csM = vec(std(CtrM; dims = 2)) .+ 1e-14 * std(CtrM)
    for sd in 1:NCURVE
        model, ps, st = build_deeponet(; hidden = 60,
                                       rng = StableRNG(SEED + 7300 + 10j + sd))
        ps, _ = train_deeponet!(model, ps, st, XtrM, XY, YtrM;
                                epochs = EPOCHS, verbose = false)
        curve["don"][sd, j] =
            mean_relative_l2(Matrix(unscale(predict_deeponet(model, ps, st, Xte, XY))),
                             Ute)

        mlp, mps, mst = build_pod_mlp(DEFAULT_R;
                                      rng = StableRNG(SEED + 7400 + 10j + sd))
        mps, _ = train!(mlp, mps, mst, XtrM, CtrM ./ csM;
                        epochs = EPOCHS, verbose = false)
        curve["mlp"][sd, j] =
            mean_relative_l2(Matrix(reconstruct(P, csM .* predict(mlp, mps, mst, Xte),
                                                DEFAULT_R)), Ute)
        @printf("  M=%3d seed %d: DeepONet %.3e  POD-MLP %.3e\n",
                M, sd, curve["don"][sd, j], curve["mlp"][sd, j])
    end
end

# ----- registered thresholds, scored --------------------------------------------------
POD_DEP_RESID = 6.53e-3
POD_ENS_REL = 1.35e-3
m_rel(a) = mean(arms[a]["rels"]); m_res(a) = mean(arms[a]["resids"])
deg_don = mean(curve["don"][:, 1]) / mean(curve["don"][:, end])
deg_mlp = mean(curve["mlp"][:, 1]) / mean(curve["mlp"][:, end])
d1 = m_res("equal") >= 10 * POD_DEP_RESID && m_res("3x") >= 3 * POD_DEP_RESID
d2 = m_rel("equal") <= 5 * POD_ENS_REL
d3 = m_res("frozen") <= 3 * POD_DEP_RESID && m_res("equal") / m_res("frozen") >= 3
d4 = deg_don > deg_mlp
println("\n=== registered predictions, scored ===")
@printf("D1 residual:   arm1 %.3e (≥ 6.5e-2?), arm2 %.3e (≥ 2.0e-2?)  → %s\n",
        m_res("equal"), m_res("3x"), d1 ? "HIT" : "MISS")
@printf("D2 accuracy:   arm1 rel %.3e (≤ 6.75e-3?)                    → %s\n",
        m_rel("equal"), d2 ? "HIT" : "MISS")
@printf("D3 mechanism:  frozen resid %.3e (≤ 2.0e-2?), ratio %.1f (≥ 3?) → %s\n",
        m_res("frozen"), m_res("equal") / m_res("frozen"), d3 ? "HIT" : "MISS")
@printf("D4 sample eff: deg DON %.2f vs MLP %.2f (strictly larger?)   → %s\n",
        deg_don, deg_mlp, d4 ? "HIT" : "MISS")

# ----- save ---------------------------------------------------------------------------
mlp_ref = load_mlp(model_path("pod_mlp_r$(DEFAULT_R)"))
jldsave(joinpath(DATA_DIR, "deeponet.jld2");
        arms, curve, epochs = EPOCHS, nseeds = NSEEDS, ncurve = NCURVE,
        scores = Dict("D1" => d1, "D2" => d2, "D3" => d3, "D4" => d4),
        pod_dep_resid = POD_DEP_RESID, pod_ens_rel = POD_ENS_REL,
        mlp_losses = mlp_ref.losses)

# ----- figure -------------------------------------------------------------------------
fig = Figure(size = (1200, 380))
ax1 = Axis(fig[1, 1]; yscale = log10, xlabel = "epoch",
           ylabel = "training loss / initial",
           title = "Optimization dynamics (different losses; see caption)")
lines!(ax1, arms["equal"]["losses"] ./ arms["equal"]["losses"][1],
       label = "DeepONet equal")
lines!(ax1, mlp_ref.losses ./ mlp_ref.losses[1], label = "POD-MLP (whitened coef)")
axislegend(ax1; position = :rt)

ax2 = Axis(fig[1, 2]; xticks = (1:4, ["POD-MLP", "DON equal", "DON 3×", "frozen Φ"]),
           yscale = log10, ylabel = "mean rel. residual",
           title = "Physics consistency (N = $(NSEEDS) seeds)")
for (i, v) in enumerate((fill(POD_DEP_RESID, 1), arms["equal"]["resids"],
                         arms["3x"]["resids"], arms["frozen"]["resids"]))
    boxplot!(ax2, fill(i, length(v)), v)
end

ax3 = Axis(fig[1, 3]; xscale = log2, yscale = log10, xlabel = "training snapshots M",
           ylabel = "held-out rel. L²", title = "Sample efficiency (N = $(NCURVE))",
           xticks = (collect(MS), string.(collect(MS))))
scatterlines!(ax3, collect(MS), vec(mean(curve["don"]; dims = 1)), label = "DeepONet")
scatterlines!(ax3, collect(MS), vec(mean(curve["mlp"]; dims = 1)), label = "POD-MLP")
axislegend(ax3; position = :rt)
save(joinpath(FIG_DIR, "27_deeponet.png"), fig)

println("\nsaved data/deeponet.jld2 and figures/27_deeponet.png")
