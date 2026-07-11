# Small-data follow-up to the DeepONet baseline: does the crossover exist?
#
# ============================ PRE-REGISTERED (2026-07-11) ============================
# Written before this script's first run. Follow-up to scripts/28's D4 result, which
# fenced its claim at the measured range: DeepONet's held-out error is flat in M
# (architecture-set floor ≈ 3.3–4.4e-3) while the POD-MLP converts data into accuracy;
# POD-MLP led at every tested M ∈ {102, 205, 410} by 2.1–2.6×. Below M = 102 nothing
# was measured. This script measures it.
#
# REGISTERED QUESTION: at M ∈ {13, 26, 51} training snapshots, does the DeepONet's
# flat floor undercut the POD-MLP's growing small-sample error?
#
# COMPETING HYPOTHESES (registered; no confident directional bet is available):
#   H-A (crossover): at some tested M the DeepONet ensemble-mean rel L² drops below
#       the POD-MLP's. Mechanism: the DeepONet fits one global smooth (x, μ) surface
#       from M × 1891 field samples, while the POD-MLP interpolates two coefficient
#       functions from only M points with no regularization beyond keep-best.
#   H-B (no crossover): the POD-MLP leads at every tested M. Mechanism: the manifold
#       is rank 3, so even ~13 snapshots span an essentially exact basis, and the
#       2-function fit stays easy; the DeepONet must learn basis and coefficients
#       jointly from the same scraps.
# REGISTERED PRIOR, with its confidence stated: weak lean to H-B at M = 51 and
# M = 26; no call at M = 13 (assessed as a coin flip before running).
#
# DECISION CRITERION: per-M comparison of ensemble means (N = 5 init-only seeds),
# same architectures, budgets, and training schedule as scripts/28 (DeepONet arm-1
# (2,60,60,8) vs POD-MLP r = 2; 6,000 full-batch Adam epochs; scaled-field targets
# for the DeepONet, whitened coefficients with per-subset cscales for the POD-MLP,
# exactly as each trains in production). Crossover at M := DON mean < MLP mean.
#
# INCLUSION RULE, registered before the run: a CONCLUSIVE result
# enters report §7.4 regardless of which method it favors; an inconclusive one is
# recorded in this script's output and data/ only, and is not framed as a miss
# anywhere: a question answered "insufficient power at N = 5" is a null, not a miss.
# CONCLUSIVENESS CRITERION, fixed before the data exists: at a given M the result is
# conclusive iff |mean_DON − mean_MLP| > 2 × max(std_DON, std_MLP) at that M (N = 5
# each); the experiment is conclusive overall iff that holds at ≥ 2 of the 3 tested M.
#
# Caveat registered up front: at M = 13 the per-subset cscales are computed from 13
# samples and are noisy; that is part of the production recipe being tested, not a
# confound to correct.
#
# Artifacts: data/deeponet_small_m.jld2, figures/28_deeponet_small_m.png; printed by
# the §7.4 block of scripts/18_report_numbers.jl.
#
# RESULT (run 2026-07-11, appended after the run; registered text above unchanged):
#   M=13: DON 3.274e-2 ± 5.1e-3 vs MLP 3.499e-2 ± 7.5e-3  (DON nominally ahead; inconclusive)
#   M=26: DON 1.288e-2 ± 8.2e-4 vs MLP 1.329e-2 ± 1.4e-3  (DON nominally ahead; inconclusive)
#   M=51: DON 6.616e-3 ± 8.9e-4 vs MLP 5.568e-3 ± 4.9e-4  (MLP nominally ahead; inconclusive)
#   INCONCLUSIVE overall per the registered criterion. Per the inclusion rule this is
#   a null, not a miss: recorded here and in data/ only, with no report, printer, or
#   README entry (this supersedes the artifact sentence above about the §7.4 printer
#   block). The registered prior is unresolved. For any future registration: the two
#   methods are statistically indistinguishable at N = 5 for M ≤ 26, and the
#   DeepONet's 102–410 flatness does NOT extend below M = 102 (3.7e-3 → 3.3e-2 from
#   410 down to 13 snapshots): both methods are data-limited there, so D4's fence at
#   M = 102 was drawn where the regime actually changes.
#
# AMENDMENT (2026-07-11, appended after the RESULT): report §7.4 gains a one-sentence
#   existence pointer to this probe (no numbers, no direction claimed), and the README
#   script inventory lists this file. The registered
#   "no report entry" clause was mis-scoped: it was meant to bar claims, and silence
#   would have left §7.4's "below M = 102 nothing is measured" standing false once
#   this script entered the repo. The no-claims clause stands: the outcome is not
#   framed as a finding anywhere.
# =====================================================================================

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(PROJECT_DIR, "src", "models_deeponet.jl"))
using CairoMakie
CairoMakie.activate!()

println("=== 29_deeponet_small_m ===")

fom = build_steady_fom(CFG)
d = load_snapshots()
pod = load_pod()
P = pod.P
tr, te = d["train_idx"], d["test_idx"]
Xtr, Xte = d["X"][:, tr], d["X"][:, te]
Utr, Ute = d["U"][:, tr], d["U"][:, te]

xs, ys = node_axes(fom)
XY = Matrix(reduce(hcat, [[standardize(x, extrema(xs)...),
                           standardize(y, extrema(ys)...)]
                          for y in ys for x in xs]))
ū = P.mean
s = std(Utr .- ū)

MS = (13, 26, 51)
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
                                       rng = StableRNG(SEED + 7500 + 10j + sd))
        ps, _ = train_deeponet!(model, ps, st, XtrM, XY, YtrM;
                                epochs = 6000, verbose = false)
        curve["don"][sd, j] =
            mean_relative_l2(Matrix(ū .+ s .* predict_deeponet(model, ps, st, Xte, XY)),
                             Ute)

        mlp, mps, mst = build_pod_mlp(DEFAULT_R;
                                      rng = StableRNG(SEED + 7600 + 10j + sd))
        mps, _ = train!(mlp, mps, mst, XtrM, CtrM ./ csM;
                        epochs = 6000, verbose = false)
        curve["mlp"][sd, j] =
            mean_relative_l2(Matrix(reconstruct(P, csM .* predict(mlp, mps, mst, Xte),
                                                DEFAULT_R)), Ute)
        @printf("  M=%2d seed %d: DeepONet %.3e  POD-MLP %.3e\n",
                M, sd, curve["don"][sd, j], curve["mlp"][sd, j])
    end
end

don_m = vec(mean(curve["don"]; dims = 1))
mlp_m = vec(mean(curve["mlp"]; dims = 1))
don_s = vec(std(curve["don"]; dims = 1))
mlp_s = vec(std(curve["mlp"]; dims = 1))
crossover = [don_m[j] < mlp_m[j] for j in eachindex(don_m)]
concl = [abs(don_m[j] - mlp_m[j]) > 2 * max(don_s[j], mlp_s[j])
         for j in eachindex(don_m)]
conclusive_overall = count(concl) >= 2
println("\n=== registered question, answered ===")
for (j, M) in enumerate(MS)
    @printf("M=%2d: DON %.3e ± %.1e vs MLP %.3e ± %.1e  → %s leads, %s\n",
            M, don_m[j], don_s[j], mlp_m[j], mlp_s[j],
            crossover[j] ? "DeepONet" : "POD-MLP",
            concl[j] ? "conclusive" : "inconclusive")
end
println(conclusive_overall ?
        "CONCLUSIVE overall: H-$(any(crossover .& concl) ? "A (crossover exists)" : "B (no crossover)")" :
        "INCONCLUSIVE overall at N = $(NCURVE): recorded here, not a report item.")

jldsave(joinpath(DATA_DIR, "deeponet_small_m.jld2");
        curve, ncurve = NCURVE, crossover, concl, conclusive_overall)

fig = Figure(size = (520, 380))
ax = Axis(fig[1, 1]; xscale = log2, yscale = log10, xlabel = "training snapshots M",
          ylabel = "held-out rel. L²",
          title = "Small-data regime (N = $(NCURVE) seeds)",
          xticks = (collect(MS), string.(collect(MS))))
scatterlines!(ax, collect(MS), don_m; label = "DeepONet")
scatterlines!(ax, collect(MS), mlp_m; label = "POD-MLP")
axislegend(ax; position = :rt)
save(joinpath(FIG_DIR, "28_deeponet_small_m.png"), fig)

println("\nsaved data/deeponet_small_m.jld2 and figures/28_deeponet_small_m.png")
