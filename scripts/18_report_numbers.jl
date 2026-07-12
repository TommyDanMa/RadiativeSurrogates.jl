# Prints every headline table in report-markdown form, straight from
# data/*.jld2. Diff this output against report/report.md whenever numbers are
# touched; frozen numbers can then never silently drift (the parent project's
# freeze lesson, institutionalized).

include(joinpath(@__DIR__, "config.jl"))

println("=== 18_report_numbers ===\n")

need(f) = isfile(f) || (println("missing $(basename(f)); run its script first\n"); false)

if need(eval_path())
    ev = load(eval_path())
    println("## §7.2 steady evaluation (deployed r = $(DEFAULT_R))")
    @printf("POD floor: rel L² %.1e, residual %.1e; FOM residual floor %.1e; FOM %.0f ms\n",
            ev["floor_rel"], ev["floor_resid"], ev["fom_resid"], ev["fom_ms"])
    for name in ("direct_mlp", "pod_mlp", "pod_kan")
        r = ev["results"][name]
        @printf("| %s | %.2e | %.2e | %s | %.1e | %.1f | %d |\n", name,
                r[:rel], r[:resid], isnan(r[:coef]) ? "—" : @sprintf("%.1e", r[:coef]),
                r[:dP], r[:dT], r[:params])
    end
    sw = ev["sweep"]
    println("rank sweep:")
    for (i, r) in enumerate(sw["ranks"])
        @printf("| %d | %.2e / %.2e | %.2e / %.2e |\n", r,
                sw["mlp_rel"][i], sw["mlp_resid"][i], sw["kan_rel"][i], sw["kan_resid"][i])
    end
    println()
end

if need(joinpath(DATA_DIR, "benchmark.jld2"))
    b = load(joinpath(DATA_DIR, "benchmark.jld2"))
    println("## §9.1 benchmark (batch = $(b["batch"]), FOM $(round(b["t_fom_ms"], digits=1)) ms)")
    haskey(b, "cpu") && @printf("hardware: %s | %d Julia threads | %d BLAS threads | Julia %s\n",
                                b["cpu"], b["threads"], b["blas_threads"], b["julia"])
    for (i, nm) in enumerate(b["names"])
        @printf("| %s | %.3f | %.4f | %.0f× | %.0f× | %d |\n", nm,
                b["t1"][i], b["tb"][i], b["s1"][i], b["sb"][i], b["allocs"][i])
    end
    println()
end

if need(eval_t_path())
    et = load(eval_t_path())
    println("## §8.2 transient sweep (deployed r = $(et["deployed_r"]))")
    for (i, r) in enumerate(et["ranks"])
        @printf("| %d | %.2e | %.1e |\n", r, et["sweep_rel"][i], et["floor_rel"][i])
    end
    @printf("peak-T err: median %.1f K, max %.1f K; P_rad tracking: median %.1e, max %.1e\n\n",
            median(et["dTpk"]), maximum(et["dTpk"]), median(et["dPq"]), maximum(et["dPq"]))
end

if need(joinpath(DATA_DIR, "galerkin_transient.jld2"))
    gt = load(joinpath(DATA_DIR, "galerkin_transient.jld2"))
    println("## §8.4 transient Galerkin floor (trajectory metric, N = $(gt["ntest"]) held-out)")
    for (i, r) in enumerate(gt["ranks"])
        @printf("| %d | %.2e | %.2e | %.2e |\n", r, gt["mapper_rel"][i],
                gt["gal_rel"][i], gt["proj_floor_traj"][i])
    end
    @printf("r=8 QoI: peak-T err median %.3f K max %.3f K; P_rad median %.2e max %.2e\n",
            median(gt["dTpk"]), maximum(gt["dTpk"]), median(gt["dPq"]), maximum(gt["dPq"]))
    i8 = findfirst(==(DEFAULT_R_T), gt["ranks"])
    @printf("stiffness (r=8, audit traj): |Re λ| ∈ [%.2e, %.2e] 1/s, per-period %.2e\n",
            gt["stiff_min"][i8], gt["stiff_max"][i8], gt["stiff_per_period"])
    @printf("tol check (1e-9 vs 1e-10): max Δ %.1e; scores: %s\n\n",
            gt["tolcheck_max"],
            join(("$k $(gt["scores"][k] ? "HIT" : "MISS")" for k in ("P-G1", "P-G2", "P-G3")), "  "))
end

if need(joinpath(DATA_DIR, "robustness.jld2"))
    rb = load(joinpath(DATA_DIR, "robustness.jld2"))
    println("## A.1 robustness (seed ensembles, N = $(length(rb["mlp_rels"])))")
    seedline(label, r, dep, rels) =
        @printf("%s r=%d: deployed %.2e, mean %.2e ± %.1e, [%.2e, %.2e]\n",
                label, r, dep, mean(rels), std(rels), minimum(rels), maximum(rels))
    seedline("POD-MLP  ", DEFAULT_R, rb["dep_mlp"], rb["mlp_rels"])
    seedline("POD-KAN  ", DEFAULT_R, rb["dep_kan"], rb["kan_rels"])
    seedline("POD-MLP-t", DEFAULT_R_T, rb["dep_t"], rb["t_rels"])
    @printf("OOD: in-box median %.1e; outside median %.1e, worst %.1e\n\n",
            median(rb["in_rel"]), median(rb["ood_rel"]), maximum(rb["ood_rel"]))
end

if need(joinpath(DATA_DIR, "grid_independence.jld2"))
    g = load(joinpath(DATA_DIR, "grid_independence.jld2"))
    println("## §4 grid independence (σ₃/σ₁)")
    @printf("60×30 P1 %.1e | 120×60 P1 %.1e | 60×30 P2 %.1e\n\n",
            g["d_base"][3], g["d_fine"][3], g["d_p2"][3])
end

if need(joinpath(DATA_DIR, "design_study.jld2"))
    ds = load(joinpath(DATA_DIR, "design_study.jld2"))
    println("## §9.2 design studies")
    for op in ds["ops"]
        @printf("Qp=%.0f, per=%.0f: %d cands, FOM %.0f s vs sur %.2f s (%.0f×), Spearman %.4f, top-5 %d/5, QoI err med %.1f K max %.1f K\n",
                op[:Qp], op[:period], length(op[:fomQ]), op[:t_fom], op[:t_sur],
                op[:t_fom] / op[:t_sur], op[:spearman], op[:top5],
                median(abs.(op[:surQ] .- op[:fomQ])),
                maximum(abs.(op[:surQ] .- op[:fomQ])))
    end
    println()
end

if need(joinpath(DATA_DIR, "mapper_diagnostics.jld2"))
    md_ = load(joinpath(DATA_DIR, "mapper_diagnostics.jld2"))
    println("## §8.2 learning curve")
    for (M, e) in zip(md_["sizes"], md_["lc"])
        @printf("| %d trajectories | %.2e |\n", M, e)
    end
    @printf("slopes: %s\n\n", join((@sprintf("%.2f", s) for s in md_["slopes"]), "  "))
end

if need(joinpath(DATA_DIR, "rank_prediction.jld2"))
    rp = load(joinpath(DATA_DIR, "rank_prediction.jld2"))
    println("## §6 rank predictions (note §4)")
    @printf("P1 (ε,Q1,Q2): σ3 %.1e  σ4 %.1e  σ5 %.1e\n",
            rp["d_p1"][3], rp["d_p1"][4], rp["d_p1"][5])
    @printf("P2 (ε,Q,k):   σ3 %.1e  σ4 %.1e\n", rp["d_p2"][3], rp["d_p2"][4])
    @printf("P3 σ3 ε-shrink: %s | Q-shrink: %s\n",
            join((@sprintf("%.1e", v) for v in rp["sig3_eps"]), " "),
            join((@sprintf("%.1e", v) for v in rp["sig3_Q"]), " "))
    if haskey(rp, "slopes_eps")
        @printf("P3 fitted slopes (N = %d dataset draws): ε %.2f ± %.2f | Q %.2f ± %.2f\n",
                rp["n_draws"], mean(rp["slopes_eps"]), std(rp["slopes_eps"]),
                mean(rp["slopes_Q"]), std(rp["slopes_Q"]))
    end
    println()
end

if need(joinpath(DATA_DIR, "cliff_reconstruction.jld2"))
    cr = load(joinpath(DATA_DIR, "cliff_reconstruction.jld2"))
    println("## §6 cliff reconstruction (note §4b; weighted collinearity vs measured POD)")
    for i in eachindex(cr["names"])
        @printf("%-12s r*=%d  recon %.1e  meas %.1e  (recon/meas %.2f;  naive %.1e)\n",
                cr["names"][i], cr["rstar"][i], cr["recon"][i], cr["measured"][i],
                cr["recon"][i] / cr["measured"][i], cr["naive"][i])
    end
    println()
end

if need(joinpath(DATA_DIR, "statistics.jld2"))
    st = load(joinpath(DATA_DIR, "statistics.jld2"))
    println("## A.1 statistics (B = $(st["nboot"]))")
    for c in st["sp_cis"]
        @printf("Spearman (Qp=%.0f): %.4f [%.4f, %.4f]\n", c[:Qp], c[:m], c[:lo], c[:hi])
    end
    @printf("ablations r=%d: whitening off %.2e | random basis %.2e (floor %.2e)\n\n",
            DEFAULT_R, st["e_raw"], st["e_rand"], st["floor_rand"])
end

if need(joinpath(DATA_DIR, "galerkin_floor.jld2"))
    gf = load(joinpath(DATA_DIR, "galerkin_floor.jld2"))
    println("## §7.3 error decomposition")
    for (i, r) in enumerate(gf["ranks"])
        @printf("| %d | %.2e | %.2e | %.2e |\n", r,
                gf["proj_floor"][i], gf["gal_rel"][i], gf["mlp_rel"][i])
    end
end

if need(joinpath(DATA_DIR, "eigenmode_diagnostic.jld2"))
    eg = load(joinpath(DATA_DIR, "eigenmode_diagnostic.jld2"))
    println("\n## §6 eigenmode diagnostic (note §4b)")
    @printf("power-law fit α ~ λ^-%.2f; excited modes: %s\n",
            eg["p_powerlaw"], join(string.(eg["excited"]), ","))
    @printf("collinearity svals: %s\n",
            join((@sprintf("%.1e", s) for s in eg["func_svals"][1:min(4, end)]), "  "))
end

if need(joinpath(DATA_DIR, "gci.jld2"))
    gc = load(joinpath(DATA_DIR, "gci.jld2"))
    gci_ok(v) = filter(!isnan, v)
    println("\n## A.2 GCI error bars (Fs = $(gc["fs"]), grids 60×30 → 240×120)")
    @printf("T_probe p̂: %s\n",
            join((@sprintf("%.2f", p) for p in gci_ok(gc["phat"][:, 1])), ", "))
    @printf("u_L2 p̂: %s\n",
            join((@sprintf("%.2f", p) for p in gci_ok(gc["phat"][:, 2])), ", "))
    bc = gci_ok(gc["band_coarse"][:, 1])
    isempty(bc) ||
        @printf("T_probe band on deployed 60×30 (= reported T_peak there): median %.3f K, max %.3f K (N = %d)\n",
                median(bc), maximum(bc), length(bc))
    @printf("P_rad grid-exact: worst |P_rad − Q|/Q %.1e; T_edge balance-pinned: worst coarse-vs-fine %.1e\n",
            gc["prad_dev"], gc["tedge_dev"])
end

if need(joinpath(DATA_DIR, "precision_audit.jld2"))
    pa = load(joinpath(DATA_DIR, "precision_audit.jld2"))
    println("\n## A.3 precision audit (N = $(pa["ntest"]) test samples)")
    @printf("F64: rel L² %.4e | ΔT_peak max %.2f K | ΔP/P max %.2e\n",
            pa["rel"]["F64"], pa["dT_max"]["F64"], pa["dP_max"]["F64"])
    @printf("F32: rel L² %.4e | ΔT_peak max %.2f K | ΔP/P max %.2e | field drift vs F64 %.1e K\n",
            pa["rel"]["Float32"], pa["dT_max"]["Float32"], pa["dP_max"]["Float32"],
            pa["drift"]["Float32"])
    @printf("F16: finite fractions net %.3f / coeffs %.3f / fields %.3f\n",
            pa["finite_net"]["Float16"], pa["finite_coef"]["Float16"],
            pa["finite_field"]["Float16"])
    @printf("rad4 in F16 overflows for T ≥ %.1f K (fraction of field values: %.3f)\n",
            pa["rad4_f16_overflow_K"], pa["rad4_f16_overflow_frac"])
end

if need(joinpath(DATA_DIR, "attractor_check.jld2"))
    at = load(joinpath(DATA_DIR, "attractor_check.jld2"))
    println("\n## A.4 attractor check ($(at["ncycles"]) FOM cycles, N = $(length(at["points"])) points)")
    for (i, p) in enumerate(at["points"])
        dev, chg = at["dev"][i], at["chg"][i]
        @printf("ε=%.2f Qp=%.0f duty=%.2f per=%.0f: dev(3) %.2f K, dev c3–20 ∈ [%.2f, %.2f] K; FOM settle c1→2 %.3f K, c19→20 %.4f K\n",
                p[:eps_r], p[:Qp], p[:duty], p[:period], dev[3],
                minimum(dev[3:end]), maximum(dev[3:end]), chg[1], chg[19])
    end
end

if need(joinpath(DATA_DIR, "forward_uq.jld2"))
    uq = load(joinpath(DATA_DIR, "forward_uq.jld2"))
    @printf("\n## A.5 forward UQ (ε ~ N(%.2f, %.2f), Q ~ N(%.0f, %.0f) W/m; sur N = %d, FOM N = %d)\n",
            uq["eps_mean"], uq["eps_std"], uq["q_mean"], uq["q_std"],
            uq["n_sur"], uq["n_fom"])
    for (i, q) in enumerate(uq["quants"])
        @printf("| %.1f%% | %.2f | %.2f | %+.2f | %.2f |\n", 100q, uq["qs_fom"][i],
                uq["qs_sur_paired"][i], uq["qs_sur_paired"][i] - uq["qs_fom"][i],
                uq["qs_sur_full"][i])
    end
    @printf("paired |sur − FOM| (N = %d): median %.2f K, max %.2f K\n",
            uq["n_fom"], uq["d_med"], uq["d_max"])
end

if need(joinpath(DATA_DIR, "deeponet.jld2"))
    dn = load(joinpath(DATA_DIR, "deeponet.jld2"))
    println("\n## §7.4 DeepONet baseline (N = $(dn["nseeds"]) seeds/arm, " *
            "$(dn["epochs"]) epochs, matched budget)")
    for name in ("equal", "3x", "frozen")
        a = dn["arms"][name]
        @printf("%-6s %6d params | rel %.2e ± %.1e | resid %.2e ± %.1e | dep: rel %.2e resid %.2e ΔP/P %.1e ΔT %.1f K\n",
                name, a["params"], mean(a["rels"]), std(a["rels"]),
                mean(a["resids"]), std(a["resids"]),
                a["dep_rel"], a["dep_resid"], a["dep_dP"], a["dep_dT"])
    end
    c = dn["curve"]
    @printf("sample efficiency (N = %d seeds): M = %s | DON %s | MLP %s\n",
            dn["ncurve"], join(c["Ms"], "/"),
            join((@sprintf("%.2e", v) for v in vec(mean(c["don"]; dims = 1))), " "),
            join((@sprintf("%.2e", v) for v in vec(mean(c["mlp"]; dims = 1))), " "))
    s = dn["scores"]
    println("registered scores: " *
            join(("$k $(s[k] ? "HIT" : "MISS")" for k in ("D1", "D2", "D3", "D4")), "  "))
end
