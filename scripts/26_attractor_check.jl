# Attractor check for the transient surrogate's deployment contract. The
# design QoI (§8.6) reads the surrogate's *final training cycle* (cycle 3) as
# "the settled cycle". Here the FOM runs to 20 load cycles at two parameter
# points and the surrogate's phase-mapped settled-cycle prediction is compared
# against every FOM cycle: per-cycle peak-temperature deviation. The mapper is
# feedforward in (μ, τ, phase), so autoregressive drift is structurally
# impossible; what CAN fail is the contract itself (cycle 3 not yet settled,
# or the attractor drifting away from the cycle-3 state). Normalized time
# τ > 1 is out of distribution for the mapper and is never queried here:
# prediction is by phase mapping onto the final training cycle.
#
# PRE-REGISTERED (header written before first run):
#   (1) The FOM settles onto a periodic attractor: the cycle-to-cycle change
#       of the peak-T trace decays geometrically after the first cycles;
#       < 0.5 K by cycle 10 and < 0.05 K at cycle 19→20, for both points
#       (the run starts at the steady state of the cycle-mean load, already
#       near the attractor).
#   (2) The surrogate's settled-cycle deviation is FLAT in the cycle index:
#       for every cycle c ≥ 3, dev(c) stays within 2× of dev(3) with no
#       monotone growth; its level is the surrogate's ordinary prediction
#       error at that μ (report §8.4), not a drift term.
#   (3) Deployment contract to document: phase features are valid ON the
#       attractor; τ > 1 queries are OOD and guarded, exactly like the
#       (ε, Q) box guard for the steady surrogate.

include(joinpath(@__DIR__, "config.jl"))
using NonlinearSolve
using CairoMakie
CairoMakie.activate!()

println("=== 26_attractor_check ===")

const NCYC_LONG = 20
POINTS = ((eps_r = 0.60, Qp = 650.0, duty = 0.50, period = 5400.0),
          (eps_r = 0.35, Qp = 400.0, duty = 0.40, period = 9000.0))

fom = build_steady_fom(CFG)
pod = load(pod_t_path())
P = PODBasis(pod["mean"], pod["modes"], pod["svals"])
m = load_mlp(model_path("pod_mlp_t_r$(DEFAULT_R_T)"))
nt_long = NCYC_LONG * NPHASE + 1

results = []
for pt in POINTS
    ε, Qp, duty, period = pt.eps_r, pt.Qp, pt.duty, pt.period
    T_end = NCYC_LONG * period
    u0 = solve_steady(fom; eps_r = ε, Q = duty * Qp, abstol = 1e-9).x
    t = @elapsed sol = solve_transient(fom; eps_r = ε,
                                       load = square_wave(Qp, period, duty),
                                       u0, tspan = (0.0, T_end),
                                       tstops = load_switch_times(period, duty, 0.0, T_end),
                                       saveat = collect(range(0.0, T_end; length = nt_long)),
                                       abstol = 1e-7, reltol = 1e-7)
    @assert NonlinearSolve.SciMLBase.successful_retcode(sol)
    U = Array(sol)
    pk = vec(maximum(U; dims = 1))

    # Surrogate settled-cycle trace: final training cycle, states j = 41..60
    # (phases 0..19 at τ ∈ [2/3, 59/60]).
    μst = standardize_mu_t(ε, Qp, duty, period)
    js = (2 * NPHASE + 1):(3 * NPHASE)
    X = reduce(hcat, [transient_features(μst, j) for j in js])
    Ĉ = m.cscales .* predict(m.model, m.ps, m.st, X)
    sur_pk = vec(maximum(Matrix(reconstruct(P, Ĉ, DEFAULT_R_T)); dims = 1))

    trace(c) = pk[(c - 1) * NPHASE .+ (1:NPHASE)]
    dev = [maximum(abs.(trace(c) .- sur_pk)) for c in 1:NCYC_LONG]
    chg = [maximum(abs.(trace(c + 1) .- trace(c))) for c in 1:(NCYC_LONG - 1)]

    @printf("\nμ = (ε %.2f, Qp %.0f, duty %.2f, per %.0f s): FOM %d cycles in %.1f s\n",
            ε, Qp, duty, period, NCYC_LONG, t)
    @printf("FOM settling |Δtrace| cycle→cycle: c1→2 %.3f K, c9→10 %.3f K, c19→20 %.4f K\n",
            chg[1], chg[9], chg[19])
    @printf("surrogate settled-cycle deviation: dev(3) %.2f K, dev(10) %.2f K, dev(20) %.2f K\n",
            dev[3], dev[10], dev[20])
    @printf("flatness over c = 3..20: max dev %.2f K, min dev %.2f K (ratio to dev(3): %.2f–%.2f)\n",
            maximum(dev[3:end]), minimum(dev[3:end]),
            minimum(dev[3:end]) / dev[3], maximum(dev[3:end]) / dev[3])

    push!(results, (; pt, pk, sur_pk, dev, chg, t))
end

jldsave(joinpath(DATA_DIR, "attractor_check.jld2");
        points = [Dict(pairs(r.pt)) for r in results],
        pk = [r.pk for r in results], sur_pk = [r.sur_pk for r in results],
        dev = [r.dev for r in results], chg = [r.chg for r in results],
        ncycles = NCYC_LONG)

# ----- figure ---------------------------------------------------------------------
fig = Figure(size = (980, 640))
for (i, r) in enumerate(results)
    pt = r.pt
    ax1 = Axis(fig[1, i]; xlabel = "phase index within cycle", ylabel = "peak T (K)",
               title = @sprintf("ε=%.2f, Qp=%.0f, duty=%.2f, per=%.0fs",
                                pt.eps_r, pt.Qp, pt.duty, pt.period))
    for c in 1:NCYC_LONG
        lines!(ax1, 1:NPHASE, r.pk[(c - 1) * NPHASE .+ (1:NPHASE)];
               color = (:black, 0.1 + 0.9 * c / NCYC_LONG))
    end
    lines!(ax1, 1:NPHASE, r.sur_pk; color = :tomato, linewidth = 2,
           label = "surrogate settled cycle")
    axislegend(ax1; position = :rt, labelsize = 10)
    ax2 = Axis(fig[2, i]; xlabel = "cycle index", ylabel = "K", yscale = log10,
               title = "per-cycle deviation (flat = no drift) and FOM settling")
    scatterlines!(ax2, 1:NCYC_LONG, max.(r.dev, 1e-6); label = "max |FOM − surrogate|")
    scatterlines!(ax2, 1:(NCYC_LONG - 1), max.(r.chg, 1e-6);
                  label = "FOM cycle-to-cycle change")
    axislegend(ax2; position = :rc, labelsize = 10)
end
save(joinpath(FIG_DIR, "25_attractor_check.png"), fig)

println("\nsaved data/attractor_check.jld2 and figures/25_attractor_check.png")
