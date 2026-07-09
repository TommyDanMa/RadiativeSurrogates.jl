# Grid-convergence error bars (GCI, ASME V&V 20 / Roache) on the FOM design
# QoIs. Three systematically refined grids (60×30 → 120×60 → 240×120, ratio 2),
# five Sobol (ε, Q) points. Per point and QoI: observed order
# p̂ = log2((f_coarse − f_mid)/(f_mid − f_fine)), Richardson extrapolation
# f_RE, and Fs = 1.25 error bands on the fine grid and on the *deployed*
# 60×30 grid (the grid every snapshot, surrogate, and QoI in this project
# lives on). Steady only; transient GCI is deferred and said so in the report.
#
# PRE-REGISTERED, round 1 (header written before first run):
#   (1) Observed order p̂ for T_peak and T_edge falls in the MMS band
#       [1.7, 2.2] at every point (code verification pinned O(h²) = 2.000).
#   (2) P_rad shows NO grid convergence to measure: the chip patch is
#       face-aligned on all three grids and config B satisfies the exact
#       discrete energy balance, so P_rad = P_in at Newton tolerance on every
#       grid. Grid-to-grid differences at ~1e-9 relative; p̂/GCI not
#       applicable. This degeneracy is itself a structural verification.
#   (3) No magnitude prediction for the T_peak band; it is compared after the
#       fact against the surrogate's 2.4 K median design-QoI error (§8.6).
#
# ROUND-1 SCORECARD (measured 2026-07-09, before the round-2 changes below):
#   (2) HIT: worst |P_rad − Q|/Q = 1.6e-12 over 15 solves.
#   (1) MISS twice, instructively. T_peak (nodal max) converges
#       NON-MONOTONICALLY (−33 mK then +1 mK on the first point): the argmax
#       node drifts as the grid refines, so the raw max is not a smooth
#       functional of h and Richardson does not apply to it. T_edge shows no
#       grid dependence at all (~1e-9 relative): the exact discrete energy
#       balance pins ∮u⁴ on every grid and the near-isothermal edge transfers
#       the pin to the mean edge temperature; a THIRD structural identity,
#       not an order control.
#
# ROUND 2 (registered after the round-1 output, before the round-2 run):
#   T_probe := u_h at the deployed grid's hot node (a vertex of all nested
#   grids; on 60×30 it IS the reported T_peak) replaces the nodal max as the
#   banding target; u_L2 := ‖u_h‖_L²(Ω) is the smooth order control. Both are
#   predicted at p̂ ∈ [1.7, 2.2]; the T_probe band on 60×30 is the honest
#   "T_peak discretization error" statement. T_peak (max) and T_edge stay in
#   the table, reported for what they are.

include(joinpath(@__DIR__, "config.jl"))
using Gridap
using QuasiMonteCarlo
using CairoMakie
CairoMakie.activate!()

println("=== 24_gci ===")

const FS = 1.25                              # GCI safety factor (3-grid study)
NPTS = 5
lb = [EPS_RANGE[1], Q_RANGE[1]]
ub = [EPS_RANGE[2], Q_RANGE[2]]
mus = QuasiMonteCarlo.sample(NPTS, lb, ub, SobolSample())

grids = ((60, 30), (120, 60), (240, 120))    # coarse (deployed) → fine
qoi_names = ("T_probe", "u_L2", "T_peak", "P_rad", "T_edge")
banded = (true, true, false, false, false)   # round 2: which QoIs Richardson applies to

"DOF index of the grid node at coordinates (x, y) (must be a vertex)."
function node_dof(fom, x, y)
    hx, hy = fom.cfg.Lx / fom.cfg.nx, fom.cfg.Ly / fom.cfg.ny
    gi, gj = round(Int, x / hx) + 1, round(Int, y / hy) + 1
    d = findfirst(k -> fom.gi[k] == gi && fom.gj[k] == gj, eachindex(fom.gi))
    @assert d !== nothing "probe point ($x, $y) is not a grid vertex"
    return d
end

foms = [build_steady_fom(RadiativeConfig(; nx, ny)) for (nx, ny) in grids]

# Fixed probe per point: the deployed grid's argmax node (a vertex everywhere).
xs_c, ys_c = node_axes(foms[1])
probe_xy = Vector{NTuple{2,Float64}}(undef, NPTS)

f = Array{Float64,3}(undef, NPTS, length(grids), length(qoi_names))
for (g, fom) in enumerate(foms)
    x_warm = nothing
    t = @elapsed for k in 1:NPTS
        s = solve_steady(fom, (mus[1, k], mus[2, k]); abstol = 1e-9, u0 = x_warm)
        @assert s.converged
        x_warm = s.x
        if g == 1
            i_star = argmax(s.x)
            probe_xy[k] = (xs_c[fom.gi[i_star]], ys_c[fom.gj[i_star]])
        end
        T_probe = s.x[node_dof(fom, probe_xy[k]...)]
        u_L2 = sqrt(sum(∫(s.uh * s.uh) * fom.dΩ))
        T_edge = sum(∫(s.uh) * fom.dΓ) / fom.cfg.Ly
        f[k, g, :] .= (T_probe, u_L2, maximum(s.x),
                       radiated_power(fom, s.uh, mus[1, k]), T_edge)
    end
    @printf("grid %3d×%-3d  n = %6d   %d solves in %.1f s\n",
            fom.cfg.nx, fom.cfg.ny, fom.n, NPTS, t)
end

# Richardson / GCI per (point, QoI). Differences below the solver-tolerance
# floor mean there is no discretization error to measure; a sign flip means
# the functional is not smooth in h (both expected: see header scorecard).
phat = fill(NaN, NPTS, length(qoi_names))
f_re = fill(NaN, NPTS, length(qoi_names))
band_fine = fill(NaN, NPTS, length(qoi_names))
band_coarse = fill(NaN, NPTS, length(qoi_names))
noise_floor(ffine) = max(1e-8 * abs(ffine), 1e-8)
gci_ok(v) = filter(!isnan, v)

println()
for (q, qn) in enumerate(qoi_names)
    println("QoI $qn:")
    for k in 1:NPTS
        fc, fm, ff = f[k, 1, q], f[k, 2, q], f[k, 3, q]
        d12, d23 = fc - fm, fm - ff
        why = abs(d23) < noise_floor(ff) ? "Δ below solver-tolerance floor" :
              d12 * d23 <= 0 ? "non-monotone in h (sign flip)" : ""
        if !isempty(why)
            @printf("  μ=(%.3f, %5.0f)  f = %.6f / %.6f / %.6f   %s → GCI n/a\n",
                    mus[1, k], mus[2, k], fc, fm, ff, why)
            continue
        end
        p = log2(d12 / d23)
        fre = ff - d23 / (2^p - 1)
        phat[k, q] = p
        f_re[k, q] = fre
        band_fine[k, q] = FS * abs(d23) / (2^p - 1)
        band_coarse[k, q] = FS * abs(fc - fre)
        @printf("  μ=(%.3f, %5.0f)  f = %.4f / %.4f / %.4f   p̂ = %.2f   f_RE = %.4f   band 60×30 ± %.3f\n",
                mus[1, k], mus[2, k], fc, fm, ff, p, fre, band_coarse[k, q])
    end
end

# P_rad grid-exactness, stated directly (round-1 prediction (2)).
prad_dev = maximum(abs(f[k, g, 4] - mus[2, k]) / mus[2, k]
                   for k in 1:NPTS, g in eachindex(grids))
@printf("\nP_rad vs injected Q: worst relative deviation %.1e over %d solves (grid-exact)\n",
        prad_dev, NPTS * length(grids))
tedge_dev = maximum(abs(f[k, 3, 5] - f[k, 1, 5]) / abs(f[k, 3, 5]) for k in 1:NPTS)
@printf("T_edge coarse-vs-fine relative deviation: worst %.1e (pinned by the energy balance)\n",
        tedge_dev)

for (q, qn) in ((1, "T_probe"), (2, "u_L2"))
    ps = gci_ok(phat[:, q])
    bc = gci_ok(band_coarse[:, q])
    @printf("%s: p̂ = %s (N = %d of %d points banded)\n", qn,
            join((@sprintf("%.2f", p) for p in ps), ", "), length(ps), NPTS)
    isempty(bc) ||
        @printf("%s discretization error, deployed 60×30 grid (Fs = %.2f): median %.3f, max %.3f (N = %d)\n",
                qn, FS, median(bc), maximum(bc), length(bc))
end

jldsave(joinpath(DATA_DIR, "gci.jld2");
        mus, grids = [collect(g) for g in grids], qoi_names = collect(qoi_names),
        banded = collect(banded), probe_xy = [collect(p) for p in probe_xy],
        f, phat, f_re, band_fine, band_coarse, prad_dev, tedge_dev, fs = FS)

# ----- figure: convergence of T_probe toward the Richardson limit ----------------
fig = Figure(size = (560, 420))
ax = Axis(fig[1, 1]; xscale = log2, yscale = log10,
          xlabel = "grid spacing h / h_fine", ylabel = "|T_probe − T_RE| (K)",
          title = "Grid convergence of the hot-node temperature (5 Sobol points)")
hs = [4.0, 2.0, 1.0]
for k in 1:NPTS
    isnan(f_re[k, 1]) && continue
    err = abs.(f[k, :, 1] .- f_re[k, 1])
    scatterlines!(ax, hs, max.(err, 1e-12);
                  label = @sprintf("ε=%.2f, Q=%.0f", mus[1, k], mus[2, k]))
end
ref = gci_ok(band_coarse[:, 1])
isempty(ref) || lines!(ax, hs, (maximum(ref) / FS) .* (hs ./ 4) .^ 2;
                       linestyle = :dash, color = :gray, label = "slope 2")
axislegend(ax; position = :rb, labelsize = 10)
save(joinpath(FIG_DIR, "23_gci.png"), fig)

println("saved data/gci.jld2 and figures/23_gci.png")
