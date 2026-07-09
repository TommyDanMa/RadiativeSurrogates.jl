# Forward uncertainty propagation through the steady surrogate, with an exact
# FOM Monte Carlo reference. Input uncertainty (stated here, used nowhere
# else): ε ~ N(0.85, 0.02) (coating emissivity tolerance around a high-ε
# design point) and Q ~ N(600, 50) W/m (load uncertainty), both truncated to
# the training box by rejection; StableRNG(SEED + 2700). QoI: steady peak
# temperature. Surrogate MC N = 100,000; FOM reference N = 10,000 (the first
# 10,000 of the same sample stream, so surrogate-vs-FOM deltas are PAIRED and
# free of MC noise). T_space is a model-form limitation (fixed 3 K deep-space
# sink; an Earth-shine LEO environment shifts the effective sink), NOT a
# random input: the surrogate was trained at fixed T_space.
#
# PRE-REGISTERED (header written before first run):
#   (1) Paired quantile deltas |surrogate − FOM| at 50/90/99/99.7% are all
#       < 4 K, i.e. within the steady surrogate's max peak-T error over the
#       102-sample test box (3.6 K); expected well under, since the UQ cloud
#       is concentrated in-box.
#   (2) The N = 100,000 surrogate quantiles match the N = 10,000 FOM
#       reference within that same band plus MC error.
#   (3) 100% of accepted samples pass in_training_box (true by construction;
#       printed as the guard statistic).

include(joinpath(@__DIR__, "config.jl"))
using CairoMakie
CairoMakie.activate!()

println("=== 27_forward_uq ===")

const EPS_MEAN, EPS_STD = 0.85, 0.02
const Q_MEAN, Q_STD = 600.0, 50.0
const N_SUR = 100_000
const N_FOM = 10_000
const QUANTS = (0.5, 0.9, 0.99, 0.997)

rng = StableRNG(SEED + 2700)

"Truncated-normal sample of size N by rejection; returns (values, acceptance)."
function trunc_normal(rng, μ, σ, lo, hi, N)
    out = Vector{Float64}(undef, N)
    acc, tries = 0, 0
    while acc < N
        x = μ + σ * randn(rng)
        tries += 1
        if lo <= x <= hi
            acc += 1
            out[acc] = x
        end
    end
    return out, N / tries
end

eps_s, acc_eps = trunc_normal(rng, EPS_MEAN, EPS_STD, EPS_RANGE..., N_SUR)
Q_s, acc_Q = trunc_normal(rng, Q_MEAN, Q_STD, Q_RANGE..., N_SUR)
inbox = count(in_training_box.(eps_s, Q_s))
@printf("ε ~ N(%.2f, %.2f), Q ~ N(%.0f, %.0f) W/m, truncated to the training box\n",
        EPS_MEAN, EPS_STD, Q_MEAN, Q_STD)
@printf("acceptance: ε %.4f, Q %.4f; in_training_box %d / %d\n",
        acc_eps, acc_Q, inbox, N_SUR)

# ----- surrogate MC (chunked reconstruction to bound memory) ----------------------
pod = load_pod()
m = load_mlp(model_path("pod_mlp_r$(DEFAULT_R)"))
X = vcat(standardize.(eps_s, EPS_RANGE...)', standardize.(Q_s, Q_RANGE...)')

T_sur = Vector{Float64}(undef, N_SUR)
t_sur = @elapsed for lo in 1:5000:N_SUR
    hi = min(lo + 4999, N_SUR)
    Ĉ = m.cscales .* predict(m.model, m.ps, m.st, X[:, lo:hi])
    T_sur[lo:hi] = vec(maximum(reconstruct(pod.P, Ĉ, DEFAULT_R); dims = 1))
end
@printf("surrogate MC: N = %d in %.2f s (warm)\n", N_SUR, t_sur)

# ----- FOM reference MC on the first N_FOM points, warm-started -------------------
fom = build_steady_fom(CFG)
T_fom = Vector{Float64}(undef, N_FOM)
x_warm = nothing
t0 = time()
for k in 1:N_FOM
    s = solve_steady(fom, (eps_s[k], Q_s[k]); abstol = 1e-9, u0 = x_warm)
    @assert s.converged
    global x_warm = s.x
    T_fom[k] = maximum(s.x)
    k % 2000 == 0 && @printf("  FOM %5d / %d (%.0f s elapsed)\n", k, N_FOM, time() - t0)
end
t_fom = time() - t0
@printf("FOM MC: N = %d in %.0f s (median %.1f ms/solve, warm-started)\n",
        N_FOM, t_fom, 1e3 * t_fom / N_FOM)

# ----- quantile table --------------------------------------------------------------
qs_fom = [quantile(T_fom, q) for q in QUANTS]
qs_sur_paired = [quantile(T_sur[1:N_FOM], q) for q in QUANTS]
qs_sur_full = [quantile(T_sur, q) for q in QUANTS]
d_pair = T_sur[1:N_FOM] .- T_fom

println("\nworst-case (peak) steady T quantiles:")
@printf("%-8s %-14s %-20s %-16s %s\n",
        "q", "FOM (N=1e4)", "surrogate paired", "Δ paired (K)", "surrogate (N=1e5)")
for (i, q) in enumerate(QUANTS)
    @printf("%-8s %-14.2f %-20.2f %+-16.2f %.2f\n", @sprintf("%.1f%%", 100q),
            qs_fom[i], qs_sur_paired[i], qs_sur_paired[i] - qs_fom[i],
            qs_sur_full[i])
end
@printf("pointwise |surrogate − FOM| on the paired N = %d: median %.2f K, max %.2f K\n",
        N_FOM, median(abs.(d_pair)), maximum(abs.(d_pair)))
@printf("3σ statement: the 99.7%% worst-case-T quantile is %.1f K (FOM) vs %.1f K (surrogate, N = 1e5), Δ %+.2f K\n",
        qs_fom[end], qs_sur_full[end], qs_sur_full[end] - qs_fom[end])
println("T_space fixed at 3 K: model-form limitation (Earth-shine shifts the effective sink), not a sampled input")

jldsave(joinpath(DATA_DIR, "forward_uq.jld2");
        eps_mean = EPS_MEAN, eps_std = EPS_STD, q_mean = Q_MEAN, q_std = Q_STD,
        n_sur = N_SUR, n_fom = N_FOM, quants = collect(QUANTS),
        qs_fom, qs_sur_paired, qs_sur_full,
        d_med = median(abs.(d_pair)), d_max = maximum(abs.(d_pair)),
        T_fom, T_sur, acc_eps, acc_Q, t_fom, t_sur)

# ----- figure ---------------------------------------------------------------------
fig = Figure(size = (640, 420))
ax = Axis(fig[1, 1]; xlabel = "worst-case (peak) steady T (K)", ylabel = "density",
          title = @sprintf("Forward UQ: ε ~ N(%.2f, %.2f), Q ~ N(%.0f, %.0f) W/m",
                           EPS_MEAN, EPS_STD, Q_MEAN, Q_STD))
hist!(ax, T_fom; bins = 60, normalization = :pdf, color = (:black, 0.35),
      label = "FOM (N = 10,000)")
hist!(ax, T_sur; bins = 120, normalization = :pdf, color = (:tomato, 0.45),
      label = "surrogate (N = 100,000)")
vlines!(ax, [qs_fom[end]]; color = :black, linestyle = :dash, label = "99.7% FOM")
vlines!(ax, [qs_sur_full[end]]; color = :tomato, linestyle = :dot,
        label = "99.7% surrogate")
axislegend(ax; position = :rt, labelsize = 10)
save(joinpath(FIG_DIR, "26_forward_uq.png"), fig)

println("saved data/forward_uq.jld2 and figures/26_forward_uq.png")
