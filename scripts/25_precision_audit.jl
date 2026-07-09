# Precision audit of the deployed steady surrogate (POD-MLP r = 2): cast the
# whole inference pipeline (inputs, network weights, whitening scales, POD mean
# and modes) to Float32 and Float16 and re-evaluate the 102-sample held-out
# test set against the Float64 reference: relative L², QoI deltas, and where
# non-finite values first appear. No FOM solves (edge-compute question: what
# arithmetic does deployment actually need?).
#
# PRE-REGISTERED (from notes/vv-plan.md, header written before first run):
#   (1) Float32 inference is essentially lossless: held-out rel L² and both
#       QoI errors match the Float64 pipeline at the precision the report
#       quotes (9.18e-4 rel L², 3.6 K max ΔT_peak).
#   (2) Float16 fails at RECONSTRUCTION: the raw (unwhitened) coefficient
#       scale is ~1e4–1e5 against the 65,504 Float16 ceiling, so the
#       unwhitening product cscales .* ĉ is where Inf first appears (or, if
#       marginally finite, eps(Float16) at that magnitude is O(10), i.e. O(1%)
#       coefficient error at best).
#   (3) Any u⁴ evaluation in Float16 overflows for T ≥ 65504^(1/4) ≈ 16.0 K,
#       i.e. at every temperature in the operating regime (fields span
#       ~296–973 K), so residual/QoI physics can never be evaluated in F16.

include(joinpath(@__DIR__, "config.jl"))

println("=== 25_precision_audit ===")

fom = build_steady_fom(CFG)
d = load_snapshots()
pod = load_pod()
te = d["test_idx"]
Xte, Ute = d["X"][:, te], d["U"][:, te]
mus_te = [(d["mus"][1, k], d["mus"][2, k]) for k in te]
ntest = length(te)
m = load_mlp(model_path("pod_mlp_r$(DEFAULT_R)"))
Φ = Matrix(modes_r(pod.P, DEFAULT_R))

@printf("test set N = %d; whitening scales cscales = %s\n",
        ntest, join((@sprintf("%.3e", c) for c in m.cscales), ", "))

# Recursive parameter cast (NamedTuple trees of arrays, as Lux stores them).
cast(::Type{T}, x::AbstractArray) where {T} = T.(x)
cast(::Type{T}, x::NamedTuple) where {T} = map(v -> cast(T, v), x)
cast(::Type{T}, x::Tuple) where {T} = map(v -> cast(T, v), x)
cast(::Type{T}, x) where {T} = x

"Run the full inference pipeline in element type `T`; returns per-stage arrays."
function pipeline(::Type{T}) where {T}
    psT, stT = cast(T, m.ps), cast(T, m.st)
    XT = T.(Xte)
    csT = T.(m.cscales)
    ΦT, meanT = T.(Φ), T.(pod.P.mean)
    ĉ_white = predict(m.model, psT, stT, XT)      # network output (whitened)
    Ĉ = csT .* ĉ_white                            # unwhitening
    Û = ΦT * Ĉ .+ meanT                           # reconstruction
    return (; ĉ_white, Ĉ, Û)
end

finite_frac(A) = count(isfinite, A) / length(A)

"Metrics of a reconstruction matrix vs FOM truth (computed in Float64)."
function metrics(Û)
    Û64 = Float64.(Û)
    rels = [relative_l2_error(Û64[:, k], Ute[:, k]) for k in 1:ntest]
    dT = [abs(maximum(Û64[:, k]) - maximum(Ute[:, k])) for k in 1:ntest]
    dP = [qoi_errors(fom, Û64[:, k], Ute[:, k]; eps_r = mus_te[k][1]).dP_rel
          for k in 1:ntest]
    return (; rel = mean(rels), dT_max = maximum(dT), dT_med = median(dT),
            dP_max = maximum(dP))
end

ref = pipeline(Float64)
res = Dict{String,Any}("F64" => metrics(ref.Û))
rows = []
for T in (Float32, Float16)
    p = pipeline(T)
    ff = (finite_frac(p.ĉ_white), finite_frac(p.Ĉ), finite_frac(p.Û))
    @printf("\n%s finite fractions: net output %.3f | unwhitened coeffs %.3f | fields %.3f\n",
            T, ff...)
    if ff[3] == 1.0
        mt = metrics(p.Û)
        drift = maximum(abs.(Float64.(p.Û) .- ref.Û)) # vs F64 pipeline, K
        @printf("%s: rel L² %.4e | ΔT_peak max %.2f K med %.2f K | ΔP/P max %.2e | max field drift vs F64 %.2e K\n",
                T, mt.rel, mt.dT_max, mt.dT_med, mt.dP_max, drift)
        res[string(T)] = (; mt..., drift, ff)
    else
        ninf = count(!isfinite, p.Û)
        @printf("%s: %d / %d field values non-finite → reconstruction unusable\n",
                T, ninf, length(p.Û))
        res[string(T)] = (; rel = NaN, dT_max = NaN, dT_med = NaN, dP_max = NaN,
                          drift = NaN, ff)
    end
    push!(rows, (T, ff))
end

r64 = res["F64"]
@printf("\nF64 reference: rel L² %.4e | ΔT_peak max %.2f K med %.2f K | ΔP/P max %.2e\n",
        r64.rel, r64.dT_max, r64.dT_med, r64.dP_max)

# Prediction (3): u⁴ in Float16 overflows at every operating temperature.
T_over = 65504.0^0.25
frac_over = count(u -> !isfinite(rad4(Float16(u))), Ute) / length(Ute)
@printf("\nrad4 in Float16 overflows for T ≥ %.1f K; fraction of truth-field values overflowing: %.3f\n",
        T_over, frac_over)

jldsave(joinpath(DATA_DIR, "precision_audit.jld2");
        ntest, cscales = m.cscales,
        rel = Dict(k => v.rel for (k, v) in res),
        dT_max = Dict(k => v.dT_max for (k, v) in res),
        dT_med = Dict(k => v.dT_med for (k, v) in res),
        dP_max = Dict(k => v.dP_max for (k, v) in res),
        drift = Dict(k => get(v, :drift, 0.0) for (k, v) in res),
        finite_net = Dict(string(T) => ff[1] for (T, ff) in rows),
        finite_coef = Dict(string(T) => ff[2] for (T, ff) in rows),
        finite_field = Dict(string(T) => ff[3] for (T, ff) in rows),
        rad4_f16_overflow_K = T_over, rad4_f16_overflow_frac = frac_over)

println("saved data/precision_audit.jld2")
