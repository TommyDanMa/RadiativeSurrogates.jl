# Stabilized learned-vector-field arm for the transient latent dynamics.
#
# ============================ PRE-REGISTERED (2026-07-12) ============================
# Written BEFORE this script's first run. Script 32 round 4 was aborted after about
# 13 hours based on training-infrastructure telemetry only: 143 dt_NaN warnings were
# observed. No latent_ode artifact was produced and no held-out L1-L4 score existed.
# This is a distinct, prospectively specified stabilized arm, not a continuation or
# post-result adjustment of script 32.
#
# FROZEN REFERENCES AND SCORECARD: reuse scripts/32 exactly: mapper trajectory rel L²
# 1.737e-2 at r=8; mapper-at-24 5.2e-2; exact-Galerkin floor 3.719e-6; projection
# floor 2.949e-6; the frozen attractor and design-study references; L1 < 1.737e-2,
# L2 < 5.2e-2, L3 settled-cycle deviation at most 2× dev(3), and L4 Spearman ≥
# 0.998, top-5 ≥ 4/5, same optimum, at both operating points. Any rollout failure
# in a prediction's evaluation prevents a hit.
#
# DATA, MODEL, AND CLOCK RULE: the same train/test split, deployed whitening, model
# 13→64→64→8 (5,576 parameters), Stage-A baseline, seed SEED+3100 for deployed and
# SEED+3200+s for ensemble initialization only, and N=20 iff deployed total training
# wall time is at most 30 minutes, otherwise N=5. Inputs and phase-time convention
# are unchanged from script 32. Fresh outputs only: data/latent_ode_stabilized.jld2,
# data/model_latent_ode_stabilized_r8.jld2, figures/32_latent_ode_stabilized.png,
# and checkpoints under data/latent_ode_stabilized/.
#
# STAGE A: retain the Euclidean-coordinate derivative target
#   c_dot = Phi' M^-1(force - residual),
# because c=Phi'(u-mean). This is intentionally different from script 30's Galerkin
# target (Phi'MPhi)^-1 Phi'f. The raw Lux output is bounded modewise as
# fscale*tanh(raw/fscale), where fscale_i=max(1.25*max(abs(target_i)), 1e-6), using
# training targets only and fixed before Stage B. The bounded model is itself fit
# for 3,000 full-batch epochs, lr 1e-3→1e-5, keep-best.
#
# STAGE B: multiple-shooting horizons G=5,11,21,61 with 60,40,30,20 epochs and
# clipped Adam lr 3e-5→1e-6. The fixed mixed loss is
#   rollout + 0.10*derivative_anchor + 0.01*state_barrier.
# Rollout and derivative terms are normalized per mode and time point. Their scales
# use training data only and are fixed. The smooth barrier uses each mode's training
# Ctilde interval expanded by 25% of its range. Solver stop bounds expand that same
# interval by 200% of its range (minimum half-width 1). A failed segment contributes
# the constant 100 penalty, but the nonzero full derivative anchor is always present,
# so every mixed loss retains a parameter gradient. No held-out data tune any choice.
#
# SOLVERS AND GUARDS: training uses the one-step L-stable
# OrdinaryDiffEqSDIRK.KenCarp4 at abstol=reltol=1e-6; independent evaluation uses
# FBDF at 1e-9. Every known load switch is an explicit split-chain boundary.
# unstable_check stops nonfinite or out-of-stop-bound states; maxiters=10_000 and
# SciMLLogging.None() (the installed v7 replacement for verbose=false) bound failure
# behavior. Retcodes, failed counts, dimensions,
# finiteness, and bounds are checked. Stage B uses
# InterpolatingAdjoint(autojacvec=ZygoteVJP(), checkpointing=false). The installed
# SciMLSensitivity 7.114 constructor supports checkpointing, but its split-chain
# endpoint replay raises a bounds error on the required switch-crossing preflight;
# the supported non-checkpointed API is therefore pre-registered for this arm.
# BacksolveAdjoint is not used.
#
# PREFLIGHT: before deployed production, deterministic training-only probes require
# a finite bounded Stage-A fit and rollout; KenCarp4/FBDF agreement ≤5e-4 normalized
# RMS; adjoint directional derivatives within max(5e-3 absolute, 8e-2 relative) of
# centered finite differences for one non-switch and one switch-crossing plan; a
# finite mixed gradient under intentional termination; and checkpoint round-trip.
# Stable PROBE_PASS/PROBE_FAIL sentinels are printed and flushed. Smoke mode uses
# tiny budgets and separate smoke checkpoints, runs two members plus resume, and
# never writes production aggregate/model/figure artifacts.
#
# RECOVERY: prep/deployed runs serially, independent ensemble members run in isolated
# one-thread Julia children (at most three concurrently), and finalize is the sole
# aggregate writer. Stage A and every completed G horizon are checkpointed with
# protocol, seed, parameters, restart data, histories, and solver/sensitivity/
# environment metadata. Reliable resume is from completed Stage A or a full member;
# partial-Adam horizon checkpoints are receipts only and retrain Stage B from Stage A.
# Workers atomically rename unique temporary JLD2 files. Finalize refuses missing or
# stale members.
# =====================================================================================

include(joinpath(@__DIR__, "config.jl"))
using Gridap
using OrdinaryDiffEq
using OrdinaryDiffEqSDIRK
using SciMLSensitivity
using ComponentArrays
using Optimisers
using Zygote
using Lux
using CairoMakie
CairoMakie.activate!()

const PROTOCOL_ID = "latent-ode-stabilized-v1-2026-07-12"
const PHASE = get(ENV, "LATENT33_PHASE", "prep")
const SMOKE = get(ENV, "LATENT33_SMOKE", "0") == "1"
const MEMBER_INDEX = parse(Int, get(ENV, "LATENT33_MEMBER", "0"))
const R = DEFAULT_R_T
const GS = (5, 11, 21, 61)
const EPOCHS_B = (60, 40, 30, 20)
const TOL_TRAIN = 1e-6
const TOL_EVAL = 1e-9
const LOSS_WEIGHTS = (rollout = 1.0, anchor = 0.10, barrier = 0.01)
const FAILURE_PENALTY = 100.0
const MAXITERS_SEGMENT = 10_000
const CHECKPOINT_ROOT = joinpath(DATA_DIR,
                                 SMOKE ? "latent_ode_stabilized_smoke" :
                                 "latent_ode_stabilized")
const MANIFEST_PATH = joinpath(CHECKPOINT_ROOT, "manifest.jld2")
const SENSE = InterpolatingAdjoint(autojacvec = ZygoteVJP(), checkpointing = false)
const SENSE_NAME = "InterpolatingAdjoint(ZygoteVJP,checkpointing=false)"

println("=== 33_latent_ode_stabilized phase=$PHASE smoke=$SMOKE ===")
flush(stdout)
mkpath(CHECKPOINT_ROOT)
LinearAlgebra.BLAS.set_num_threads(parse(Int, get(ENV, "OPENBLAS_NUM_THREADS", "3")))

struct SubPlan
    a::Float64
    b::Float64
    w::Float64
    saveat::Vector{Float64}
    keep::Vector{Int}
end

struct SegPlan
    k::Int
    j0::Int
    j1::Int
    subs::Vector{SubPlan}
end

mutable struct RunTelemetry
    solves::Int
    failures::Int
    retcodes::Dict{String,Int}
end

RunTelemetry() = RunTelemetry(0, 0, Dict{String,Int}())
telemetry_dict(t) = Dict("solves" => t.solves, "failures" => t.failures,
                         "retcodes" => t.retcodes)

function environment_metadata()
    return Dict(
        "julia_version" => string(VERSION),
        "sciml_sensitivity" => "7.114",
        "solver_train" => "OrdinaryDiffEqSDIRK.KenCarp4",
        "solver_eval" => "FBDF",
        "sensealg" => SENSE_NAME,
        "julia_threads" => Threads.nthreads(),
        "blas_threads" => LinearAlgebra.BLAS.get_num_threads(),
        "openblas_num_threads" => get(ENV, "OPENBLAS_NUM_THREADS", ""),
        "omp_num_threads" => get(ENV, "OMP_NUM_THREADS", ""),
        "mkl_num_threads" => get(ENV, "MKL_NUM_THREADS", ""),
    )
end

function atomic_jldsave(path; kwargs...)
    tmp = path * ".tmp.$(getpid()).$(time_ns())"
    jldsave(tmp; kwargs...)
    mv(tmp, path; force = true)
    return path
end

function checkpoint_path(label)
    return joinpath(CHECKPOINT_ROOT, "$label.jld2")
end

function matching_checkpoint(path, seed, kind)
    isfile(path) || return false
    d = load(path)
    return get(d, "protocol_id", "") == PROTOCOL_ID &&
           get(d, "seed", -1) == seed && get(d, "kind", "") == kind &&
           get(d, "smoke", !SMOKE) == SMOKE
end

function save_checkpoint(path; seed, kind, ps, st, fscale, losses_a = Float64[],
                         losses_b = Float64[], failure_history = Int[],
                         retcode_history = Dict{String,Int}(), wall = 0.0,
                         completed_g = 0, complete = false,
                         restart_data = Dict{String,Any}())
    atomic_jldsave(path; protocol_id = PROTOCOL_ID, seed, kind, smoke = SMOKE,
                   ps = deepcopy(NamedTuple(ps)), st, fscale, losses_a, losses_b,
                   failure_history, retcode_history, wall, completed_g, complete,
                   restart_data,
                   restart = "restart Stage B from completed Stage A",
                   gs = collect(GS), epochs_b = collect(EPOCHS_B),
                   loss_weights = Dict(pairs(LOSS_WEIGHTS)),
                   failure_penalty = FAILURE_PENALTY,
                   tol_train = TOL_TRAIN, tol_eval = TOL_EVAL,
                   environment = environment_metadata())
end

function load_context()
    fom = build_steady_fom(CFG)
    dt = load(snapshots_t_path())
    pod = load(pod_t_path())
    P = PODBasis(pod["mean"], pod["modes"], pod["svals"])
    U3, mus = dt["U"], dt["mus"]
    ttr, tte = collect(dt["train_idx"]), collect(dt["test_idx"])
    n, nt, _ = size(U3)
    cs_dep = load_mlp(model_path("pod_mlp_t_r$(R)")).cscales
    Φ = Matrix(modes_r(P, R))
    ū = P.mean
    sgrid = collect(range(0.0, Float64(NCYCLES); length = nt))
    M = mass_matrix(fom)
    # Euclidean POD coordinates require Φ'M⁻¹, not the Galerkin mass inverse.
    Z = cholesky(Symmetric(M)) \ Φ
    Funit = load_vector(fom, 1.0)
    return (; fom, dt, P, U3, mus, ttr, tte, n, nt, cs_dep, Φ, ū, sgrid, Z, Funit)
end

function stage_a_data(ctx, k, csv)
    ε, Qp, duty, period = ctx.mus[:, k]
    μst = collect(standardize_mu_t(ε, Qp, duty, period))
    q = square_wave(Qp, period, duty)
    aop = Gridap.FESpaces.get_algebraic_operator(
        FEOperator(steady_form(ctx.fom; eps_r = ε, Q = 0.0),
                   ctx.fom.U, ctx.fom.V))
    rb = similar(ctx.ū)
    X = zeros(13, ctx.nt)
    Y = zeros(R, ctx.nt)
    Ctilde = Matrix(project(ctx.P, ctx.U3[:, :, k], R)) ./ csv
    for j in 1:ctx.nt
        tj = ctx.sgrid[j] * period
        Gridap.Algebra.residual!(rb, aop, collect(ctx.U3[:, j, k]))
        cdot = ctx.Z' * (q(tj) .* ctx.Funit .- rb)
        X[:, j] = vcat(Ctilde[:, j], μst, q(tj) > 0 ? 1.0 : 0.0)
        Y[:, j] = (cdot ./ csv) .* period
    end
    return (; X, Y, Ctilde, μst, duty, period)
end

function assemble_data(ctx, ks, csv)
    println("assembling exact Stage-A targets for $(length(ks)) trajectories...")
    flush(stdout)
    return Dict(k => stage_a_data(ctx, k, csv) for k in ks)
end

load_on(s, duty) = mod(s, 1.0) < duty ? 1.0 : 0.0

function build_chain(s0, s1, duty, targets)
    boundaries = [s0]
    for k0 in floor(Int, s0):ceil(Int, s1)
        for switch in (Float64(k0), k0 + duty)
            s0 < switch < s1 && push!(boundaries, switch)
        end
    end
    push!(boundaries, s1)
    boundaries = sort(unique(boundaries))
    subs = SubPlan[]
    for i in 1:(length(boundaries) - 1)
        a, b = boundaries[i], boundaries[i + 1]
        in_window = [s for s in targets if a < s <= b]
        saveat = sort(unique(vcat(in_window, [b])))
        keep = findall(s -> any(==(s), in_window), saveat)
        push!(subs, SubPlan(a, b, load_on((a + b) / 2, duty), saveat, keep))
    end
    return subs
end

function build_plans(ctx, dat, ks, G)
    return [SegPlan(k, j0, j0 + G - 1,
                    build_chain(ctx.sgrid[j0], ctx.sgrid[j0 + G - 1],
                                dat[k].duty,
                                ctx.sgrid[(j0 + 1):(j0 + G - 1)]))
            for k in ks for j0 in 1:(G - 1):(ctx.nt - G + 1)]
end

bounded_output(model, st, ps, x, fscale) =
    fscale .* tanh.(first(Lux.apply(model, x, ps, st)) ./ fscale)

function segment_field(model, st, μst, w, fscale, parameter_axes)
    return (c, pflat, s) -> begin
        ps = ComponentArray(pflat, parameter_axes)
        bounded_output(model, st, ps, vcat(c, μst, w), fscale)
    end
end

function record_solve!(telemetry, retcode, ok)
    telemetry === nothing && return
    Zygote.ignore() do
        code = string(retcode)
        telemetry.solves += 1
        telemetry.failures += !ok
        telemetry.retcodes[code] = get(telemetry.retcodes, code, 0) + 1
    end
end

function exec_chain(model, st, ps, μst, c0, subs, fscale, stop_lo, stop_hi;
                    training, sensealg = nothing, telemetry = nothing,
                    validate = true)
    c = c0
    blocks = ()
    parameter_axes = getaxes(ps)
    pflat = collect(ps)
    for sub in subs
        field = segment_field(model, st, μst, sub.w, fscale, parameter_axes)
        problem = ODEProblem(field, c, (sub.a, sub.b), pflat)
        # The same keyword is forwarded to the augmented adjoint solve. Bounds are
        # physical-state guards, so apply them only to the R-state forward solve;
        # the augmented adjoint still receives the nonfinite guard.
        unstable = (dt, u, p, t) ->
            any(x -> !isfinite(x), u) ||
            (length(u) == R &&
             (any(u .< stop_lo) || any(u .> stop_hi)))
        common = (; saveat = sub.saveat, save_start = false, dense = true,
                  abstol = training ? TOL_TRAIN : TOL_EVAL,
                  reltol = training ? TOL_TRAIN : TOL_EVAL,
                  unstable_check = unstable, maxiters = MAXITERS_SEGMENT,
                  verbose = OrdinaryDiffEq.SciMLLogging.None())
        sol = if training
            OrdinaryDiffEq.solve(problem, OrdinaryDiffEqSDIRK.KenCarp4();
                                 common..., sensealg)
        else
            OrdinaryDiffEq.solve(problem, FBDF(); common...)
        end
        A = Array(sol)
        if validate
            ok = Zygote.ignore() do
                OrdinaryDiffEq.SciMLBase.successful_retcode(sol) &&
                    size(A) == (R, length(sub.saveat)) && all(isfinite, A) &&
                    all(A .>= stop_lo) && all(A .<= stop_hi)
            end
            record_solve!(telemetry, sol.retcode, ok)
            ok || return nothing
        end
        c = A[:, end]
        isempty(sub.keep) || (blocks = (blocks..., A[:, sub.keep]))
    end
    values = isempty(blocks) ? zeros(eltype(c), R, 0) : reduce(hcat, blocks)
    return values
end

function training_scales(dat, ks)
    XA = reduce(hcat, (dat[k].X for k in ks))
    YA = reduce(hcat, (dat[k].Y for k in ks))
    C = reduce(hcat, (dat[k].Ctilde for k in ks))
    fscale = max.(1.25 .* vec(maximum(abs.(YA); dims = 2)), 1e-6)
    yscale = max.(sqrt.(vec(mean(abs2, YA; dims = 2))), 1e-6)
    cscale = max.(sqrt.(vec(mean(abs2, C; dims = 2))), 1e-3)
    cmin = vec(minimum(C; dims = 2))
    cmax = vec(maximum(C; dims = 2))
    widths = max.(cmax .- cmin, 1e-3)
    barrier_lo = cmin .- 0.25 .* widths
    barrier_hi = cmax .+ 0.25 .* widths
    stop_lo = cmin .- 2.0 .* max.(widths, 1.0)
    stop_hi = cmax .+ 2.0 .* max.(widths, 1.0)
    return (; XA, YA, fscale, yscale, cscale, barrier_lo, barrier_hi,
            stop_lo, stop_hi)
end

function stage_a_loss(model, st, ps, scales)
    prediction = bounded_output(model, st, ps, scales.XA, scales.fscale)
    return mean(abs2, (prediction .- scales.YA) ./ scales.yscale)
end

function train_stage_a(model, st, ps0, scales; epochs)
    ps = ComponentArray(ps0)
    opt = Optimisers.setup(Optimisers.Adam(1e-3), ps)
    gamma = epochs > 1 ? (1e-5 / 1e-3)^(1 / (epochs - 1)) : 1.0
    lr = 1e-3
    best_loss, best_ps = Inf, deepcopy(ps)
    losses = Float64[]
    for epoch in 1:epochs
        result = Zygote.withgradient(p -> stage_a_loss(model, st, p, scales), ps)
        grad = result.grad[1]
        isfinite(result.val) && grad !== nothing && isfinite(norm(grad)) ||
            error("nonfinite Stage-A objective or gradient at epoch $epoch")
        push!(losses, result.val)
        if result.val < best_loss
            best_loss = result.val
            best_ps = deepcopy(ps)
        end
        opt, ps = Optimisers.update(opt, ps, grad)
        lr *= gamma
        Optimisers.adjust!(opt, lr)
    end
    return NamedTuple(best_ps), losses
end

softplus(x) = log1p(exp(-abs(x))) + max(x, zero(x))

function state_barrier(C, scales)
    below = softplus.((scales.barrier_lo .- C) ./ scales.cscale)
    above = softplus.((C .- scales.barrier_hi) ./ scales.cscale)
    return mean(abs2, below) + mean(abs2, above)
end

function mixed_loss(model, st, ps, dat, plans, scales;
                    stop_lo = scales.stop_lo, stop_hi = scales.stop_hi,
                    telemetry = nothing)
    rollout_sum = zero(eltype(ps))
    barrier_sum = zero(eltype(ps))
    point_count = 0
    for plan in plans
        d = dat[plan.k]
        result = exec_chain(model, st, ps, d.μst, d.Ctilde[:, plan.j0],
                            plan.subs, scales.fscale, stop_lo, stop_hi;
                            training = true, sensealg = SENSE, telemetry)
        npoints = plan.j1 - plan.j0
        if result !== nothing
            target = d.Ctilde[:, (plan.j0 + 1):plan.j1]
            rollout_sum += sum(abs2, (result .- target) ./ scales.cscale)
            barrier_sum += npoints * state_barrier(result, scales)
        else
            rollout_sum += FAILURE_PENALTY * R * npoints
        end
        point_count += npoints
    end
    rollout = rollout_sum / (R * point_count)
    barrier = barrier_sum / point_count
    anchor = stage_a_loss(model, st, ps, scales)
    return LOSS_WEIGHTS.rollout * rollout +
           LOSS_WEIGHTS.anchor * anchor +
           LOSS_WEIGHTS.barrier * barrier
end

function train_stage_b(model, st, ps_a, dat, ks, scales; epochs_b, checkpoint_label,
                       seed, verbose)
    ps = ComponentArray(ps_a)
    opt = Optimisers.setup(
        Optimisers.OptimiserChain(Optimisers.ClipNorm(1.0), Optimisers.Adam(3e-5)),
        ps)
    total_epochs = sum(epochs_b)
    gamma = total_epochs > 1 ? (1e-6 / 3e-5)^(1 / (total_epochs - 1)) : 1.0
    lr = 3e-5
    best_loss, best_ps = Inf, deepcopy(ps)
    losses = Float64[]
    failures = Int[]
    retcodes = Dict{String,Int}()
    for (round, G) in enumerate(GS)
        epochs_b[round] == 0 && continue
        plans = build_plans(CONTEXT[], dat, ks, G)
        SMOKE && (plans = plans[1:1])
        for epoch in 1:epochs_b[round]
            telemetry = RunTelemetry()
            result = Zygote.withgradient(
                p -> mixed_loss(model, st, p, dat, plans, scales; telemetry), ps)
            grad = result.grad[1]
            grad === nothing && error("mixed objective unexpectedly has no gradient")
            push!(losses, result.val)
            push!(failures, telemetry.failures)
            for (code, count) in telemetry.retcodes
                retcodes[code] = get(retcodes, code, 0) + count
            end
            if isfinite(result.val) && isfinite(norm(grad)) && result.val < best_loss
                best_loss = result.val
                best_ps = deepcopy(ps)
            end
            isfinite(norm(grad)) ||
                error("nonfinite mixed gradient at G=$G epoch=$epoch")
            opt, ps = Optimisers.update(opt, ps, grad)
            lr *= gamma
            Optimisers.adjust!(opt, lr)
            if verbose && (epoch == 1 || epoch % 10 == 0)
                @printf("  G=%2d epoch=%3d loss=%.3e |g|=%.2e fails=%d/%d\n",
                        G, epoch, result.val, norm(grad), telemetry.failures,
                        telemetry.solves)
                flush(stdout)
            end
        end
        receipt = checkpoint_path("$(checkpoint_label)_G$(G)")
        save_checkpoint(receipt; seed, kind = "horizon_receipt",
                        ps = best_ps, st, fscale = scales.fscale,
                        losses_b = losses, failure_history = failures,
                        retcode_history = retcodes, completed_g = G,
                        restart_data = restart_metadata(scales, ks))
    end
    return NamedTuple(best_ps), losses, failures, retcodes
end

const CONTEXT = Ref{Any}(nothing)

function restart_metadata(scales, ks)
    return Dict{String,Any}(
        "train_indices" => collect(ks),
        "yscale" => scales.yscale,
        "cscale" => scales.cscale,
        "barrier_lo" => scales.barrier_lo,
        "barrier_hi" => scales.barrier_hi,
        "stop_lo" => scales.stop_lo,
        "stop_hi" => scales.stop_hi,
        "architecture" => "13→64→64→8",
        "parameter_count" => 5576,
    )
end

function member_path(index)
    return checkpoint_path(index == 0 ? "deployed" : "member_$(lpad(index, 2, '0'))")
end

function train_member(ctx, seed, dat, ks; label, force = false)
    path = label == "deployed" ? member_path(0) :
           member_path(parse(Int, split(label, "_")[end]))
    if isfile(path) && !matching_checkpoint(path, seed, "member")
        error("stale checkpoint rejected: $path")
    end
    if !force && matching_checkpoint(path, seed, "member")
        d = load(path)
        get(d, "complete", false) || error("incomplete member checkpoint: $path")
        println("RESUME_PASS $label")
        flush(stdout)
        return d
    end
    model, ps0, st = build_pod_mlp(R; indim = 13, hidden = 64, depth = 2,
                                   rng = StableRNG(seed))
    length(ComponentArray(ps0)) == 5576 ||
        error("registered architecture must contain 5,576 parameters")
    scales = training_scales(dat, ks)
    epochs_a = SMOKE ? 3 : 3000
    epochs_b = SMOKE ? (1, 0, 0, 0) : EPOCHS_B
    started = time()
    prior_wall = 0.0
    stage_a_path = checkpoint_path("$(label)_stage_a")
    if isfile(stage_a_path) && !matching_checkpoint(stage_a_path, seed, "stage_a")
        error("stale Stage-A checkpoint rejected: $stage_a_path")
    end
    if matching_checkpoint(stage_a_path, seed, "stage_a")
        stage_a = load(stage_a_path)
        ps_a, losses_a = stage_a["ps"], stage_a["losses_a"]
        prior_wall = stage_a["wall"]
        stage_a["fscale"] == scales.fscale ||
            error("Stage-A checkpoint has mismatched training-only fscale")
        println("RESUME_STAGE_A $label")
        flush(stdout)
    else
        ps_a, losses_a =
            train_stage_a(model, st, ps0, scales; epochs = epochs_a)
        save_checkpoint(stage_a_path; seed, kind = "stage_a", ps = ps_a, st,
                        fscale = scales.fscale, losses_a,
                        wall = time() - started,
                        restart_data = restart_metadata(scales, ks))
    end
    ps_b, losses_b, failures, retcodes =
        train_stage_b(model, st, ps_a, dat, ks, scales; epochs_b,
                      checkpoint_label = label, seed,
                      verbose = label == "deployed")
    wall = prior_wall + time() - started
    save_checkpoint(path; seed, kind = "member", ps = ps_b, st,
                    fscale = scales.fscale, losses_a, losses_b,
                    failure_history = failures, retcode_history = retcodes,
                    wall, completed_g = SMOKE ? GS[1] : GS[end], complete = true,
                    restart_data = restart_metadata(scales, ks))
    return load(path)
end

function evaluation_bounds(dat, ks)
    scales = training_scales(dat, ks)
    return scales.stop_lo, scales.stop_hi
end

function eval_rollouts(ctx, model, st, ps, dat, csv, fscale, train_ks; ks = ctx.tte)
    stop_lo, stop_hi = evaluation_bounds(dat, train_ks)
    rels = Float64[]
    failures = 0
    telemetry = RunTelemetry()
    for k in ks
        d = dat[k]
        subs = build_chain(ctx.sgrid[1], ctx.sgrid[end], d.duty, ctx.sgrid[2:end])
        result = exec_chain(model, st, ComponentArray(ps), d.μst,
                            d.Ctilde[:, 1], subs, fscale, stop_lo, stop_hi;
                            training = false, telemetry)
        if result === nothing
            failures += 1
            continue
        end
        Uhat = Matrix(reconstruct(ctx.P,
                                  csv .* hcat(d.Ctilde[:, 1], result), R))
        push!(rels, norm(Uhat - ctx.U3[:, :, k]) / norm(ctx.U3[:, :, k]))
    end
    return rels, failures, telemetry
end

safemean(x) = isempty(x) ? NaN : mean(x)

function probe_pass(name, condition, detail)
    sentinel = condition ? "PROBE_PASS" : "PROBE_FAIL"
    println("$sentinel $name $detail")
    flush(stdout)
    condition || error("preflight probe failed: $name")
end

function probe_plan(ctx, dat, k; crossing)
    duty = dat[k].duty
    if crossing
        j0 = findfirst(j -> ctx.sgrid[j] < duty < ctx.sgrid[min(j + 2, ctx.nt)],
                       1:(ctx.nt - 2))
        j0 === nothing && error("could not construct switch-crossing probe")
        j1 = min(j0 + 2, ctx.nt)
    else
        j0 = findfirst(j -> begin
            a, b = ctx.sgrid[j], ctx.sgrid[j + 1]
            !any(s -> a < s < b, (floor(a) + duty, ceil(a)))
        end, 1:(ctx.nt - 1))
        j0 === nothing && error("could not construct non-switch probe")
        j1 = j0 + 1
    end
    return SegPlan(k, j0, j1,
                   build_chain(ctx.sgrid[j0], ctx.sgrid[j1], duty,
                               ctx.sgrid[(j0 + 1):j1]))
end

function directional_probe(model, st, ps, dat, scales, plan, name)
    theta = ComponentArray(ps)
    # A coordinate direction avoids changing ComponentArray axes during finite
    # differencing while still testing the complete solve pullback.
    index = length(theta)
    objective(p) = mixed_loss(model, st, p, dat, [plan], scales)
    grad = Zygote.gradient(objective, theta)[1]
    ad = grad[index]
    h = 2e-5
    plus = deepcopy(theta)
    minus = deepcopy(theta)
    plus[index] += h
    minus[index] -= h
    fd = (objective(plus) - objective(minus)) / (2h)
    err = abs(ad - fd)
    tol = max(5e-3, 8e-2 * max(abs(ad), abs(fd)))
    probe_pass(name, isfinite(ad) && isfinite(fd) && err <= tol,
               @sprintf("ad=%.3e fd=%.3e err=%.2e tol=%.2e", ad, fd, err, tol))
end

function run_probes(ctx, dat, ks)
    k = first(ks)
    model, ps0, st = build_pod_mlp(R; indim = 13, hidden = 64, depth = 2,
                                   rng = StableRNG(SEED + 3399))
    scales = training_scales(dat, ks)
    ps_a, losses = train_stage_a(model, st, ps0, scales; epochs = 3)
    plan_plain = probe_plan(ctx, dat, k; crossing = false)
    result_train = exec_chain(model, st, ComponentArray(ps_a), dat[k].μst,
                              dat[k].Ctilde[:, plan_plain.j0], plan_plain.subs,
                              scales.fscale, scales.stop_lo, scales.stop_hi;
                              training = true, sensealg = SENSE)
    probe_pass("stage_a_finite", all(isfinite, losses) && result_train !== nothing,
               "loss=$(last(losses))")

    result_eval = exec_chain(model, st, ComponentArray(ps_a), dat[k].μst,
                             dat[k].Ctilde[:, plan_plain.j0], plan_plain.subs,
                             scales.fscale, scales.stop_lo, scales.stop_hi;
                             training = false)
    agreement = result_train !== nothing && result_eval !== nothing ?
                sqrt(mean(abs2, (result_train .- result_eval) ./
                                scales.cscale)) : Inf
    probe_pass("solver_agreement", agreement <= 5e-4,
               @sprintf("normalized_rms=%.3e", agreement))

    directional_probe(model, st, ps_a, dat, scales, plan_plain,
                      "adjoint_non_switch")
    plan_switch = probe_plan(ctx, dat, k; crossing = true)
    directional_probe(model, st, ps_a, dat, scales, plan_switch,
                      "adjoint_switch_crossing")

    tiny_lo = dat[k].Ctilde[:, plan_plain.j0] .+ 1e-12
    tiny_hi = tiny_lo .+ 1e-12
    grad_bad = Zygote.gradient(
        p -> mixed_loss(model, st, p, dat, [plan_plain], scales;
                        stop_lo = tiny_lo, stop_hi = tiny_hi),
        ComponentArray(ps_a))[1]
    probe_pass("terminated_segment_gradient",
               grad_bad !== nothing && all(isfinite, grad_bad) && norm(grad_bad) > 0,
               @sprintf("gradient_norm=%.3e", norm(grad_bad)))

    probe_file = checkpoint_path("probe_roundtrip")
    save_checkpoint(probe_file; seed = SEED + 3399, kind = "probe",
                    ps = ps_a, st, fscale = scales.fscale, losses_a = losses,
                    complete = true)
    loaded = load(probe_file)
    roundtrip = loaded["protocol_id"] == PROTOCOL_ID &&
                loaded["seed"] == SEED + 3399 &&
                ComponentArray(loaded["ps"]) == ComponentArray(ps_a)
    probe_pass("checkpoint_roundtrip", roundtrip, "protocol_and_parameters_match")
end

function prep_phase(ctx)
    prep_ks = SMOKE ? ctx.ttr[1:2] : ctx.ttr
    needed = unique(vcat(prep_ks, SMOKE ? Int[] : ctx.tte))
    dat = assemble_data(ctx, needed, ctx.cs_dep)
    run_probes(ctx, dat, prep_ks)
    deployed = train_member(ctx, SEED + 3100, dat, prep_ks; label = "deployed")
    nens = SMOKE ? 2 : (deployed["wall"] <= 1800 ? 20 : 5)
    atomic_jldsave(MANIFEST_PATH; protocol_id = PROTOCOL_ID, smoke = SMOKE, nens,
                   deployed_wall = deployed["wall"], train_indices = prep_ks,
                   environment = environment_metadata())
    println("PREP_DONE NENS=$nens deployed_wall=$(deployed["wall"])")
    flush(stdout)
end

function member_phase(ctx, index)
    index >= 1 || error("LATENT33_MEMBER must be positive")
    manifest = load(MANIFEST_PATH)
    manifest["protocol_id"] == PROTOCOL_ID || error("stale manifest protocol")
    index <= manifest["nens"] || error("member $index exceeds NENS")
    ks = collect(manifest["train_indices"])
    dat = assemble_data(ctx, ks, ctx.cs_dep)
    train_member(ctx, SEED + 3200 + index, dat, ks;
                 label = "member_$index")
    println("MEMBER_DONE index=$index")
    flush(stdout)
end

function load_member(index, expected_seed)
    path = member_path(index)
    matching_checkpoint(path, expected_seed, "member") ||
        error("missing or stale member checkpoint: $path")
    d = load(path)
    get(d, "complete", false) || error("member is incomplete: $path")
    return d
end

function final_evaluations(ctx, deployed, dat, train_ks)
    model, _, _ = build_pod_mlp(R; indim = 13, hidden = 64, depth = 2,
                                rng = StableRNG(SEED + 3100))
    ps = deployed["ps"]
    st = deployed["st"]
    fscale = deployed["fscale"]
    rels_dep, fail_dep, tele_dep =
        eval_rollouts(ctx, model, st, ps, dat, ctx.cs_dep, fscale, train_ks)

    stage_a = load(checkpoint_path("deployed_stage_a"))
    rels_a, fail_a, tele_a =
        eval_rollouts(ctx, model, stage_a["st"], stage_a["ps"], dat, ctx.cs_dep,
                      stage_a["fscale"], train_ks)
    return (; model, ps, st, fscale, rels_dep, fail_dep, tele_dep,
            rels_a, fail_a, tele_a)
end

function l2_retrain(ctx)
    ks24 = ctx.ttr[1:24]
    C24 = Matrix(project(ctx.P, reshape(ctx.U3[:, :, ks24], ctx.n, :), R))
    s24 = vec(std(C24; dims = 2))
    cs24 = max.(s24, 0.01 * s24[1])
    dat24 = assemble_data(ctx, vcat(ks24, ctx.tte), cs24)
    mem24 = train_member(ctx, SEED + 3100, dat24, collect(ks24);
                         label = "member_24", force = true)
    model, _, _ = build_pod_mlp(R; indim = 13, hidden = 64, depth = 2,
                                rng = StableRNG(SEED + 3100))
    rels24, fail24, telemetry =
        eval_rollouts(ctx, model, mem24["st"], mem24["ps"], dat24, cs24,
                      mem24["fscale"], collect(ks24))
    return (; rels24, fail24, telemetry, cs24)
end

function l3_evaluation(ctx, evals, dat, train_ks)
    at = load(joinpath(DATA_DIR, "attractor_check.jld2"))
    ncycles = at["ncycles"]
    grid = collect(range(0.0, Float64(ncycles);
                         length = ncycles * NPHASE + 1))
    stop_lo, stop_hi = evaluation_bounds(dat, train_ks)
    deviations = Vector{Vector{Float64}}()
    failures = 0
    telemetry = RunTelemetry()
    for (i, point) in enumerate(at["points"])
        steady = solve_steady(ctx.fom; eps_r = point[:eps_r],
                             Q = point[:duty] * point[:Qp])
        c0 = (ctx.Φ' * (steady.x .- ctx.ū)) ./ ctx.cs_dep
        μst = collect(standardize_mu_t(point[:eps_r], point[:Qp],
                                       point[:duty], point[:period]))
        subs = build_chain(grid[1], grid[end], point[:duty], grid[2:end])
        result = exec_chain(evals.model, evals.st, ComponentArray(evals.ps),
                            μst, c0, subs, evals.fscale, stop_lo, stop_hi;
                            training = false, telemetry)
        if result === nothing
            failures += 1
            push!(deviations, fill(Inf, ncycles))
            continue
        end
        Uhat = Matrix(reconstruct(ctx.P,
                                  ctx.cs_dep .* hcat(c0, result), R))
        peaks = vec(maximum(Uhat; dims = 1))
        settled = peaks[((ncycles - 1) * NPHASE + 1):(ncycles * NPHASE)]
        reference = at["pk"][i]
        trace(cycle) = reference[(cycle - 1) * NPHASE .+ (1:NPHASE)]
        push!(deviations,
              [maximum(abs.(trace(cycle) .- settled)) for cycle in 1:ncycles])
    end
    hit = failures == 0 &&
          all(d -> maximum(d[3:end]) <= 2 * d[3], deviations)
    return deviations, failures, hit, telemetry
end

function qoi_grid(ctx, evals, dat, train_ks, op, mode, ds)
    pods = load_pod()
    steady_model = load_mlp(model_path("pod_mlp_r$(DEFAULT_R)"))
    eps_grid, duty_grid = ds["eps_grid"], ds["duty_grid"]
    Qhat = zeros(length(eps_grid), length(duty_grid))
    telemetry = RunTelemetry()
    stop_lo, stop_hi = evaluation_bounds(dat, train_ks)
    last_cycle = (2 * NPHASE + 1):ctx.nt
    for (i, ε) in enumerate(eps_grid), (j, duty) in enumerate(duty_grid)
        mean_load = duty * op[:Qp]
        u0 = mode === :mlp ?
             vec(reconstruct(
                 pods.P,
                 steady_model.cscales .* predict(
                     steady_model.model, steady_model.ps, steady_model.st,
                     collect(standardize_mu(ε, mean_load))),
                 DEFAULT_R)) :
             solve_steady(ctx.fom; eps_r = ε, Q = mean_load).x
        c0 = (ctx.Φ' * (u0 .- ctx.ū)) ./ ctx.cs_dep
        μst = collect(standardize_mu_t(ε, op[:Qp], duty, op[:period]))
        subs = build_chain(ctx.sgrid[1], ctx.sgrid[end], duty, ctx.sgrid[2:end])
        result = exec_chain(evals.model, evals.st, ComponentArray(evals.ps),
                            μst, c0, subs, evals.fscale, stop_lo, stop_hi;
                            training = false, telemetry)
        Qhat[i, j] = result !== nothing ?
                     maximum(maximum(view(
                         Matrix(reconstruct(
                             ctx.P,
                             ctx.cs_dep .* hcat(c0, result), R)),
                         :, q)) for q in last_cycle) : NaN
    end
    return Qhat, telemetry
end

rank_of(v) = invperm(sortperm(v))
spearman(a, b) = cor(float.(rank_of(vec(a))), float.(rank_of(vec(b))))

function l4_evaluation(ctx, evals, dat, train_ks)
    ds = load(joinpath(DATA_DIR, "design_study.jld2"))
    results = []
    telemetry_all = []
    for op in ds["ops"]
        Qhat, telemetry = qoi_grid(ctx, evals, dat, train_ks, op, :mlp, ds)
        Qhat_fom, telemetry_fom =
            qoi_grid(ctx, evals, dat, train_ks, op, :fom, ds)
        fom_q = op[:fomQ]
        sp = any(isnan, Qhat) ? NaN : spearman(Qhat, fom_q)
        sp_fom = any(isnan, Qhat_fom) ? NaN : spearman(Qhat_fom, fom_q)
        top5 = any(isnan, Qhat) ? 0 :
               length(intersect(sortperm(vec(Qhat))[1:5],
                                sortperm(vec(fom_q))[1:5]))
        same = !any(isnan, Qhat) && argmin(vec(Qhat)) == argmin(vec(fom_q))
        push!(results, (; Qp = op[:Qp], sp, sp_fom, top5, same,
                        Qhat, Qhat_fom, fom_q))
        push!(telemetry_all, (; Qp = op[:Qp], mlp = telemetry,
                              fom = telemetry_fom))
    end
    hit = all(x -> x.sp >= 0.998 && x.top5 >= 4 && x.same, results)
    return results, hit, telemetry_all
end

function save_figure(evals, deviations, l4results)
    fig = Figure(size = (1240, 400))
    ax1 = Axis(fig[1, 1]; yscale = log10,
               ylabel = "traj rel L² (N = $(length(CONTEXT[].tte)))",
               title = "Stabilized latent ODE at r = 8",
               xticks = (1:5, ["mapper", "Stage A", "stabilized",
                                "Galerkin", "projection"]))
    values = [1.737e-2, safemean(evals.rels_a), safemean(evals.rels_dep),
              3.719e-6, 2.949e-6]
    barplot!(ax1, 1:5, [isfinite(x) ? x : 1.0 for x in values])
    ax2 = Axis(fig[1, 2]; xlabel = "FOM cycle",
               ylabel = "peak-T deviation (K)", yscale = log10,
               title = "20-cycle attractor check")
    for (i, deviation) in enumerate(deviations)
        scatterlines!(ax2, 1:length(deviation), max.(deviation, 1e-6);
                      label = "point $i")
    end
    axislegend(ax2; position = :rt)
    ax3 = Axis(fig[1, 3]; xlabel = "FOM QoI (K)",
               ylabel = "stabilized latent-ODE QoI (K)",
               title = "Design grids")
    for result in l4results
        scatter!(ax3, vec(result.fom_q), vec(result.Qhat); markersize = 5,
                 label = @sprintf("Qp=%.0f", result.Qp))
    end
    ablines!(ax3, 0, 1; color = (:gray, 0.6))
    axislegend(ax3; position = :rb)
    save(joinpath(FIG_DIR, "32_latent_ode_stabilized.png"), fig)
end

function finalize_smoke(ctx, manifest)
    for index in 1:manifest["nens"]
        load_member(index, SEED + 3200 + index)
    end
    deployed = load_member(0, SEED + 3100)
    ks = collect(manifest["train_indices"])
    dat = assemble_data(ctx, ks, ctx.cs_dep)
    model, _, _ = build_pod_mlp(R; indim = 13, hidden = 64, depth = 2,
                                rng = StableRNG(SEED + 3100))
    result = exec_chain(model, deployed["st"], ComponentArray(deployed["ps"]),
                        dat[first(ks)].μst, dat[first(ks)].Ctilde[:, 1],
                        build_chain(ctx.sgrid[1], ctx.sgrid[2],
                                    dat[first(ks)].duty, [ctx.sgrid[2]]),
                        deployed["fscale"],
                        training_scales(dat, ks).stop_lo,
                        training_scales(dat, ks).stop_hi; training = false)
    probe_pass("smoke_finalize", result !== nothing,
               "two_members_and_deployed_loaded")
    println("SMOKE_DONE no_production_artifacts_written")
    flush(stdout)
end

function finalize_phase(ctx)
    manifest = load(MANIFEST_PATH)
    manifest["protocol_id"] == PROTOCOL_ID || error("stale manifest")
    manifest["smoke"] == SMOKE || error("smoke/production manifest mismatch")
    SMOKE && return finalize_smoke(ctx, manifest)

    nens = manifest["nens"]
    members = [load_member(index, SEED + 3200 + index) for index in 1:nens]
    deployed = load_member(0, SEED + 3100)
    train_ks = collect(ctx.ttr)
    dat = assemble_data(ctx, vcat(train_ks, ctx.tte), ctx.cs_dep)
    evals = final_evaluations(ctx, deployed, dat, train_ks)
    ensemble_rels = Float64[]
    ensemble_failures = Int[]
    ensemble_telemetry = []
    for member in members
        model, _, _ = build_pod_mlp(R; indim = 13, hidden = 64, depth = 2,
                                    rng = StableRNG(member["seed"]))
        rels, failures, telemetry =
            eval_rollouts(ctx, model, member["st"], member["ps"], dat,
                          ctx.cs_dep, member["fscale"], train_ks)
        push!(ensemble_rels, safemean(rels))
        push!(ensemble_failures, failures)
        push!(ensemble_telemetry, telemetry_dict(telemetry))
    end
    l2 = l2_retrain(ctx)
    deviations, l3_failures, l3_hit, l3_telemetry =
        l3_evaluation(ctx, evals, dat, train_ks)
    l4results, l4_hit, l4_telemetry =
        l4_evaluation(ctx, evals, dat, train_ks)
    l1_hit = evals.fail_dep == 0 && safemean(evals.rels_dep) < 1.737e-2
    l2_hit = l2.fail24 == 0 && safemean(l2.rels24) < 5.2e-2

    println("\n=== registered stabilized-arm predictions, scored ===")
    @printf("L1 %s: %.3e (%d failed)\n",
            l1_hit ? "HIT " : "MISS", safemean(evals.rels_dep), evals.fail_dep)
    @printf("L2 %s: %.3e (%d failed)\n",
            l2_hit ? "HIT " : "MISS", safemean(l2.rels24), l2.fail24)
    @printf("L3 %s: %d failed\n", l3_hit ? "HIT " : "MISS", l3_failures)
    @printf("L4 %s\n", l4_hit ? "HIT " : "MISS")

    result_path = joinpath(DATA_DIR, "latent_ode_stabilized.jld2")
    model_path_out = joinpath(DATA_DIR, "model_latent_ode_stabilized_r8.jld2")
    atomic_jldsave(result_path; protocol_id = PROTOCOL_ID,
                   rels_dep = evals.rels_dep, rel_dep = safemean(evals.rels_dep),
                   fail_dep = evals.fail_dep, rels_stage_a = evals.rels_a,
                   rel_stage_a = safemean(evals.rels_a), fail_stage_a = evals.fail_a,
                   ensemble_rels, ensemble_failures, ensemble_telemetry, nens,
                   rels24 = l2.rels24, rel24 = safemean(l2.rels24),
                   fail24 = l2.fail24, deviations, l3_failures,
                   evaluation_telemetry = Dict(
                       "deployed" => telemetry_dict(evals.tele_dep),
                       "stage_a" => telemetry_dict(evals.tele_a),
                       "l2" => telemetry_dict(l2.telemetry),
                       "l3" => telemetry_dict(l3_telemetry),
                       "l4" => [Dict("Qp" => x.Qp,
                                     "mlp" => telemetry_dict(x.mlp),
                                     "fom" => telemetry_dict(x.fom))
                                for x in l4_telemetry]),
                   l4 = [Dict(pairs(x)) for x in l4results],
                   scores = Dict("L1" => l1_hit, "L2" => l2_hit,
                                 "L3" => l3_hit, "L4" => l4_hit),
                   solver_train = "KenCarp4", solver_eval = "FBDF",
                   sensealg = SENSE_NAME, environment = environment_metadata())
    atomic_jldsave(model_path_out; protocol_id = PROTOCOL_ID,
                   kind = "latent_ode_stabilized", indim = 13, outdim = R,
                   hidden = 64, depth = 2, ps = deployed["ps"],
                   st = deployed["st"], fscale = deployed["fscale"],
                   cscales = ctx.cs_dep,
                   losses_a = deployed["losses_a"],
                   losses_b = deployed["losses_b"],
                   failure_history = deployed["failure_history"],
                   retcode_history = deployed["retcode_history"],
                   train_time = deployed["wall"], seed = SEED + 3100,
                   time_convention = "phase time s=t/period; bounded f outputs dc̃/ds",
                   solver = "KenCarp4", sensealg = SENSE_NAME)
    save_figure(evals, deviations, l4results)
    println("FINALIZE_DONE saved stabilized aggregate, model, and figure")
    flush(stdout)
end

function main()
    ctx = load_context()
    CONTEXT[] = ctx
    if PHASE == "prep"
        prep_phase(ctx)
    elseif PHASE == "member"
        member_phase(ctx, MEMBER_INDEX)
    elseif PHASE == "finalize"
        finalize_phase(ctx)
    elseif PHASE == "probe"
        ks = SMOKE ? ctx.ttr[1:2] : ctx.ttr
        dat = assemble_data(ctx, ks, ctx.cs_dep)
        run_probes(ctx, dat, collect(ks))
    else
        error("unknown LATENT33_PHASE=$PHASE; use prep, member, finalize, or probe")
    end
end

main()
