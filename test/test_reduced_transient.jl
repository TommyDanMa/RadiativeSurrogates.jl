# Reduced transient dynamics: the Galerkin mass matrix is SPD, and a one-trajectory
# exact-Galerkin rollout at the deployed rank stays within a wide bound of the frozen
# scripts/30 result (mean 3.72e-6 over the held-out set; bound covers per-trajectory
# spread).
using OrdinaryDiffEq

@testset "reduced transient (regression)" begin
datadir_rt = normpath(joinpath(@__DIR__, "..", "data"))
needed_rt = joinpath.(datadir_rt, ["snapshots_transient.jld2", "pod_transient.jld2"])
if !all(isfile, needed_rt)
    @info "reduced-transient regression skipped (data/ artifacts not present)"
    @test true
else
    @isdefined(CFG) || include(joinpath(@__DIR__, "..", "scripts", "config.jl"))

    fom_rt = build_steady_fom(CFG)
    dtr = load(snapshots_t_path())
    podt = load(pod_t_path())
    Pt = PODBasis(podt["mean"], podt["modes"], podt["svals"])
    rrt = DEFAULT_R_T
    Φt = Matrix(modes_r(Pt, rrt))
    Mrt = Φt' * (mass_matrix(fom_rt) * Φt)
    @test norm(Mrt - Mrt') / norm(Mrt) < 1e-12
    @test isposdef(Symmetric(Mrt))

    U3t, must = dtr["U"], dtr["mus"]
    kt = dtr["test_idx"][1]
    ntt = size(U3t, 2)
    εt, Qpt, dutyt, periodt = must[:, kt]
    ūt = Pt.mean
    qrt = square_wave(Qpt, periodt, dutyt)
    Tend = NCYCLES * periodt
    aopt = Gridap.FESpaces.get_algebraic_operator(
        FEOperator(steady_form(fom_rt; eps_r = εt, Q = 0.0), fom_rt.U, fom_rt.V))
    Mch = cholesky(Symmetric(Mrt))
    Frt = Φt' * load_vector(fom_rt, 1.0)
    xb = similar(ūt)
    rbb = similar(ūt)
    Jf = Gridap.Algebra.allocate_jacobian(aopt, collect(U3t[:, 1, kt]))
    f_rt! = (dc, c, p, t) -> begin
        mul!(xb, Φt, c)
        xb .+= ūt
        Gridap.Algebra.residual!(rbb, aopt, xb)
        dc .= Mch \ (qrt(t) .* Frt .- Φt' * rbb)
        return nothing
    end
    jac_rt! = (Jr, c, p, t) -> begin
        mul!(xb, Φt, c)
        xb .+= ūt
        Gridap.Algebra.jacobian!(Jf, aopt, xb)
        Jr .= .-(Mch \ (Φt' * (Jf * Φt)))
        return nothing
    end
    c0t = Φt' * (U3t[:, 1, kt] .- ūt)
    probt = ODEProblem(
        ODEFunction(f_rt!; jac = jac_rt!, jac_prototype = zeros(rrt, rrt)),
        c0t, (0.0, Tend))
    solt = OrdinaryDiffEq.solve(probt, FBDF();
                                saveat = collect(range(0.0, Tend; length = ntt)),
                                tstops = load_switch_times(periodt, dutyt, 0.0, Tend),
                                abstol = 1e-9, reltol = 1e-9)
    Ût = Matrix(reconstruct(Pt, Array(solt), rrt))
    relt = norm(Ût - U3t[:, :, kt]) / norm(U3t[:, :, kt])
    @test relt < 3.0e-5
end
end
