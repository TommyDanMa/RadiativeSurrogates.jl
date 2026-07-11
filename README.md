# RadiativeSurrogates: reduced-basis neural surrogates with a nonlinear radiation boundary condition

[![CI](https://github.com/TommyDanMa/RadiativeSurrogates.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/TommyDanMa/RadiativeSurrogates.jl/actions/workflows/ci.yml)
[![ColPrac: Contributor's Guide](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

*(Non-specialists: start with [EXPLAINER.md](EXPLAINER.md), a plain-language tour.)*

This project grew out of [TommyDanMa/reduced-basis-neural-surrogates](https://github.com/TommyDanMa/reduced-basis-neural-surrogates), where the
linear-elliptic reduced-basis study (a *Treball de Recerca*) lives; this repo is the
standalone continuation. The problem: the heat equation on a 2D radiator section whose
dominant heat rejection is **Stefan–Boltzmann radiation to deep space** (T_space = 3 K),
with parametric emissivity ε and electronics load cycles, a controlled prototype for
AI1-class orbital-compute thermal design (`docs/src/assets/`). **Gridap assembles, SciML
solves**: Gridap.jl builds the FE residual/Jacobian, NonlinearSolve.jl drives the steady
Newton solves, OrdinaryDiffEq.jl (mass-matrix ImplicitEuler/Trapezoid/FBDF) drives the
transients.

![AI1 schema](docs/src/assets/spacex-starmind-ai1-schema.jpg)

*The inspiration: AI1-class orbital compute (SpaceX reveal, June 2026). Its spec numbers
set our parameter regime; this prototype is a methods study, not their tool.*

**TL;DR**
- **Question:** does "learn the POD coefficients, not the field" survive a strongly
  nonlinear boundary condition, and what does the radiation regime do to the rank?
- **Core result:** for fixed load geometry the *steady* manifold is numerically **rank 3**
  (σ₃/σ₁ ≈ 7e-9); all nonlinearity lives in the coefficient map c(μ), and a whitened
  8.6k-parameter MLP hits 9.2e-4 relative L² (PDE residual 6.5e-3 vs **38.7** for a direct
  field MLP) at ~5,300× single-query / ~8,900× batched throughput over the 23 ms Newton
  FOM. Load-cycle *transients* re-thicken the manifold (σ₈/σ₁ ≈ 3e-5 vs the steady cliff),
  and the (μ, t) ↦ c surrogate, an honest 1.5–1.7% on raw trajectories, reproduces
  full-order **design rankings** on a 10×10 (ε, duty) grid: Spearman 0.999, top-5 overlap
  5/5, identical optimum, QoI error median 2.4 K on a 382–785 K range, 221× end-to-end
  (≈8,100× warm), replicated at a second operating point (Spearman 0.9988).
- **Why it matters:** not because anyone needs a stand-in for a 23 ms solve; a prime runs
  the FOM and has the answer over a coffee. The honest value: (i) the *diagnostic*, when
  coefficient learning is the right factorization and why this manifold is rank 3
  ([notes/rank-note.md](notes/rank-note.md)); (ii) *many-query outer loops* (MDO,
  optimization under uncertainty), where the surrogate propagates 10⁵ scenarios in ~7 s,
  FOM-verified at every reported quantile (report A.5); (iii) *onboard/edge autonomy*,
  where an 8.6k-parameter, Float32-lossless model with an explicit validity guard fits
  hardware the FOM never will (report A.3).
- **Main caveat:** 2D, fixed load geometry, unity view factor, in-distribution queries:
  a controlled prototype, not a general theorem. The rank-3 collapse is a property of the
  fixed geometry, not of radiation physics.

## Quickstart

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. test/runtests.jl              # test suite (CI runs this too)
julia --project=. scripts/00_run_all.jl         # full reproduction, ~2 h
# or step through scripts/01 ... 29 individually; the live console:
julia --project=. scripts/14_run_console.jl
```

`scripts/18_report_numbers.jl` reprints every headline table straight from `data/*.jld2`,
so the report can always be diffed against the data. Programmatic queries outside the
training box should check `in_training_box(ε, Q)`; measured out-of-distribution
degradation is in report appendix A.1.

**The two-page technical note** distilling the project's core finding is
[notes/rank-note.md](notes/rank-note.md): a predictive argument for why the steady
manifold is rank three, with pre-registered tests (a second heater adds exactly one mode;
conductivity adds none, where naive parameter counting says it should; the box-scaling
prediction missed, and the miss is reported), plus the exact-Galerkin error decomposition
showing that 100.00% of the deployed surrogate's error is coefficient-map error at r = 2.

The console solves the steady FOM live while you drag ε and Q (a ~23 ms Newton solve is
interactive), next to the surrogate, the error field, QoI readouts and both σ-spectra; the
transient panel updates the surrogate's peak-T trace instantly, with the true FOM one
button-press away:

![Design console](figures/15_console.png)

## Layout

- `src/`: `physics.jl` (weak forms, u|u|³ radiation term), `fom_steady.jl`
  (Gridap→NonlinearSolve glue, warm starts, residual traces), `fom_transient.jl`
  (mass-matrix ODE form, square-wave loads via `tstops`), `postprocess.jl` (radiated power,
  peak T, energy balance), plus `pod.jl` / `models.jl` / `metrics.jl` ported from the parent
  project (`models_kan.jl` / `models_deeponet.jl` stay out of the core module).
- `scripts/01–29`: numbered, deterministic (StableRNGs + Sobol): validate steady, steady
  dataset/POD/mappers/eval/benchmark, validate transient, transient dataset/space-time
  POD/mapper/eval, Nr study, design study, console (01–14); robustness, grid
  independence, mapper diagnostics, rank predictions, statistics, Galerkin floor,
  eigenmode diagnostic, cliff reconstruction (15–23); V&V tightening: GCI error bars,
  precision audit, attractor check, forward UQ (24–27); pre-registered
  operator-learning baseline, DeepONet (28), and its small-M probe (29).
  `18_report_numbers.jl` reprints
  every headline table from `data/`.
- `test/`: eight named test files (51 assertions): MMS rates 1.9996/1.9999/2.0000 through
  the nonlinear BC, AD = FD = hand-written Jacobian, exact discrete energy balance,
  θ-family dt-orders 0.95–0.98 / 2.08–2.00, nodally exact 1D radiative equilibrium, plus a
  skip-guarded regression file pinning the deployed surrogates with 3× margin (17 more
  assertions with `data/` present, 68 in total).
- `report/report.md`: the write-up, figures included; `figures/`: all generated figures.

## Headline (steady, held-out; deployed rank r = 2, pinned by the rank sweep)

| model | rel L² | PDE residual ‖R(û)‖/‖F‖ | params | single× | batched× |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct MLP | 5.8e-3 | 3.9e+1 | 131,427 | 1,796× | 7,259× |
| **POD-MLP** | **9.2e-4** | **6.5e-3** | 8,642 | 5,270× | 8,852× |
| POD-KAN | 2.9e-3 | 2.1e-2 | **880** | 4,159× | 9,104× |

POD floor at r=2: 1.1e-9 (basis ≠ bottleneck). Speed-ups: batched throughput over FOM Newton
solves (23.1 ms median), incl. reconstruction, excl. plotting/model loading, after warm-up
(BenchmarkTools medians, batch = 102). Hardware for all timings: AMD Ryzen AI 9 HX 370
(CPU only), 1 Julia thread, 12 BLAS threads, Julia 1.12.6. Coefficients are trained **whitened**; the rank cliff gives c a ~1e9
dynamic range. Robustness (report A.1; N = 20 fresh seeds varying network init only):
POD-MLP 1.35e-3 ± 0.18e-3 (the deployed 9.2e-4 sits at the favourable edge of the
ensemble), POD-KAN 2.86e-3 ± 0.49e-3, transient 1.89e-2 ± 0.20e-2; the rank-3 cliff
replicates at 120×60 and with P2
elements; out-of-distribution the median error grows ~17× just outside the box, so queries
should stay inside `in_training_box`.

Operator-learning baseline, pre-registered and run (scripts/28; DeepONet via
NeuralOperators.jl, equal 8,656-parameter and 3× budgets, matched data-only training,
N = 20 seeds/arm):
competitive on rel L² (3.3e-3) but 283× worse on PDE residual (1.85 vs 6.5e-3); with
the trunk frozen to the POD basis the residual gap collapses 15×, so the learned basis
carries it. Two of four registered predictions missed and are reported as misses
(report §7.4).

New in the V&V pass (report appendix A): forward UQ through the surrogate (N = 10⁵ in
~7 s) matches an exact 10⁴-solve FOM Monte Carlo within 0.71 K at every reported
quantile; Float32 inference is lossless at reported precision (Float16 physics
evaluation overflows: u⁴); GCI bands the deployed grid's own QoI discretization error at
≤ 0.01 K (observed orders 1.99–2.00); and a 20-cycle FOM run verifies the settled-cycle
deployment contract (per-cycle deviation flat within 1.41×; the τ > 1 region stays OOD
and guarded).

## Key figures

![Steady vs space-time σ-decay](figures/11_spacetime_decay.png)

*The project's keystone: the steady rank-3 cliff against the rich space-time spectrum.
Fixed load geometry collapses the steady manifold; switching transients re-thicken it.*

![Transient surrogate evaluation](figures/12_transient_eval.png)

*Radiated-power tracking through load switches, surrogate vs FOM, shown for the worst
held-out trajectory.*

![Design study](figures/14_design_study.png)

*The (ε, duty) design-ranking grid: worst-case chip temperature, FOM vs surrogate,
Spearman 0.999 with the same optimum (star).*

![Load-cycle response](figures/10_load_cycle_response.png)

*The radiator's thermal inertia filters a 0↔600 W square wave down to a ~20 W ripple in
radiated power while the chip sawtooths over ~60 K.*

Relationship to the parent project: this executes its §11 "Gridap.jl FEM" future-work item,
keeps its conventions (numbered scripts, MMS acceptance band, honest benchmark captions), and
drops one of its claims honestly: with a radiation BC there is no "exact BC by construction",
so QoI fidelity (radiated power, peak temperature) takes that role.
