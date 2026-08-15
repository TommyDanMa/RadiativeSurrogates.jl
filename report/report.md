---
bibliography: ../paper/paper.bib
link-citations: true
---

# When Is Coefficient Learning the Right Factorization?

**A diagnostic study of reduced-basis neural surrogates for the heat equation with a
nonlinear radiation boundary condition**

*A Gridap × SciML prototype for orbital compute thermal design.*

## Abstract

This report asks when learning reduced-basis coefficients ($\mu \mapsto c$, never the
full field) is the right factorization for a neural surrogate, and answers it twice on
one problem: the heat equation on a 2D radiator section rejecting heat by
Stefan–Boltzmann radiation to deep space ($T_\text{space} = 3\,$K), with parametric
emissivity and electronics load cycles.
#### Steady case: yes, and provably map-limited.
For fixed load geometry the solution manifold is numerically **rank 3**
($\sigma_3/\sigma_1 \approx 7\cdot10^{-9}$, grid- and element-independent): the
parameters move temperature levels and amplitudes of essentially fixed spatial shapes,
so, on this fixed-geometry parameter box, the observed nonlinear variation is compressed
into the coefficient map $c(\mu)$. Two exact parameter collapses and a measured
functional collinearity account quantitatively for the collapse; of three predictions
pre-registered to test the explanation, two hit (a second heater adds one dominant mode;
conductivity as a third parameter adds no strong mode, against naive parameter counting)
and one missed (box-shrink scaling). An exploratory follow-up after that miss reconstructs
the cliff to ~25% across seven configurations.
An exact reduced-Galerkin floor certifies the rank-2 factorization on this box: at the
deployed rank,
100% of the surrogate's error is coefficient-map error, and an 8,642-parameter
whitened MLP reaches $9.2\cdot10^{-4}$ relative $L^2$ with a PDE residual ~6,000×
smaller than a direct field MLP's, at thousands of times the FOM's throughput. A
pre-registered operator-learning baseline (DeepONet, matched budgets and data) is
competitive in $L^2$, but its N = 20 mean PDE residual is 1.85: 283× the deployed
POD-MLP reference at $6.53\cdot10^{-3}$. Freezing its trunk to the POD basis lowers
the N = 20 mean residual 14.9× under the same field loss and parameter budget.
#### Transient case: not yet, for a measured reason.
Load switching re-thickens the manifold ($\sigma_8/\sigma_1 \approx 3\cdot10^{-5}$); the $(\mu, t) \mapsto c$ mapper
plateaus at 1.7%. Its error remains strongly responsive to added trajectories (learning
curve slopes 0.82/1.10) but nearly insensitive to rank, so the next tests increase data
or replace direct time regression with latent dynamics. The plateaued surrogate still
reproduces full-order design rankings (Spearman 0.9987/0.9988 at two operating points,
top-5 5/5, identical optimum). A controlled 2D prototype, not a general theorem; the
verification machinery lives in Appendix A.

## 1. The question

For fixed load geometry, the steady manifold of the heat equation with a
Stefan–Boltzmann $u^4$ boundary is numerically rank 3: $\sigma_3/\sigma_1 =
7.2\cdot10^{-9}$ over 410 solutions whose peak temperatures span 296–973 K, and the
number survives grid and element refinement (§4). Nothing in the problem statement announces it.

Everything else in this report hangs on that observation. Reduced-basis surrogates
learn $r$ coefficients instead of $n$ field values; whether that factorization is
sound rests on exactly the manifold dimension that is usually assumed rather than
measured. Here it
is measured, structurally constrained, and relied on: §5 gives two exact parameter
collapses and a measured functional-collinearity account; §6 tests that account with
predictions pre-registered before the scripts ran, two hits and one recorded miss; §7
draws the consequence, an exact Galerkin floor certifying that 100% of the deployed
surrogate's error is coefficient-map error, none of it basis.

A diagnostic earns the name only if it can say no. The transient problem supplies the
no: load switching re-thickens the manifold ($\sigma_8/\sigma_1 \approx 3\cdot10^{-5}$),
the $(\mu, t) \mapsto c$ mapper plateaus at 1.7%, and its learning curve and rank sweep
identify snapshot count, rather than basis rank, as the immediate constraint at this
budget (§8). The two verdicts use the same physics, solver and pipeline, and each is
checked against held-out metrics and downstream design QoIs. The resulting diagnostic
workflow is shown in @fig:diagnostic-workflow.

![Diagnostic workflow for choosing a surrogate
factorization.](../docs/src/assets/diagnostic-flowchart.svg){#fig:diagnostic-workflow width=72%}

The report walks this diagnostic twice. Every numbered box is an artifact in the
repository: box 1 is the test suite (§3); 2 and 2b are the snapshot POD and its grid-,
element- and reconstruction-floor checks (§4); 3 is the parameter collapses and
coefficient-function collinearity (§§5–6); 4 is the pre-registered score card (§6); 5
is the Galerkin floor, seed spread, OOD guard and GCI reference band (§7, §9.3). The
return arrows are gates, not generic iteration: solver failure forbids surrogate work;
mesh dependence means that the measured spectrum belongs to the discretization; a
prediction miss sends an incomplete structural account back for revision. The steady
problem exits through the yes-branch, while switching sends the transient problem
through the slow-decay no-branch (§8). Section 9 collects the engineering consequences,
and Appendix A contains the verification machinery.

**Physical motivation.** In vacuum heat leaves only by radiation. The concrete prompt
is AI1-class orbital compute (SpaceX reveal, June 2026; see `docs/src/assets/`):
150 kW-peak compute through ~110 m² of deployable radiators, about 1.4 kW/m², which
our parameter regime brackets; the spec numbers are motivation, not model inputs. One
reason to use the radiation problem as a diagnostic is that its strong nonlinearity is
confined to one boundary term, with the remaining operator linear.

**What the surrogate is for here.** This FOM is deliberately small: one Newton solve
takes 23 ms on the laptop used for §9.1, so a single query does not justify a surrogate.
The primary product is the diagnostic itself (`notes/rank-note.md`): deciding whether
coefficient learning is the right factorization and locating its error when it is.
The many-query experiments then test what follows from that decision, rather than
establishing an operational need: 100,000 surrogate UQ scenarios take 7.5 s, while
10,000 paired FOM scenarios take 444 s (§9.3, A.5). The deployment audit establishes
two narrower software properties: the 8,642-parameter model is Float32-lossless and
ships with an explicit `in_training_box` guard (A.3). It does not establish that this
small FOM could not run on a particular edge target.

## 2. The problem

Strong form on $\Omega = (0,1) \times (0,0.5)$ m:

$$\rho c\, \partial_t u = \nabla\!\cdot\!(k \nabla u) + q(x;\mu,t), \qquad
-k\,\partial_n u = \varepsilon(\mu)\,\sigma\,(u^4 - T_\text{space}^4) \ \text{on } \Gamma_r,$$

with the remaining edges insulated (canonical config B) and the steady problem obtained by
dropping the time term. In code the radiation nonlinearity is written $u|u|^3 - T_s^4$: it
coincides with $u^4 - T_s^4$ for $u > 0$ and is monotone on all of $\mathbb{R}$, so Newton
iterates that momentarily dip negative still feel a restoring flux of the correct sign.

Two structural facts drive the verification suite. First, with no Dirichlet DOFs the constant
$v \equiv 1$ lies in the test space, so the solved discrete system satisfies the **exact
discrete energy balance** $P_\text{in} = \oint \varepsilon\sigma (u_h^4 - T_s^4)\,d\Gamma$ up
to the Newton tolerance. Second, the boundary Jacobian is analytic,
$4\varepsilon\sigma|u|^3$ on $\Gamma_r$, giving a hand-written matrix to check the assembled
AD Jacobian against. The dimensionless **radiation number** $\mathrm{Nr} = \varepsilon\sigma
T^3 L / k$ measures how strongly radiation couples relative to conduction; over our parameter
box it spans roughly 0.05–1.5.

POD/SVD compression is inherited verbatim from the parent project (plain $\ell^2$ inner
product on the uniform grid; Eckart–Young optimality; mass-weighted POD remains future
work), following the standard reduced-basis construction [@Quarteroni2016].
One difference: the parent's "reconstructions satisfy the BCs exactly by construction"
selling point does **not** carry over, since the radiation BC is part of the residual. QoI
fidelity (radiated power, peak temperature) takes over that role.

## 3. A verified solver

The FE machinery is Gridap.jl [@Badia2020]: a
`CartesianDiscreteModel` (default 60×30, P1 Lagrangian; element order is a config knob), the
weak residual

$$R(u; v) = \int_\Omega k \nabla v \cdot \nabla u \; - \int_\Omega v\, q
\; + \oint_{\Gamma_r} v\, \varepsilon\sigma\,(u|u|^3 - T_s^4),$$

boundary quadrature of degree 6 (the $v\,u^4$ integrand is degree 5 for P1), and Gridap's
autodiff for the Jacobian. Everything downstream of assembly is the SciML stack
[@Pal2024; @Rackauckas2017]:

- **Steady:** the algebraic operator is wrapped as a `NonlinearFunction` (in-place residual
  and Jacobian, sparse `jac_prototype`) and solved by `NonlinearSolve.NewtonRaphson`,
  warm-started along parameter sweeps and initialised at the analytic radiative-equilibrium
  temperature $(Q/(\varepsilon\sigma|\Gamma_r|) + T_s^4)^{1/4}$. Every residual evaluation is
  logged, so Newton traces are first-class artifacts.
- **Transient:** method of lines, $M \dot u = -R_\text{space}(u) + Q(t) F_\text{unit}$, with
  the consistent $\int \rho c\, u v$ mass matrix, integrated by OrdinaryDiffEq mass-matrix
  methods: `ImplicitEuler` and `Trapezoid` (the θ-family the scope document asked for), plus
  adaptive `FBDF` for production runs. Square-wave load switches are passed as `tstops`, so
  discontinuities are hit exactly rather than smeared.

This division of labour is the point of the architecture. "Residuals and Jacobians under
control" means we hand the *assembled* $R$ and $J$ to the solver ourselves, and can therefore
test them independently of any solver behaviour.

Eight named test files (51 assertions) verify the discretization and solver before any
ML enters, plus a skip-guarded regression file (17 more whenever the data artifacts are
present, 68 in total) that bounds the *deployed* surrogates' quality with 3× margin:

| Test | What it pins | Result |
| --- | --- | --- |
| `test_steady_solver` | tags/geometry, ε=0 linear limit vs direct sparse solve, NewtonRaphson vs TrustRegion, warm starts | pass (agreement ≤ 1e-9) |
| `test_jacobian` | AD Jacobian vs central FD in random directions; vs hand-written $A_\text{cond} + 4\varepsilon\sigma|u|^3$ boundary block | pass (≤ 1e-6 / 1e-9 rel) |
| `test_energy_balance` | exact discrete balance (config B); reaction-recovered wall influx (config A) | rel ≤ 1e-8 (measured ~1e-10) |
| `test_manufactured_solution` | MMS **through the nonlinear BC** (inhomogeneous radiation data), P1 L² order 2, band [1.7, 2.2] | rates **1.9996, 1.9999, 2.0000** |
| `test_analytic_equilibrium` | closed-form 1D profile $u_1 = (q_0 L/\varepsilon\sigma + T_s^4)^{1/4}$ | nodally exact (2.6e-12 K) |
| `test_transient` | steady long-time limit; fixed-Δt orders vs tight FBDF reference; energy bookkeeping $\Delta E = \int (P_\text{in} - P_\text{rad})$ | BE rates 0.95–0.98, CN 2.08–2.00; bookkeeping < 1% |
| `test_pod_models` | POD orthonormality/round-trip, mapper shapes, determinism, trainability | pass |

: Solver and discretization tests. {#tbl:test-suite}

![MMS convergence](../figures/01_mms_convergence.png){#fig:mms-convergence}
![Analytic radiative equilibrium](../figures/02_analytic_equilibrium.png){#fig:analytic-equilibrium}

The time integrators get the same treatment. Fixed-Δt convergence against a tight FBDF
reference (smooth sinusoidal load): ImplicitEuler rates 0.954/0.967/0.980, Trapezoid
2.077/2.017/2.004; the θ-family behaves exactly as advertised through the nonlinear
boundary term, and long-horizon constant-load runs reproduce the steady solution to the
integrator tolerance.

![θ-family dt-convergence](../figures/09_transient_convergence.png){#fig:transient-convergence}

Two details are worth recording. The 1D uniform-load test is *nodally exact* (P1 nodal
superconvergence via the piecewise-linear Green's function), a far stronger check than the
intended $O(h^2)$ tolerance. And all energy bookkeeping must use the *discretely injected*
power $Q \cdot \int \hat q\, d\Omega_h$: on grids where the chip patch does not align with
cell faces the quadrature captures only part of the nominal $Q$, and the discrete system
conserves the quadratured value, not the nominal one.

## 4. The observation: a rank-3 manifold

![Steady temperature fields, configs A and B](../figures/03_steady_fields.png){#fig:steady-fields}

512 Sobol samples over $\mu = (\varepsilon, Q) \in [0.1, 0.95] \times [100, 800]$ W/m, one
converged Newton solve each (median 49 ms as logged by the machine that generated the
frozen dataset, 23.1 ms re-measured on the current hardware, §9.1; worst energy balance
6e-12, peak temperatures 296–973 K). The POD of the 410 training snapshots delivers the
observation the rest of the report answers to:

| $r$ | $\sigma_r/\sigma_1$ | reconstruction error (test) |
| ---: | ---: | ---: |
| 1 | 1 | 1.1e-1 |
| 2 | 1.15e-1 | **1.1e-9** |
| 3 | 7.2e-9 | 6.2e-16 |
| 5 | 6.1e-16 | 1.8e-16 |

: Steady singular values and held-out reconstruction errors. {#tbl:steady-spectrum}

![POD σ-decay and reconstruction floor](../figures/06_pod_decay.png){#fig:pod-decay}

**The steady manifold is numerically rank three** (and rank *two* for every practical
purpose): two modes reconstruct held-out fields to 1.1e-9, and the third singular value
already sits eight orders below the first.

The measurement itself is verified before it is interpreted
(@fig:diagnostic-workflow, box 2b).
The collapse is not a discretization artifact: replicating the POD on a 2× finer grid
gives $\sigma_3/\sigma_1 = 6.7\cdot10^{-9}$ (120×60, P1) and with quadratic elements
$6.8\cdot10^{-9}$ (60×30, P2), against $7.2\cdot10^{-9}$ at the baseline
(`scripts/16_grid_independence.jl`). The spectrum is a property of the solution
manifold, not of the mesh.

![Grid independence of the σ-decay](../figures/17_grid_independence.png){#fig:grid-independence}

Nothing in the problem statement guarantees low rank. Temperatures span 296–973 K, the boundary term is quartic, and a two-parameter solution family need not lie near a two-dimensional linear subspace. Yet two modes reconstruct held-out fields to 1.1×10⁻⁹. This discrepancy requires a structural explanation, which §6 tests through pre-registered predictions.

## 5. Candidate explanation

Three structural facts, in increasing order of what they explain.

### Exact parameter collapses

The interior equation is linear: each fixed load shape
contributes one fixed direction to the inhomogeneity, scaled by its amplitude, and
$(Q, k)$ enter the particular solution only through $Q/k$. This does not prove one
manifold response direction per load, because the nonlinear boundary condition couples
the response through the total temperature. The exact statement is that the solution
depends on $(\varepsilon, Q, k)$ only through $(Q/k, \varepsilon/k)$: the *effective*
parameter count is two, whatever the nominal count is. The harmonic remainder, driven
by near-uniform radiating-edge flux, spans very few measured shapes.

### Analyticity

The parameter-to-solution map is analytic on the box (the radiation
term is analytic and monotone in $u$), and analytic parametric maps have exponentially
decaying Kolmogorov widths; the smoothness-to-decay half of the argument is standard
[@CohenDeVoreSchwab2011; @CohenDeVore2016; @Lassila2013;
@BachmayrCohen2017; @ChkifaCohenSchwab2015]. This explains *fast* decay. It does not
explain where the cliff sits.

### Observed compression

With the load geometry fixed, varying $Q$ scales the source
term exactly.
Empirically, the resulting solution variation is almost confined to fixed spatial
shapes; varying $\varepsilon$ primarily moves the radiative-equilibrium level and their
amplitudes. Those amplitudes depend on the parameters strongly and nonlinearly, while
POD rank sees only the shapes they multiply. At the measured accuracy on this box, the
nonlinear parameter dependence is therefore compressed into the coefficient map
$c(\mu)$, which is what a learned mapper is for.
(What re-thickens the manifold is transients, not parameters; §8.)

If this explanation is right, it predicts. Three tests were pre-registered in
`notes/rank-note.md` before `scripts/19_rank_prediction.jl` first ran:

- **P1**: predict that a second heater patch produces one additional dominant centered
  mode. This is a falsifiable response-rank claim, not an exact consequence of the
  source rank.
- **P2**: conductivity as a third parameter produces no new $O(\sigma_2)$-scale mode,
  because of the $Q/k$ collapse; naive $d+1$ parameter counting predicts a fourth one.
- **P3**: shrinking the parameter box shrinks $\sigma_3$ with a log–log slope in
  $[1, 2]$ (analyticity-driven decay).

## 6. The predictions: two hits, one miss, and what the miss exposed

Each prediction targets a different way the explanation could fail: P1, whether a new
load shape produces one additional dominant response direction; P2, whether the $Q/k$
collapse predicts the absence of the mode expected from parameter counting; P3, whether
analyticity predicts the *scale* of the cliff. A hit supports the corresponding account;
a miss localizes which part is not yet understood. Scored against
`scripts/19_rank_prediction.jl`:

- **P1, hit.** The cliff moves down by exactly one mode:
  $\sigma_4/\sigma_1 = 8.3\cdot10^{-6}$.
- **P2, strong hit.** $\sigma_4/\sigma_1 = 3\cdot10^{-14}$: no fourth mode where naive
  parameter counting predicts one. The absence agrees with the exact $Q/k$ collapse
  and falsifies the naive $d+1$ rule on this test.
- **P3, miss.** The box-shrink scaling came out sub-quadratic: fitted log–log slopes
  $0.22 \pm 0.00$ on the ε-axis and $0.66 \pm 0.01$ on the Q-axis (mean ± std over
  N = 3 independent dataset draws), against the predicted $[1, 2]$. A miss, recorded;
  too tightly replicated to be a sampling artifact.

![Rank predictions](../figures/20_rank_prediction.png){#fig:rank-predictions}

The miss says the *scale* of the cutoff was not understood, and it looped the diagnosis
back once (@fig:diagnostic-workflow, miss edge). An eigenmode diagnostic
(`scripts/22_eigenmode_diagnostic.jl`, suggested by external review, exploratory, not
pre-registered) rejects the proposed $1/\lambda$ spectral-decay explanation and instead
finds **functional collinearity**: 12 parity-allowed eigenmodes of the linearized operator
are excited (amplitudes $\sim \lambda^{-1.9}$), but their coefficient functions over the
parameter box have normalized singular values 2.5, 2.3, 3.9e-5. Many spatial modes, two
independent parameter functions. Details in `notes/rank-note.md` §4b.

![Eigenmode diagnostic](../figures/21_eigenmode_diagnostic.png){#fig:eigenmode-diagnostic}

The quantitative version generalizes (`scripts/23_cliff_reconstruction.jl`): the
singular value of the amplitude-weighted excited coefficient functions reproduces the
measured POD cliff to a uniform ~25% across seven configurations (base, both
3-parameter variants with $r^*$ detected automatically, and the four shrink boxes),
while the naive leakage × total-amplitude product overpredicts by up to three orders
of magnitude (deflation matters). Within this decomposition, the cliff is the
non-collinearity left after weighting each coefficient function by the amplitude its
spatial mode carries. This locates the cutoff algebraically; it does not explain why
those coefficient functions are nearly collinear.
This measured linear-width decomposition is distinct from nonlinear or compressive
reduced-basis constructions [@BalloutMadayPrudhomme2024;
@CohenDolbeaultSomacalDahmen2023].

![Cliff reconstruction across configurations](../figures/22_cliff_reconstruction.png){#fig:cliff-reconstruction}

## 7. The consequence: coefficient learning, certified

### 7.1 Models and metrics

The measured rank and Galerkin floor support the factorization $\hat G = R \circ N$: a fixed linear
reconstruction $R(c) = \bar u + \Phi_r c$ and a small learned map
$N: \tilde\mu \mapsto c$ (inputs standardized to $[-1,1]^d$). Three mappers are
compared: a POD-MLP (tanh, 64×3), a POD-KAN (`KolmogorovArnold.KDense`
[@Liu2024], kept as a
*compact coefficient-mapper baseline*, not a structural ingredient), and a direct MLP
$\mu \mapsto u$ baseline trained on scaled fields.

One training consequence follows directly from the rank cliff: the coefficient dynamic
range is $c_1 : c_3 \approx 10^9$, so raw MSE weights the small-mode gradients at
roughly $10^{-18}$ of the leading scale under this training protocol. All mappers are
therefore trained on **whitened
coefficients** $\tilde c_i = c_i / \mathrm{std}(c_i)$, with the scales stored in the
model file and re-applied at prediction time. In the same-seed ablation, turning
whitening off degrades the mapper 200× to 2.1e-1. Separately, swapping in a random
orthonormal basis caps accuracy at exactly its own projection floor, 2.13e-1
(ablations in A.1).

The contrast that motivates non-intrusive reduction here: an intrusive POD-Galerkin ROM
of this problem would need a Newton solve *in the reduced space* at every query, with
the $u^4$ boundary term evaluated by hyper-reduction such as DEIM
[@Chaturantabut2010] to stay independent of
$n$. The coefficient mapper sidesteps both, because the FOM's nonlinearity is baked into
the training data. The price is being purely data-fit inside the sampled box, with the
assembled nonlinear residual retained as an *a posteriori* physics check.

Metrics, per test sample: relative L² field error; coefficient error; the **nonlinear
PDE residual** $\|R(\hat u)\|_2 / \|F_\text{load}\|_2$ evaluated with the same Gridap
assembly as the solver (the nonlinear counterpart of the parent's
$\|A\hat u - b\|/\|b\|$); and two design QoIs, relative radiated-power error and
absolute peak-temperature error (K).

### 7.2 Held-out accuracy

Held-out evaluation (102 samples), deployed rank **r = 2** (pinned by
@fig:rank-sweep;
seeds are rank-keyed so the deployed models *are* the sweep models):

| model | rel L² | rel residual | coef err | ΔP/P | ΔT_peak (K) | params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| direct MLP | 5.80e-3 | **3.87e+1** | — | 1.7e-2 | 16.3 | 131,427 |
| **POD-MLP** | **9.18e-4** | **6.53e-3** | 6.2e-3 | 3.5e-3 | 3.6 | 8,642 |
| POD-KAN | 2.92e-3 | 2.09e-2 | 1.5e-2 | 1.1e-2 | 6.4 | **880** |

: Held-out steady-surrogate metrics at deployed rank. {#tbl:steady-models}

POD floor at r=2: rel L² 1.1e-9, residual 5.1e-7; FOM residual floor 4.2e-12. The basis sits
five to six orders below the network error; *the basis is not the bottleneck*, in an even
more extreme form than the parent project. The direct MLP's residual (38.7: its fields
violate the discretized PDE badly despite a decent L² error) is the physics-consistency gap
in one number.

Rank sweep (rel L² / residual):

| $r$ | POD-MLP | POD-KAN |
| ---: | :---: | :---: |
| 1 | 1.73e-2 / 7.41e-1 | 1.82e-2 / 7.44e-1 |
| **2** | **9.18e-4 / 6.53e-3** | 2.92e-3 / 2.09e-2 |
| 3 | 1.30e-3 / 8.93e-3 | 2.93e-3 / 1.87e-2 |
| 5 | 1.74e-3 / 1.18e-2 | 3.56e-3 / 2.27e-2 |

: Relative $L^2$ error and residual by basis rank. {#tbl:steady-rank-sweep}

![Accuracy and residual vs rank](../figures/07_rank_sweep.png){#fig:rank-sweep}
![Worst-case steady prediction](../figures/08_prediction_worstcase.png){#fig:steady-worst-case}

The parent's non-monotonicity lesson recurs *in extremis*: mode 3 is numerically dead
($\sigma_3/\sigma_1 \sim 10^{-8}$) and modes 4–5 are pure round-off, so ranks beyond 2 make
the network fit noise and measurably degrade both metrics. Model selection by held-out
error, not by σ-decay alone, picks r = 2.

Seed robustness, inline (full machinery in A.1): twenty fresh initializations give
1.35e-3 ± 0.18e-3, range [9.69e-4, 1.67e-3] (N = 20). The deployed model's 9.18e-4
beats all twenty and is therefore a favorable seed relative to this ensemble. Every
tested initialization preserves MLP < KAN, both ≪ direct, and the r = 2 optimum.
Separately, paired bootstrap differences over the same 102 test parameters exclude
zero for the deployed models: KAN − MLP = +2.0e-3 [1.6, 2.4]e-3 and direct − MLP =
+4.9e-3 [4.4, 5.4]e-3 rel L² (B = 10,000).

### 7.3 The certificate: an exact Galerkin floor

The error decomposition uses the exact Galerkin floor
(`scripts/21_galerkin_floor.jl`). Solving the exact reduced-Galerkin system
(Newton on $\Phi_r^\top R(\bar u + \Phi_r c) =
0$; at $n = 1891$ no hyper-reduction is needed offline, DEIM being the online path) bounds
any surrogate confined to the POD subspace. Mean over the 102 test parameters:

| rank | L² projection floor | Galerkin floor | POD-MLP error | map share |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.7e-2 | 2.4e-2 | 1.7e-2 | basis-limited |
| **2** | 1.14e-9 | **1.14e-9** | 9.2e-4 | **100%** |
| 3 | 6.2e-16 | 2.8e-13 | 1.3e-3 | 100% |
| 5 | 1.8e-16 | 2.8e-13 | 1.7e-3 | 100% |

: Projection, exact-Galerkin and learned-map errors by rank. {#tbl:steady-galerkin-floor}

At and above the manifold rank the Galerkin floor equals the L² projection floor to every
digit measured (@tbl:steady-galerkin-floor:
$1.14\text{e-}9 = 1.14\text{e-}9$ at $r=2$), so none of the deployed
surrogate's error is basis error: all of it is in the coefficient map. Below it ($r=1$)
the mapper already sits at the basis floor. The crossover tells you, per rank, whether to
spend effort on the basis or on the map.

### 7.4 The registered operator-learning baseline: DeepONet (run 2026-07-11)

The protocol registered here ran with predictions D1–D4 pre-registered in the script
header before the first run and the scorecard appended after
(`scripts/28_deeponet_baseline.jl`): an operator-learning baseline, DeepONet
[@Lu2021] (NeuralOperators.jl v0.6.4; branch on
$\tilde\mu$, trunk on $x$, p = 8 latent functions) on the same 410 snapshots, at equal
and 3× parameter budgets, the POD-MLP's training schedule byte-for-byte (6,000
full-batch Adam epochs, geometric lr anneal, keep-best), scaled-field MSE for every
arm, N = 20 init-only seeds per arm, a trunk-frozen-to-$\Phi_2$ arm isolating
learned-basis vs fixed-basis (the published fixed-basis POD-DeepONet configuration
[@Lu2022]), and a sample-efficiency curve
at $M \in \{102, 205, 410\}$ (N = 5). Two predictions hit, two missed; the misses are the interesting part.

| arm | params | rel L² | residual ‖R(û)‖/‖F‖ |
| --- | ---: | ---: | ---: |
| POD-MLP r = 2 (reference) | 8,642 | 9.18e-4 deployed; 1.35e-3 ± 0.18e-3 (N = 20) | 6.53e-3 deployed |
| DeepONet, equal budget | 8,656 | 3.30e-3 ± 0.50e-3 | 1.85 ± 0.28 |
| DeepONet, 3× budget | 25,936 | 2.44e-3 ± 0.33e-3 | 1.54 ± 0.17 |
| DeepONet, trunk = $\Phi_2$ | 8,642 | 1.90e-2 ± 0.04e-2 | 0.125 ± 0.004 |

: Registered DeepONet comparison. DeepONet entries are mean ± std over N = 20
initializations. The POD row reports the deployed residual; a POD residual ensemble was
not evaluated. {#tbl:deeponet}

The residuals form a ladder: a free field (the direct MLP of §7.2) sits at 38.7;
sharing eight learned shapes brings it to 1.85; replacing the learned shapes with the
POD basis at the same field loss and parameter budget brings it to 0.125; training the
same POD branch under the whitened-coefficient loss brings it to 6.53e-3; exact
projection coefficients reach 5.1e-7 (§7.2). The whole ladder is not a one-factor
ablation: the first step changes the output factorization. The next two comparisons
are targeted: learned vs fixed basis at matched field loss, then field loss vs
whitened-coefficient loss with the POD basis and branch architecture fixed.

#### D1: Hit on the primary metric

The registered D1 comparison is the equal-budget DeepONet's N = 20 mean against the
deployed POD-MLP reference, not an ensemble-to-ensemble ratio. At matched data,
training schedule and parameter budget, that comparison is 283× on PDE residual
(1.85 against 6.53e-3). Tripling the DeepONet's parameters buys 17%, not orders
(1.54), so width is not the immediate limit within this tested range. Field MSE does
not directly penalize the
assembled residual or curvature: second differences can amplify percent-level field
error to order one. POD modes inherit the curvature of the solutions they combine;
the learned trunk is not constrained to do so. Shared shapes reduce the direct MLP's
38.7 residual to 1.85, but do not close the gap.

#### D2: Hit

Rel L² 3.30e-3, 2.4× the POD-MLP ensemble mean: competitive on the metric
DeepONet optimizes.

#### D3: Miss

The frozen-trunk arm was registered to land within 3× of the POD-MLP's
residual; it landed at 19× (0.125). The registered bound ignored the loss-weighting
effect A.1 had already measured: field MSE under-weights mode 2 by
$(\sigma_1/\sigma_2)^2 \approx 76$, expressed here as a 158.4 K peak-temperature
error. The bound is recorded as a miss. Its relative clause held: at the same field
loss and parameter budget, the POD trunk beats the learned trunk by 14.9× on residual,
identifying basis choice as a contributor under this protocol. The two targeted
comparisons separate two effects: learned vs fixed basis at matched field loss, then
field loss vs whitened-coefficient loss with the POD basis and branch architecture
fixed. The POD-MLP combines both choices.

#### D4: Miss; the frame was wrong, not the numbers

Registered: DeepONet degrades more
than the POD-MLP as the training set drops 410 → 102. Measured: DeepONet barely moves
(4.36 → 3.29 → 3.67e-3 across $M$, flat within seed noise) while the POD-MLP converts
data into accuracy (2.09 → 1.46 → 1.43e-3). DeepONet is not data-hungry here; it is
*saturated* at a protocol-set floor over this range. Within the
measured range the POD-MLP leads at every $M$ (2.1–2.6×). Below $M = 102$ a registered
probe ($M \in \{13, 26, 51\}$, N = 5; `scripts/29`) came back inconclusive by its own
registered criterion: the two methods are indistinguishable there, both data-limited;
no direction is claimed. "Worse sample efficiency" mispredicted the direction of the phenomenon: at
larger $M$ the gap widens. The flat curve also shows that the DeepONet reaches its
measured floor with a quarter of the data; what it pays is the height of that floor.

![DeepONet baseline](../figures/27_deeponet.png){#fig:deeponet-baseline}

*Loss-curve overlay (normalized to initial value; the two losses differ, scaled-field
MSE against whitened-coefficient MSE, so the overlay shows optimization dynamics
only), DeepONet residual ensembles with the deployed POD reference, and the
sample-efficiency curve.*

(The parent project's exact-BC wedge does not apply here: config B has no Dirichlet
BC.)

Scope: this comparison runs on the diagnostic's yes-branch, favorable terrain for a
fixed linear basis, using NeuralOperators.jl and the fixed-basis POD-DeepONet variant
[@Lu2022]. It measures residual differences among these factorizations and
losses at matched budget on a rank-3 manifold; it is not a claim about operator
learning in general, and the no-branch (§8) points toward learnable dynamics.
One more fence: every arm trains on snapshots alone, so the 283× is a statement about
factorizations under a data-only loss, not about the best any DeepONet can do. A
physics-informed DeepONet changes the information budget (residual evaluations are
operator access); the reduced side's version of that upgrade is already in §7.3, the
exact Galerkin coefficients solved from the operator instead of learned from data. A
PI-DeepONet arm is the natural extension if the residual gap is ever the contested
claim.

## 8. The transient branch: same diagnosis, opposite verdict

### 8.1 Switching transients re-thicken the manifold

The transient problem drives the radiator with square-wave load cycles. The physics sets
the scales: the radiated power lags the load with the expected
$\tau_\text{rad} \approx \rho c (V/A) / 4\varepsilon\sigma T^3 \sim 10^4$ s relaxation,
a sensible regime against the 1.8–10.8 ks load periods.

![Square-wave load cycle response](../figures/10_load_cycle_response.png){#fig:load-cycle}

*The radiator's thermal inertia filters the 0↔600 W square wave down to a ~20 W ripple in
radiated power: the radiating edge effectively sees the orbit-average load while the chip
sawtooths over ~60 K.*

96 Sobol trajectories over $\mu = (\varepsilon, Q_p, \text{duty}, \text{period})$, three
load periods each, 61 phase-uniform states, started from the steady state at the cycle-mean
load (median transient FOM solve 13 s on the dataset-generating machine, about 270× its
steady solve; this cost asymmetry is what makes a transient surrogate worth having). The
space-time POD delivers the promised contrast to §4: instead of a rank-3 cliff, a smooth
spectrum with $\sigma_4/\sigma_1 = 6.4\cdot10^{-3}$, $\sigma_8/\sigma_1 = 3.0\cdot10^{-5}$,
$\sigma_{16}/\sigma_1 = 6.9\cdot10^{-8}$. **Switching transients re-thicken the manifold.**

![Steady vs space-time σ-decay](../figures/11_spacetime_decay.png){#fig:spacetime-decay}

#### Radiation-strength stratification

The scope document asked how the mode count
responds to the radiation regime. Conditioning the snapshot sets on emissivity bins (Nr
from the mean radiating-edge temperature per bin):

| ε bin | Nr (transient) | steady modes @1e-6 | space-time @1e-4 | @1e-6 |
| --- | ---: | ---: | ---: | ---: |
| [0.10, 0.38] | 0.21 | 2 | 6 | 12 |
| [0.38, 0.67] | 0.23 | 2 | 6 | 12 |
| [0.67, 0.95] | 0.27 | 2 | 7 | 12 |

: Mode counts across emissivity bins. {#tbl:radiation-number-modes}

![σ-decay by emissivity bin and modes vs Nr](../figures/13_nr_study.png){#fig:radiation-number-study}

Over this parameter box the mode count grows only weakly with Nr; the dominant axis is
**steady vs transient** (2 modes vs ~6–12), not the sampled variation in radiation
strength. Together with P1, this is consistent with fixed load geometry as the primary
driver of the steady collapse in the tested regime; it does not establish independence
from radiation strength outside that regime.

### 8.2 The mapper plateaus at the current data budget

The $(\tilde\mu, t)$ mapper (inputs: 4 parameters, normalized time, two phase harmonics;
128×3 MLP) uses *capped* whitening: full whitening would upweight $\sigma_8$-level modes by
~3·10⁴ and concentrates training on the least-resolved switching detail, so the scales
are floored at 1% of $\sigma_1$.

| $r$ | traj rel L² | POD floor |
| ---: | ---: | ---: |
| 4 | 1.69e-2 | 2.5e-4 |
| **8** | 1.74e-2 | 2.1e-6 |
| 16 | 1.76e-2 | 1.1e-8 |
| 32 | 1.45e-2 | 1.2e-10 |

: Transient mapper error and trajectory projection floor by rank. {#tbl:transient-rank-sweep}

The sweep is nearly flat: at this data budget, increasing rank does not materially
improve the mapper, while every POD floor remains four to eight orders lower.

#### Learning-curve and mode diagnostics

`scripts/17_mapper_diagnostics.jl` tests what currently limits the mapper. The learning
curve is steep: 24 → 48 → 77 training trajectories give errors
5.2e-2 → 3.0e-2 → 1.8e-2, slopes 0.82 and 1.10 in $M$ (near $M^{-1}$, far steeper than the
statistical $M^{-1/2}$). Thus error remains strongly responsive to snapshot count over
the measured range, whereas adding basis modes buys little at the current budget. The
per-mode view is consistent with that reading: modes 1–4 are learned to 7–22% relative
error, mode 6 to 87%, and modes 7–8 have relative error above one.

![Learning curve and per-mode diagnostics](../figures/18_mode_diagnostics.png){#fig:mapper-diagnostics}

QoI tracking at the deployed $r=8$ (N = 19 held-out trajectories): peak-temperature error
median 14.3 K / max 42.6 K along trajectories; radiated-power tracking error median 14% /
max 39%. The worst case is
instructive: it is the *low-load* corner ($Q_p = 136$ W/m), where the true
$P_\text{rad} \approx 94$ W is nearly constant because the radiator's thermal inertia filters
the square wave almost completely, so a small absolute error is large relative to the local
signal. On design-relevant loads the same surrogate ranks designs almost perfectly (§9.2).

![Transient surrogate evaluation](../figures/12_transient_eval.png){#fig:transient-evaluation}

*Error vs time, peak-temperature trace, and radiated power vs time for a load cycle
(surrogate vs FOM), shown for the worst held-out trajectory.*

### 8.3 Verdict and exits

This is the flowchart's no-branch, reached with measurements in hand. The manifold is not
intrinsically low-dimensional at the accuracy the steady problem enjoys; the basis is four
to eight orders better than the map at every rank; and the binding constraint is data
(learning-curve slopes near $M^{-1}$). At this data budget, $(\mu, t) \mapsto c$
regression is not the right factorization: it works to 1.7%, and its own diagnostics say
which exits exist. In the order the measurements license them: a larger trajectory budget
(the near-$M^{-1}$ slope pays directly), and latent *dynamics*, learning
$\dot c = f(c, \mu, \text{load}(t))$ through the solver instead of regressing $c(t)$
against time (§§8.4–8.5 measure the floor and the first snapshots-only attempt; §11
carries the learned-$f$ build). More modes is measurably not an exit.

What survives the plateau is ranking. The 1.7% surrogate still orders designs almost
perfectly, which is the engineering result the next section quantifies.

### 8.4 The floor under the exit: exact-Galerkin reduced dynamics (run 2026-07-12)

Before learning $f$, measure what any 8-dimensional Markovian model could achieve at
best. The POD-Galerkin ROM, $M_r \dot c = -\Phi^T R(\bar u + \Phi c) + q(t)\,\Phi^T
F_\text{unit}$ with $M_r = \Phi^T M \Phi$, integrates each held-out trajectory from
the projected data IC (FBDF, tstops at the switches, tolerance 1e-9), with the
projected residual assembled exactly. No hyper-reduction, and "exactly" scopes the
assembly: evaluating $R$ at $\bar u + \Phi c$ instead of at the true full state is
itself the closure approximation under test. Scored in the mapper's own trajectory
metric, with the projection floor recomputed in that metric. Three predictions were
registered in `scripts/30_transient_galerkin_floor.jl` before the run; all three hit.

| $r$ | mapper | Galerkin (exact assembly) | projection floor (traj metric) |
| ---: | ---: | ---: | ---: |
| 4 | 1.69e-2 | 3.90e-4 | 3.31e-4 |
| 8 | 1.74e-2 | 3.72e-6 | 2.95e-6 |
| 16 | 1.76e-2 | 3.47e-7 | 1.49e-8 |
| 32 | 1.45e-2 | 3.45e-7 | 2.66e-10 |

: Exact transient Galerkin floor by rank. {#tbl:transient-galerkin-floor}

Closure is nearly free where the reference data can resolve it: at $r = 8$ the exact
dynamics land at 3.72e-6, 1.26× the projection floor (P-G1); at $r = 4$, 1.2×
(P-G2). At $r = 16$ and 32 the readout saturates at 3.5e-7, the same value at both
ranks. This matches the integration-error floor expected from trajectories generated
at tolerance 1e-7 and is insensitive to this script's tighter tolerance: changing
1e-9 to 1e-10 moves the result by only 2.3e-12. The saturation is therefore reported
as a data-resolution floor, not as evidence about higher-rank closure.

The headroom number the exit hangs on (P-G3): the deployed mapper sits 4,670× above
the dynamics floor at the same rank. This is analogous to the steady map-vs-Galerkin
headroom in §7.3, although the transient metric and closure approximation differ. In
engineering units, the exact reduced dynamics track peak temperature to 0.005 K
median / 0.021 K max on held-out trajectories, where the mapper's tracking error is
14.3 K median / 42.6 K max (§8.2).

One measurement was registered with no bet attached: the reduced spectrum. Along the
audit trajectory at $r = 8$, |Re λ| spans 4.31e-5 to 8.15e-2 1/s (ratio ~1.9e3,
~3.9e2 per load period): the latent training problem is genuinely stiff, which was
the premise of the exit and is now a measurement. It motivates an implicit training
integrator for the learned-$f$ arm.

### 8.5 The first learned-dynamics arm: quadratic operator inference (run 2026-07-12)

On the 19 held-out trajectories, the Galerkin result constructs an eight-state Markovian
reduced model near the projection floor, but it uses the assembled operator. The first
non-intrusive question is how far snapshots alone can go. The tested quadratic form
follows operator inference [@PeherstorferWillcox2016].
`scripts/31_operator_inference.jl` was pre-registered
before its run: fit a structured quadratic vector field to finite-difference derivatives
of the whitened POD coefficients, then roll it out on the same 19 held-out trajectories
used above.

The model is

$$\dot{\tilde c} = (A_0 + \varepsilon A_1)\tilde c + (H_0 + \varepsilon H_1)(\tilde c \otimes \tilde c)_\text{sym} + (b_0 + \varepsilon b_1) + B\tilde q(t).$$

The load channel is linear, matching the FOM structure; emissivity enters the inferred
operators affinely. The degree-two truncation is the named approximation: projecting
$u|u|^3$ into coefficient space does not produce a quadratic polynomial. Central
differences use only phase-grid points whose two-sided stencil crosses no load switch.
The deterministic fit uses 3,850 rows, 91 features and a ridge fixed before the run at
$10^{-8}\mathrm{tr}(G^\top G)/91 = 9.93\cdot10^{-6}$. Evaluation uses FBDF,
exact switch `tstops`, and absolute and relative tolerances $10^{-9}$.

| method | information used | successful trajectories | trajectory rel $L^2$ |
| --- | --- | ---: | ---: |
| direct $(\mu,t)\mapsto c$ mapper | snapshots | 19/19 | $1.737\cdot10^{-2}$ (N = 19) |
| quadratic operator inference | snapshot finite differences | **1/19** | $9.20\cdot10^{-4}$ (N = 1 successful only) |
| exact Galerkin, $r=8$ | assembled projected residual | 19/19 | $3.719\cdot10^{-6}$ (N = 19) |
| trajectory projection floor, $r=8$ | reference snapshots | 19/19 | $2.949\cdot10^{-6}$ (N = 19) |

: Transient reduced-dynamics ladder. {#tbl:transient-dynamics}

The registered failure mode is the result: **18 of 19 rollouts escape in finite time**,
with the adaptive step size collapsing at $t \approx 0.4$–$6.4$ ks. The survivor's
$9.20\cdot10^{-4}$ error lies between the Galerkin floor and the mapper, but N = 1 makes
the registered accuracy band vacuous; it is reported only conditional on a successful
rollout. No aggregate accuracy claim is made.

The signature is consistent with three mechanisms that this experiment does not
distinguish: quadratic extrapolation outside the fitted neighborhood without the
contraction of the projected radiation term; under-constrained post-switch fast content
from the registered stencil exclusions; and instability of the unconstrained polynomial
rollout outside its fitted region. The narrow verdict is therefore not that latent dynamics
fail. Exact Galerkin already constructs stable reduced dynamics near the projection
floor. It is that this unconstrained, degree-two, snapshots-only fit fails the required
rollout-stability test; stability-constrained operator inference or a stabilized learned
vector field is the next rung.

## 9. Engineering consequences

### 9.1 Timing: latency vs throughput

BenchmarkTools medians, batch size = 102. Speed-ups are batched throughput over FOM Newton
solves, including POD reconstruction (and unscaling for the direct model), excluding plotting
and model loading, after warm-up; the FOM reference is the full Newton solve described in §3.
*Hardware, for every timing in this report:* AMD Ryzen
AI 9 HX 370 (CPU only; the iGPU is unused), 1 Julia thread, 12 BLAS threads, Julia 1.12.6,
re-measured 2026-07-09. The frozen data artifacts were generated on earlier hardware,
whose dataset log shows a median 49 ms Newton solve (§4); all timings quoted here are
from the current machine.

| method | single (ms) | batched (ms/q) | single× | batched× | allocs (1-q) |
| --- | ---: | ---: | ---: | ---: | ---: |
| FOM solve | 23.1 | — | 1× | — | 188,945 |
| direct MLP | 0.013 | 0.0032 | 1,796× | 7,259× | 22 |
| POD-MLP | 0.004 | 0.0026 | 5,270× | 8,852× | 31 |
| POD-KAN | 0.006 | 0.0025 | 4,159× | 9,104× | 90 |

: Warm steady-query timings and throughput. {#tbl:steady-timings}

The 10³–10⁴× factors are larger than the parent's because the FOM is now a nonlinear
solve. Their relevance is limited to many-query or deployment settings; the headline
of the method remains the structure (rank-2 basis, physics metrics), not the raw
speed-up.

### 9.2 Design ranking at throughput

The capstone reproduces the project's stated success criterion, a tool for testing subtle
design changes. Candidate designs: a 10×10 grid over emissivity coating $\varepsilon \in [0.15, 0.9]$ and
load $\text{duty} \in [0.25, 0.75]$; QoI = worst-case chip temperature over the final (settled) load
cycle. Two operating points guard against the result being a one-off.

| | OP1: $Q_p$=650 W/m, 5400 s | OP2: $Q_p$=400 W/m, 9000 s |
| --- | ---: | ---: |
| FOM sweep (100 candidates) | 390 s | 390 s |
| surrogate sweep | 1.77 s (incl. Julia JIT) | **0.048 s** (warm) |
| throughput | 221× | ≈8,100× |
| QoI error | median 2.4 K, max 16.5 K | median 2.2 K, max 16.8 K |
| QoI range | 382–785 K | 320–645 K |
| Spearman rank correlation | **0.9987** | **0.9988** |
| top-5 designs recovered | **5 / 5** | **5 / 5** |
| best design | identical (ε=0.90, duty=0.25) | identical (ε=0.90, duty=0.25) |

: Design-study performance at two operating points. {#tbl:design-study}

The throughput asymmetry is compilation, reported rather than hidden: the first sweep pays
Julia's JIT once; warm sweeps cost ~0.05 s for 100 candidates. The conservative end-to-end
number is 221×; the warm number is the deployment-relevant one. The FOM reference these
errors are measured against is itself solid: the steady GCI study (A.2) bands the
deployed grid's discretization error on peak temperature at 0.005–0.010 K, and the
transient reference is integrator-tolerance-controlled (§3), both far below the
2.2–2.4 K median surrogate QoI error quoted here.

![Design study: worst-case chip temperature maps](../figures/14_design_study.png){#fig:design-study}

*Worst-case chip temperature over a 10×10 (ε, duty) grid. The surrogate reproduces the FOM
ranking with Spearman correlation 0.999 at over 220× higher throughput (≈8×10³× warm), with
the same best design point (star).*

![Second operating point](../figures/19_design_study_op2.png){#fig:design-study-op2}

The claim defended is deliberately *ranking fidelity*, not pointwise truth: a screening tool
must order candidates correctly and hand the shortlist to the full-order model, which this
one does, in-distribution, with margin.

### 9.3 The deployment contract

This surrogate ships with four measured clauses; the machinery behind each lives in Appendix A.

#### Validity domain

Outside the training box the surrogate degrades *fast*, and
measurably: on 60 points in a physically sensible ring (ε down to 0.02, Q up to
1000 W/m), the median error grows from 7.0e-4 in-box to 1.2e-2, with a worst case of
19.5% at only 9% normalized exceedance. `in_training_box(ε, Q)` in `scripts/config.jl`
is the guard; screen inside the box, hand anything near the boundary to the FOM (A.1).

#### Settled-cycle clause

The design QoI reads cycle 3 of a 3-cycle run as "the
settled cycle". A 20-cycle FOM check at two points (A.4): the FOM settles geometrically
(cycle-to-cycle change 0.028/0.022 K by c19→20); the surrogate's settled-cycle deviation
stays flat within 1.41× of its cycle-3 value; and the measured qualification is that
cycle 3 itself sits 1.4–3 K short of the true attractor. That bias is shared by the FOM
sweep and the surrogate in the design study because both read cycle 3, so it does not
enter their reported agreement at that horizon. Whether cycle-3 and true-attractor
rankings agree across the full design grid was not tested; absolute temperatures quoted
at "the settled cycle" carry the measured 1.4–3 K qualification.
Normalized time $\tau > 1$ is out of distribution and guarded, the same way
`in_training_box` guards the steady box.

#### Arithmetic

Float32 inference is lossless at every reported digit; Float16 inference
survives (rel L² 1.06e-3, ~1 K worst-case field drift), but any $u^4$ evaluation
overflows Float16 at $T \ge 16.0$ K, i.e. at 100% of operating temperatures, so the
physics-residual check and any QoI integrand containing $u^4$ must run in Float32 or
wider (A.3). Thus Float32 is sufficient for the measured inference pipeline; no
hardware-exclusion claim is made for the small FOM.

#### Many-query readiness

Forward UQ through the surrogate propagates N = 100,000
scenarios in 7.5 s; against an exact 10,000-solve FOM Monte Carlo on the same sample
stream, every reported quantile agrees within 0.71 K (paired; A.5). The per-sample cost
sits ~590× below the FOM's (44 ms against 75 µs).

### 9.4 Interactive console

`scripts/14_run_console.jl` (GLMakie) solves the steady FOM *live* while you drag
$(\varepsilon, Q)$; a ~23 ms Newton solve (§9.1) is interactive. Next to it sit the surrogate's
prediction, the error field, QoI readouts, and both σ-spectra; the transient panel updates
the surrogate's peak-temperature trace instantly and overlays the true FOM trajectory on
demand. `CONSOLE_SMOKE=1` renders the same layout headlessly for CI. The console is a
scoping or calibration tool, not the science.

![Console](../figures/15_console.png){#fig:console}

## 10. Discussion

### Results by branch

1. For the tested fixed load geometry, the steady manifold is rank-2/3 and its measured
   nonlinear variation is represented through $c(\mu)$. This is favorable terrain for
   coefficient mappers; an intrusive online method would still need hyper-reduction.
   The transient branch gives the opposite verdict on the same discretization and solver.
2. The PDE residual separates the direct and reduced surrogates more strongly than L²:
   their L² errors differ by ~6× and their residuals by ~6,000×.
3. The surrogate preserves design rankings (worst-case temperature) at over two orders of
   magnitude higher throughput end-to-end, making it a practical screening tool inside
   many-query design loops, in-distribution; batched steady field queries reach 10³–10⁴×.
4. The workflow combines a non-intrusive ROM of a strongly nonlinear-BC problem with
   an exact Galerkin error decomposition and pre-registered diagnostics, implemented
   with the cited general-purpose FE and solver ecosystems
   [@Badia2020; @Pal2024; @Rackauckas2017]. The intrusive online alternative and its
   need for hyper-reduction follow the standard reduced-basis and DEIM constructions
   cited below.

### Additional verification findings

The verification pass produced three structural identities
in one family: the exact discrete energy balance (§2), $P_\text{rad}$ grid-exact to
1.6e-12 as its corollary, and the mean radiating-edge temperature nodally pinned to
1.0e-13 by an exact 1D reduction of the y-summed Q1 system (A.2). A pre-registered
Float16 overflow prediction that missed in the surrogate's favour (A.3). And the
settled-cycle bias, measured at 1.4–3 K at two points; both sides of the reported design
comparison use cycle 3, while true-attractor ranking across the grid remains untested
(A.4).
Together, these checks distinguish the steady and transient verdicts from unresolved
discretization, solver-tolerance or sampling effects within the tested protocol.
### Caveats

The model is 2D, with fixed geometry, unity view factor (deep space only), constant properties,
effective (not measured) $\rho c$, in-distribution queries only (measured: median error
grows ~17× just outside the training box, worst ~20% at 9% exceedance; A.1), and a single
PDE family: a controlled prototype, not a general theorem. The steady rank-3 result is
documented only for fixed load geometry and should not be attributed to radiation
physics alone; parametric load *placement* is expected to re-thicken the steady
manifold and remains future work.

### V&V status

Using ASME V&V 20 vocabulary, every comparison in
this report is *verification*, a computation checked against a more trusted computation
or an exact structural identity; *validation* would require physical measurement, which
this prototype does not attempt. The tiers:

| tier | evidence | status |
| --- | --- | --- |
| code verification | MMS through the nonlinear BC, observed L² orders 1.9996–2.0000 (§3); nodally exact 1D radiative equilibrium | complete |
| solution verification | exact discrete energy balance (measured ~1e-10, §3); nonlinear residual metric on every prediction (§7.1); grid-convergence (GCI) error bands on the FOM QoIs (A.2) | complete |
| surrogate verification | held-out errors, rank sweeps, N = 20 seed ensembles, measured OOD degradation, exact Galerkin floor (§7, A.1); note surrogate-vs-FOM is verification, not validation | complete |
| model validation | **unaddressed by design**: 2D section, unity view factor, fixed 3 K deep-space sink, effective ρc; no comparison against measurement or a 3D code | open |
| uncertainty quantification | forward UQ over (ε, Q) with an exact FOM Monte Carlo reference (A.5) | complete |

: Verification and validation status. {#tbl:vv-status}

The named model-form gap: $T_\text{space}$ is fixed at 3 K (deep space); in LEO,
Earth-shine and albedo raise the *effective* sink temperature to roughly 250 K, which
shifts absolute temperatures. The surrogate is trained at fixed $T_\text{space}$, so this
is a model-form limitation to re-train against, not a random input to sample (the 3 K
sink term itself is numerically irrelevant, $T_s^4/u^4 \le 10^{-8}$ over the box). A 3D
cross-code benchmark is declined on purpose: it would test model form, not this study's
claims, and MMS is the stronger code-verification instrument; the fin-model variant in
future work is the natural home for one.

## 11. Future work

The operator-learning baseline has run (§7.4: two hits, two instructive misses). The
transient dynamics floor and the snapshots-only quadratic attempt have also run
(§§8.4–8.5): the floor leaves 4,670× headroom under the mapper, while 18/19
operator-inference rollouts escape. First, complete the remaining exit identified in §8.3:
latent dynamics for $c(t)$, learning
$\dot c = f(c, \mu, \text{load}(t))$ through the solver instead of the $(\mu, t)$
regression, as a neural ODE / UDE on the coefficients or as operator inference on the
reduced system [@PeherstorferWillcox2016]; the load switches make the training problem
stiff [@Kim2021], so the crux is adjoint gradients through a stiff integrator. Then:
thin-plate fin variant (radiation as a volumetric $2\varepsilon\sigma(u^4 - T_s^4)/t$
sink; same Newton machinery); parametric load geometry; view factors and orbital
environment (solar/albedo as boundary sources); full DEIM for online reduced-Galerkin;
residual-informed training; unstructured GridapGmsh geometries; 3D;
mass-matrix-weighted POD.

## Appendix A. Verification machinery

Sorting rule for what lives here: checks whose failure would flip a conclusion are cited
inline in the section that rests on them; their machinery and full tables sit in this
appendix. Every number is regenerated by the numbered scripts and reprinted from the
frozen artifacts by `scripts/18_report_numbers.jl`.

### A.1 Seed ensembles, out-of-distribution behaviour, statistics

Are the headlines one lucky seed? Twenty fresh training seeds per deployed mapper
(`scripts/15_robustness.jl`; deployed models keep their rank-keyed seeds, the ensemble is
reported, not deployed). The seeds vary the network parameter initialization **only**:
training is full-batch Adam on the fixed Sobol dataset with a deterministic learning-rate
schedule, so there is no minibatch or data-draw stochasticity to vary.

| mapper | deployed | ensemble mean ± std (N = 20) | ensemble [min, max] |
| --- | ---: | ---: | ---: |
| POD-MLP r=2 | 9.18e-4 | 1.35e-3 ± 0.18e-3 | [9.69e-4, 1.67e-3] |
| POD-KAN r=2 | 2.92e-3 | 2.86e-3 ± 0.49e-3 | [1.90e-3, 3.56e-3] |
| POD-MLP-t r=8 | 1.74e-2 | 1.89e-2 ± 0.20e-2 | [1.53e-2, 2.25e-2] |

: Initialization-seed ensembles. {#tbl:seed-ensembles}

Across these 20 initializations, the ordering (MLP < KAN, both ≪ direct), the r = 2
optimum and the transient plateau are unchanged. The deployed steady MLP sits at the
favorable edge of this ensemble, below all 20 fresh seeds, which stay within 1.9× of
it. The initialization-ensemble summary is 1.35e-3 ± 0.18e-3, quoted alongside the
deployed result in the README.

Outside the training box the surrogate degrades *fast*, and now measurably: on 60 points in
a physically sensible ring (ε down to 0.02, Q up to 1000 W/m), the median error grows from
7.0e-4 in-box to 1.2e-2, with a worst case of 19.5% at only 9% normalized exceedance.
`in_training_box(ε, Q)` in `scripts/config.jl` is the guard; the usage contract is:
screen inside the box, and hand anything near the boundary to the FOM.

#### Confidence intervals and ablations

`scripts/20_statistics.jl` uses 10,000-resample bootstraps with
StableRNG(SEED+9000). Design-ranking Spearman: 0.9984 [0.9960, 0.9994] at
OP1 and 0.9983 [0.9965, 0.9992] at OP2 (N = 100 designs each). Paired per-sample
differences on the same 102 test parameters: KAN − MLP = +2.0e-3 [1.6, 2.4]e-3 and
direct − MLP = +4.9e-3 [4.4, 5.4]e-3 rel L²; both intervals exclude zero for the
deployed models. The N = 20 initialization ensembles in @tbl:seed-ensembles separately
show that the
model ordering survives every tested seed. Ablations at r = 2: turning whitening
**off** (same seed) degrades the mapper 200× to 2.1e-1, and swapping in a random
orthonormal basis caps accuracy at exactly its own projection floor (2.13e-1). Thus
whitening is load-bearing for this same-seed training comparison, while the random-basis
result attains its representational ceiling. The full StableRNG seed table is printed by
the script and in `data/statistics.jld2`.

![Seed robustness and OOD degradation](../figures/16_robustness.png){#fig:robustness}

### A.2 Grid-convergence error bars on the FOM (GCI)

How much of the "FOM truth" is discretization error? Three systematically refined grids
(60×30 / 120×60 / 240×120, ratio 2), five Sobol $(\varepsilon, Q)$ points, safety factor
$F_s = 1.25$ (`scripts/24_gci.jl`; predictions pre-registered in the script header, two
rounds, misses kept). The smooth QoIs converge at exactly the code-verification order:
the hot-node probe temperature (a fixed vertex of all three grids, equal to the reported
$T_\text{peak}$ on the deployment grid) shows observed order $\hat p = 1.99$ at all five
points, and $\|u_h\|_{L^2}$ shows $\hat p = 2.00$ (N = 5 each). Richardson banding puts
the **deployment-grid discretization error of the peak temperature at median 0.005 K,
max 0.010 K**: roughly 500× below the surrogate's 2.4 K median design-QoI error (§9.2),
so the reference the surrogate is judged against is nowhere the limiting error.

The pre-registered round-1 scorecard: one hit, two instructive misses.
$P_\text{rad}$ is grid-exact as predicted (worst $|P_\text{rad} - Q|/Q = 1.6\cdot
10^{-12}$ over 15 solves): the chip patch is face-aligned on all three grids and the
discrete energy balance holds on each, so there is no discretization error to measure.
The other two round-1 QoIs missed the predicted $\hat p \in [1.7, 2.2]$ band, each for a
structural reason. The raw nodal-max $T_\text{peak}$ converges *non-monotonically* (−33
then +1 mK between refinements): the argmax node drifts under refinement, a max is not a
smooth functional of $h$, and Richardson does not apply to it; the fixed-probe form is
the correct banding target. And the mean radiating-edge temperature shows no measurable
grid dependence at all (worst coarse-vs-fine deviation $1.0\cdot10^{-13}$): on this
tensor-product Q1 grid the $y$-summed discrete system reduces exactly to a 1D problem
whose data (the $y$-integrated load and the exactly balanced edge flux) are
grid-independent, pinning the edge mean nodally. A third structural identity in the same
family as the discrete energy balance. Transient GCI is deferred: the transient
reference is integrator-tolerance-controlled (§3) and is not banded here.

![Grid convergence of the hot-node temperature](../figures/23_gci.png){#fig:gci}

### A.3 Precision audit: what arithmetic does deployment need?

Casting the entire steady inference pipeline (inputs, network weights, whitening scales,
POD mean and modes) to reduced precision and re-evaluating the 102-sample test set
(`scripts/25_precision_audit.jl`; predictions pre-registered in the script header):

| pipeline | rel L² | max ΔT_peak (K) | max field drift vs F64 (K) |
| --- | ---: | ---: | ---: |
| Float64 | 9.1757e-4 | 3.56 | — |
| Float32 | 9.1757e-4 | 3.56 | 1.3e-4 |
| Float16 | 1.0586e-3 | 3.37 | 1.07 |

: Reduced-precision inference audit. {#tbl:precision-audit}

Float32 is lossless at every reported digit, as pre-registered. The Float16 prediction
**missed, in the surrogate's favour**: we predicted overflow at reconstruction
(coefficient scale against the 65,504 Float16 ceiling), but the whitening scales are
$4.97\cdot10^3$ and $5.71\cdot10^2$, every intermediate stays finite, and the Float16
pipeline still delivers rel L² 1.06e-3 with ~1 K worst-case field drift: usable,
marginally degraded. What does fail in Float16, exactly as pre-registered: any $u^4$
evaluation overflows for $T \ge 65504^{1/4} \approx 16.0$ K, i.e. at 100% of operating
temperatures. Edge-compute reading: field *inference* is Float32-safe (even
Float16-tolerant at the $10^{-3}$ level), but the physics-residual check and any QoI
integrand containing $u^4$ must run in Float32 or wider.

### A.4 Attractor check: 20 cycles against the 3-cycle deployment contract

The design QoI (§9.2) reads cycle 3 of a 3-cycle run as "the settled cycle". The mapper
is feedforward in $(\mu, \tau, \text{phase})$, so autoregressive drift is structurally
impossible; what could fail is the contract itself. Two parameter points, FOM run to 20
cycles, surrogate predicting its fixed settled-cycle trace by phase mapping
(`scripts/26_attractor_check.jl`; N = 2, predictions pre-registered in the script
header; $\tau > 1$ is never queried, it is OOD by the deployment contract):

- **FOM settling (pre-registered, hit).** The cycle-to-cycle peak-T trace change decays
  geometrically at both points: 22.4 K (c1→2) → 0.34 K (c9→10) → 0.028 K (c19→20), and
  15.3 → 0.29 → 0.022 K, under the registered 0.5 K-by-cycle-10 and 0.05 K-at-19→20
  bounds.
- **Deviation flatness (hit on the registered 2× bound, with one measured
  qualification).** The per-cycle deviation between the surrogate's fixed settled-cycle
  trace and FOM cycles 3–20 stays within [0.81, 1.41]× and [0.84, 1.13]× of its cycle-3
  value (7.4 K and 10.2 K, consistent with the §8.2 error level). It is not constant,
  though: it rises by +3.0 K and +1.4 K before saturating, and the rise tracks the FOM's
  own settling tail. Read structurally: the training data's cycle 3 sits 1.4–3 K
  short of the true attractor at these points. That bias is shared by the FOM sweep and
  the surrogate in the design study because both read cycle 3, so it does not enter their
  reported agreement at that horizon. The 20-cycle check covers only N = 2 points and
  does not establish that cycle-3 and true-attractor rankings agree across the design grid.

The deployment contract, now explicit and measured: phase features are valid on (or
near) the attractor; normalized time $\tau > 1$ is out of distribution and guarded, the
same way `in_training_box` guards the steady $(\varepsilon, Q)$ box.

![Attractor check](../figures/25_attractor_check.png){#fig:attractor-check}

### A.5 Forward UQ with an exact FOM reference

Input uncertainty, stated here and used nowhere else: $\varepsilon \sim
\mathcal N(0.85, 0.02)$ (coating tolerance around a high-emissivity design point) and
$Q \sim \mathcal N(600, 50)$ W/m (load uncertainty), both truncated to the training box
by rejection (acceptance 1.0000 each; 100,000 / 100,000 samples pass `in_training_box`
by construction). QoI: steady worst-case (peak) temperature
(`scripts/27_forward_uq.jl`; StableRNG(SEED + 2700); predictions pre-registered in the
script header). The surrogate propagates N = 100,000 samples in 7.5 s warm; the exact
FOM reference is N = 10,000 Newton solves (444 s, warm-started, median 44 ms/solve) on
the *first 10,000 points of the same sample stream*, so the surrogate-vs-FOM deltas
in @tbl:forward-uq are paired and carry no Monte Carlo noise:

| quantile | FOM (N = 10⁴) | surrogate, paired (N = 10⁴) | Δ paired (K) | surrogate (N = 10⁵) |
| ---: | ---: | ---: | ---: | ---: |
| 50% | 616.49 | 616.27 | −0.22 | 616.44 |
| 90% | 649.51 | 649.36 | −0.15 | 650.24 |
| 99% | 678.24 | 678.93 | +0.68 | 678.68 |
| 99.7% | 690.66 | 691.36 | +0.71 | 689.10 |

: Forward-UQ temperature quantiles. {#tbl:forward-uq}

Pre-registered: every paired quantile delta under 4 K (the steady surrogate's max
peak-T error over the test box). Measured: $|\Delta| \le 0.71$ K at every reported
quantile, pointwise paired error median 0.29 K, max 0.86 K (N = 10,000), an order of
magnitude inside the registered bound. The 3σ statement a designer would use: the 99.7%
quantile of worst-case temperature is 690.7 K by FOM against 689.1 K by the surrogate at
N = 10⁵, and the paired column shows only +0.7 K of that gap is surrogate error; the
rest is FOM-side Monte Carlo noise in a thin tail at N = 10⁴. For tail quantiles the
surrogate's N = 10⁵ estimate is the better-converged one, at a per-sample cost ~590×
below the FOM's (44 ms against 75 µs). $T_\text{space}$ stays fixed at 3 K throughout: a
model-form limitation
(§10), not a random input; re-training at an Earth-shine effective sink, not sampling
the sink, is the correct treatment.

![Forward UQ](../figures/26_forward_uq.png){#fig:forward-uq}

## References
