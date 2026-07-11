# Why is the steady radiative manifold rank three? A predictive argument, pre-registered

*Working note for discussion. Everything here is reproducible from
`scripts/`; the pre-registered predictions in §3 were written and committed before
`scripts/19_rank_prediction.jl` was run (see git history).*

## 1. The finding

For the steady heat equation on a fixed 2D radiator section with a Stefan–Boltzmann
boundary condition ($-k\,\partial_n u = \varepsilon\sigma(u^4 - T_s^4)$, deep-space sink),
sampled over $\mu = (\varepsilon, Q) \in [0.1, 0.95] \times [100, 800]$ W/m, the centered
snapshot matrix has singular values

$$\sigma_2/\sigma_1 = 1.15\cdot 10^{-1}, \qquad \sigma_3/\sigma_1 = 7.2\cdot 10^{-9},
\qquad \sigma_5/\sigma_1 = 6\cdot 10^{-16}.$$

The manifold is numerically **rank 2 after centering** (three shapes counting the mean),
despite temperatures spanning 300–970 K and a strongly nonlinear BC. This is not a
discretization artifact: the ratios replicate on a 2× finer grid (6.7e-9) and with P2
elements (6.8e-9), and every downstream conclusion survives 20-seed retraining (N = 20,
mean ± std in report A.1; seeds vary the network initialization only, since training is
full-batch on a fixed Sobol dataset; StableRNG seed table in the report appendix).

## 2. The argument

Two ingredients, of very different logical strength.

**(a) Exact parameter collapses (rigorous; no approximation).** Divide the whole problem
by $k$: the interior equation becomes $-\nabla^2 u = (Q/k)\,\hat q$ and the BC becomes
$-\partial_n u = (\varepsilon/k)\,\sigma(u^4 - T_s^4)$. The solution therefore depends on
$(\varepsilon, Q, k)$ **only through $(Q/k, \varepsilon/k)$**: conductivity is not a new
parameter, it rescales the other two, and the 3-parameter manifold is exactly a
2-parameter manifold (this is what P2 tests). Similarly, each disjoint load shape enters
the inhomogeneity as its own fixed spatial mode $(Q_i/k)\,\hat q_i$, which is what P1's
"+1 direction per heater" rests on. One caution the first draft got wrong: the
*inhomogeneity* is rank-1 per load, but the *response* to it is not automatically
one direction, because the BC couples everything through the total temperature. The
collapse arguments bound the number of effective parameters, not yet the rank.

**(b) From effective parameters to rank (standard, plus an open gap).** Linearizing the
BC around a reference state gives a Robin problem with coefficient
$4\varepsilon_0\sigma u_0^3$ and smooth (indeed analytic) dependence on the effective
parameters; Kolmogorov $n$-widths of such families decay exponentially. This half is
15 years of standard theory: Cohen–DeVore–Schwab (2011) for analytic parametric
ellipticity, Cohen–DeVore (2016) for widths under holomorphic maps,
Lassila–Manzoni–Quarteroni–Rozza (2013) for parametric operators with a dominant affine
term, Bachmayr–Cohen (2017) for the low-rank story. What the standard theory predicts is
*fast smooth decay*; what we observe is a **hard cutoff** ($\sigma_{d+1}/\sigma_1 \sim
10^{-8}$, then machine floor), and the failed Taylor scaling in P3 says the cutoff
magnitude is not a local second-order remainder either. The mechanism that sets the
cutoff scale is the genuinely open part; §4b below tests one candidate (spectral-gap
suppression: the parametric response may excite eigenmodes of the linearized operator at
amplitudes falling off with the eigenvalue).

## 3. Pre-registered predictions (written before running the experiments)

The argument is only worth discussing if it predicts. Three experiments, three *different*
expected outcomes:

**P1 -second heater patch.** $\mu = (\varepsilon, Q_1, Q_2)$ with a second, disjoint
source patch. By (a), each load contributes its own exact direction. **Predict: centered
rank 3** (two source responses + level), i.e. $\sigma_3/\sigma_1 = O(10^{-1\pm 1})$ now a
*real* mode, and the remainder cliff moves to $\sigma_4/\sigma_1 = O(10^{-8\pm 1})$, with
$\sigma_5$ near machine precision.

**P2 -conductivity as the third parameter.** $\mu = (\varepsilon, Q, k)$, $k \in [2, 10]$.
The naive "$d+1$ shapes" rule says rank 4. The structural argument says otherwise: by (a),
$Q$ and $k$ enter the particular solution only through $Q/k$, and the level is set by a
$k$-free energy balance, so $k$ mostly re-parametrizes existing directions. **Predict: no
new $O(\sigma_2)$-scale mode; the spectrum looks like the 2-parameter one, with
$\sigma_4/\sigma_1 < 10^{-5}$** (weak new curvature is admitted, a hard fourth mode is
not). This is the falsification test that distinguishes the structural argument from
parameter counting: if a strong fourth mode appears, §2(a) is wrong as stated.

**P3 -box shrinking.** Shrink the $\varepsilon$-box (halfwidth full/½/¼ around 0.525) at
fixed $Q$-box, and separately the $Q$-box (around 450) at fixed $\varepsilon$-box. If
$\sigma_3$ is a second-order remainder of the joint map, halving a box side that
participates in the dominant curvature should reduce $\sigma_3/\sigma_1$ by ×2–×4
(order between 1 and 2); a box axis that does not participate should leave it nearly
unchanged. **Predict: log–log slope in [1, 2] on at least one axis; the pair of slopes
identifies which curvature (ε-, Q-, or cross-term) dominates the remainder.** We commit to
reporting both slopes whatever they are.

## 4. Results (scorecard, misses included)

![Rank predictions: spectra overlays for the 3-parameter variants and box-shrink scaling of the remainder](../figures/20_rank_prediction.png)

**P1 -hit on structure, miss on remainder magnitude.** With the second heater,
$\sigma_3/\sigma_1 = 3.5\cdot 10^{-2}$ becomes a genuine mode and the cliff moves to
$\sigma_4/\sigma_1 = 8.3\cdot 10^{-6}$, $\sigma_5 = 2.2\cdot 10^{-9}$: centered rank 3 as
predicted. The new remainder is however ~2 orders above the predicted $O(10^{-8\pm1})$
band; in hindsight the two loads double the total-load excursion, and the second-order
term grows with it. Structural prediction confirmed, magnitude underestimated.

**P2 -strong hit; the falsification test passed.** With $\mu = (\varepsilon, Q, k)$ the
spectrum is indistinguishable in shape from the 2-parameter one:
$\sigma_3/\sigma_1 = 2.3\cdot 10^{-8}$ (remainder level), $\sigma_4/\sigma_1 =
3.2\cdot 10^{-14}$. The naive $d{+}1$ rule predicts a fourth $O(\sigma_2)$ mode;
the $Q/k$-collapse argument predicted none; none appears, with nine orders of margin.
This is the experiment that distinguishes the structural argument from parameter counting.

**P3 -informative miss.** Shrinking boxes reduces the remainder far more slowly than a
second-order Taylor term should: fitted log–log slopes $0.22 \pm 0.00$ on the ε-axis and
$0.66 \pm 0.01$ on the Q-axis, mean ± std over N = 3 independent dataset draws
(registered prediction: at least one axis in [1, 2]). The replication was added at
external review; the sampling uncertainty is negligible, so the miss is not a dataset
artifact. The Q-axis dominance matches the edge-flux-nonuniformity mechanism, but the
scaling says the parameter box is not in the asymptotic small-perturbation regime, so
$\sigma_3$ is governed by something coarser than the local quadratic remainder. We
report this as a miss; §4b locates what actually sets the scale.

## 4b. Exploratory follow-up (not pre-registered): what sets the cutoff?

External review proposed a spectral-gap mechanism: the response might excite eigenmodes of
the linearized operator at amplitudes $\sim 1/\lambda_k$, making the cliff an
eigenvalue-gap effect. We measured it (`scripts/22_eigenmode_diagnostic.jl`):
generalized eigenbasis of $A_\text{cond} +
4\varepsilon_0\sigma|u_0|^3$ at the box-center reference, centered snapshots expanded in
it. Three findings:

1. **A parity selection rule.** The problem is symmetric about the midline $y = L_y/2$,
   so odd-parity eigenmodes carry exactly zero ($\alpha_k/\alpha_1 \sim 10^{-13}$); only
   12 of the first 20 modes are excited at all.
2. **Amplitudes fall as $\lambda^{-1.9}$**, steeper than the hypothesized $1/\lambda$.
3. **The decisive one: the cliff is functional collinearity, not spectral content.**
   Sizable eigen-content ($\alpha_4/\alpha_1 = 3.6\cdot10^{-2}$) coexists with POD
   $\sigma_3/\sigma_1 = 7\cdot10^{-9}$. Row-normalizing the 12 excited coefficient
   functions $\mu \mapsto \langle v_k, u(\mu)\rangle$ and taking their singular values
   gives 2.5, 2.3, $3.9\cdot10^{-5}$, then machine floor: twelve spatial modes, but only
   **two independent parameter functions** driving them.

![Eigenmode diagnostic: response energy vs linearized-operator eigenvalue, with the rejected 1/λ reference and the λ^-1.9 fit](../figures/21_eigenmode_diagnostic.png)

**The quantitative version survives across configurations**
(`scripts/23_cliff_reconstruction.jl`, exploratory, run after the round-2 review): the
$(r^*{+}1)$-th singular value of the *amplitude-weighted* excited coefficient functions
reproduces the measured POD cliff to a uniform ~25% on all seven configurations tested:
base $5.8$ vs $7.2\cdot10^{-9}$; P1 $6.6$ vs $8.3\cdot10^{-6}$, with $r^* = 3$ detected
automatically, as the collapse argument demands; P2 $1.9$ vs $2.3\cdot10^{-8}$, $r^*=2$;
and the four P3 shrink boxes track the box-size dependence. The cliff is therefore
entirely a property of the dozen excited coefficient functions; nothing beyond the first
20 eigenmodes contributes at cliff level. One correction to the naive product formula we
sketched first: leakage × total amplitude of the leaking direction overpredicts by
$\sim 10^3$, because the equal-weight leaking direction mixes into the large-amplitude
modes; the cliff is carried by its component *outside* the span of the top-$r^*$ weighted
directions, which lives at the $\sim 10^{-4}$ relative amplitude of the small modes. In
one line: **cliff scale = parametric collinearity leakage × the amplitude the independent
direction actually carries.**

![Cliff reconstruction: predicted (weighted collinearity) vs measured POD cliff across seven configurations](../figures/22_cliff_reconstruction.png)

So the plain spectral-gap hypothesis is rejected, and the open question sharpens: why do
the coefficient functions of a dozen spatial modes span (almost exactly) a 2-dimensional
function space? The §2(a) collapses bound the *effective parameter count* at two, and
analyticity in those two effective parameters is presumably what compresses the function
space; that reconnects to the standard width theory, but posed in parameter-function
space rather than state space. This is the thread we would most like a read on.

## 5. The transient contrast

Under switching load cycles the space-time manifold re-thickens ($\sigma_8/\sigma_1
\approx 3\cdot 10^{-5}$, smooth decay over 60 modes). This is *consistent* with the
argument: each trajectory sweeps a continuum of transient states that no small fixed
family spans, and the switching makes the phase dependence non-smooth, which is exactly
when Kolmogorov widths decay slowly; but we do not claim it as a prediction of the
linearization, which is a steady, smooth-family statement.

## 6. Where the surrogate error actually lives

Exact reduced-Galerkin solves (Newton on $\Phi_r^\top R(\bar u + \Phi_r c) = 0$; no
hyper-reduction needed offline at $n = 1891$; DEIM would be the online path) give the
honest floor for *any* surrogate confined to the POD subspace. Mean over 102 held-out
parameters, all solves converged:

| rank | L² projection floor | Galerkin floor | POD-MLP error | map share of error |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.7e-2 | 2.4e-2 | 1.7e-2 | basis-limited |
| **2** | 1.14e-9 | **1.14e-9** | 9.2e-4 | **100.00%** |
| 3 | 6.2e-16 | 2.8e-13 | 1.3e-3 | 100.00% |
| 5 | 1.8e-16 | 2.8e-13 | 1.7e-3 | 100.00% |

Two readings. At the manifold rank and above, the Galerkin floor *equals* the projection
floor, so every last part of the deployed surrogate's error is coefficient-map error;
the basis contributes nothing measurable. Below the manifold rank ($r = 1$) the picture
inverts: Galerkin is even slightly worse than L² projection (it optimizes the projected
residual, not the error) and the neural mapper already sits at the basis floor. The
crossover is the diagnostic: it tells you, per rank, whether to spend effort on the basis
or on the map.

Supporting statistics (all in `scripts/20_statistics.jl`; N stated, StableRNG seed table
in the report appendix): design-ranking Spearman with 10,000-resample bootstrap CIs,
0.9984 [0.9960, 0.9994] and 0.9983 [0.9965, 0.9992] at the two operating points; paired
bootstrap on per-sample errors, KAN − MLP = +2.0e-3 [1.6, 2.4]e-3 and direct − MLP =
+4.9e-3 [4.4, 5.4]e-3 (both exclude zero); ablations at r = 2: whitening off degrades
the mapper 200× (2.1e-1), and a random orthonormal basis caps accuracy at exactly its
own projection floor (2.13e-1), the basis-quality control.

## 7. Questions (updated after two rounds of external review)

Positioning adopted. The smoothness-to-exponential-decay half of §2(b) is standard
(Cohen–DeVore–Schwab 2011; Cohen–DeVore 2016; Lassila–Manzoni–Quarteroni–Rozza 2013;
Bachmayr–Cohen 2017; the analytic-in-parameters machinery is Chkifa–Cohen–Schwab 2015).
The §4b picture is also known *in shape*: many excited spatial modes with few independent
parameter functions is the linear-width vs intrinsic-dimension gap behind nonlinear and
compressive reduced bases (Prud'homme–Maday–Ballout 2024; Cohen–Dolbeault–Somacal–Dahmen
2023 is the closest low-width example in spirit to the $Q/k$ collapse). We read
Prud'homme–Maday–Ballout in full before claiming anything: they establish complexity
scaling with the parameter count rather than the $N$-width, but contain no quantitative
cliff decomposition, no SVD analysis of normalized coefficient functions, and no symmetry
constraints. What survives as plausibly ours: (i) the cross-configuration cliff
reconstruction of §4b (weighted collinearity = POD cliff to ~25% on seven configurations);
(ii) the parity selection rule as an exact structural constraint; (iii) the §2(a)
collapses run as pre-registered falsification tests, a move the width literature does not
make. Baseline plan: DeepONet only, on the same snapshots; full DEIM dropped, since §6
already delivers the Galerkin floor it would provide.

For discussion:
1. The §4b reconstruction (cliff = amplitude-weighted collinearity of the excited
   coefficient functions, verified across seven configurations): worth a standalone short
   note, or a paragraph in the larger write-up? We believe it is not in
   Prud'homme–Maday–Ballout 2024; a second pair of eyes on that would help.
2. The transient thread. The direct $(\mu, t) \to c$ mapper plateaus at 1.7% and is
   measurably data-limited (learning-curve slopes 0.8–1.1, report §8.2), so the next
   formulation is probably latent dynamics: $\dot c = f(c, \mu, \text{load}(t))$, eight
   unknowns, trained through the ODE solver against the same trajectories. The load
   switches put this squarely in the stiff-neural-ODE regime where naive training is
   known to fail (Kim, Ji, Deng, Ma & Rackauckas 2021: output scaling, loss choice,
   stabilized gradients). Is that the right next formulation here, and what training
   protocol (solver, sensitivity method, loss scaling) would you start from?
3. DeepONet protocol: same 410 snapshots, equal and 3× parameter budgets, matched
   gradient-step budget, loss curves overlaid, PDE residual as the primary metric, and a
   third arm with the trunk frozen to the POD basis to isolate the basis choice (config B
   has no Dirichlet BC, so the classical exact-BC wedge does not apply here). Anything
   missing?

## References

- Cohen, DeVore & Schwab, *Analytic regularity and polynomial approximation of parametric
  and stochastic elliptic PDEs*, Analysis and Applications, 2011.
- Cohen & DeVore, *Kolmogorov widths under holomorphic mappings*, IMA Journal of
  Numerical Analysis, 2016.
- Lassila, Manzoni, Quarteroni & Rozza, *Generalized reduced basis methods and n-width
  estimates for the approximation of the solution manifold of parametric PDEs*, 2013.
- Bachmayr & Cohen, *Kolmogorov widths and low-rank approximations of parametric elliptic
  PDEs*, Mathematics of Computation, 2017.
- Chkifa, Cohen & Schwab, *Breaking the curse of dimensionality in sparse polynomial
  approximation of parametric PDEs*, J. Math. Pures Appl., 2015.
- Cohen, Dolbeault, Somacal & Dahmen, *Reduced order modeling for elliptic problems with
  high contrast diffusion coefficients*, arXiv:2304.10971, 2023.
- Prud'homme, Maday & Ballout, *Nonlinear compressive reduced basis approximation for
  multi-parameter elliptic problem*, arXiv:2407.03769, 2024.
- Chaturantabut & Sorensen, *Nonlinear model reduction via discrete empirical
  interpolation*, SIAM J. Sci. Comput., 2010.
- Kim, Ji, Deng, Ma & Rackauckas, *Stiff neural ordinary differential equations*,
  Chaos, 2021.
- Peherstorfer & Willcox, *Data-driven operator inference for nonintrusive
  projection-based model reduction*, CMAME, 2016.
