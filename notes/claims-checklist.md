# Scientific claims and academic-tone checklist

*Created by GPT-5.6 during an editorial audit of this repository and retained as a
transparent record of the prose-review method. Scientific responsibility remains with
the project author.*

Use this before freezing an abstract, results section, figure caption or README headline.
The goal is not to weaken claims. It is to make each sentence exactly as strong as its
evidence, while keeping standard knowledge concise and attributed.

## 1. Classify the claim before wording it

### Exact identity or proved consequence

Use **exact**, **proves**, **must**, **cannot** or **independent of** only when the result
follows from equations or logic under stated assumptions.

- State the assumptions and domain.
- Show or cite the derivation.
- Check that the conclusion is the actual logical consequence, not a nearby intuition.
- Do not soften an exact result merely to sound cautious.

Example pattern:

> Dividing by k shows that the solution depends on (Q, ε, k) only through
> (Q/k, ε/k), for the stated model with fixed geometry and material laws.

### Measured result

Use **measured**, **observed**, **on this test set**, **over this box** or
**within this protocol**.

- State N, range, metric and failure count.
- Separate interpolation from extrapolation.
- Do not convert repeated measurements into a theorem without a derivation.

### Controlled causal comparison

Use **identifies**, **isolates**, **causes** or **attributable to** only when the
comparison changes the claimed factor and holds relevant alternatives fixed.

- Same data and split?
- Same loss and optimization budget?
- Same parameter budget or a justified capacity control?
- Same evaluation metric?
- Only the intended architecture component changed?
- Enough seeds to distinguish the effect from initialization?

If several factors change, report a ladder or association, not a unique cause.

### Mechanistic interpretation

Distinguish three levels:

1. **Algebraic localization:** a decomposition shows where a quantity appears.
2. **Empirical consistency:** observations match a proposed mechanism.
3. **Causal mechanism:** interventions discriminate it from plausible alternatives.

Do not use level 3 language for level 1 or 2 evidence. A decomposition may locate a
cutoff without explaining why its ingredients have those values.

### Standard or previously known result

If the statement is standard:

- Say so once.
- Cite the appropriate source at first use.
- State only the part needed for the argument.
- Describe this work as applying, measuring or testing it, not discovering it.

Avoid narrating a textbook fact as a revelation. The contribution is usually the
problem-specific measurement, falsification test or quantitative consequence.

### Engineering decision

Use **motivates**, **supports**, **we choose** or **is the next tested option**.
A useful choice need not be mathematically necessary or sufficient.

## 2. Logic checks

Before keeping **therefore**, **because**, **hence**, **the reason**, **the mechanism**,
**not X but Y**, or **one-to-one**, ask:

- Is the stated condition necessary, sufficient, both, or neither?
- Could X and Y both bind at different budgets or ranks?
- Does absence of evidence for X establish Y?
- Does a low-dimensional input imply a low-rank response? Usually not.
- Does one source direction imply one solution-manifold direction? Not under a nonlinear
  coupled operator.
- Does matching a floor identify the source of every error, or only rule out one
  bottleneck at the measured scale?
- Is a percentage an additive error decomposition, or only a ratio of two norms?
- Does a failed training protocol reject the model class, or only that optimization and
  architecture?
- Does a successful result on one box imply independence outside it?

## 3. Scope without performative humility

Scope the claim where the evidence changes, not everywhere.

Good:

> At r = 2 on the held-out box, the exact Galerkin floor equals the projection floor;
> the deployed surrogate is map-limited.

Weak and unnecessary:

> This may perhaps suggest that the basis might not be the main issue.

Overbroad:

> Reduced-basis surrogates are map-limited.

Precise scope is not sandbagging. It protects the strong statement inside that scope.

## 4. Ablation language

### If one factor is isolated

> At matched loss, data and parameter budget, replacing the learned trunk with the POD
> trunk reduces the residual by 14.9×; basis choice contributes materially under this
> protocol.

### If several factors change

> Residuals form a monotone ladder across the tested factorizations. Because architecture
> and objective change together in some steps, the full ladder is descriptive rather
> than a one-factor causal decomposition.

### If the control misses its registered bound

Report both facts:

- the bound missed;
- any independently registered relative comparison that still passed.

Do not rescue the original prediction by silently replacing it with the surviving
comparison.

## 5. Failure and negative-result language

- Report denominator and failure count before conditional accuracy.
- Never average only successful runs without labeling the statistic conditional.
- Name the rejected object narrowly: solver configuration, training protocol, polynomial
  ansatz, parameter box or model class.
- Preserve constructions that prove the broader class remains viable.

Pattern:

> Eighteen of 19 rollouts escape. The one successful rollout has error 9.2×10⁻⁴
> (N = 1, conditional). This rejects the unconstrained quadratic fit as a deployable
> model here; it does not reject eight-state Markovian dynamics.

## 6. Novelty and attribution etiquette

- Cite standard theory when it carries an inference.
- Cite the method origin when using an established construction.
- Cite the closest prior mechanism before describing what remains distinctive.
- Do not claim novelty from absence in papers that were not searched systematically.
- Prefer a concrete contribution statement:

> We pre-register the collapse as a falsification test and quantify the remaining cliff
> across seven configurations.

Avoid:

> We reveal the fundamental mechanism behind low-rank physics.

If attribution is uncertain, mark it for literature review rather than writing around it.

## 7. Tone checks

Delete or reconsider:

- “locating the physics”;
- “the physics lives in”;
- “the lesson is” when the result is a comparison;
- “honest” as self-certification;
- “obviously”, “clearly” or “of course”;
- historical sermons such as “ten years ago this required...” without evidence;
- praise for standard tooling that substitutes for a technical claim;
- “the community's own method” or other adversarial framing.

Prefer:

- measured quantity first;
- comparison and controls second;
- interpretation third;
- limitation only where it changes the conclusion.

## 8. Insider-smell test

Ask how a specialist could object in one sentence:

- “That is standard; where is the citation?”
- “That ablation changes two things.”
- “You showed consistency, not causation.”
- “That follows only on your parameter box.”
- “The result is conditional on the successful run.”
- “A sufficient condition was presented as necessary.”
- “The floor is not an additive error decomposition.”
- “This is a useful engineering choice, not a theorem.”

If the objection is correct, fix the claim. If it is not, add the missing derivation,
control or scope so the strong wording is visibly earned.

## 9. Final five-minute pass

Search for:

`all`, `none`, `only`, `exact`, `proves`, `confirms`, `therefore`, `because`,
`the mechanism`, `the reason`, `the bottleneck`, `sets`, `locates`, `physics`,
`cannot`, `must`, `independent`, `general`, `standard`, `known`, `honest`.

For every match:

1. Label it exact, measured, causal, mechanistic, standard or engineering.
2. Point to the equation, artifact, control or citation that earns the wording.
3. Add scope only if the evidence has scope.
4. Keep strong wording when the proof or controlled comparison supports it.
5. Replace rhetoric with the measured relationship when it does not.

## 10. Release gate

- [ ] Every headline number states N or points to a table that does.
- [ ] Failures and conditional metrics are reported together.
- [ ] Exact claims include their assumptions.
- [ ] Causal claims have a true control.
- [ ] Mechanism claims distinguish localization, consistency and causation.
- [ ] Necessary and sufficient conditions are not interchanged.
- [ ] Parameter-box findings are not promoted to universal laws.
- [ ] Standard results are cited and not presented as discoveries.
- [ ] Novel contributions are stated concretely.
- [ ] Strong justified claims remain strong.
- [ ] No self-congratulatory, adversarial or performatively deferential framing remains.
