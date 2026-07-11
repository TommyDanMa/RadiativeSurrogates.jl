# Contributing

Contributions are welcome. This repository follows the SciML community
practices:

- **Conduct and process**: [ColPrac](https://github.com/SciML/ColPrac), the
  contributor's guide on collaborative practices for community packages.
- **Code style**: [SciMLStyle](https://github.com/SciML/SciMLStyle). In short:
  4-space indent, 92-character lines, `CamelCase` types, `snake_case`
  functions, comments above code, prefer renaming over commenting.

## Running the tests

Run the suite the way CI does:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. test/runtests.jl
```

Do **not** use `Pkg.test()`: it sandboxes a test environment in which the
skip-guarded `test/test_deployed_models.jl` cannot include `scripts/config.jl`.
The suite is 51 assertions on a fresh clone and 68 with `data/` artifacts
present.

## Data and the frozen-number discipline

`data/` is gitignored and regenerated end to end by
`julia --project=. scripts/00_run_all.jl` (all scripts are deterministic:
StableRNGs + Sobol sampling). Every headline number in `report/report.md` is
reprinted from the data artifacts by `scripts/18_report_numbers.jl`.

If your change affects any numerical result:

1. Regenerate the affected artifacts by running the corresponding numbered
   script(s).
2. Run `scripts/18_report_numbers.jl` and diff its output against
   `report/report.md`.
3. Update the report so the diff is clean, and say so in the PR.

PRs that change numbers without updating the report will fail review. New
experiments should follow the house pattern: a numbered script with any
prediction pre-registered in its header comment before the first run, results
saved to `data/`, figures to `figures/`, and a printer section added to
`scripts/18_report_numbers.jl`.

## Benchmarks

Timing numbers are hardware-bound. If you re-measure, state the hardware (CPU,
Julia version, thread counts) as done in report §9.1, and do not mix timings
from different machines in one table.
