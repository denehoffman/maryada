# maryada
[![Crates.io Version](https://img.shields.io/crates/v/maryada?style=for-the-badge&logo=rust&logoColor=%23D34516&labelColor=%231E2650&color=%23D34516)](https://crates.io/crates/maryada) [![docs.rs](https://img.shields.io/docsrs/maryada?style=for-the-badge&logo=docs.rs&logoColor=%23D34516&labelColor=%231E2650)](https://docs.rs/maryada/latest/maryada/) [![Codecov](https://img.shields.io/codecov/c/github/denehoffman/maryada?style=for-the-badge&logo=codecov)](https://app.codecov.io/gh/denehoffman/maryada) [![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/denehoffman/maryada/build-clippy-test.yml?style=for-the-badge&logo=github)](https://github.com/denehoffman/maryada/actions/workflows/build-clippy-test.yml)

maryada is a `no_std` binary64 interval arithmetic library conforming to IEEE Std 1788.1-2017. It provides bare and decorated real intervals, outward-rounded elementary operations, and text and binary interchange. Rectangular complex intervals are available through an optional feature.

The conformance claim applies to the real interval API. The rectangular complex interval extension is outside the scope of the standard. See [CONFORMANCE.md](CONFORMANCE.md) for the operation accuracy declarations, required features, implementation details, and test coverage.

```rust
use maryada::{Interval, IntervalOps};

let x = Interval::new(1.0, 2.0);
let y = x.sqr();

assert_eq!(y.bounds(), (1.0, 4.0));
```

## Features

- `complex` enables the `ComplexBox` rectangular complex interval API.
- `num-complex` enables `complex` plus interoperability with `num_complex::Complex64`.
- `linalg` enables the linear algebra API, namely `IntervalMatrix` and its associated operations.
- `complex-linalg` enables allocation-backed complex interval matrices and verified complex solves.
- `branch` enables the `no_std` branch-and-bound engine and its allocation-backed work queues.

Complex systems use the same matrix-facing API as real systems. The default
verified solve first applies a residual-centered complex epsilon-inflation
method and transparently retries through a conservative 2n real embedding when
the direct sufficient test does not certify an enclosure. Use `try_solve` when
you need a typed `SolveError`, or `solve` for the concise `Option` form.

The `branch` feature provides a resumable generic `BranchAndBound` scheduler,
pluggable evaluators, pruners, branching rules, and work queues. Its
`GlobalMinimizer` convenience layer implements best-first interval subdivision;
domains can be real intervals, complex boxes, vectors, or custom types that
implement `Bisect` and optionally `Midpoint`. A new minimizer has finite
value-width, parameter-width, and global-gap defaults and no step limit:

```rust,ignore
let result = GlobalMinimizer::new(domain, objective).solve()?;
```

Use `with_value_tolerance`, `with_domain_tolerance`, `with_gap_tolerance`, and
`with_max_steps` to build a different policy. `advance(n)` is available when
an explicit bounded, resumable step is desired. Disabled tolerances are
represented by `None`; `solve` rejects a configuration with neither a
tolerance nor a step limit.

In a `GlobalMinimizationResult`, `minimum` is the rigorous enclosure of the
global minimum value. `best_point`, `best_domain`, and `best_value` identify the
point that supplied the current upper bound, the associated parameter domain,
and the objective enclosure over that domain, respectively. `best_value` is
therefore not expected to equal `minimum`: the lower global bound can come
from a different queued, unresolved, or certified box. `unresolved` contains
live domains when the result stopped before the queue was exhausted.

I have added some basic support for complex-valued operations (not defined in the IEEE standard). Note that complex functions transform interval spaces in nontrivial ways, so while the resulting image from this library will contain the true image, it is not guaranteed to be equal to it, though it may be equal under certain operations.

Linear algebra has been implemented mostly by following the work of Jaroslav Horáček's dissertation:

```bibtex
@phdthesis{Horacek2019Interval,
  author = {Jaroslav Hor{\'a}{\v c}ek},
  title  = {Interval Linear and Nonlinear Systems},
  school = {Charles University, Faculty of Mathematics and Physics},
  year   = {2019},
  address = {Prague, Czech Republic},
  url    = {https://dspace.cuni.cz/handle/20.500.11956/111301}
}
```

# Future plans

- Support for alternate complex interval space formulations, like disks and polyarcs
- Linear algebra methods (a lot of this is done, but there are still some thing I want to add)
- Potentially a PyO3 binding if I find it useful

# Licensing

maryada is available under either the MIT License or the Apache License 2.0.
