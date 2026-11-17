# maryada
[![Crates.io Version](https://img.shields.io/crates/v/maryada?style=for-the-badge&logo=rust&logoColor=%23D34516&labelColor=%231E2650&color=%23D34516)](https://crates.io/crates/maryada) [![docs.rs](https://img.shields.io/docsrs/maryada?style=for-the-badge&logo=docs.rs&logoColor=%23D34516&labelColor=%231E2650)](https://docs.rs/maryada/latest/maryada/) [![Codecov](https://img.shields.io/codecov/c/github/denehoffman/maryada?style=for-the-badge&logo=codecov)](https://app.codecov.io/gh/denehoffman/maryada) [![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/denehoffman/maryada/build-clippy-test.yml?style=for-the-badge&logo=github)](https://github.com/denehoffman/maryada/actions/workflows/build-clippy-test.yml)

maryada is a `no_std` binary64 interval arithmetic library conforming to IEEE Std 1788.1-2017. It provides bare and decorated real intervals, outward-rounded elementary operations, and text and binary interchange. Rectangular complex intervals are available through an optional feature.

The conformance claim applies to the real interval API. The rectangular complex interval extension is outside the scope of the standard. See [CONFORMANCE.md](CONFORMANCE.md) for the operation accuracy declarations, required features, implementation details, and test coverage.

```rust
use maryada::{EnclosureOps, Interval, IntervalOps};

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
- `serde` enables serialization of `Interval` and `DecoratedInterval` without requiring `std` or an allocator in maryada.

Serde writes intervals as exact IEEE interval literals in human-readable formats.
For example, JSON encodes `[1,2]` as `"[0x1p+0,0x1p+1]"`. Deserialization accepts
literal strings that parse without IEEE exception flags. In binary formats,
Serde writes a byte string with the same big-endian payload as `to_be_bytes`.
The payload contains 16 bytes for an `Interval` and 17 for a `DecoratedInterval`.
The outer byte-string framing is provided by the format.

Complex systems use the same matrix-facing API as real systems. The default
verified solve first applies a residual-centered complex epsilon-inflation
method and transparently retries through a conservative 2n real embedding when
the direct sufficient test does not certify an enclosure. Use `try_solve` when
you need a typed `SolveError`, or `solve` for the concise `Option` form.

The `branch` feature provides a resumable generic `BranchAndBound` scheduler,
pluggable evaluators, pruners, branching rules, and work queues. Its
`GlobalMinimizer` convenience layer implements best-first interval subdivision;
domains can be real intervals, complex boxes, vectors, or custom types that
implement `Bisect` and optionally `Midpoint`.

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

## Benchmarks

```sh
cargo bench --locked --bench performance --features branch,linalg
```

Criterion measures five scalar input classes, 4×4 and 32×32 matrix products
and verified inverses, repeated solves with and without reused setup, and an
eight-dimensional search with gap checks enabled and disabled. The prepared
solve timings exclude setup. Time is reported per batch of inputs. Throughput
counts one operation per scalar call, matrix product, inverse, solve, or
512-step search.
The defaults are a 250 ms warmup, 30 samples, and a one-second measurement
target per workload. Criterion's command-line flags can override them.
Add `-- --test` to exercise the workloads once for a smoke check.

Save a named baseline before changing the code, then compare the same workloads
afterward. The commands below select the search workloads.

```sh
cargo bench --locked --bench performance --features branch,linalg -- 'search/' --save-baseline before
cargo bench --locked --bench performance --features branch,linalg -- 'search/' --baseline before
```

Use the same machine and Rust toolchain for before-and-after measurements. Run
`cargo test --locked --all-features` alongside benchmark comparisons to check
exactness and containment.

# Future plans

- Support for alternate complex interval space formulations, like disks and polyarcs
- Linear algebra methods (a lot of this is done, but there are still some thing I want to add)
- Potentially a PyO3 binding if I find it useful

# Licensing

maryada is available under either the MIT License or the Apache License 2.0.
