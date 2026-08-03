# maryada
[![Crates.io Version](https://img.shields.io/crates/v/maryada?style=for-the-badge&logo=rust&logoColor=%23D34516&labelColor=%231E2650&color=%23D34516)](https://crates.io/crates/maryada) [![docs.rs](https://img.shields.io/docsrs/maryada?style=for-the-badge&logo=docs.rs&logoColor=%23D34516&labelColor=%231E2650)](https://docs.rs/maryada/latest/maryada/) [![Codecov](https://img.shields.io/codecov/c/github/denehoffman/maryada?style=for-the-badge&logo=codecov)](https://app.codecov.io/gh/denehoffman/maryada) [![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/denehoffman/maryada/build-clippy-test.yml?style=for-the-badge&logo=github)](https://github.com/denehoffman/maryada/actions/workflows/build-clippy-test.yml)
maryada is a `no_std` binary64 interval arithmetic library conforming to IEEE Std 1788.1-2017. It provides bare and decorated real intervals, outward-rounded elementary operations, text and binary interchange, and rectangular complex intervals.

The conformance claim applies to the real interval API. The rectangular complex interval extension is outside the scope of the standard. See [CONFORMANCE.md](CONFORMANCE.md) for the operation accuracy declarations, required features, implementation details, and test coverage.

```rust
use maryada::Interval;

let x = Interval::new(1.0, 2.0);
let y = x.sqr();

assert_eq!(y.bounds(), (1.0, 4.0));
```

Additionally, I have added some basic support for complex-valued operations (not defined in the IEEE standard). Note that complex functions transform interval spaces in nontrivial ways, so while the resulting image from this library will contain the true image, it is not guaranteed to be equal to it, though it may be equal under certain operations.

# Future plans

- Support for alternate complex interval space formulations, like disks and polyarcs
- Linear algebra methods
- Potentially a PyO3 binding if I find it useful

# Licensing

maryada is available under either the MIT License or the Apache License 2.0.
