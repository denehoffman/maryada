# maryada
![Crates.io Version](https://img.shields.io/crates/v/maryada?style=for-the-badge&logo=rust&logoColor=%23D34516&labelColor=%231E2650&color=%23D34516&link=https%3A%2F%2Fcrates.io%2Fcrates%2Fmaryada) ![docs.rs](https://img.shields.io/docsrs/maryada?style=for-the-badge&logo=docs.rs&logoColor=%23D34516&labelColor=%231E2650&link=https%3A%2F%2Fdocs.rs%2Fmaryada%2Flatest%2Fmaryada%2F) ![Codecov](https://img.shields.io/codecov/c/github/denehoffman/maryada?style=for-the-badge&logo=codecov&link=https%3A%2F%2Fapp.codecov.io%2Fgh%2Fdenehoffman%2Fmaryada) ![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/denehoffman/maryada/build-clippy-test.yml?style=for-the-badge&logo=github&link=https%3A%2F%2Fgithub.com%2Fdenehoffman%2Fmaryada%2Factions%2Fworkflows%2Fbuild-clippy-test.yml) 





maryada is a `no_std` binary64 interval arithmetic library compliant with the IEEE 1788.1 standard. It provides bare and decorated real intervals, outward-rounded elementary operations, text and binary interchange, and rectangular complex intervals.

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
