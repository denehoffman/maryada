# Third-party notices

## IntervalArithmetic.jl test suite

`tests/itf1788/` contains test vectors ported from the test suite of
[IntervalArithmetic.jl](https://github.com/JuliaIntervals/IntervalArithmetic.jl),
revision `17865d8f58634dfc604c60c0009e64181dc20e5a`.

IntervalArithmetic.jl is distributed under the MIT License:

> Copyright (c) 2014 David P. Sanders and Luis Benet
> Copyright (c) 2023 Luis Benet, Luca Ferranti, Olivier Hénot and Benoît Richard

The portable ITF1788 vectors copied into this repository carry their own
copyright and Apache License 2.0 notices. The complete license is retained at
`tests/itf1788/LICENSE.md`, and the per-file notices are retained verbatim.

maryada's Rust harness in `tests/itf1788.rs` is an adaptation written for this
project; the underlying test cases remain under their stated upstream terms.
