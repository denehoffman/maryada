# ITF1788 conformance vectors

These fixtures were copied from `IntervalArithmetic.jl` revision
`17865d8f58634dfc604c60c0009e64181dc20e5a`. The Rust harness executes the
operations that Maryada currently exposes and ignores ITL operations without a
Maryada equivalent (for example `rootn`, overlap classification, and vector
reductions).

ITL interval endpoints denote binary64 operands, so the harness decodes those
tokens and passes them to Maryada's `nums_to_interval` and `set_dec` APIs. It
does not use `FromStr` for fixture operands: Maryada correctly interprets
decimal interval text as exact real input requiring outward enclosure, which is
different from ITL's already-rounded endpoint semantics. Hexadecimal tokens do
delegate to Maryada's interval parser.

The upstream vectors specify tight results. Maryada currently promises valid
outward enclosures, so interval assertions require the upstream expected result
to be contained in Maryada's result. Decorations may be conservatively weaker
than the upstream expectation. Empty intervals, NaI, Boolean results, and
numeric results remain exact checks.

The `libieeep1788_*` fixtures retain their Apache-2.0 headers and license. The
`atan2.itl` file retains its original permissive notice. LGPL-derived C-XSC and
MPFI fixtures from the upstream suite are intentionally not included.
