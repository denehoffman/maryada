# IEEE Std 1788.1-2017 conformance

maryada conforms to IEEE Std 1788.1-2017 for its binary64 real interval API. This document describes the scope of that conformance, the declared accuracy modes, and the implementation of the standard's required features. The crate's rectangular complex interval extension is outside the scope of the standard.

## Declared operation accuracy

The ranges below comprise all binary64 interval inputs, including empty and unbounded inputs. A decorated operation has the same accuracy as its interval part.

| Operations | Declared mode |
| --- | --- |
| `neg`, `add`, `sub`, `mul`, `div`, `recip`, `sqr`, `sqrt`, `fma` | tightest |
| `sign`, `ceil`, `floor`, `trunc`, `roundTiesToEven`, `roundTiesToAway` | tightest |
| `abs`, `min`, `max` | tightest |
| `cancelMinus`, `cancelPlus` | result prescribed by 6.7.3 |
| `intersection`, `convexHull` | tightest |
| `pown`, `pow`, `exp`, `exp2`, `exp10`, `log`, `log2`, `log10` | valid |
| `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2` | valid |
| `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh` | valid |

"Tightest", "valid", and the accuracy of a decorated operation have the meanings defined in 6.5. The valid declarations are deliberately weaker than the standard's recommended accurate mode. Accurate mode is a recommendation ("should") in 6.5.2, whereas valid enclosure is the conformance requirement for these operations.

For finite bounded inputs, `mid` uses Rust's binary64 midpoint operation. It avoids overflow in the sum and returns the binary64 midpoint rounded to nearest. The other cases of `mid`, and `inf`, `sup`, `wid`, `rad`, `mag`, and `mig`, follow 6.7.6 directly.

`intervalToText` recognizes an absent conversion specifier, the empty string, and `hex`. They all select the same exact hexadecimal interval-literal layout. An unrecognized specifier falls back to that layout, which still exactly encloses the input.

## Mandatory features

| Requirement | Implementation | Conformance coverage |
| --- | --- | --- |
| 5.2 decorations | `Decoration` provides `ill`, `trv`, `def`, `dac`, and `com` in propagation order. | `tests/decorations.rs` covers ordering, permitted combinations, and propagation. |
| 6.7.1 constants | Bare and decorated Empty and Entire constants use the prescribed initial decorations. | `tests/interval_api.rs` covers the required constants and initial decorations. |
| 6.7.2 elementary functions | Every operation in Table 4.1 has bare and decorated entry points. | The ITF1788 runner covers every operation family. Tightest kernels use exact directed algorithms; valid kernels use directed compositions with explicit remainder or source-error bounds. |
| 6.7.3 cancellative operations | `cancelMinus` and `cancelPlus`, including trivial decorated versions. | The portable ITF1788 cancellation cases cover both operations. |
| 6.7.4 set operations | `intersection` and `convexHull`, including trivial decorated versions. | The portable ITF1788 set cases cover both operations. |
| 6.7.5 constructors | Bare and decorated numeric and text constructors, with required failure values and signals. | `tests/text_interchange.rs` and `tests/signals.rs` cover the grammar and exception behavior. |
| 6.7.6 numeric functions | `inf`, `sup`, `mid`, `rad`, `wid`, `mag`, and `mig`; `midRad` is also provided. | The portable ITF1788 numeric cases cover the required functions. |
| 6.7.7 boolean functions | `isEmpty`, `isEntire`, `equal`, `subset`, `interior`, `disjoint`, and decorated `isNaI`. | The portable ITF1788 boolean cases cover the required functions. |
| 6.7.8 decoration operations | Decoration comparisons, `newDec`, `intervalPart`, `decorationPart`, and `setDec`. | `tests/decorations.rs` and `tests/signals.rs` cover initialization, permitted combinations, the min rule, local classes, NaI propagation, and signals. |
| 6.8.2 text input | Required number, bracket, uncertain, decorated, Empty, Entire, and NaI forms. | `tests/text_interchange.rs` covers accepted and rejected forms, accuracy-relaxed inputs, overflow, and signals. |
| 6.8.3 text output | Exact hexadecimal enclosure layout for bare and decorated intervals; invalid specifiers still produce valid enclosing literals. | `tests/text_interchange.rs` covers round trips and the recognized specifiers documented above. |
| 7.3 interchange | Standard inf-sup and inf-sup-decoration encodings in both byte orders. | `tests/text_interchange.rs` covers canonical special objects, every decoration octet, round trips, and malformed-object signals. |

## Implementation details

### Numerical enclosure mechanisms

The basic directed kernels do not infer an error direction from a rounded hardware result. Finite multiplication is converted to an exact dyadic product of at most 106 significand bits; division retains 74 quotient bits and a remainder flag; square root uses an integer square root and exact remainder; and fused multiply-add combines the exact product with the addend in a 127-bit fixed-point window. Their common packer rounds the retained exact dyadic value toward the requested direction, including subnormal and overflow cases. Addition uses an error-free two-sum decomposition. These mechanisms are the basis for the `tightest` declarations for the basic operations.

The pinned `libm` sources give explicit finite-result error bounds below one ulp for `exp` and `log`. maryada widens these results by two representable steps and handles exact values, infinities, underflow, and overflow separately. Nonintegral `exp2` uses directed multiplication by an enclosure of `ln(2)` followed by `exp`. `log2` and `log10` use directed division of a `log` enclosure by adjacent binary64 enclosures of `ln(2)` and `ln(10)`. `sinh` uses a positive-term directed series with a geometric remainder bound near zero and its exponential identity elsewhere; `cosh` uses its exponential identity throughout. `tanh`, `asinh`, `acosh`, and `atanh` use stable directed identities built from those kernels. `atan2` uses directed division, `atan`, and outward enclosures of the required multiples of pi; `atan` itself uses range reduction to a directed alternating series with an explicit next-term remainder bound. `asin` and `acos` reduce to directed `atan` and square root identities. `exp10` and `pow` are likewise interval compositions of directed kernels rather than calls to unchecked point implementations.

For `sin`, `cos`, and `tan`, endpoint evaluation uses an integer multiple of an outward-enclosed `pi/2`, verifies that the resulting remainder is contained in `[-0.8, 0.8]`, and evaluates directed Taylor series with alternating-remainder bounds. If binary64 range reduction is too imprecise, `sin` and `cos` return their full mathematical range and `tan` returns Entire. Thus every finite input has a valid enclosure; the fallback deliberately makes no stronger accuracy claim.

The conformance runner includes 4,642 portable ITF1788 cases. It requires exact interval bounds and exact decorations for the operation groups whose accuracy is required to be tightest. For the remaining elementary functions it checks containment of the tight reference result. These cases exercise sampled enclosures, while the directed construction of the kernels provides the guarantee over all binary64 inputs.

The `libm` dependency is pinned to version 0.2.16 so a later compatible release cannot silently change the elementary kernels used by the implementation.

Additional integration tests cover the required decimal, hexadecimal, rational, bracket, uncertain, decorated, empty, entire, and NaI literal forms; malformed and accuracy-relaxed inputs; text-output round trips; canonical and malformed big- and little-endian interchange encodings; and the required exception conditions.

## Conformance statement

maryada provides all four conformance requirements in 1.5: the five decorations; all required operations in 6.7 with the mandatory accuracy modes; text input and output under 6.8.2 and 6.8.3; and the interchange representation under 7.3. The operation table supplies the tightness documentation required by 6.5.3 over the complete input range.

The elementary functions declared valid may also be raised to the standard's recommended accurate mode in the future. Accurate mode is not required for conformance, and the current valid implementations satisfy the applicable requirements.
