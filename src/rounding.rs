use core::cmp::Ordering;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Direction {
    Down,
    Up,
}

impl Direction {
    #[inline]
    fn exact_zero(&self) -> f64 {
        match self {
            Direction::Down => -0.0,
            Direction::Up => 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum FloatClass {
    Zero { negative: bool },
    Finite(Dyadic),
    Infinity { negative: bool },
    NaN,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Dyadic {
    pub negative: bool,
    pub significand: u64,
    pub exponent: i32,
}

impl Dyadic {
    #[inline]
    fn normalize(&self) -> Dyadic {
        let mut value = Dyadic {
            negative: self.negative,
            significand: self.significand,
            exponent: self.exponent,
        };
        let bits = 64 - value.significand.leading_zeros();
        let shift = 53 - bits;
        value.significand <<= shift;
        value.exponent -= shift as i32;
        value
    }
}

#[derive(Clone, Copy, Debug)]
enum WideFloatClass {
    Zero { negative: bool },
    Finite(WideDyadic),
    Infinity { negative: bool },
    NaN,
}

#[derive(Clone, Copy, Debug)]
struct WideDyadic {
    negative: bool,
    significand: u128,
    exponent: i32,
}

impl From<Dyadic> for WideDyadic {
    fn from(value: Dyadic) -> Self {
        Self {
            negative: value.negative,
            significand: u128::from(value.significand),
            exponent: value.exponent,
        }
    }
}

impl WideDyadic {
    /// Central binary64 rounding and packing operation.
    ///
    /// `sticky` means additional discarded nonzero bits exist below the
    /// supplied significand.
    fn pack(&self, sticky: bool, direction: Direction) -> f64 {
        #[inline]
        fn overflow(negative: bool, direction: Direction) -> f64 {
            let sign = (negative as u64) << 63;

            let toward_infinity = match direction {
                Direction::Down => negative,
                Direction::Up => !negative,
            };

            let magnitude = if toward_infinity {
                0x7ff0_0000_0000_0000
            } else {
                0x7fef_ffff_ffff_ffff
            };

            f64::from_bits(sign | magnitude)
        }

        #[inline]
        fn round_right(
            significand: u128,
            shift: u32,
            sticky: bool,
            negative: bool,
            direction: Direction,
        ) -> u128 {
            let truncated = significand.checked_shr(shift).unwrap_or(0);
            let discarded = match shift {
                0 => false,
                1..=127 => {
                    let mask = (1u128 << shift) - 1;
                    significand & mask != 0
                }
                _ => significand != 0,
            };
            let inexact = sticky || discarded;
            let increment_magnitude = inexact
                && match direction {
                    Direction::Down => negative,
                    Direction::Up => !negative,
                };
            truncated + u128::from(increment_magnitude)
        }
        let sign = (self.negative as u64) << 63;
        assert!(
            self.significand != 0 || !sticky,
            "zero significand cannot have an unspecified nonzero tail",
        );
        if self.significand == 0 {
            return f64::from_bits(sign);
        }
        let bit_length = 128i64 - i64::from(self.significand.leading_zeros());
        let mut top_exponent = i64::from(self.exponent) + bit_length - 1;
        if top_exponent > 1023 {
            return overflow(self.negative, direction);
        }
        let subnormal = top_exponent < -1022;
        let quantum_exponent = if subnormal { -1074 } else { top_exponent - 52 };
        let shift = quantum_exponent - i64::from(self.exponent);
        assert!(
            !sticky || shift >= 0,
            "sticky information is insufficient when left-shifting",
        );
        let mut rounded = if shift >= 0 {
            round_right(
                self.significand,
                u32::try_from(shift).expect("right shift must fit in u32"),
                sticky,
                self.negative,
                direction,
            )
        } else {
            self.significand << u32::try_from(-shift).expect("left shift must fit in u32")
        };
        if subnormal {
            debug_assert!(rounded <= 1u128 << 52);
            return f64::from_bits(sign | rounded as u64);
        }
        if rounded == 1u128 << 53 {
            rounded >>= 1;
            top_exponent += 1;
            if top_exponent > 1023 {
                return overflow(self.negative, direction);
            }
        }
        debug_assert!((1u128 << 52) <= rounded && rounded < (1u128 << 53),);
        let biased_exponent =
            u64::try_from(top_exponent + 1023).expect("validated binary64 exponent");
        f64::from_bits(sign | (biased_exponent << 52) | ((rounded as u64) & ((1u64 << 52) - 1)))
    }

    fn round_sum(x: WideDyadic, y: WideDyadic, direction: Direction) -> f64 {
        if x.negative == y.negative {
            Self::round_same_sign_sum(x, y, direction)
        } else {
            match Self::magnitude_cmp(x, y) {
                Ordering::Equal => direction.exact_zero(),

                Ordering::Greater => Self::round_difference(x, y, direction),

                Ordering::Less => Self::round_difference(y, x, direction),
            }
        }
    }

    fn round_same_sign_sum(x: WideDyadic, y: WideDyadic, direction: Direction) -> f64 {
        // Retain a 127-bit fixed-point window. The operand with the
        // greatest top exponent is represented exactly; the other may
        // contribute a sticky tail.
        let top = x.top_exponent().max(y.top_exponent());
        let quantum = top - 126;
        let (x_significand, x_sticky) = Self::project(x, quantum);
        let (y_significand, y_sticky) = Self::project(y, quantum);
        debug_assert!(!(x_sticky && y_sticky));
        WideDyadic {
            negative: x.negative,
            significand: x_significand + y_significand,
            exponent: i32::try_from(quantum).expect("FMA exponent fits in i32"),
        }
        .pack(x_sticky || y_sticky, direction)
    }

    fn round_difference(larger: WideDyadic, smaller: WideDyadic, direction: Direction) -> f64 {
        debug_assert_eq!(Self::magnitude_cmp(larger, smaller), Ordering::Greater,);
        let quantum = larger.top_exponent() - 126;
        let (larger_significand, larger_sticky) = Self::project(larger, quantum);
        let (smaller_significand, smaller_sticky) = Self::project(smaller, quantum);
        // The larger operand determines the window and is exact.
        debug_assert!(!larger_sticky);
        let difference = larger_significand - smaller_significand;
        if smaller_sticky {
            // If the discarded part is ε, with 0 < ε < 1:
            //
            //     A - (B + ε)
            //       = (A - B - 1) + (1 - ε)
            //
            // This converts a negative discarded correction into the
            // positive-tail convention expected by pack().
            assert!(difference > 1);
            WideDyadic {
                negative: larger.negative,
                significand: difference - 1,
                exponent: i32::try_from(quantum).expect("FMA exponent fits in i32"),
            }
            .pack(true, direction)
        } else {
            debug_assert_ne!(difference, 0);
            WideDyadic {
                negative: larger.negative,
                significand: difference,
                exponent: i32::try_from(quantum).expect("FMA exponent fits in i32"),
            }
            .pack(false, direction)
        }
    }

    fn project(value: WideDyadic, quantum: i64) -> (u128, bool) {
        let shift = i64::from(value.exponent) - quantum;
        if shift >= 0 {
            let shift = u32::try_from(shift).expect("projection shift fits in u32");
            debug_assert!(shift < 128);
            (value.significand << shift, false)
        } else {
            let shift = u32::try_from(-shift).expect("projection shift fits in u32");
            let truncated = value.significand.checked_shr(shift).unwrap_or(0);
            let discarded = match shift {
                0 => false,
                1..=127 => {
                    let mask = (1_u128 << shift) - 1;
                    value.significand & mask != 0
                }
                _ => value.significand != 0,
            };
            (truncated, discarded)
        }
    }

    fn magnitude_cmp(x: WideDyadic, y: WideDyadic) -> Ordering {
        match x.top_exponent().cmp(&y.top_exponent()) {
            Ordering::Equal => {}
            ordering => return ordering,
        }
        let common_exponent = x.exponent.min(y.exponent);
        let x_shift = u32::try_from(x.exponent - common_exponent).expect("nonnegative shift");
        let y_shift = u32::try_from(y.exponent - common_exponent).expect("nonnegative shift");
        debug_assert!(x_shift < 128);
        debug_assert!(y_shift < 128);
        (x.significand << x_shift).cmp(&(y.significand << y_shift))
    }

    fn top_exponent(&self) -> i64 {
        debug_assert_ne!(self.significand, 0);
        let bit_length = 128 - self.significand.leading_zeros();
        i64::from(self.exponent) + i64::from(bit_length) - 1
    }
}

impl FloatClass {
    /// Exact binary64 decomposition.
    ///
    /// finite value = (-1)^negative * significand * 2^exponent
    pub(crate) fn new(value: f64) -> FloatClass {
        let bits = value.to_bits();
        let negative = bits >> 63 != 0;
        let raw_exp = ((bits >> 52) & 0x7ff) as i32;
        let fraction = bits & ((1u64 << 52) - 1);
        match (raw_exp, fraction) {
            (0, 0) => FloatClass::Zero { negative },
            (0, _) => FloatClass::Finite(Dyadic {
                negative,
                significand: fraction,
                exponent: -1074,
            }),
            (0x7ff, 0) => FloatClass::Infinity { negative },
            (0x7ff, _) => FloatClass::NaN,
            _ => FloatClass::Finite(Dyadic {
                negative,
                significand: fraction | (1u64 << 52),
                exponent: raw_exp - 1075,
            }),
        }
    }
}

impl WideFloatClass {
    fn exact_product(x: f64, y: f64) -> Self {
        match (FloatClass::new(x), FloatClass::new(y)) {
            (FloatClass::NaN, _) | (_, FloatClass::NaN) => Self::NaN,
            (FloatClass::Zero { .. }, FloatClass::Infinity { .. })
            | (FloatClass::Infinity { .. }, FloatClass::Zero { .. }) => Self::NaN,
            (FloatClass::Infinity { negative: x }, FloatClass::Infinity { negative: y })
            | (
                FloatClass::Infinity { negative: x },
                FloatClass::Finite(Dyadic { negative: y, .. }),
            )
            | (
                FloatClass::Finite(Dyadic { negative: x, .. }),
                FloatClass::Infinity { negative: y },
            ) => Self::Infinity { negative: x ^ y },
            (FloatClass::Zero { negative: x }, FloatClass::Zero { negative: y })
            | (FloatClass::Zero { negative: x }, FloatClass::Finite(Dyadic { negative: y, .. }))
            | (FloatClass::Finite(Dyadic { negative: x, .. }), FloatClass::Zero { negative: y }) => {
                Self::Zero { negative: x ^ y }
            }
            (FloatClass::Finite(x), FloatClass::Finite(y)) => Self::Finite(WideDyadic {
                negative: x.negative ^ y.negative,

                // Each input has at most 53 significand bits,
                // so the exact product has at most 106 bits.
                significand: u128::from(x.significand) * u128::from(y.significand),

                exponent: x.exponent + y.exponent,
            }),
        }
    }
}

#[inline]
fn finish(rounded: f64, error: f64, direction: Direction) -> f64 {
    if error == 0.0 || !rounded.is_finite() {
        return rounded;
    }
    match direction {
        Direction::Down if error < 0.0 => rounded.next_down(),
        Direction::Up if error > 0.0 => rounded.next_up(),
        _ => rounded,
    }
}

#[inline]
fn signed_zero(negative: bool) -> f64 {
    f64::from_bits((negative as u64) << 63)
}

#[inline]
fn infinity(negative: bool) -> f64 {
    f64::from_bits(((negative as u64) << 63) | 0x7ff0_0000_0000_0000)
}

#[inline]
fn overflow(negative: bool, direction: Direction) -> f64 {
    let outward = match direction {
        Direction::Down => negative,
        Direction::Up => !negative,
    };

    if outward {
        infinity(negative)
    } else {
        f64::MAX.copysign(if negative { -1.0 } else { 1.0 })
    }
}

fn add_zero(x_negative: bool, y_negative: bool, direction: Direction) -> f64 {
    let negative = if x_negative == y_negative {
        x_negative
    } else {
        matches!(direction, Direction::Down)
    };
    signed_zero(negative)
}

pub(crate) fn neg(x: f64) -> f64 {
    -x
}

pub(crate) fn add(x: f64, y: f64, direction: Direction) -> f64 {
    let sum = x + y;
    if sum.is_nan() {
        return sum;
    }
    if sum.is_infinite() && x.is_finite() && y.is_finite() {
        return overflow(sum.is_sign_negative(), direction);
    }
    if !sum.is_finite() {
        return sum;
    }
    let y_v = sum - x;
    let error = (x - (sum - y_v)) + (y - y_v);
    if sum == 0.0 && error == 0.0 {
        return direction.exact_zero();
    }
    finish(sum, error, direction)
}

pub(crate) fn sub(x: f64, y: f64, direction: Direction) -> f64 {
    add(x, neg(y), direction)
}

pub(crate) fn mul(x: f64, y: f64, direction: Direction) -> f64 {
    match (FloatClass::new(x), FloatClass::new(y)) {
        (FloatClass::NaN, _) | (_, FloatClass::NaN) => f64::NAN,
        (FloatClass::Zero { .. }, FloatClass::Infinity { .. })
        | (FloatClass::Infinity { .. }, FloatClass::Zero { .. }) => f64::NAN,
        (FloatClass::Infinity { negative: a }, FloatClass::Infinity { negative: b })
        | (FloatClass::Infinity { negative: a }, FloatClass::Finite(Dyadic { negative: b, .. }))
        | (FloatClass::Finite(Dyadic { negative: a, .. }), FloatClass::Infinity { negative: b }) => {
            infinity(a ^ b)
        }
        (FloatClass::Zero { negative: a }, FloatClass::Zero { negative: b })
        | (FloatClass::Zero { negative: a }, FloatClass::Finite(Dyadic { negative: b, .. }))
        | (FloatClass::Finite(Dyadic { negative: a, .. }), FloatClass::Zero { negative: b }) => {
            signed_zero(a ^ b)
        }
        (FloatClass::Finite(a), FloatClass::Finite(b)) => WideDyadic {
            negative: a.negative ^ b.negative,
            significand: u128::from(a.significand) * u128::from(b.significand),
            exponent: a.exponent + b.exponent,
        }
        .pack(false, direction),
    }
}

fn div_finite(numerator: Dyadic, denominator: Dyadic, direction: Direction) -> f64 {
    let numerator = numerator.normalize();
    let denominator = denominator.normalize();
    let scaled = u128::from(numerator.significand) << 74;
    let divisor = u128::from(denominator.significand);
    let quotient = scaled / divisor;
    let remainder = scaled % divisor;
    WideDyadic {
        negative: numerator.negative ^ denominator.negative,
        significand: quotient,
        exponent: numerator.exponent - denominator.exponent - 74,
    }
    .pack(remainder != 0, direction)
}

pub(crate) fn div(x: f64, y: f64, direction: Direction) -> f64 {
    match (FloatClass::new(x), FloatClass::new(y)) {
        (FloatClass::NaN, _)
        | (_, FloatClass::NaN)
        | (FloatClass::Zero { .. }, FloatClass::Zero { .. })
        | (FloatClass::Infinity { .. }, FloatClass::Infinity { .. }) => f64::NAN,
        (FloatClass::Infinity { negative: a }, FloatClass::Zero { negative: b })
        | (FloatClass::Infinity { negative: a }, FloatClass::Finite(Dyadic { negative: b, .. }))
        | (FloatClass::Finite(Dyadic { negative: a, .. }), FloatClass::Zero { negative: b }) => {
            infinity(a ^ b)
        }
        (FloatClass::Zero { negative: a }, FloatClass::Infinity { negative: b })
        | (FloatClass::Zero { negative: a }, FloatClass::Finite(Dyadic { negative: b, .. }))
        | (FloatClass::Finite(Dyadic { negative: a, .. }), FloatClass::Infinity { negative: b }) => {
            signed_zero(a ^ b)
        }
        (FloatClass::Finite(a), FloatClass::Finite(b)) => div_finite(a, b, direction),
    }
}

pub(crate) fn recip(x: f64, direction: Direction) -> f64 {
    div(1.0, x, direction)
}

pub(crate) fn sqr(x: f64, direction: Direction) -> f64 {
    mul(x, x, direction)
}

fn sqrt_finite(value: Dyadic, direction: Direction) -> f64 {
    let value = value.normalize();
    let mut significand = u128::from(value.significand);
    let mut exponent = value.exponent;
    if exponent & 1 != 0 {
        significand <<= 1;
        exponent -= 1;
    }
    let radicand = significand << 72;
    let root = radicand.isqrt();
    let remainder = radicand - root * root;
    WideDyadic {
        negative: false,
        significand: root,
        exponent: exponent / 2 - (72 / 2),
    }
    .pack(remainder != 0, direction)
}

pub(crate) fn sqrt(x: f64, direction: Direction) -> f64 {
    match FloatClass::new(x) {
        FloatClass::NaN => f64::NAN,
        FloatClass::Infinity { negative: true } => f64::NAN,
        FloatClass::Infinity { negative: false } => f64::INFINITY,
        FloatClass::Zero { negative } => signed_zero(negative),
        FloatClass::Finite(Dyadic { negative: true, .. }) => f64::NAN,
        FloatClass::Finite(value) => sqrt_finite(value, direction),
    }
}

pub(crate) fn fma(x: f64, y: f64, z: f64, direction: Direction) -> f64 {
    let product = WideFloatClass::exact_product(x, y);
    match (product, FloatClass::new(z)) {
        (WideFloatClass::NaN, _) | (_, FloatClass::NaN) => f64::NAN,
        (
            WideFloatClass::Infinity {
                negative: product_negative,
            },
            FloatClass::Infinity {
                negative: z_negative,
            },
        ) => {
            if product_negative == z_negative {
                infinity(product_negative)
            } else {
                f64::NAN
            }
        }
        (WideFloatClass::Infinity { negative }, _) => infinity(negative),
        (_, FloatClass::Infinity { negative }) => infinity(negative),
        (
            WideFloatClass::Zero {
                negative: product_negative,
            },
            FloatClass::Zero {
                negative: z_negative,
            },
        ) => add_zero(product_negative, z_negative, direction),
        (WideFloatClass::Zero { .. }, FloatClass::Finite(_)) => z,
        (WideFloatClass::Finite(product), FloatClass::Zero { .. }) => {
            product.pack(false, direction)
        }
        (WideFloatClass::Finite(product), FloatClass::Finite(z)) => {
            WideDyadic::round_sum(product, WideDyadic::from(z), direction)
        }
    }
}

// Power functions.

pub(crate) fn pown(x: f64, p: i32, direction: Direction) -> f64 {
    todo!()
}

pub(crate) fn pow(x: f64, y: f64, direction: Direction) -> f64 {
    todo!()
}

pub(crate) fn exp(x: f64, direction: Direction) -> f64 {
    todo!()
}

pub(crate) fn exp2(x: f64, direction: Direction) -> f64 {
    todo!()
}

pub(crate) fn exp10(x: f64, direction: Direction) -> f64 {
    todo!()
}

pub(crate) fn log(x: f64, direction: Direction) -> f64 {
    todo!()
}

pub(crate) fn log2(x: f64, direction: Direction) -> f64 {
    todo!()
}

pub(crate) fn log10(x: f64, direction: Direction) -> f64 {
    todo!()
}

// Trigonometric functions.

pub(crate) fn sin(x: f64, direction: Direction) -> f64 {
    todo!()
}

pub(crate) fn cos(x: f64, direction: Direction) -> f64 {
    todo!()
}

pub(crate) fn tan(x: f64, direction: Direction) -> f64 {
    todo!()
}

pub(crate) fn asin(x: f64, direction: Direction) -> f64 {
    todo!()
}

pub(crate) fn acos(x: f64, direction: Direction) -> f64 {
    todo!()
}

pub(crate) fn atan(x: f64, direction: Direction) -> f64 {
    todo!()
}

pub(crate) fn atan2(y: f64, x: f64, direction: Direction) -> f64 {
    todo!()
}

// Hyperbolic functions.

pub(crate) fn sinh(x: f64, direction: Direction) -> f64 {
    todo!()
}

pub(crate) fn cosh(x: f64, direction: Direction) -> f64 {
    todo!()
}

pub(crate) fn tanh(x: f64, direction: Direction) -> f64 {
    todo!()
}

pub(crate) fn asinh(x: f64, direction: Direction) -> f64 {
    todo!()
}

pub(crate) fn acosh(x: f64, direction: Direction) -> f64 {
    todo!()
}

pub(crate) fn atanh(x: f64, direction: Direction) -> f64 {
    todo!()
}

// Required tightest integer-valued functions.

pub(crate) fn ceil(x: f64) -> f64 {
    todo!()
}

pub(crate) fn floor(x: f64) -> f64 {
    todo!()
}

pub(crate) fn trunc(x: f64) -> f64 {
    todo!()
}

pub(crate) fn round_ties_to_even(x: f64) -> f64 {
    todo!()
}

pub(crate) fn round_ties_to_away(x: f64) -> f64 {
    todo!()
}

// Numeric interval queries.

pub(crate) fn midpoint(inf: f64, sup: f64) -> f64 {
    todo!()
}

pub(crate) fn radius(inf: f64, sup: f64, midpoint: f64) -> f64 {
    todo!()
}

// Exact parsing of all required number literal forms.

pub(crate) fn number_literal(literal: &str, direction: Direction) -> Option<f64> {
    // Must support:
    //
    // - decimal literals;
    // - C99 hexadecimal floating constants;
    // - rational literals p/q;
    // - inf and infinity.
    //
    // The result must be correctly directed even for arbitrarily long
    // literals. This can be implemented without allocation by repeatedly
    // scanning the input slice and retaining guard/round/sticky data.
    todo!()
}

// Certified argument-reduction predicates used by interval kernels.

pub(crate) fn contains_sin_maximum(inf: f64, sup: f64) -> bool {
    todo!()
}

pub(crate) fn contains_sin_minimum(inf: f64, sup: f64) -> bool {
    todo!()
}

pub(crate) fn contains_cos_maximum(inf: f64, sup: f64) -> bool {
    todo!()
}

pub(crate) fn contains_cos_minimum(inf: f64, sup: f64) -> bool {
    todo!()
}

pub(crate) fn contains_tan_pole(inf: f64, sup: f64) -> bool {
    todo!()
}
