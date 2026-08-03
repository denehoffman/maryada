use core::{
    cmp::Ordering,
    f64::consts::{FRAC_PI_2, FRAC_PI_4, LN_2, LN_10, PI, TAU},
};

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

    fn opposite(&self) -> Direction {
        match self {
            Direction::Down => Direction::Up,
            Direction::Up => Direction::Down,
        }
    }

    fn constant_bound(&self, value: f64) -> f64 {
        match self {
            Direction::Down => value.next_down(),
            Direction::Up => value.next_up(),
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
        debug_assert!(((1u128 << 52)..(1u128 << 53)).contains(&rounded));
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
    if p == 0 {
        return 1.0;
    }
    if x.is_nan() {
        return f64::NAN;
    }
    let negative = x.is_sign_negative() && p & 1 != 0;
    let magnitude_direction = if negative {
        direction.opposite()
    } else {
        direction
    };
    let mut base = if p < 0 {
        recip(x.abs(), magnitude_direction)
    } else {
        x.abs()
    };
    let mut exponent = p.unsigned_abs();
    let mut magnitude = 1.0;
    while exponent != 0 {
        if exponent & 1 != 0 {
            magnitude = mul(magnitude, base, magnitude_direction);
        }
        exponent >>= 1;
        if exponent != 0 {
            base = sqr(base, magnitude_direction);
        }
    }
    if negative { -magnitude } else { magnitude }
}

fn outward_positive_approximation(value: f64, direction: Direction) -> f64 {
    // Some libm transcendental kernels are documented only as nearly rounded.
    // Two representable steps conservatively cover the observed binary64 error;
    // exact zeros and overflow require separate one-sided handling.
    if value == 0.0 {
        return match direction {
            Direction::Down => -0.0,
            Direction::Up => f64::from_bits(1),
        };
    }
    if value == f64::INFINITY {
        return match direction {
            Direction::Down => f64::MAX,
            Direction::Up => f64::INFINITY,
        };
    }
    outward_finite_approximation_ulps(value, direction, 2)
}

fn outward_finite_approximation_ulps(mut value: f64, direction: Direction, ulps: usize) -> f64 {
    debug_assert!(value.is_finite());
    for _ in 0..ulps {
        value = match direction {
            Direction::Down => value.next_down(),
            Direction::Up => value.next_up(),
        };
    }
    value
}

pub(crate) fn exp(x: f64, direction: Direction) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == f64::NEG_INFINITY {
        return direction.exact_zero();
    }
    if x == f64::INFINITY {
        return f64::INFINITY;
    }
    if x == 0.0 {
        return 1.0;
    }
    outward_positive_approximation(libm::exp(x), direction)
}

pub(crate) fn exp2(x: f64, direction: Direction) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == f64::NEG_INFINITY {
        return direction.exact_zero();
    }
    if x == f64::INFINITY {
        return f64::INFINITY;
    }
    if (-1074.0..=-1023.0).contains(&x) && x == (x as i32) as f64 {
        return f64::from_bits(1u64 << ((x as i32 + 1074) as u32));
    }
    if (-1022.0..=1023.0).contains(&x) && x == (x as i32) as f64 {
        return f64::from_bits(((x as i32 + 1023) as u64) << 52);
    }
    let ln_2 = match (x.is_sign_negative(), direction) {
        (false, Direction::Down) | (true, Direction::Up) => LN_2.next_down(),
        (false, Direction::Up) | (true, Direction::Down) => LN_2.next_up(),
    };
    exp(mul(x, ln_2, direction), direction)
}

pub(crate) fn exp10(x: f64, direction: Direction) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == f64::NEG_INFINITY {
        return direction.exact_zero();
    }
    if x == f64::INFINITY {
        return f64::INFINITY;
    }
    if x == 0.0 {
        return 1.0;
    }
    if (-323.0..=308.0).contains(&x) && x == (x as i32) as f64 {
        return pown(10.0, x as i32, direction);
    }
    let ln_10 = match (x.is_sign_negative(), direction) {
        (false, Direction::Down) | (true, Direction::Up) => LN_10.next_down(),
        (false, Direction::Up) | (true, Direction::Down) => LN_10.next_up(),
    };
    exp(mul(x, ln_10, direction), direction)
}

pub(crate) fn log(x: f64, direction: Direction) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x == f64::INFINITY {
        return f64::INFINITY;
    }
    if x == 1.0 {
        return direction.exact_zero();
    }
    outward_finite_approximation_ulps(libm::log(x), direction, 2)
}

pub(crate) fn log2(x: f64, direction: Direction) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x == f64::INFINITY {
        return f64::INFINITY;
    }
    if let FloatClass::Finite(value) = FloatClass::new(x)
        && value.significand.is_power_of_two()
    {
        let exponent = value.exponent + value.significand.trailing_zeros() as i32;
        return if exponent == 0 {
            direction.exact_zero()
        } else {
            f64::from(exponent)
        };
    }
    log_over_constant(x, LN_2, direction)
}

pub(crate) fn log10(x: f64, direction: Direction) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x == f64::INFINITY {
        return f64::INFINITY;
    }
    if x == 1.0 {
        return direction.exact_zero();
    }
    log_over_constant(x, LN_10, direction)
}

fn log_over_constant(x: f64, constant: f64, direction: Direction) -> f64 {
    // The standard constants are correctly rounded binary64 values, so the
    // adjacent values enclose the exact logarithm of the base. Division by
    // both positive endpoints handles either sign of log(x).
    let numerator = log(x, direction);
    let constant_lower = constant.next_down();
    let constant_upper = constant.next_up();
    match direction {
        Direction::Down => f64::min(
            div(numerator, constant_lower, Direction::Down),
            div(numerator, constant_upper, Direction::Down),
        ),
        Direction::Up => f64::max(
            div(numerator, constant_lower, Direction::Up),
            div(numerator, constant_upper, Direction::Up),
        ),
    }
}

// Trigonometric functions.

pub(crate) fn sin(x: f64, direction: Direction) -> f64 {
    if !x.is_finite() {
        return f64::NAN;
    }
    if x == 0.0 {
        return direction.exact_zero();
    }
    let Some(((lower, upper), _)) = reduced_trig_bounds(x) else {
        return match direction {
            Direction::Down => -1.0,
            Direction::Up => 1.0,
        };
    };
    match direction {
        Direction::Down => lower,
        Direction::Up => upper,
    }
}

pub(crate) fn cos(x: f64, direction: Direction) -> f64 {
    if !x.is_finite() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    let Some((_, (lower, upper))) = reduced_trig_bounds(x) else {
        return match direction {
            Direction::Down => -1.0,
            Direction::Up => 1.0,
        };
    };
    match direction {
        Direction::Down => lower,
        Direction::Up => upper,
    }
}

pub(crate) fn tan(x: f64, direction: Direction) -> f64 {
    if !x.is_finite() {
        return f64::NAN;
    }
    if x == 0.0 {
        return direction.exact_zero();
    }
    let Some(((sin_lower, sin_upper), (cos_lower, cos_upper))) = reduced_trig_bounds(x) else {
        return match direction {
            Direction::Down => f64::NEG_INFINITY,
            Direction::Up => f64::INFINITY,
        };
    };
    if cos_lower <= 0.0 && cos_upper >= 0.0 {
        return match direction {
            Direction::Down => f64::NEG_INFINITY,
            Direction::Up => f64::INFINITY,
        };
    }
    let candidates = [
        div(sin_lower, cos_lower, direction),
        div(sin_lower, cos_upper, direction),
        div(sin_upper, cos_lower, direction),
        div(sin_upper, cos_upper, direction),
    ];
    match direction {
        Direction::Down => candidates.into_iter().fold(f64::INFINITY, f64::min),
        Direction::Up => candidates.into_iter().fold(f64::NEG_INFINITY, f64::max),
    }
}

type Bounds = (f64, f64);

fn reduced_trig_bounds(x: f64) -> Option<(Bounds, Bounds)> {
    debug_assert!(x.is_finite());
    // Any exactly represented integer multiple is valid for the reduction.
    // The subsequent remainder check rejects a quotient whose binary64
    // selection or pi enclosure is too imprecise.
    let multiple = libm::round(x / FRAC_PI_2);
    if multiple.abs() > (1_u64 << 52) as f64 {
        return None;
    }
    let multiple_integer = multiple as i64;
    let half_pi_lower = FRAC_PI_2.next_down();
    let half_pi_upper = FRAC_PI_2.next_up();
    let product_lower = f64::min(
        mul(multiple, half_pi_lower, Direction::Down),
        mul(multiple, half_pi_upper, Direction::Down),
    );
    let product_upper = f64::max(
        mul(multiple, half_pi_lower, Direction::Up),
        mul(multiple, half_pi_upper, Direction::Up),
    );
    let remainder_lower = sub(x, product_upper, Direction::Down);
    let remainder_upper = sub(x, product_lower, Direction::Up);
    if remainder_lower < -0.8 || remainder_upper > 0.8 {
        return None;
    }

    let sine = (
        sin_series(remainder_lower, Direction::Down),
        sin_series(remainder_upper, Direction::Up),
    );
    let cosine_lower = f64::min(
        cos_series(remainder_lower, Direction::Down),
        cos_series(remainder_upper, Direction::Down),
    );
    let cosine_upper = if remainder_lower <= 0.0 && remainder_upper >= 0.0 {
        1.0
    } else {
        f64::max(
            cos_series(remainder_lower, Direction::Up),
            cos_series(remainder_upper, Direction::Up),
        )
    };
    let cosine = (cosine_lower, cosine_upper);
    let negate = |(lower, upper): Bounds| (-upper, -lower);
    Some(match multiple_integer.rem_euclid(4) {
        0 => (sine, cosine),
        1 => (cosine, negate(sine)),
        2 => (negate(sine), negate(cosine)),
        _ => (negate(cosine), sine),
    })
}

fn sin_series(x: f64, direction: Direction) -> f64 {
    if x.is_sign_negative() {
        return -sin_series(-x, direction.opposite());
    }
    debug_assert!((0.0..=0.8).contains(&x));
    let square_lower = mul(x, x, Direction::Down);
    let square_upper = mul(x, x, Direction::Up);
    let mut term_lower = x;
    let mut term_upper = x;
    let mut lower = 0.0;
    let mut upper = 0.0;
    for index in 0..14 {
        if index & 1 == 0 {
            lower = add(lower, term_lower, Direction::Down);
            upper = add(upper, term_upper, Direction::Up);
        } else {
            lower = sub(lower, term_upper, Direction::Down);
            upper = sub(upper, term_lower, Direction::Up);
        }
        let denominator = f64::from(((2 * index + 2) * (2 * index + 3)) as u32);
        term_lower = div(
            mul(term_lower, square_lower, Direction::Down),
            denominator,
            Direction::Down,
        );
        term_upper = div(
            mul(term_upper, square_upper, Direction::Up),
            denominator,
            Direction::Up,
        );
    }
    // Fourteen terms end on a negative term; the next positive term bounds
    // the alternating remainder from above.
    upper = add(upper, term_upper, Direction::Up);
    match direction {
        Direction::Down => lower,
        Direction::Up => upper,
    }
}

fn cos_series(x: f64, direction: Direction) -> f64 {
    let x = x.abs();
    debug_assert!((0.0..=0.8).contains(&x));
    let square_lower = mul(x, x, Direction::Down);
    let square_upper = mul(x, x, Direction::Up);
    let mut term_lower = 1.0;
    let mut term_upper = 1.0;
    let mut lower = 0.0;
    let mut upper = 0.0;
    for index in 0..14 {
        if index & 1 == 0 {
            lower = add(lower, term_lower, Direction::Down);
            upper = add(upper, term_upper, Direction::Up);
        } else {
            lower = sub(lower, term_upper, Direction::Down);
            upper = sub(upper, term_lower, Direction::Up);
        }
        let denominator = f64::from(((2 * index + 1) * (2 * index + 2)) as u32);
        term_lower = div(
            mul(term_lower, square_lower, Direction::Down),
            denominator,
            Direction::Down,
        );
        term_upper = div(
            mul(term_upper, square_upper, Direction::Up),
            denominator,
            Direction::Up,
        );
    }
    upper = add(upper, term_upper, Direction::Up);
    match direction {
        Direction::Down => lower,
        Direction::Up => upper,
    }
}

pub(crate) fn asin(x: f64, direction: Direction) -> f64 {
    if x.is_nan() || !(-1.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    if x == 0.0 {
        return direction.exact_zero();
    }
    if x == 1.0 {
        return direction.constant_bound(FRAC_PI_2);
    }
    if x == -1.0 {
        return -direction.opposite().constant_bound(FRAC_PI_2);
    }
    // asin(x) = atan(x / sqrt(1-x^2))
    let square_lower = sqr(x, Direction::Down);
    let square_upper = sqr(x, Direction::Up);
    let radicand_lower = sub(1.0, square_upper, Direction::Down);
    let radicand_upper = sub(1.0, square_lower, Direction::Up);
    let denominator_lower = sqrt(radicand_lower, Direction::Down);
    let denominator_upper = sqrt(radicand_upper, Direction::Up);
    let ratio = match direction {
        Direction::Down => f64::min(
            div(x, denominator_lower, Direction::Down),
            div(x, denominator_upper, Direction::Down),
        ),
        Direction::Up => f64::max(
            div(x, denominator_lower, Direction::Up),
            div(x, denominator_upper, Direction::Up),
        ),
    };
    atan(ratio, direction)
}

pub(crate) fn acos(x: f64, direction: Direction) -> f64 {
    if x.is_nan() || !(-1.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    if x == 1.0 {
        return direction.exact_zero();
    }
    sub(
        direction.constant_bound(FRAC_PI_2),
        asin(x, direction.opposite()),
        direction,
    )
}

pub(crate) fn atan(x: f64, direction: Direction) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return direction.exact_zero();
    }
    let magnitude_direction = if x.is_sign_negative() {
        direction.opposite()
    } else {
        direction
    };
    let result = atan_positive(x.abs(), magnitude_direction);
    if x.is_sign_negative() {
        -result
    } else {
        result
    }
}

fn atan_positive(x: f64, direction: Direction) -> f64 {
    debug_assert!(x >= 0.0);
    if x.is_infinite() {
        return direction.constant_bound(FRAC_PI_2);
    }
    if x <= 0.5 {
        return atan_series_positive(x, direction);
    }
    let reverse = direction.opposite();
    if x <= 1.0 {
        // atan(x) = pi/4 - atan((1-x)/(1+x)). The transformed argument is
        // in [0, 1/3], and its bound must be opposite to the result direction.
        let numerator = sub(1.0, x, reverse);
        let denominator = add(1.0, x, direction);
        let reduced = div(numerator, denominator, reverse);
        return sub(
            direction.constant_bound(FRAC_PI_4),
            atan_series_positive(reduced, reverse),
            direction,
        );
    }
    // atan(x) = pi/2 - atan(1/x)
    let reduced = div(1.0, x, reverse);
    sub(
        direction.constant_bound(FRAC_PI_2),
        atan_positive(reduced, reverse),
        direction,
    )
}

fn atan_series_positive(x: f64, direction: Direction) -> f64 {
    debug_assert!((0.0..=0.5).contains(&x));
    let square_lower = mul(x, x, Direction::Down);
    let square_upper = mul(x, x, Direction::Up);
    let mut power_lower = x;
    let mut power_upper = x;
    let mut lower = 0.0;
    let mut upper = 0.0;
    // atan(x) = x - x^3/3 + x^5/5 - ... . On [0, 1/2] the term
    // magnitudes decrease. Enclose 28 terms, then use the magnitude of the
    // next term as the alternating-series remainder bound.
    for index in 0..28 {
        let denominator = f64::from((2 * index + 1) as u32);
        let term_lower = div(power_lower, denominator, Direction::Down);
        let term_upper = div(power_upper, denominator, Direction::Up);
        if index & 1 == 0 {
            lower = add(lower, term_lower, Direction::Down);
            upper = add(upper, term_upper, Direction::Up);
        } else {
            lower = sub(lower, term_upper, Direction::Down);
            upper = sub(upper, term_lower, Direction::Up);
        }
        power_lower = mul(power_lower, square_lower, Direction::Down);
        power_upper = mul(power_upper, square_upper, Direction::Up);
    }
    // The next (positive) term can only raise the 28-term partial sum.
    let remainder = div(power_upper, 57.0, Direction::Up);
    upper = add(upper, remainder, Direction::Up);
    match direction {
        Direction::Down => lower,
        Direction::Up => upper,
    }
}

pub(crate) fn atan2(y: f64, x: f64, direction: Direction) -> f64 {
    if x.is_nan() || y.is_nan() || (x == 0.0 && y == 0.0) {
        return f64::NAN;
    }
    let signed_constant = |value: f64, negative: bool, direction: Direction| {
        if negative {
            -direction.opposite().constant_bound(value)
        } else {
            direction.constant_bound(value)
        }
    };
    if y.is_infinite() {
        if x.is_infinite() {
            let magnitude_direction = if y.is_sign_negative() {
                direction.opposite()
            } else {
                direction
            };
            let quarter = magnitude_direction.constant_bound(FRAC_PI_4);
            let magnitude = if x.is_sign_negative() {
                mul(quarter, 3.0, magnitude_direction)
            } else {
                quarter
            };
            return if y.is_sign_negative() {
                -magnitude
            } else {
                magnitude
            };
        }
        return signed_constant(FRAC_PI_2, y.is_sign_negative(), direction);
    }
    if x.is_infinite() {
        if x.is_sign_positive() {
            return direction.exact_zero();
        }
        return signed_constant(PI, y < 0.0, direction);
    }
    if x == 0.0 {
        return signed_constant(FRAC_PI_2, y < 0.0, direction);
    }
    if x > 0.0 {
        return atan(div(y, x, direction), direction);
    }
    if y < 0.0 {
        let angle = atan(div(y.abs(), x.abs(), direction), direction);
        add(signed_constant(PI, true, direction), angle, direction)
    } else {
        let reverse = direction.opposite();
        let angle = atan(div(y, x.abs(), reverse), reverse);
        sub(direction.constant_bound(PI), angle, direction)
    }
}

// Hyperbolic functions.

pub(crate) fn sinh(x: f64, direction: Direction) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return direction.exact_zero();
    }
    if x.is_infinite() {
        return x;
    }
    let magnitude_direction = if x.is_sign_negative() {
        direction.opposite()
    } else {
        direction
    };
    let magnitude = x.abs();
    let result = if magnitude <= 1.0 {
        sinh_series_positive(magnitude, magnitude_direction)
    } else {
        // Outside the cancellation-prone region, use
        // sinh(x) = (exp(x) - exp(-x)) / 2.
        let positive = exp(magnitude, magnitude_direction);
        let negative = exp(-magnitude, magnitude_direction.opposite());
        mul(
            sub(positive, negative, magnitude_direction),
            0.5,
            magnitude_direction,
        )
    };
    if x.is_sign_negative() {
        -result
    } else {
        result
    }
}

fn sinh_series_positive(x: f64, direction: Direction) -> f64 {
    debug_assert!((0.0..=1.0).contains(&x));
    // All terms of sinh(x) are nonnegative here. Directed recurrence through
    // x^17/17! gives a lower or upper partial sum without cancellation.
    let square = mul(x, x, direction);
    let mut term = x;
    let mut sum = x;
    for index in 1..=8 {
        let denominator = f64::from((2 * index * (2 * index + 1)) as u32);
        term = div(mul(term, square, direction), denominator, direction);
        sum = add(sum, term, direction);
    }
    if direction == Direction::Down {
        return sum;
    }
    // The omitted positive tail starts with x^19/19!. Each later term is at
    // most 1/420 of its predecessor for x <= 1, so a geometric majorant is
    // rigorous and adds less than one binary64 ulp near x = 1.
    let next = div(mul(term, square, Direction::Up), 18.0 * 19.0, Direction::Up);
    let ratio = div(square, 20.0 * 21.0, Direction::Up);
    let tail = div(next, sub(1.0, ratio, Direction::Down), Direction::Up);
    add(sum, tail, Direction::Up)
}

pub(crate) fn cosh(x: f64, direction: Direction) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }
    if x == 0.0 {
        return 1.0;
    }
    let magnitude = x.abs();
    // cosh(x) = (exp(|x|) + exp(-|x|)) / 2. Each operand and operation is
    // rounded in the requested direction, so the composition inherits the
    // enclosure guarantee of exp without relying on a separate cosh kernel.
    let sum = add(
        exp(magnitude, direction),
        exp(-magnitude, direction),
        direction,
    );
    f64::max(mul(sum, 0.5, direction), 1.0)
}

pub(crate) fn tanh(x: f64, direction: Direction) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return direction.exact_zero();
    }
    if x == f64::NEG_INFINITY {
        return -1.0;
    }
    if x == f64::INFINITY {
        return 1.0;
    }
    let magnitude_direction = if x.is_sign_negative() {
        direction.opposite()
    } else {
        direction
    };
    let magnitude = x.abs();
    let result = div(
        sinh(magnitude, magnitude_direction),
        cosh(magnitude, magnitude_direction.opposite()),
        magnitude_direction,
    );
    let result = match magnitude_direction {
        Direction::Down => f64::max(result, 0.0),
        Direction::Up => f64::min(result, 1.0),
    };
    if x.is_sign_negative() {
        -result
    } else {
        result
    }
}

pub(crate) fn asinh(x: f64, direction: Direction) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return direction.exact_zero();
    }
    if x.is_infinite() {
        return x;
    }
    let magnitude_direction = if x.is_sign_negative() {
        direction.opposite()
    } else {
        direction
    };
    let magnitude = x.abs();
    let result = if magnitude <= 1.0 {
        let square = sqr(magnitude, magnitude_direction);
        let root = sqrt(add(1.0, square, magnitude_direction), magnitude_direction);
        log(
            add(magnitude, root, magnitude_direction),
            magnitude_direction,
        )
    } else {
        // asinh(x) = log(x) + log(1 + sqrt(1 + 1/x^2)); this form avoids
        // overflow in x^2 for large x.
        let inverse = div(1.0, magnitude, magnitude_direction);
        let square = sqr(inverse, magnitude_direction);
        let root = sqrt(add(1.0, square, magnitude_direction), magnitude_direction);
        add(
            log(magnitude, magnitude_direction),
            log(add(1.0, root, magnitude_direction), magnitude_direction),
            magnitude_direction,
        )
    };
    if x.is_sign_negative() {
        -result
    } else {
        result
    }
}

pub(crate) fn acosh(x: f64, direction: Direction) -> f64 {
    if x.is_nan() || x < 1.0 {
        return f64::NAN;
    }
    if x == 1.0 {
        return direction.exact_zero();
    }
    if x == f64::INFINITY {
        return f64::INFINITY;
    }
    // acosh(x) = log(x) + log(1 + sqrt(1 - 1/x^2)). The reciprocal is
    // evaluated oppositely because it is subtracted inside the square root.
    let reverse = direction.opposite();
    let inverse = div(1.0, x, reverse);
    let square = sqr(inverse, reverse);
    let radicand = sub(1.0, square, direction);
    let root = sqrt(radicand, direction);
    add(
        log(x, direction),
        log(add(1.0, root, direction), direction),
        direction,
    )
}

pub(crate) fn atanh(x: f64, direction: Direction) -> f64 {
    if x.is_nan() || !(-1.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    if x == -1.0 {
        return f64::NEG_INFINITY;
    }
    if x == 1.0 {
        return f64::INFINITY;
    }
    if x == 0.0 {
        return direction.exact_zero();
    }
    let magnitude_direction = if x.is_sign_negative() {
        direction.opposite()
    } else {
        direction
    };
    let magnitude = x.abs();
    let numerator = add(1.0, magnitude, magnitude_direction);
    let denominator = sub(1.0, magnitude, magnitude_direction.opposite());
    let ratio = div(numerator, denominator, magnitude_direction);
    let result = mul(log(ratio, magnitude_direction), 0.5, magnitude_direction);
    if x.is_sign_negative() {
        -result
    } else {
        result
    }
}

// NOTE: The required tightest integer-valued functions are all basically just direct libm
// implementations, so no need to have crate-only methods here.

// Numeric interval queries.

pub(crate) fn radius(inf: f64, sup: f64, midpoint: f64) -> f64 {
    // Each distance must be rounded upward independently: rounding the
    // width first and then halving can underestimate after two roundings.
    let below = sub(midpoint, inf, Direction::Up);
    let above = sub(sup, midpoint, Direction::Up);
    f64::max(below, above)
}

// Exact parsing of all required number literal forms.

pub(crate) fn number_literal(literal: &str, direction: Direction) -> Option<f64> {
    let bytes = literal.as_bytes();
    if bytes.is_empty() || bytes.iter().any(|byte| byte.is_ascii_whitespace()) {
        return None;
    }

    let (negative, unsigned) = strip_number_sign(bytes)?;
    if eq_ascii_case(unsigned, b"inf") || eq_ascii_case(unsigned, b"infinity") {
        return Some(if negative {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        });
    }

    if bytes.contains(&b'/') {
        return parse_rational_literal(bytes, direction);
    }
    if unsigned.len() >= 2 && unsigned[0] == b'0' && unsigned[1].eq_ignore_ascii_case(&b'x') {
        return parse_hex_literal(bytes, direction);
    }
    parse_decimal_literal(literal, direction)
}

#[derive(Clone, Copy)]
struct DecimalLiteral<'a> {
    negative: bool,
    mantissa: &'a [u8],
    digits_before_point: usize,
    leading_zero_digits: usize,
    significant_digits: usize,
    exponent: i64,
    nonzero: bool,
}

#[derive(Clone, Copy)]
struct IntegerLiteral<'a> {
    negative: bool,
    digits: &'a [u8],
}

struct ExactDecimal {
    /// Decimal digits in little-endian order.
    digits: [u8; 800],
    len: usize,
    /// The value is `digits * 10^shift`.
    shift: i32,
}

fn strip_number_sign(bytes: &[u8]) -> Option<(bool, &[u8])> {
    match bytes.first().copied()? {
        b'-' => Some((true, &bytes[1..])),
        b'+' => Some((false, &bytes[1..])),
        _ => Some((false, bytes)),
    }
}

fn eq_ascii_case(left: &[u8], right: &[u8]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.eq_ignore_ascii_case(right))
}

fn parse_signed_decimal_exponent(bytes: &[u8]) -> Option<i64> {
    let (negative, digits) = strip_number_sign(bytes)?;
    if digits.is_empty() || !digits.iter().all(u8::is_ascii_digit) {
        return None;
    }
    let mut value = 0i64;
    for &digit in digits {
        value = value
            .saturating_mul(10)
            .saturating_add(i64::from(digit - b'0'));
    }
    Some(if negative {
        value.saturating_neg()
    } else {
        value
    })
}

fn analyze_decimal_literal(literal: &str) -> Option<DecimalLiteral<'_>> {
    let bytes = literal.as_bytes();
    let (negative, unsigned) = strip_number_sign(bytes)?;
    if unsigned.is_empty() {
        return None;
    }
    let exponent_index = unsigned
        .iter()
        .position(|byte| byte.eq_ignore_ascii_case(&b'e'));
    let (mantissa, exponent) = if let Some(index) = exponent_index {
        if unsigned[index + 1..]
            .iter()
            .any(|byte| byte.eq_ignore_ascii_case(&b'e'))
        {
            return None;
        }
        (
            &unsigned[..index],
            parse_signed_decimal_exponent(&unsigned[index + 1..])?,
        )
    } else {
        (unsigned, 0)
    };

    let mut point = None;
    let mut digit_count = 0usize;
    let mut digits_before_point = 0usize;
    let mut leading_zero_digits = 0usize;
    let mut nonzero = false;
    for (index, &byte) in mantissa.iter().enumerate() {
        if byte == b'.' {
            if point.replace(index).is_some() {
                return None;
            }
        } else if byte.is_ascii_digit() {
            if point.is_none() {
                digits_before_point += 1;
            }
            digit_count += 1;
            if !nonzero {
                if byte == b'0' {
                    leading_zero_digits += 1;
                } else {
                    nonzero = true;
                }
            }
        } else {
            return None;
        }
    }
    if digit_count == 0 {
        return None;
    }

    Some(DecimalLiteral {
        negative,
        mantissa,
        digits_before_point,
        leading_zero_digits,
        significant_digits: digit_count - leading_zero_digits,
        exponent,
        nonzero,
    })
}

fn exact_decimal_from_float(value: f64) -> ExactDecimal {
    debug_assert!(value.is_finite() && value != 0.0);
    let dyadic = match FloatClass::new(value.abs()) {
        FloatClass::Finite(value) => value,
        _ => unreachable!(),
    };
    let mut result = ExactDecimal {
        digits: [0; 800],
        len: 0,
        shift: 0,
    };
    let mut significand = dyadic.significand;
    while significand != 0 {
        result.digits[result.len] = (significand % 10) as u8;
        result.len += 1;
        significand /= 10;
    }
    let (factor, count) = if dyadic.exponent < 0 {
        result.shift = dyadic.exponent;
        (5, dyadic.exponent.unsigned_abs())
    } else {
        (2, dyadic.exponent as u32)
    };
    for _ in 0..count {
        let mut carry = 0u16;
        for digit in &mut result.digits[..result.len] {
            let product = u16::from(*digit) * factor + carry;
            *digit = (product % 10) as u8;
            carry = product / 10;
        }
        while carry != 0 {
            debug_assert!(result.len < result.digits.len());
            result.digits[result.len] = (carry % 10) as u8;
            result.len += 1;
            carry /= 10;
        }
    }
    result
}

fn compare_decimal_to_float(decimal: DecimalLiteral<'_>, value: f64) -> Ordering {
    if !decimal.nonzero {
        return if value == 0.0 {
            Ordering::Equal
        } else if value > 0.0 {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    if value == f64::INFINITY {
        return Ordering::Less;
    }
    if value == f64::NEG_INFINITY {
        return Ordering::Greater;
    }
    if value == 0.0 || decimal.negative != value.is_sign_negative() {
        return if decimal.negative {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }

    let exact = exact_decimal_from_float(value);
    let decimal_position = (decimal.digits_before_point as i64)
        .saturating_add(decimal.exponent)
        .saturating_sub(decimal.leading_zero_digits as i64);
    let exact_position = exact.len as i64 + i64::from(exact.shift);
    let mut magnitude_order = decimal_position.cmp(&exact_position);
    if magnitude_order == Ordering::Equal {
        let mut input = decimal
            .mantissa
            .iter()
            .copied()
            .filter(u8::is_ascii_digit)
            .skip(decimal.leading_zero_digits);
        let count = decimal.significant_digits.max(exact.len);
        for index in 0..count {
            let input_digit = input.next().map_or(0, |digit| digit - b'0');
            let exact_digit = if index < exact.len {
                exact.digits[exact.len - 1 - index]
            } else {
                0
            };
            match input_digit.cmp(&exact_digit) {
                Ordering::Equal => {}
                order => {
                    magnitude_order = order;
                    break;
                }
            }
        }
    }
    if decimal.negative {
        magnitude_order.reverse()
    } else {
        magnitude_order
    }
}

fn select_directed_neighbor(value: f64, exact_order: Ordering, direction: Direction) -> f64 {
    match (exact_order, direction) {
        (Ordering::Equal, _) => value,
        (Ordering::Less, Direction::Down) | (Ordering::Greater, Direction::Up) => {
            if exact_order == Ordering::Less {
                value.next_down()
            } else {
                value.next_up()
            }
        }
        _ => value,
    }
}

fn parse_decimal_literal(literal: &str, direction: Direction) -> Option<f64> {
    let decimal = analyze_decimal_literal(literal)?;
    let nearest = literal.parse::<f64>().ok()?;
    Some(select_directed_neighbor(
        nearest,
        compare_decimal_to_float(decimal, nearest),
        direction,
    ))
}

fn parse_integer_literal(bytes: &[u8], allow_negative: bool) -> Option<IntegerLiteral<'_>> {
    let (negative, unsigned) = strip_number_sign(bytes)?;
    if negative && !allow_negative {
        return None;
    }
    if unsigned.is_empty() || !unsigned.iter().all(u8::is_ascii_digit) {
        return None;
    }
    let first_nonzero = unsigned
        .iter()
        .position(|&digit| digit != b'0')
        .unwrap_or(unsigned.len());
    Some(IntegerLiteral {
        negative: negative && first_nonzero != unsigned.len(),
        digits: &unsigned[first_nonzero..],
    })
}

fn product_digit(
    q: IntegerLiteral<'_>,
    value: &ExactDecimal,
    position: usize,
    carry: &mut u64,
) -> u8 {
    let mut sum = *carry;
    let first = position.saturating_sub(value.len.saturating_sub(1));
    let last = position.min(q.digits.len().saturating_sub(1));
    if !q.digits.is_empty() && first <= last {
        for q_index in first..=last {
            let q_digit = u64::from(q.digits[q.digits.len() - 1 - q_index] - b'0');
            let value_digit = u64::from(value.digits[position - q_index]);
            sum += q_digit * value_digit;
        }
    }
    *carry = sum / 10;
    (sum % 10) as u8
}

fn compare_integer_to_product(
    numerator: IntegerLiteral<'_>,
    numerator_shift: usize,
    denominator: IntegerLiteral<'_>,
    value: &ExactDecimal,
    product_shift: usize,
) -> Ordering {
    let end = numerator_shift.saturating_add(numerator.digits.len()).max(
        product_shift
            .saturating_add(denominator.digits.len())
            .saturating_add(value.len)
            .saturating_add(1),
    );
    let mut carry = 0u64;
    let mut order = Ordering::Equal;
    for position in 0..end {
        let left = if position >= numerator_shift {
            let index = position - numerator_shift;
            if index < numerator.digits.len() {
                numerator.digits[numerator.digits.len() - 1 - index] - b'0'
            } else {
                0
            }
        } else {
            0
        };
        let right = if position >= product_shift {
            product_digit(denominator, value, position - product_shift, &mut carry)
        } else {
            0
        };
        if left != right {
            order = left.cmp(&right);
        }
    }
    debug_assert_eq!(carry, 0);
    order
}

fn compare_rational_magnitude_to_float(
    numerator: IntegerLiteral<'_>,
    denominator: IntegerLiteral<'_>,
    value: f64,
) -> Ordering {
    debug_assert!(!numerator.negative && !denominator.negative && value >= 0.0);
    if numerator.digits.is_empty() {
        return if value == 0.0 {
            Ordering::Equal
        } else {
            Ordering::Less
        };
    }
    if value == 0.0 {
        return Ordering::Greater;
    }
    if value == f64::INFINITY {
        return Ordering::Less;
    }
    let exact = exact_decimal_from_float(value);
    if exact.shift >= 0 {
        compare_integer_to_product(numerator, 0, denominator, &exact, exact.shift as usize)
    } else {
        compare_integer_to_product(
            numerator,
            exact.shift.unsigned_abs() as usize,
            denominator,
            &exact,
            0,
        )
    }
}

fn parse_rational_literal(bytes: &[u8], direction: Direction) -> Option<f64> {
    let slash = bytes.iter().position(|&byte| byte == b'/')?;
    if bytes[slash + 1..].contains(&b'/') {
        return None;
    }
    let numerator = parse_integer_literal(&bytes[..slash], true)?;
    let denominator = parse_integer_literal(&bytes[slash + 1..], false)?;
    if denominator.digits.is_empty() {
        return None;
    }
    if numerator.digits.is_empty() {
        return Some(if numerator.negative { -0.0 } else { 0.0 });
    }
    let magnitude_numerator = IntegerLiteral {
        negative: false,
        digits: numerator.digits,
    };

    let mut lower_bits = 0u64;
    let mut upper_bits = f64::INFINITY.to_bits();
    while upper_bits - lower_bits > 1 {
        let middle_bits = lower_bits + (upper_bits - lower_bits) / 2;
        let middle = f64::from_bits(middle_bits);
        match compare_rational_magnitude_to_float(magnitude_numerator, denominator, middle) {
            Ordering::Less => upper_bits = middle_bits,
            Ordering::Greater => lower_bits = middle_bits,
            Ordering::Equal => {
                let exact = if numerator.negative { -middle } else { middle };
                return Some(exact);
            }
        }
    }
    let lower = f64::from_bits(lower_bits);
    let upper = f64::from_bits(upper_bits);
    Some(match (numerator.negative, direction) {
        (false, Direction::Down) => lower,
        (false, Direction::Up) => upper,
        (true, Direction::Down) => -upper,
        (true, Direction::Up) => -lower,
    })
}

fn parse_hex_literal(bytes: &[u8], direction: Direction) -> Option<f64> {
    let (negative, unsigned) = strip_number_sign(bytes)?;
    if unsigned.len() < 4 || unsigned[0] != b'0' || !unsigned[1].eq_ignore_ascii_case(&b'x') {
        return None;
    }
    let exponent_index = unsigned[2..]
        .iter()
        .position(|byte| byte.eq_ignore_ascii_case(&b'p'))?
        + 2;
    if unsigned[exponent_index + 1..]
        .iter()
        .any(|byte| byte.eq_ignore_ascii_case(&b'p'))
    {
        return None;
    }
    let exponent = parse_signed_decimal_exponent(&unsigned[exponent_index + 1..])?;
    let mantissa = &unsigned[2..exponent_index];
    let mut point_seen = false;
    let mut digit_count = 0usize;
    let mut fractional_digits = 0usize;
    let mut significant_started = false;
    let mut significant_digits = 0usize;
    let mut retained_digits = 0usize;
    let mut significand = 0u128;
    let mut sticky = false;
    for &byte in mantissa {
        if byte == b'.' {
            if point_seen {
                return None;
            }
            point_seen = true;
            continue;
        }
        let digit = match byte {
            b'0'..=b'9' => byte - b'0',
            b'a'..=b'f' => byte - b'a' + 10,
            b'A'..=b'F' => byte - b'A' + 10,
            _ => return None,
        };
        digit_count += 1;
        if point_seen {
            fractional_digits += 1;
        }
        if digit != 0 {
            significant_started = true;
        }
        if significant_started {
            significant_digits += 1;
            if retained_digits < 31 {
                significand = (significand << 4) | u128::from(digit);
                retained_digits += 1;
            } else {
                sticky |= digit != 0;
            }
        }
    }
    if digit_count == 0 {
        return None;
    }
    if !significant_started {
        return Some(if negative { -0.0 } else { 0.0 });
    }
    let omitted_digits = significant_digits - retained_digits;
    let binary_exponent = exponent
        .saturating_sub((fractional_digits as i64).saturating_mul(4))
        .saturating_add((omitted_digits as i64).saturating_mul(4));
    if binary_exponent > 4096 {
        return Some(match (negative, direction) {
            (false, Direction::Down) => f64::MAX,
            (false, Direction::Up) => f64::INFINITY,
            (true, Direction::Down) => f64::NEG_INFINITY,
            (true, Direction::Up) => -f64::MAX,
        });
    }
    if binary_exponent < -4096 {
        return Some(match (negative, direction) {
            (false, Direction::Down) => -0.0,
            (false, Direction::Up) => f64::from_bits(1),
            (true, Direction::Down) => -f64::from_bits(1),
            (true, Direction::Up) => 0.0,
        });
    }
    Some(
        WideDyadic {
            negative,
            significand,
            exponent: binary_exponent as i32,
        }
        .pack(sticky, direction),
    )
}

// Certified argument-reduction predicates used by interval kernels.

fn contains_periodic_point(
    inf: f64,
    sup: f64,
    phase_lower: f64,
    phase_upper: f64,
    period_lower: f64,
    period_upper: f64,
) -> bool {
    if !inf.is_finite() || !sup.is_finite() {
        return true;
    }

    let numerator_lower = sub(inf, phase_upper, Direction::Down);
    let numerator_upper = sub(sup, phase_lower, Direction::Up);
    let quotient_lower = f64::min(
        div(numerator_lower, period_lower, Direction::Down),
        div(numerator_lower, period_upper, Direction::Down),
    );
    let quotient_upper = f64::max(
        div(numerator_upper, period_lower, Direction::Up),
        div(numerator_upper, period_upper, Direction::Up),
    );
    libm::ceil(quotient_lower) <= libm::floor(quotient_upper)
}

pub(crate) fn contains_sin_maximum(inf: f64, sup: f64) -> bool {
    contains_periodic_point(
        inf,
        sup,
        FRAC_PI_2.next_down(),
        FRAC_PI_2.next_up(),
        TAU.next_down(),
        TAU.next_up(),
    )
}

pub(crate) fn contains_sin_minimum(inf: f64, sup: f64) -> bool {
    contains_periodic_point(
        inf,
        sup,
        -FRAC_PI_2.next_up(),
        -FRAC_PI_2.next_down(),
        TAU.next_down(),
        TAU.next_up(),
    )
}

pub(crate) fn contains_cos_maximum(inf: f64, sup: f64) -> bool {
    contains_periodic_point(inf, sup, -0.0, 0.0, TAU.next_down(), TAU.next_up())
}

pub(crate) fn contains_cos_minimum(inf: f64, sup: f64) -> bool {
    contains_periodic_point(
        inf,
        sup,
        PI.next_down(),
        PI.next_up(),
        TAU.next_down(),
        TAU.next_up(),
    )
}

pub(crate) fn contains_tan_pole(inf: f64, sup: f64) -> bool {
    contains_periodic_point(
        inf,
        sup,
        FRAC_PI_2.next_down(),
        FRAC_PI_2.next_up(),
        PI.next_down(),
        PI.next_up(),
    )
}
