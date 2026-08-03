use core::{cmp::Ordering, fmt, ops, str::FromStr};
use impl_ops::*;

use crate::{DecoratedInterval, Interval, ParseIntervalError, Signal, SignalFlags, SignalSink};

impl Interval {
    /// Constructs an interval from binary64 endpoints, returning empty for invalid bounds.
    pub fn new(inf: f64, sup: f64) -> Self {
        Self::nums_to_interval(inf, sup, &mut ())
    }
    /// Converts this bare interval to a decorated interval.
    pub fn decorate<S: SignalSink>(&self, signals: &mut S) -> DecoratedInterval {
        match Interval::from_nums(self.inf_raw(), self.sup_raw()) {
            Some(value) => DecoratedInterval::new_dec_raw(value),
            None => {
                signals.raise(Signal::UndefinedOperation);
                DecoratedInterval::NAI
            }
        }
    }

    /// Encodes this interval in the big-endian interchange format.
    pub fn to_be_bytes(self) -> [u8; crate::INTERVAL_ENCODED_LEN] {
        crate::interval_to_be_bytes(self)
    }

    /// Encodes this interval in the little-endian interchange format.
    pub fn to_le_bytes(self) -> [u8; crate::INTERVAL_ENCODED_LEN] {
        crate::interval_to_le_bytes(self)
    }

    /// Decodes a big-endian interval, reporting invalid input to `signals`.
    pub fn from_be_bytes<S: SignalSink>(bytes: &[u8], signals: &mut S) -> Self {
        crate::interval_from_be_bytes(bytes, signals)
    }

    /// Decodes a little-endian interval, reporting invalid input to `signals`.
    pub fn from_le_bytes<S: SignalSink>(bytes: &[u8], signals: &mut S) -> Self {
        crate::interval_from_le_bytes(bytes, signals)
    }
}

impl From<f64> for Interval {
    fn from(value: f64) -> Self {
        Self::new(value, value)
    }
}

impl From<&f64> for Interval {
    fn from(value: &f64) -> Self {
        (*value).into()
    }
}

impl From<&Interval> for DecoratedInterval {
    fn from(value: &Interval) -> Self {
        value.decorate(&mut ())
    }
}

impl From<Interval> for DecoratedInterval {
    fn from(value: Interval) -> Self {
        value.decorate(&mut ())
    }
}

impl From<&DecoratedInterval> for Interval {
    fn from(value: &DecoratedInterval) -> Self {
        value.interval_raw()
    }
}

impl From<DecoratedInterval> for Interval {
    fn from(value: DecoratedInterval) -> Self {
        value.interval_raw()
    }
}

impl DecoratedInterval {
    /// Constructs a decorated interval, returning NaI for invalid bounds.
    pub fn new(inf: f64, sup: f64) -> Self {
        Self::nums_to_interval(inf, sup, &mut ())
    }

    /// Encodes this decorated interval in the big-endian interchange format.
    pub fn to_be_bytes(self) -> [u8; crate::DECORATED_INTERVAL_ENCODED_LEN] {
        crate::decorated_interval_to_be_bytes(self)
    }

    /// Encodes this decorated interval in the little-endian interchange format.
    pub fn to_le_bytes(self) -> [u8; crate::DECORATED_INTERVAL_ENCODED_LEN] {
        crate::decorated_interval_to_le_bytes(self)
    }

    /// Decodes a big-endian decorated interval, reporting invalid input.
    pub fn from_be_bytes<S: SignalSink>(bytes: &[u8], signals: &mut S) -> Self {
        crate::decorated_interval_from_be_bytes(bytes, signals)
    }

    /// Decodes a little-endian decorated interval, reporting invalid input.
    pub fn from_le_bytes<S: SignalSink>(bytes: &[u8], signals: &mut S) -> Self {
        crate::decorated_interval_from_le_bytes(bytes, signals)
    }
}

impl From<f64> for DecoratedInterval {
    fn from(value: f64) -> Self {
        Self::new(value, value)
    }
}

impl From<&f64> for DecoratedInterval {
    fn from(value: &f64) -> Self {
        (*value).into()
    }
}

fn display_interval<T: crate::IntervalDatum>(
    value: T,
    formatter: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    let mut output = [0u8; 128];
    let len = crate::interval_to_text(value, None, &mut output).map_err(|_| fmt::Error)?;
    let text = core::str::from_utf8(&output[..len]).map_err(|_| fmt::Error)?;
    formatter.write_str(text)
}

impl fmt::Display for Interval {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        display_interval(*self, formatter)
    }
}

impl fmt::Display for DecoratedInterval {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        display_interval(*self, formatter)
    }
}

impl FromStr for Interval {
    type Err = ParseIntervalError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let mut signals = SignalFlags::NONE;
        let result = Self::text_to_interval(value, &mut signals);
        if signals.contains(Signal::UndefinedOperation) || signals.contains(Signal::InvalidOperand)
        {
            Err(ParseIntervalError)
        } else {
            Ok(result)
        }
    }
}

impl FromStr for DecoratedInterval {
    type Err = ParseIntervalError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let mut signals = SignalFlags::NONE;
        let result = Self::text_to_interval(value, &mut signals);
        if signals.contains(Signal::UndefinedOperation) || signals.contains(Signal::InvalidOperand)
        {
            Err(ParseIntervalError)
        } else {
            Ok(result)
        }
    }
}

impl_op_ex!(-|a: &Interval| -> Interval { crate::neg(*a) });
impl_op_ex!(+ |a: &Interval, b: &Interval| -> Interval {crate::add(*a, *b)});
impl_op_ex!(-|a: &Interval, b: &Interval| -> Interval { crate::sub(*a, *b) });
impl_op_ex!(*|a: &Interval, b: &Interval| -> Interval { crate::mul(*a, *b) });
impl_op_ex!(/|a: &Interval, b: &Interval| -> Interval { crate::div(*a, *b) });

impl_op_ex!(+ |a: &Interval, b: &f64| -> Interval {crate::add(*a, b.into())});
impl_op_ex!(-|a: &Interval, b: &f64| -> Interval { crate::sub(*a, b.into()) });
impl_op_ex!(*|a: &Interval, b: &f64| -> Interval { crate::mul(*a, b.into()) });
impl_op_ex!(/|a: &Interval, b: &f64| -> Interval { crate::div(*a, b.into()) });

impl_op_ex!(+ |a: &f64, b: &Interval| -> Interval {crate::add(a.into(), *b)});
impl_op_ex!(-|a: &f64, b: &Interval| -> Interval { crate::sub(a.into(), *b) });
impl_op_ex!(*|a: &f64, b: &Interval| -> Interval { crate::mul(a.into(), *b) });
impl_op_ex!(/|a: &f64, b: &Interval| -> Interval { crate::div(a.into(), *b) });

impl_op_ex!(-|a: &DecoratedInterval| -> DecoratedInterval { crate::neg(*a) });
impl_op_ex!(+ |a: &DecoratedInterval, b: &DecoratedInterval| -> DecoratedInterval {crate::add(*a, *b)});
impl_op_ex!(
    -|a: &DecoratedInterval, b: &DecoratedInterval| -> DecoratedInterval { crate::sub(*a, *b) }
);
impl_op_ex!(
    *|a: &DecoratedInterval, b: &DecoratedInterval| -> DecoratedInterval { crate::mul(*a, *b) }
);
impl_op_ex!(/|a: &DecoratedInterval, b: &DecoratedInterval| -> DecoratedInterval { crate::div(*a, *b) });

impl Interval {
    /// Returns the reciprocal enclosure; see [`crate::recip`].
    pub fn recip(&self) -> Self {
        crate::recip(*self)
    }

    /// Returns the square enclosure; see [`crate::sqr`].
    pub fn sqr(&self) -> Self {
        crate::sqr(*self)
    }

    /// Returns the square-root enclosure; see [`crate::sqrt`].
    pub fn sqrt(&self) -> Self {
        crate::sqrt(*self)
    }

    /// Encloses `self * y + z` using fused interval arithmetic.
    pub fn mul_add(&self, y: &Self, z: &Self) -> Self {
        crate::fma(*self, *y, *z)
    }

    /// Raises this interval to the integer power `p`.
    pub fn pown(&self, p: i32) -> Self {
        crate::pown(*self, p)
    }

    /// Alias for [`Interval::pown`].
    pub fn powi(&self, i: i32) -> Self {
        self.pown(i)
    }

    /// Encloses powers with bases in `self` and exponents in `other`.
    pub fn pow(&self, other: &Self) -> Self {
        crate::pow(*self, *other)
    }

    /// Applies the natural exponential function.
    pub fn exp(&self) -> Self {
        crate::exp(*self)
    }

    /// Applies the base-two exponential function.
    pub fn exp2(&self) -> Self {
        crate::exp2(*self)
    }

    /// Applies the base-ten exponential function.
    pub fn exp10(&self) -> Self {
        crate::exp10(*self)
    }

    /// Applies the natural logarithm on its real domain.
    pub fn log(&self) -> Self {
        crate::log(*self)
    }

    /// Applies the base-two logarithm on its real domain.
    pub fn log2(&self) -> Self {
        crate::log2(*self)
    }

    /// Applies the base-ten logarithm on its real domain.
    pub fn log10(&self) -> Self {
        crate::log10(*self)
    }

    /// Returns the sine enclosure.
    pub fn sin(&self) -> Self {
        crate::sin(*self)
    }

    /// Returns the cosine enclosure.
    pub fn cos(&self) -> Self {
        crate::cos(*self)
    }

    /// Returns the tangent enclosure over defined values.
    pub fn tan(&self) -> Self {
        crate::tan(*self)
    }

    /// Returns the inverse-sine enclosure on `[-1, 1]`.
    pub fn asin(&self) -> Self {
        crate::asin(*self)
    }

    /// Returns the inverse-cosine enclosure on `[-1, 1]`.
    pub fn acos(&self) -> Self {
        crate::acos(*self)
    }

    /// Returns the inverse-tangent enclosure.
    pub fn atan(&self) -> Self {
        crate::atan(*self)
    }

    /// Returns the two-argument angle enclosure `atan2(self, x)`.
    pub fn atan2(&self, x: &Self) -> Self {
        crate::atan2(*self, *x)
    }

    /// Returns the hyperbolic-sine enclosure.
    pub fn sinh(&self) -> Self {
        crate::sinh(*self)
    }

    /// Returns the hyperbolic-cosine enclosure.
    pub fn cosh(&self) -> Self {
        crate::cosh(*self)
    }

    /// Returns the hyperbolic-tangent enclosure.
    pub fn tanh(&self) -> Self {
        crate::tanh(*self)
    }

    /// Returns the inverse-hyperbolic-sine enclosure.
    pub fn asinh(&self) -> Self {
        crate::asinh(*self)
    }

    /// Returns inverse hyperbolic cosine on its real domain.
    pub fn acosh(&self) -> Self {
        crate::acosh(*self)
    }

    /// Returns inverse hyperbolic tangent on its real domain.
    pub fn atanh(&self) -> Self {
        crate::atanh(*self)
    }

    /// Maps values to their signs.
    pub fn sign(&self) -> Self {
        crate::sign(*self)
    }

    /// Applies the ceiling function pointwise.
    pub fn ceil(&self) -> Self {
        crate::ceil(*self)
    }

    /// Applies the floor function pointwise.
    pub fn floor(&self) -> Self {
        crate::floor(*self)
    }

    /// Applies truncation toward zero pointwise.
    pub fn trunc(&self) -> Self {
        crate::trunc(*self)
    }

    /// Rounds pointwise to nearest integers with ties to even.
    pub fn round_ties_to_even(&self) -> Self {
        crate::round_ties_to_even(*self)
    }

    /// Rounds pointwise to nearest integers with ties away from zero.
    pub fn round_ties_to_away(&self) -> Self {
        crate::round_ties_to_away(*self)
    }

    /// Returns the absolute-value enclosure.
    pub fn abs(&self) -> Self {
        crate::abs(*self)
    }

    /// Returns the pointwise-minimum enclosure with `other`.
    pub fn min(&self, other: &Self) -> Self {
        crate::min(*self, *other)
    }

    /// Returns the pointwise-maximum enclosure with `other`.
    pub fn max(&self, other: &Self) -> Self {
        crate::max(*self, *other)
    }

    /// Encloses `sqrt(self² + other²)`.
    pub fn hypot(&self, other: &Self) -> Self {
        crate::hypot(*self, *other)
    }

    /// Returns the set intersection with `other`.
    pub fn intersection(&self, other: &Self) -> Self {
        crate::intersection(*self, *other)
    }

    /// Returns the smallest interval containing both operands.
    pub fn convex_hull(&self, other: &Self) -> Self {
        crate::convex_hull(*self, *other)
    }

    /// Extends this interval's hull to include `value`.
    pub fn hull_value(&self, value: f64) -> Self {
        crate::convex_hull(*self, Self::from(value))
    }

    /// Splits the interval at its midpoint into two covering intervals.
    pub fn bisect(&self) -> (Self, Self) {
        if self.is_empty() {
            return (*self, *self);
        }
        let midpoint = self.mid();
        (
            Self::new(self.inf(), midpoint),
            Self::new(midpoint, self.sup()),
        )
    }

    /// Returns whether this interval is a subset of `other`.
    pub fn subset(&self, other: &Self) -> bool {
        crate::subset(*self, *other)
    }

    /// Returns whether this interval lies in the interior of `other`.
    pub fn interior(&self, other: &Self) -> bool {
        crate::interior(*self, *other)
    }

    /// Returns whether this interval and `other` are disjoint.
    pub fn disjoint(&self, other: &Self) -> bool {
        crate::disjoint(*self, *other)
    }

    /// Returns the lower endpoint.
    pub fn inf(&self) -> f64 {
        crate::inf(*self)
    }

    /// Returns the upper endpoint.
    pub fn sup(&self) -> f64 {
        crate::sup(*self)
    }

    /// Returns `(lower, upper)` endpoints.
    pub fn bounds(&self) -> (f64, f64) {
        (self.inf(), self.sup())
    }

    /// Returns whether the finite scalar `value` belongs to this interval.
    pub fn contains(&self, value: f64) -> bool {
        value.is_finite() && crate::subset(Self::from(value), *self)
    }

    /// Returns a representative midpoint.
    pub fn mid(&self) -> f64 {
        crate::mid(*self)
    }

    /// Returns the upward-rounded width.
    pub fn wid(&self) -> f64 {
        crate::wid(*self)
    }

    /// Returns the upward-rounded radius.
    pub fn rad(&self) -> f64 {
        crate::rad(*self)
    }

    /// Returns the greatest absolute value in this interval.
    pub fn mag(&self) -> f64 {
        crate::mag(*self)
    }

    /// Returns the least absolute value in this interval.
    pub fn mig(&self) -> f64 {
        crate::mig(*self)
    }

    /// Returns a midpoint and radius that enclose this interval.
    pub fn mid_rad(&self) -> (f64, f64) {
        crate::mid_rad(*self)
    }

    /// Returns whether this is the empty interval.
    pub fn is_empty(&self) -> bool {
        crate::is_empty(*self)
    }

    /// Returns whether this is the entire interval.
    pub fn is_entire(&self) -> bool {
        crate::is_entire(*self)
    }

    /// Returns whether this interval contains exactly one real value.
    pub fn is_singleton(&self) -> bool {
        !self.is_empty() && self.inf() == self.sup()
    }

    /// Returns whether this interval is nonempty with finite endpoints.
    pub fn is_bounded(&self) -> bool {
        self.inf().is_finite() && self.sup().is_finite()
    }

    /// Returns whether this interval has a nonempty intersection with `other`.
    pub fn intersects(&self, other: &Self) -> bool {
        !crate::disjoint(*self, *other)
    }
}

impl PartialEq for Interval {
    fn eq(&self, other: &Self) -> bool {
        self.inf_raw() == other.inf_raw() && self.sup_raw() == other.sup_raw()
    }
}

impl Eq for Interval {}

impl PartialOrd for Interval {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self == other {
            Some(Ordering::Equal)
        } else if crate::subset(*self, *other) {
            Some(Ordering::Less)
        } else if crate::subset(*other, *self) {
            Some(Ordering::Greater)
        } else {
            None
        }
    }
}

impl DecoratedInterval {
    /// Returns the reciprocal enclosure and propagates decorations.
    pub fn recip(&self) -> Self {
        crate::recip(*self)
    }

    /// Returns the square enclosure and propagates decorations.
    pub fn sqr(&self) -> Self {
        crate::sqr(*self)
    }

    /// Returns the square-root enclosure and propagates decorations.
    pub fn sqrt(&self) -> Self {
        crate::sqrt(*self)
    }

    /// Encloses `self * y + z` and propagates decorations.
    pub fn mul_add(&self, y: &Self, z: &Self) -> Self {
        crate::fma(*self, *y, *z)
    }

    /// Raises this interval to the integer power `p`.
    pub fn pown(&self, p: i32) -> Self {
        crate::pown(*self, p)
    }

    /// Alias for [`DecoratedInterval::pown`].
    pub fn powi(&self, i: i32) -> Self {
        self.pown(i)
    }

    /// Encloses powers with exponents in `other`.
    pub fn pow(&self, other: &Self) -> Self {
        crate::pow(*self, *other)
    }

    /// Applies the natural exponential function.
    pub fn exp(&self) -> Self {
        crate::exp(*self)
    }

    /// Applies the base-two exponential function.
    pub fn exp2(&self) -> Self {
        crate::exp2(*self)
    }

    /// Applies the base-ten exponential function.
    pub fn exp10(&self) -> Self {
        crate::exp10(*self)
    }

    /// Applies the natural logarithm on its real domain.
    pub fn log(&self) -> Self {
        crate::log(*self)
    }

    /// Applies the base-two logarithm on its real domain.
    pub fn log2(&self) -> Self {
        crate::log2(*self)
    }

    /// Applies the base-ten logarithm on its real domain.
    pub fn log10(&self) -> Self {
        crate::log10(*self)
    }

    /// Returns the sine enclosure.
    pub fn sin(&self) -> Self {
        crate::sin(*self)
    }

    /// Returns the cosine enclosure.
    pub fn cos(&self) -> Self {
        crate::cos(*self)
    }

    /// Returns the tangent enclosure over defined values.
    pub fn tan(&self) -> Self {
        crate::tan(*self)
    }

    /// Returns the inverse-sine enclosure on `[-1, 1]`.
    pub fn asin(&self) -> Self {
        crate::asin(*self)
    }

    /// Returns the inverse-cosine enclosure on `[-1, 1]`.
    pub fn acos(&self) -> Self {
        crate::acos(*self)
    }

    /// Returns the inverse-tangent enclosure.
    pub fn atan(&self) -> Self {
        crate::atan(*self)
    }

    /// Returns the two-argument angle enclosure `atan2(self, x)`.
    pub fn atan2(&self, x: &Self) -> Self {
        crate::atan2(*self, *x)
    }

    /// Returns the hyperbolic-sine enclosure.
    pub fn sinh(&self) -> Self {
        crate::sinh(*self)
    }

    /// Returns the hyperbolic-cosine enclosure.
    pub fn cosh(&self) -> Self {
        crate::cosh(*self)
    }

    /// Returns the hyperbolic-tangent enclosure.
    pub fn tanh(&self) -> Self {
        crate::tanh(*self)
    }

    /// Returns the inverse-hyperbolic-sine enclosure.
    pub fn asinh(&self) -> Self {
        crate::asinh(*self)
    }

    /// Returns inverse hyperbolic cosine on its real domain.
    pub fn acosh(&self) -> Self {
        crate::acosh(*self)
    }

    /// Returns inverse hyperbolic tangent on its real domain.
    pub fn atanh(&self) -> Self {
        crate::atanh(*self)
    }

    /// Maps values to their signs.
    pub fn sign(&self) -> Self {
        crate::sign(*self)
    }

    /// Applies the ceiling function pointwise.
    pub fn ceil(&self) -> Self {
        crate::ceil(*self)
    }

    /// Applies the floor function pointwise.
    pub fn floor(&self) -> Self {
        crate::floor(*self)
    }

    /// Applies truncation toward zero pointwise.
    pub fn trunc(&self) -> Self {
        crate::trunc(*self)
    }

    /// Rounds pointwise to nearest integers with ties to even.
    pub fn round_ties_to_even(&self) -> Self {
        crate::round_ties_to_even(*self)
    }

    /// Rounds pointwise to nearest integers with ties away from zero.
    pub fn round_ties_to_away(&self) -> Self {
        crate::round_ties_to_away(*self)
    }

    /// Returns the absolute-value enclosure.
    pub fn abs(&self) -> Self {
        crate::abs(*self)
    }

    /// Returns the pointwise-minimum enclosure with `other`.
    pub fn min(&self, other: &Self) -> Self {
        crate::min(*self, *other)
    }

    /// Returns the pointwise-maximum enclosure with `other`.
    pub fn max(&self, other: &Self) -> Self {
        crate::max(*self, *other)
    }

    /// Encloses `sqrt(self² + other²)`.
    pub fn hypot(&self, other: &Self) -> Self {
        crate::hypot(*self, *other)
    }

    /// Returns the set intersection with `other`.
    pub fn intersection(&self, other: &Self) -> Self {
        crate::intersection(*self, *other)
    }

    /// Returns the smallest interval containing both operands.
    pub fn convex_hull(&self, other: &Self) -> Self {
        crate::convex_hull(*self, *other)
    }

    /// Extends this interval's hull to include `value`.
    pub fn hull_value(&self, value: f64) -> Self {
        crate::convex_hull(*self, Self::from(value))
    }

    /// Splits the interval at its midpoint into two covering intervals.
    pub fn bisect(&self) -> (Self, Self) {
        if self.is_nai() || self.is_empty() {
            return (*self, *self);
        }
        let decoration = self.decoration();
        let midpoint = self.mid();
        let left = Interval::new(self.inf(), midpoint);
        let right = Interval::new(midpoint, self.sup());
        (
            crate::set_dec(left, decoration),
            crate::set_dec(right, decoration),
        )
    }

    /// Returns whether this interval is a subset of `other`.
    pub fn subset(&self, other: &Self) -> bool {
        crate::subset(*self, *other)
    }

    /// Returns whether this interval lies in the interior of `other`.
    pub fn interior(&self, other: &Self) -> bool {
        crate::interior(*self, *other)
    }

    /// Returns whether this interval and `other` are disjoint.
    pub fn disjoint(&self, other: &Self) -> bool {
        crate::disjoint(*self, *other)
    }

    /// Returns the lower endpoint, or NaN for NaI.
    pub fn inf(&self) -> f64 {
        crate::inf(*self)
    }

    /// Returns the upper endpoint, or NaN for NaI.
    pub fn sup(&self) -> f64 {
        crate::sup(*self)
    }

    /// Returns `(lower, upper)` endpoints.
    pub fn bounds(&self) -> (f64, f64) {
        (self.inf(), self.sup())
    }

    /// Returns whether the finite scalar `value` belongs to this interval.
    pub fn contains(&self, value: f64) -> bool {
        value.is_finite() && crate::subset(Self::from(value), *self)
    }

    /// Returns a representative midpoint.
    pub fn mid(&self) -> f64 {
        crate::mid(*self)
    }

    /// Returns the upward-rounded width.
    pub fn wid(&self) -> f64 {
        crate::wid(*self)
    }

    /// Returns the upward-rounded radius.
    pub fn rad(&self) -> f64 {
        crate::rad(*self)
    }

    /// Returns the greatest absolute value in this interval.
    pub fn mag(&self) -> f64 {
        crate::mag(*self)
    }

    /// Returns the least absolute value in this interval.
    pub fn mig(&self) -> f64 {
        crate::mig(*self)
    }

    /// Returns a midpoint and radius that enclose this interval.
    pub fn mid_rad(&self) -> (f64, f64) {
        crate::mid_rad(*self)
    }

    /// Returns whether this is the empty interval; NaI is not empty.
    pub fn is_empty(&self) -> bool {
        crate::is_empty(*self)
    }

    /// Returns whether this is the entire interval.
    pub fn is_entire(&self) -> bool {
        crate::is_entire(*self)
    }

    /// Returns whether this value is Not an Interval.
    pub fn is_nai(&self) -> bool {
        crate::is_nai(*self)
    }

    /// Returns whether this interval contains exactly one real value.
    pub fn is_singleton(&self) -> bool {
        !self.is_nai() && !self.is_empty() && self.inf() == self.sup()
    }

    /// Returns whether this interval is nonempty with finite endpoints.
    pub fn is_bounded(&self) -> bool {
        self.inf().is_finite() && self.sup().is_finite()
    }

    /// Returns this interval's decoration.
    pub fn decoration(&self) -> crate::Decoration {
        crate::decoration_part(*self)
    }

    /// Returns whether this interval has a nonempty intersection with `other`.
    pub fn intersects(&self, other: &Self) -> bool {
        !crate::disjoint(*self, *other)
    }
}

impl PartialEq for DecoratedInterval {
    fn eq(&self, other: &Self) -> bool {
        self.interval_raw() == other.interval_raw()
            && self.decoration_raw() == other.decoration_raw()
    }
}

impl Eq for DecoratedInterval {}

impl PartialOrd for DecoratedInterval {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self == other {
            return Some(Ordering::Equal);
        }

        let left = self.interval_raw();
        let right = other.interval_raw();
        if left == right {
            None
        } else if crate::subset(left, right) {
            Some(Ordering::Less)
        } else if crate::subset(right, left) {
            Some(Ordering::Greater)
        } else {
            None
        }
    }
}
