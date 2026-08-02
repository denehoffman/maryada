use core::{cmp::Ordering, fmt, ops, str::FromStr};
use impl_ops::*;

use crate::{DecoratedInterval, Interval, ParseIntervalError, Signal, SignalFlags, SignalSink};

impl Interval {
    pub fn new(inf: f64, sup: f64) -> Self {
        Self::nums_to_interval(inf, sup, &mut ())
    }
    pub fn decorate<S: SignalSink>(&self, signals: &mut S) -> DecoratedInterval {
        match Interval::from_nums(self.inf_raw(), self.sup_raw()) {
            Some(value) => DecoratedInterval::new_dec_raw(value),
            None => {
                signals.raise(Signal::UndefinedOperation);
                DecoratedInterval::NAI
            }
        }
    }

    pub fn to_be_bytes(self) -> [u8; crate::INTERVAL_ENCODED_LEN] {
        crate::interval_to_be_bytes(self)
    }

    pub fn to_le_bytes(self) -> [u8; crate::INTERVAL_ENCODED_LEN] {
        crate::interval_to_le_bytes(self)
    }

    pub fn from_be_bytes<S: SignalSink>(bytes: &[u8], signals: &mut S) -> Self {
        crate::interval_from_be_bytes(bytes, signals)
    }

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
    pub fn new(inf: f64, sup: f64) -> Self {
        Self::nums_to_interval(inf, sup, &mut ())
    }

    pub fn to_be_bytes(self) -> [u8; crate::DECORATED_INTERVAL_ENCODED_LEN] {
        crate::decorated_interval_to_be_bytes(self)
    }

    pub fn to_le_bytes(self) -> [u8; crate::DECORATED_INTERVAL_ENCODED_LEN] {
        crate::decorated_interval_to_le_bytes(self)
    }

    pub fn from_be_bytes<S: SignalSink>(bytes: &[u8], signals: &mut S) -> Self {
        crate::decorated_interval_from_be_bytes(bytes, signals)
    }

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
    pub fn recip(&self) -> Self {
        crate::recip(*self)
    }

    pub fn sqr(&self) -> Self {
        crate::sqr(*self)
    }

    pub fn sqrt(&self) -> Self {
        crate::sqrt(*self)
    }

    pub fn mul_add(&self, y: &Self, z: &Self) -> Self {
        crate::fma(*self, *y, *z)
    }

    pub fn pown(&self, p: i32) -> Self {
        crate::pown(*self, p)
    }

    pub fn powi(&self, i: i32) -> Self {
        self.pown(i)
    }

    pub fn pow(&self, other: &Self) -> Self {
        crate::pow(*self, *other)
    }

    pub fn exp(&self) -> Self {
        crate::exp(*self)
    }

    pub fn exp2(&self) -> Self {
        crate::exp2(*self)
    }

    pub fn exp10(&self) -> Self {
        crate::exp10(*self)
    }

    pub fn log(&self) -> Self {
        crate::log(*self)
    }

    pub fn log2(&self) -> Self {
        crate::log2(*self)
    }

    pub fn log10(&self) -> Self {
        crate::log10(*self)
    }

    pub fn sin(&self) -> Self {
        crate::sin(*self)
    }

    pub fn cos(&self) -> Self {
        crate::cos(*self)
    }

    pub fn tan(&self) -> Self {
        crate::tan(*self)
    }

    pub fn asin(&self) -> Self {
        crate::asin(*self)
    }

    pub fn acos(&self) -> Self {
        crate::acos(*self)
    }

    pub fn atan(&self) -> Self {
        crate::atan(*self)
    }

    pub fn atan2(&self, x: &Self) -> Self {
        crate::atan2(*self, *x)
    }

    pub fn sinh(&self) -> Self {
        crate::sinh(*self)
    }

    pub fn cosh(&self) -> Self {
        crate::cosh(*self)
    }

    pub fn tanh(&self) -> Self {
        crate::tanh(*self)
    }

    pub fn asinh(&self) -> Self {
        crate::asinh(*self)
    }

    pub fn acosh(&self) -> Self {
        crate::acosh(*self)
    }

    pub fn atanh(&self) -> Self {
        crate::atanh(*self)
    }

    pub fn sign(&self) -> Self {
        crate::sign(*self)
    }

    pub fn ceil(&self) -> Self {
        crate::ceil(*self)
    }

    pub fn floor(&self) -> Self {
        crate::floor(*self)
    }

    pub fn trunc(&self) -> Self {
        crate::trunc(*self)
    }

    pub fn round_ties_to_even(&self) -> Self {
        crate::round_ties_to_even(*self)
    }

    pub fn round_ties_to_away(&self) -> Self {
        crate::round_ties_to_away(*self)
    }

    pub fn abs(&self) -> Self {
        crate::abs(*self)
    }

    pub fn min(&self, other: &Self) -> Self {
        crate::min(*self, *other)
    }

    pub fn max(&self, other: &Self) -> Self {
        crate::max(*self, *other)
    }

    pub fn hypot(&self, other: &Self) -> Self {
        crate::hypot(*self, *other)
    }

    pub fn intersection(&self, other: &Self) -> Self {
        crate::intersection(*self, *other)
    }

    pub fn convex_hull(&self, other: &Self) -> Self {
        crate::convex_hull(*self, *other)
    }

    pub fn hull_value(&self, value: f64) -> Self {
        crate::convex_hull(*self, Self::from(value))
    }

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

    pub fn subset(&self, other: &Self) -> bool {
        crate::subset(*self, *other)
    }

    pub fn interior(&self, other: &Self) -> bool {
        crate::interior(*self, *other)
    }

    pub fn disjoint(&self, other: &Self) -> bool {
        crate::disjoint(*self, *other)
    }

    pub fn inf(&self) -> f64 {
        crate::inf(*self)
    }

    pub fn sup(&self) -> f64 {
        crate::sup(*self)
    }

    pub fn bounds(&self) -> (f64, f64) {
        (self.inf(), self.sup())
    }

    pub fn contains(&self, value: f64) -> bool {
        value.is_finite() && crate::subset(Self::from(value), *self)
    }

    pub fn mid(&self) -> f64 {
        crate::mid(*self)
    }

    pub fn wid(&self) -> f64 {
        crate::wid(*self)
    }

    pub fn rad(&self) -> f64 {
        crate::rad(*self)
    }

    pub fn mag(&self) -> f64 {
        crate::mag(*self)
    }

    pub fn mig(&self) -> f64 {
        crate::mig(*self)
    }

    pub fn mid_rad(&self) -> (f64, f64) {
        crate::mid_rad(*self)
    }

    pub fn is_empty(&self) -> bool {
        crate::is_empty(*self)
    }

    pub fn is_entire(&self) -> bool {
        crate::is_entire(*self)
    }

    pub fn is_singleton(&self) -> bool {
        !self.is_empty() && self.inf() == self.sup()
    }

    pub fn is_bounded(&self) -> bool {
        self.inf().is_finite() && self.sup().is_finite()
    }

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
    pub fn recip(&self) -> Self {
        crate::recip(*self)
    }

    pub fn sqr(&self) -> Self {
        crate::sqr(*self)
    }

    pub fn sqrt(&self) -> Self {
        crate::sqrt(*self)
    }

    pub fn mul_add(&self, y: &Self, z: &Self) -> Self {
        crate::fma(*self, *y, *z)
    }

    pub fn pown(&self, p: i32) -> Self {
        crate::pown(*self, p)
    }

    pub fn powi(&self, i: i32) -> Self {
        self.pown(i)
    }

    pub fn pow(&self, other: &Self) -> Self {
        crate::pow(*self, *other)
    }

    pub fn exp(&self) -> Self {
        crate::exp(*self)
    }

    pub fn exp2(&self) -> Self {
        crate::exp2(*self)
    }

    pub fn exp10(&self) -> Self {
        crate::exp10(*self)
    }

    pub fn log(&self) -> Self {
        crate::log(*self)
    }

    pub fn log2(&self) -> Self {
        crate::log2(*self)
    }

    pub fn log10(&self) -> Self {
        crate::log10(*self)
    }

    pub fn sin(&self) -> Self {
        crate::sin(*self)
    }

    pub fn cos(&self) -> Self {
        crate::cos(*self)
    }

    pub fn tan(&self) -> Self {
        crate::tan(*self)
    }

    pub fn asin(&self) -> Self {
        crate::asin(*self)
    }

    pub fn acos(&self) -> Self {
        crate::acos(*self)
    }

    pub fn atan(&self) -> Self {
        crate::atan(*self)
    }

    pub fn atan2(&self, x: &Self) -> Self {
        crate::atan2(*self, *x)
    }

    pub fn sinh(&self) -> Self {
        crate::sinh(*self)
    }

    pub fn cosh(&self) -> Self {
        crate::cosh(*self)
    }

    pub fn tanh(&self) -> Self {
        crate::tanh(*self)
    }

    pub fn asinh(&self) -> Self {
        crate::asinh(*self)
    }

    pub fn acosh(&self) -> Self {
        crate::acosh(*self)
    }

    pub fn atanh(&self) -> Self {
        crate::atanh(*self)
    }

    pub fn sign(&self) -> Self {
        crate::sign(*self)
    }

    pub fn ceil(&self) -> Self {
        crate::ceil(*self)
    }

    pub fn floor(&self) -> Self {
        crate::floor(*self)
    }

    pub fn trunc(&self) -> Self {
        crate::trunc(*self)
    }

    pub fn round_ties_to_even(&self) -> Self {
        crate::round_ties_to_even(*self)
    }

    pub fn round_ties_to_away(&self) -> Self {
        crate::round_ties_to_away(*self)
    }

    pub fn abs(&self) -> Self {
        crate::abs(*self)
    }

    pub fn min(&self, other: &Self) -> Self {
        crate::min(*self, *other)
    }

    pub fn max(&self, other: &Self) -> Self {
        crate::max(*self, *other)
    }

    pub fn hypot(&self, other: &Self) -> Self {
        crate::hypot(*self, *other)
    }

    pub fn intersection(&self, other: &Self) -> Self {
        crate::intersection(*self, *other)
    }

    pub fn convex_hull(&self, other: &Self) -> Self {
        crate::convex_hull(*self, *other)
    }

    pub fn hull_value(&self, value: f64) -> Self {
        crate::convex_hull(*self, Self::from(value))
    }

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

    pub fn subset(&self, other: &Self) -> bool {
        crate::subset(*self, *other)
    }

    pub fn interior(&self, other: &Self) -> bool {
        crate::interior(*self, *other)
    }

    pub fn disjoint(&self, other: &Self) -> bool {
        crate::disjoint(*self, *other)
    }

    pub fn inf(&self) -> f64 {
        crate::inf(*self)
    }

    pub fn sup(&self) -> f64 {
        crate::sup(*self)
    }

    pub fn bounds(&self) -> (f64, f64) {
        (self.inf(), self.sup())
    }

    pub fn contains(&self, value: f64) -> bool {
        value.is_finite() && crate::subset(Self::from(value), *self)
    }

    pub fn mid(&self) -> f64 {
        crate::mid(*self)
    }

    pub fn wid(&self) -> f64 {
        crate::wid(*self)
    }

    pub fn rad(&self) -> f64 {
        crate::rad(*self)
    }

    pub fn mag(&self) -> f64 {
        crate::mag(*self)
    }

    pub fn mig(&self) -> f64 {
        crate::mig(*self)
    }

    pub fn mid_rad(&self) -> (f64, f64) {
        crate::mid_rad(*self)
    }

    pub fn is_empty(&self) -> bool {
        crate::is_empty(*self)
    }

    pub fn is_entire(&self) -> bool {
        crate::is_entire(*self)
    }

    pub fn is_nai(&self) -> bool {
        crate::is_nai(*self)
    }

    pub fn is_singleton(&self) -> bool {
        !self.is_nai() && !self.is_empty() && self.inf() == self.sup()
    }

    pub fn is_bounded(&self) -> bool {
        self.inf().is_finite() && self.sup().is_finite()
    }

    pub fn decoration(&self) -> crate::Decoration {
        crate::decoration_part(*self)
    }

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

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;

    #[test]
    fn constructors_reject_invalid_bounds() {
        assert!(crate::new::<Interval>(2.0, 1.0).is_empty());
        assert!(crate::singleton::<Interval>(f64::NAN).is_empty());
        assert!(crate::new::<DecoratedInterval>(2.0, 1.0).is_nai());
        assert!(crate::singleton::<DecoratedInterval>(f64::NAN).is_nai());
    }

    #[test]
    fn partial_order_represents_set_containment() {
        let small = Interval::new(1.0, 2.0);
        let large = Interval::new(0.0, 3.0);
        let disjoint = Interval::new(4.0, 5.0);

        assert_eq!(small.partial_cmp(&large), Some(Ordering::Less));
        assert_eq!(large.partial_cmp(&small), Some(Ordering::Greater));
        assert_eq!(small.partial_cmp(&disjoint), None);
        assert_eq!(disjoint.partial_cmp(&small), None);
        assert_eq!(DecoratedInterval::NAI, DecoratedInterval::NAI);
    }

    #[test]
    fn convenience_methods_delegate_to_the_correct_operations() {
        let left = Interval::new(1.0, 2.0);
        let right = Interval::new(3.0, 4.0);

        assert_eq!(left.max(&right).bounds(), (3.0, 4.0));
        assert!(left.contains(1.5));
        assert!(!left.contains(f64::NAN));
        assert!(left.disjoint(&right));

        let hull = left.hull_value(5.0);
        assert_eq!(hull.bounds(), (1.0, 5.0));
        let (first, second) = hull.bisect();
        assert_eq!(first.sup(), second.inf());
        assert_eq!(first.convex_hull(&second), hull);
    }

    #[test]
    fn text_and_byte_conveniences_round_trip() {
        let value = Interval::new(-1.25, 3.5);
        let text = std::format!("{value}");
        let parsed: Interval = text.parse().unwrap();
        assert_eq!(parsed, value);
        assert!("not an interval".parse::<Interval>().is_err());

        let bytes = value.to_be_bytes();
        assert_eq!(Interval::from_be_bytes(&bytes, &mut ()), value);
    }
}
