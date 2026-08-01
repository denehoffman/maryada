use core::{cmp::Ordering, ops};
use impl_ops::*;

use crate::{DecoratedInterval, Interval, Signal, SignalSink};

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
}

impl From<(&f64, &f64)> for Interval {
    fn from(value: (&f64, &f64)) -> Self {
        Self::new(*value.0, *value.1)
    }
}

impl From<(&f64, f64)> for Interval {
    fn from(value: (&f64, f64)) -> Self {
        Self::new(*value.0, value.1)
    }
}

impl From<(f64, &f64)> for Interval {
    fn from(value: (f64, &f64)) -> Self {
        Self::new(value.0, *value.1)
    }
}

impl From<(f64, f64)> for Interval {
    fn from(value: (f64, f64)) -> Self {
        Self::new(value.0, value.1)
    }
}

impl From<&f64> for Interval {
    fn from(value: &f64) -> Self {
        Self::new(*value, *value)
    }
}

impl From<f64> for Interval {
    fn from(value: f64) -> Self {
        Self::new(value, value)
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

impl DecoratedInterval {
    pub fn new(inf: f64, sup: f64) -> Self {
        Interval::new(inf, sup).decorate(&mut ())
    }
}

impl From<(&f64, &f64)> for DecoratedInterval {
    fn from(value: (&f64, &f64)) -> Self {
        Self::new(*value.0, *value.1)
    }
}

impl From<(&f64, f64)> for DecoratedInterval {
    fn from(value: (&f64, f64)) -> Self {
        Self::new(*value.0, value.1)
    }
}

impl From<(f64, &f64)> for DecoratedInterval {
    fn from(value: (f64, &f64)) -> Self {
        Self::new(value.0, *value.1)
    }
}

impl From<(f64, f64)> for DecoratedInterval {
    fn from(value: (f64, f64)) -> Self {
        Self::new(value.0, value.1)
    }
}

impl From<&f64> for DecoratedInterval {
    fn from(value: &f64) -> Self {
        Self::new(*value, *value)
    }
}

impl From<f64> for DecoratedInterval {
    fn from(value: f64) -> Self {
        Self::new(value, value)
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

    pub fn abs(&self) -> Self {
        crate::abs(*self)
    }

    pub fn min(&self, other: &Self) -> Self {
        crate::min(*self, *other)
    }

    pub fn max(&self, other: &Self) -> Self {
        crate::min(*self, *other)
    }

    pub fn intersection(&self, other: &Self) -> Self {
        crate::intersection(*self, *other)
    }

    pub fn convex_hull(&self, other: &Self) -> Self {
        crate::convex_hull(*self, *other)
    }

    pub fn inf(&self) -> f64 {
        crate::inf(*self)
    }

    pub fn sup(&self) -> f64 {
        crate::sup(*self)
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

    pub fn intersects(&self, other: &Self) -> bool {
        !crate::disjoint(*self, *other)
    }
}

impl PartialEq for Interval {
    fn eq(&self, other: &Self) -> bool {
        crate::equal(*self, *other)
    }
}

impl Eq for Interval {}

impl PartialOrd for Interval {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Interval {
    fn cmp(&self, other: &Self) -> Ordering {
        if crate::equal(*self, *other) {
            Ordering::Equal
        } else if crate::interior(*self, *other) {
            Ordering::Less
        } else {
            Ordering::Greater
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

    pub fn abs(&self) -> Self {
        crate::abs(*self)
    }

    pub fn min(&self, other: &Self) -> Self {
        crate::min(*self, *other)
    }

    pub fn max(&self, other: &Self) -> Self {
        crate::min(*self, *other)
    }

    pub fn intersection(&self, other: &Self) -> Self {
        crate::intersection(*self, *other)
    }

    pub fn convex_hull(&self, other: &Self) -> Self {
        crate::convex_hull(*self, *other)
    }

    pub fn inf(&self) -> f64 {
        crate::inf(*self)
    }

    pub fn sup(&self) -> f64 {
        crate::sup(*self)
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

    pub fn intersects(&self, other: &Self) -> bool {
        !crate::disjoint(*self, *other)
    }
}

impl PartialEq for DecoratedInterval {
    fn eq(&self, other: &Self) -> bool {
        crate::equal(*self, *other)
    }
}

impl Eq for DecoratedInterval {}

impl PartialOrd for DecoratedInterval {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DecoratedInterval {
    fn cmp(&self, other: &Self) -> Ordering {
        if crate::equal(*self, *other) {
            Ordering::Equal
        } else if crate::interior(*self, *other) {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    }
}
