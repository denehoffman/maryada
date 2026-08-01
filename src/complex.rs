use core::{fmt, ops};
use impl_ops::*;
use num_complex::Complex64;

use crate::{DecoratedInterval, Decoration, Interval, IntervalDatum, SignalSink};

#[derive(Copy, Clone, Debug)]
pub struct ComplexBox<I> {
    pub re: I,
    pub im: I,
}

impl<I: IntervalDatum> ComplexBox<I> {
    pub const fn new(re: I, im: I) -> Self {
        Self { re, im }
    }

    pub fn i() -> Self {
        Self::new(crate::zero(), crate::singleton(1.0))
    }

    fn is_empty_raw(&self) -> bool {
        crate::is_empty(self.re) || crate::is_empty(self.im)
    }

    fn empty_raw() -> Self {
        Self::new(I::__empty(), I::__empty())
    }

    fn neg_raw(&self) -> Self {
        Self::new(crate::neg(self.re), crate::neg(self.im))
    }

    fn add_raw(&self, other: &Self) -> Self {
        Self::new(crate::add(self.re, other.re), crate::add(self.im, other.im))
    }

    fn sub_raw(&self, other: &Self) -> Self {
        Self::new(crate::sub(self.re, other.re), crate::sub(self.im, other.im))
    }

    fn mul_raw(&self, other: &Self) -> Self {
        Self::new(
            crate::sub(crate::mul(self.re, other.re), crate::mul(self.im, other.im)),
            crate::add(crate::mul(self.re, other.im), crate::mul(self.im, other.re)),
        )
    }

    fn div_raw(&self, other: &Self) -> Self {
        let denom = crate::add(crate::sqr(other.re), crate::sqr(other.im));
        Self::new(
            crate::div(
                crate::add(crate::mul(self.re, other.re), crate::mul(self.im, other.im)),
                denom,
            ),
            crate::div(
                crate::sub(crate::mul(self.im, other.re), crate::mul(self.re, other.im)),
                denom,
            ),
        )
    }

    pub fn add_real(&self, value: I) -> Self {
        Self::new(crate::add(self.re, value), self.im)
    }

    pub fn sub_real(&self, value: I) -> Self {
        Self::new(crate::sub(self.re, value), self.im)
    }

    pub fn scale(&self, value: I) -> Self {
        Self::new(crate::mul(self.re, value), crate::mul(self.im, value))
    }

    pub fn div_real(&self, value: I) -> Self {
        Self::new(crate::div(self.re, value), crate::div(self.im, value))
    }

    pub fn conj(&self) -> Self {
        Self::new(self.re, crate::neg(self.im))
    }

    pub fn norm_sqr(&self) -> I {
        crate::add(crate::sqr(self.re), crate::sqr(self.im))
    }

    pub fn abs(&self) -> I {
        crate::hypot(self.re, self.im)
    }

    pub fn norm(&self) -> I {
        self.abs()
    }

    pub fn mid(&self) -> Complex64 {
        Complex64::new(crate::mid(self.re), crate::mid(self.im))
    }

    pub fn wid_box(&self) -> Complex64 {
        Complex64::new(crate::wid(self.re), crate::wid(self.im))
    }

    pub fn wid(&self) -> Complex64 {
        self.wid_box()
    }

    pub fn rad_box(&self) -> Complex64 {
        Complex64::new(crate::rad(self.re), crate::rad(self.im))
    }

    pub fn rad(&self) -> f64 {
        let re = crate::rad(self.re);
        let im = crate::rad(self.im);
        if re.is_nan() || im.is_nan() {
            return f64::NAN;
        }
        crate::sup(crate::hypot::<Interval>(re.into(), im.into()))
    }

    pub fn diameter(&self) -> f64 {
        let widths = self.wid_box();
        if widths.re.is_nan() || widths.im.is_nan() {
            return f64::NAN;
        }
        crate::sup(crate::hypot::<Interval>(widths.re.into(), widths.im.into()))
    }

    pub fn mag(&self) -> f64 {
        crate::sup(self.abs())
    }

    pub fn mig(&self) -> f64 {
        crate::inf(self.abs())
    }

    pub fn arg(&self) -> I {
        crate::atan2(self.im, self.re)
    }

    pub fn to_polar(&self) -> (I, I) {
        (self.abs(), self.arg())
    }

    pub fn cis(theta: I) -> Self {
        Self::new(crate::cos(theta), crate::sin(theta))
    }

    pub fn from_polar(radius: I, theta: I) -> Self {
        Self::new(
            crate::mul(radius, crate::cos(theta)),
            crate::mul(radius, crate::sin(theta)),
        )
    }

    pub fn lower_corner(&self) -> Complex64 {
        Complex64::new(crate::inf(self.re), crate::inf(self.im))
    }

    pub fn upper_corner(&self) -> Complex64 {
        Complex64::new(crate::sup(self.re), crate::sup(self.im))
    }

    pub fn corners(&self) -> [Complex64; 4] {
        let lower = self.lower_corner();
        let upper = self.upper_corner();
        [
            lower,
            Complex64::new(upper.re, lower.im),
            upper,
            Complex64::new(lower.re, upper.im),
        ]
    }

    pub fn mid_rad(&self) -> (Complex64, Complex64) {
        (self.mid(), self.rad_box())
    }

    pub fn is_empty(&self) -> bool {
        self.is_empty_raw()
    }

    pub fn is_entire(&self) -> bool {
        crate::is_entire(self.re) && crate::is_entire(self.im)
    }

    pub fn is_nai(&self) -> bool {
        self.re.__is_nai() || self.im.__is_nai()
    }

    pub fn is_singleton(&self) -> bool {
        !self.is_nai()
            && !self.is_empty()
            && crate::inf(self.re) == crate::sup(self.re)
            && crate::inf(self.im) == crate::sup(self.im)
    }

    pub fn is_bounded(&self) -> bool {
        self.lower_corner().re.is_finite()
            && self.lower_corner().im.is_finite()
            && self.upper_corner().re.is_finite()
            && self.upper_corner().im.is_finite()
    }

    pub fn is_zero(&self) -> bool {
        self.is_singleton() && crate::inf(self.re) == 0.0 && crate::inf(self.im) == 0.0
    }

    pub fn is_real(&self) -> bool {
        !self.is_empty()
            && !self.is_nai()
            && crate::inf(self.im) == 0.0
            && crate::sup(self.im) == 0.0
    }

    pub fn contains(&self, value: Complex64) -> bool {
        !value.re.is_nan()
            && !value.im.is_nan()
            && crate::subset(crate::singleton(value.re), self.re)
            && crate::subset(crate::singleton(value.im), self.im)
    }

    pub fn subset(&self, other: &Self) -> bool {
        if self.is_nai() || other.is_nai() {
            return false;
        }
        if self.is_empty() {
            return true;
        }
        if other.is_empty() {
            return false;
        }
        crate::subset(self.re, other.re) && crate::subset(self.im, other.im)
    }

    pub fn interior(&self, other: &Self) -> bool {
        if self.is_nai() || other.is_nai() {
            return false;
        }
        if self.is_empty() {
            return true;
        }
        if other.is_empty() {
            return false;
        }
        crate::interior(self.re, other.re) && crate::interior(self.im, other.im)
    }

    pub fn disjoint(&self, other: &Self) -> bool {
        crate::disjoint(self.re, other.re) || crate::disjoint(self.im, other.im)
    }

    pub fn intersects(&self, other: &Self) -> bool {
        !self.disjoint(other)
    }

    pub fn intersection(&self, other: &Self) -> Self {
        let result = Self::new(
            crate::intersection(self.re, other.re),
            crate::intersection(self.im, other.im),
        );
        if result.is_empty() {
            Self::empty_raw()
        } else {
            result
        }
    }

    pub fn convex_hull(&self, other: &Self) -> Self {
        if self.is_empty() {
            return *other;
        }
        if other.is_empty() {
            return *self;
        }
        Self::new(
            crate::convex_hull(self.re, other.re),
            crate::convex_hull(self.im, other.im),
        )
    }

    pub fn recip(&self) -> Self {
        let denom = crate::add(crate::sqr(self.re), crate::sqr(self.im));
        Self::new(
            crate::div(self.re, denom),
            crate::div(crate::neg(self.im), denom),
        )
    }

    pub fn sqr(&self) -> Self {
        Self::new(
            crate::sub(crate::sqr(self.re), crate::sqr(self.im)),
            crate::mul(crate::singleton(2.0), crate::mul(self.re, self.im)),
        )
    }

    pub fn sqrt(&self) -> Self {
        Self::from(0.5).mul_raw(&self.log()).exp()
    }

    pub fn mul_add(&self, y: &Self, z: &Self) -> Self {
        Self::new(
            crate::fma(self.re, y.re, crate::fma(crate::neg(self.im), y.im, z.re)),
            crate::fma(self.re, y.im, crate::fma(self.im, y.re, z.im)),
        )
    }

    pub fn pown(&self, p: i32) -> Self {
        if self.is_empty_raw() {
            return Self::empty_raw();
        }
        if p == 0 {
            return Self::from(1.0);
        }
        let mut base = if p < 0 { self.recip() } else { *self };
        let mut exponent = p.unsigned_abs();
        let mut result = Self::from(1.0);
        while exponent != 0 {
            if exponent & 1 != 0 {
                result = result.mul_raw(&base);
            }
            exponent >>= 1;
            if exponent != 0 {
                base = base.sqr()
            }
        }
        result
    }

    pub fn powi(&self, i: i32) -> Self {
        self.pown(i)
    }

    pub fn pow(&self, other: &Self) -> Self {
        other.mul_raw(&self.log()).exp()
    }

    pub fn exp(&self) -> Self {
        Self::new(
            crate::mul(crate::exp(self.re), crate::cos(self.im)),
            crate::mul(crate::exp(self.re), crate::sin(self.im)),
        )
    }

    pub fn exp2(&self) -> Self {
        self.mul_raw(&Self::from(crate::log(crate::singleton::<I>(2.0))))
            .exp()
    }

    pub fn exp10(&self) -> Self {
        self.mul_raw(&Self::from(crate::log(crate::singleton::<I>(10.0))))
            .exp()
    }

    pub fn log(&self) -> Self {
        Self::new(crate::log(self.abs()), self.arg())
    }

    pub fn log2(&self) -> Self {
        self.log()
            .div_raw(&Self::from(crate::log(crate::singleton::<I>(2.0))))
    }

    pub fn log10(&self) -> Self {
        self.log()
            .div_raw(&Self::from(crate::log(crate::singleton::<I>(10.0))))
    }

    pub fn sin(&self) -> Self {
        Self::new(
            crate::mul(crate::sin(self.re), crate::cosh(self.im)),
            crate::mul(crate::cos(self.re), crate::sinh(self.im)),
        )
    }

    pub fn cos(&self) -> Self {
        Self::new(
            crate::mul(crate::cos(self.re), crate::cosh(self.im)),
            crate::neg(crate::mul(crate::sin(self.re), crate::sinh(self.im))),
        )
    }

    pub fn tan(&self) -> Self {
        let denom = crate::add(
            crate::cos(crate::mul(crate::singleton(2.0), self.re)),
            crate::cosh(crate::mul(crate::singleton(2.0), self.im)),
        );
        Self::new(
            crate::div(
                crate::sin(crate::mul(crate::singleton(2.0), self.re)),
                denom,
            ),
            crate::div(
                crate::sinh(crate::mul(crate::singleton(2.0), self.im)),
                denom,
            ),
        )
    }

    pub fn asin(&self) -> Self {
        Self::i().neg_raw().mul_raw(&Self::log(
            &Self::i()
                .mul_raw(self)
                .add_raw(&((Self::from(1.0).sub_raw(&self.sqr())).sqrt())),
        ))
    }

    pub fn acos(&self) -> Self {
        Self::i().neg_raw().mul_raw(&Self::log(
            &self.add_raw(&Self::i().mul_raw(&Self::from(1.0).sub_raw(&self.sqr()).sqrt())),
        ))
    }

    pub fn atan(&self) -> Self {
        let iz = Self::i().mul_raw(self);
        Self::i().div_raw(&Self::from(2.0)).mul_raw(
            &Self::from(1.0)
                .sub_raw(&iz)
                .log()
                .sub_raw(&Self::from(1.0).add_raw(&iz).log()),
        )
    }

    pub fn sinh(&self) -> Self {
        Self::new(
            crate::mul(crate::sinh(self.re), crate::cos(self.im)),
            crate::mul(crate::cosh(self.re), crate::sin(self.im)),
        )
    }

    pub fn cosh(&self) -> Self {
        Self::new(
            crate::mul(crate::cosh(self.re), crate::cos(self.im)),
            crate::mul(crate::sinh(self.re), crate::sin(self.im)),
        )
    }

    pub fn tanh(&self) -> Self {
        let denom = crate::add(
            crate::cosh(crate::mul(crate::singleton(2.0), self.re)),
            crate::cos(crate::mul(crate::singleton(2.0), self.im)),
        );
        Self::new(
            crate::div(
                crate::sinh(crate::mul(crate::singleton(2.0), self.re)),
                denom,
            ),
            crate::div(
                crate::sin(crate::mul(crate::singleton(2.0), self.im)),
                denom,
            ),
        )
    }

    pub fn asinh(&self) -> Self {
        self.add_raw(&(self.sqr().add_raw(&Self::from(1.0))).sqrt())
            .log()
    }

    pub fn acosh(&self) -> Self {
        self.add_raw(
            &self
                .sub_raw(&Self::from(1.0))
                .sqrt()
                .mul_raw(&self.add_raw(&Self::from(1.0)).sqrt()),
        )
        .log()
    }

    pub fn atanh(&self) -> Self {
        Self::from(0.5).mul_raw(
            &self
                .add_raw(&Self::from(1.0))
                .log()
                .sub_raw(&Self::from(1.0).sub_raw(self).log()),
        )
    }
}

impl<I: IntervalDatum> From<I> for ComplexBox<I> {
    fn from(value: I) -> Self {
        Self::new(value, crate::new(0.0, 0.0))
    }
}

impl<I: IntervalDatum> From<&I> for ComplexBox<I> {
    fn from(value: &I) -> Self {
        (*value).into()
    }
}

impl<I: IntervalDatum> From<f64> for ComplexBox<I> {
    fn from(value: f64) -> Self {
        Self::new(crate::new(value, value), crate::new(0.0, 0.0))
    }
}

impl<I: IntervalDatum> From<&f64> for ComplexBox<I> {
    fn from(value: &f64) -> Self {
        (*value).into()
    }
}

impl<I: IntervalDatum> From<Complex64> for ComplexBox<I> {
    fn from(value: Complex64) -> Self {
        Self::new(crate::singleton(value.re), crate::singleton(value.im))
    }
}

impl<I: IntervalDatum> From<&Complex64> for ComplexBox<I> {
    fn from(value: &Complex64) -> Self {
        (*value).into()
    }
}

impl<I: IntervalDatum> Default for ComplexBox<I> {
    fn default() -> Self {
        Self::new(I::__zero(), I::__zero())
    }
}

impl<I: IntervalDatum + PartialEq> PartialEq for ComplexBox<I> {
    fn eq(&self, other: &Self) -> bool {
        self.re == other.re && self.im == other.im
    }
}

impl<I: IntervalDatum + Eq> Eq for ComplexBox<I> {}

impl<I: IntervalDatum + fmt::Display> fmt::Display for ComplexBox<I> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "({}) + ({})i", self.re, self.im)
    }
}

impl ComplexBox<Interval> {
    pub const EMPTY: Self = Self {
        re: Interval::EMPTY,
        im: Interval::EMPTY,
    };
    pub const ENTIRE: Self = Self {
        re: Interval::ENTIRE,
        im: Interval::ENTIRE,
    };
    pub const ZERO: Self = Self {
        re: Interval::ZERO,
        im: Interval::ZERO,
    };
    pub const ONE: Self = Self {
        re: Interval::ONE,
        im: Interval::ZERO,
    };
    pub const I: Self = Self {
        re: Interval::ZERO,
        im: Interval::ONE,
    };
}

impl ComplexBox<DecoratedInterval> {
    pub const EMPTY: Self = Self {
        re: DecoratedInterval::EMPTY,
        im: DecoratedInterval::EMPTY,
    };
    pub const ENTIRE: Self = Self {
        re: DecoratedInterval::ENTIRE,
        im: DecoratedInterval::ENTIRE,
    };
    pub const ZERO: Self = Self {
        re: DecoratedInterval::ZERO,
        im: DecoratedInterval::ZERO,
    };
    pub const ONE: Self = Self {
        re: DecoratedInterval::ONE,
        im: DecoratedInterval::ZERO,
    };
    pub const I: Self = Self {
        re: DecoratedInterval::ZERO,
        im: DecoratedInterval::ONE,
    };

    pub fn decoration(&self) -> Decoration {
        core::cmp::min(self.re.decoration_raw(), self.im.decoration_raw())
    }
}

impl_op_ex!(-|a: &ComplexBox<Interval>| -> ComplexBox<Interval> { a.neg_raw() });

impl_op_ex!(+ |a: &ComplexBox<Interval>, b: &ComplexBox<Interval>| -> ComplexBox<Interval> {
    a.add_raw(b)
});

impl_op_ex!(
    -|a: &ComplexBox<Interval>, b: &ComplexBox<Interval>| -> ComplexBox<Interval> { a.sub_raw(b) }
);

impl_op_ex!(
    *|a: &ComplexBox<Interval>, b: &ComplexBox<Interval>| -> ComplexBox<Interval> { a.mul_raw(b) }
);

impl_op_ex!(
    /|a: &ComplexBox<Interval>, b: &ComplexBox<Interval>| -> ComplexBox<Interval> {
        a.div_raw(b)
    }
);

impl_op_ex!(+ |a: &ComplexBox<Interval>, b: &f64| -> ComplexBox<Interval> {
    a.add_raw(&ComplexBox::from(*b))
});
impl_op_ex!(
    -|a: &ComplexBox<Interval>, b: &f64| -> ComplexBox<Interval> {
        a.sub_raw(&ComplexBox::from(*b))
    }
);
impl_op_ex!(
    *|a: &ComplexBox<Interval>, b: &f64| -> ComplexBox<Interval> {
        ComplexBox::new(crate::mul(a.re, (*b).into()), crate::mul(a.im, (*b).into()))
    }
);
impl_op_ex!(/|a: &ComplexBox<Interval>, b: &f64| -> ComplexBox<Interval> {
    ComplexBox::new(crate::div(a.re, (*b).into()), crate::div(a.im, (*b).into()))
});
impl_op_ex!(+ |a: &f64, b: &ComplexBox<Interval>| -> ComplexBox<Interval> {
    b.add_raw(&ComplexBox::from(*a))
});
impl_op_ex!(
    -|a: &f64, b: &ComplexBox<Interval>| -> ComplexBox<Interval> {
        ComplexBox::from(*a).sub_raw(b)
    }
);
impl_op_ex!(*|a: &f64, b: &ComplexBox<Interval>| -> ComplexBox<Interval> { b * a });
impl_op_ex!(/|a: &f64, b: &ComplexBox<Interval>| -> ComplexBox<Interval> {
    ComplexBox::from(*a).div_raw(b)
});

impl_op_ex!(-|a: &ComplexBox<DecoratedInterval>| -> ComplexBox<DecoratedInterval> { a.neg_raw() });

impl_op_ex!(+ |a: &ComplexBox<DecoratedInterval>, b: &ComplexBox<DecoratedInterval>| -> ComplexBox<DecoratedInterval> {
    a.add_raw(b)
});

impl_op_ex!(-|a: &ComplexBox<DecoratedInterval>,
              b: &ComplexBox<DecoratedInterval>|
 -> ComplexBox<DecoratedInterval> { a.sub_raw(b) });

impl_op_ex!(*|a: &ComplexBox<DecoratedInterval>,
              b: &ComplexBox<DecoratedInterval>|
 -> ComplexBox<DecoratedInterval> { a.mul_raw(b) });

impl_op_ex!(
    /|a: &ComplexBox<DecoratedInterval>, b: &ComplexBox<DecoratedInterval>| -> ComplexBox<DecoratedInterval> {
        a.div_raw(b)
    }
);

impl_op_ex!(+ |a: &ComplexBox<DecoratedInterval>, b: &f64| -> ComplexBox<DecoratedInterval> {
    a.add_raw(&ComplexBox::from(*b))
});
impl_op_ex!(
    -|a: &ComplexBox<DecoratedInterval>, b: &f64| -> ComplexBox<DecoratedInterval> {
        a.sub_raw(&ComplexBox::from(*b))
    }
);
impl_op_ex!(
    *|a: &ComplexBox<DecoratedInterval>, b: &f64| -> ComplexBox<DecoratedInterval> {
        ComplexBox::new(crate::mul(a.re, (*b).into()), crate::mul(a.im, (*b).into()))
    }
);
impl_op_ex!(/|a: &ComplexBox<DecoratedInterval>, b: &f64| -> ComplexBox<DecoratedInterval> {
    ComplexBox::new(crate::div(a.re, (*b).into()), crate::div(a.im, (*b).into()))
});
impl_op_ex!(+ |a: &f64, b: &ComplexBox<DecoratedInterval>| -> ComplexBox<DecoratedInterval> {
    b.add_raw(&ComplexBox::from(*a))
});
impl_op_ex!(
    -|a: &f64, b: &ComplexBox<DecoratedInterval>| -> ComplexBox<DecoratedInterval> {
        ComplexBox::from(*a).sub_raw(b)
    }
);
impl_op_ex!(
    *|a: &f64, b: &ComplexBox<DecoratedInterval>| -> ComplexBox<DecoratedInterval> { b * a }
);
impl_op_ex!(/|a: &f64, b: &ComplexBox<DecoratedInterval>| -> ComplexBox<DecoratedInterval> {
    ComplexBox::from(*a).div_raw(b)
});

impl ComplexBox<Interval> {
    pub fn decorate<S: SignalSink>(&self, signals: &mut S) -> ComplexBox<DecoratedInterval> {
        ComplexBox::new(self.re.decorate(signals), self.im.decorate(signals))
    }
}

impl From<&ComplexBox<Interval>> for ComplexBox<DecoratedInterval> {
    fn from(value: &ComplexBox<Interval>) -> Self {
        value.decorate(&mut ())
    }
}

impl From<ComplexBox<Interval>> for ComplexBox<DecoratedInterval> {
    fn from(value: ComplexBox<Interval>) -> Self {
        value.decorate(&mut ())
    }
}

impl From<&ComplexBox<DecoratedInterval>> for ComplexBox<Interval> {
    fn from(value: &ComplexBox<DecoratedInterval>) -> Self {
        Self::new(value.re.into(), value.im.into())
    }
}

impl From<ComplexBox<DecoratedInterval>> for ComplexBox<Interval> {
    fn from(value: ComplexBox<DecoratedInterval>) -> Self {
        Self::new(value.re.into(), value.im.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_contains(interval: Interval, value: f64) {
        assert!(
            interval.inf() <= value && value <= interval.sup(),
            "{interval:?} does not contain {value}"
        );
    }

    #[test]
    fn mul_add_combines_complex_singletons() {
        let x = ComplexBox::<Interval>::new(1.0.into(), 2.0.into());
        let y = ComplexBox::<Interval>::new(3.0.into(), 4.0.into());
        let z = ComplexBox::<Interval>::new(5.0.into(), 6.0.into());
        let result = x.mul_add(&y, &z);
        assert_eq!(result.re.bounds(), (0.0, 0.0));
        assert_eq!(result.im.bounds(), (16.0, 16.0));
    }

    #[test]
    fn corrected_complex_formulas_enclose_reference_values() {
        let square = ComplexBox::<Interval>::new(3.0.into(), 4.0.into()).sqr();
        assert_eq!(square.re.bounds(), (-7.0, -7.0));
        assert_eq!(square.im.bounds(), (24.0, 24.0));

        let half = ComplexBox::<Interval>::from(0.5);
        let two = ComplexBox::<Interval>::from(2.0);
        let atan = half.atan();
        assert_contains(atan.re, 0.5_f64.atan());
        assert_contains(atan.im, 0.0);
        let acosh = two.acosh();
        assert_contains(acosh.re, 2.0_f64.acosh());
        assert_contains(acosh.im, 0.0);
        let atanh = half.atanh();
        assert_contains(atanh.re, 0.5_f64.atanh());
        assert_contains(atanh.im, 0.0);
    }

    #[test]
    fn abs_and_arg_preserve_interval_semantics() {
        let empty = ComplexBox::new(Interval::EMPTY, Interval::ZERO);
        assert!(empty.abs().is_empty());

        let def = DecoratedInterval::set_dec_raw(3.0.into(), Decoration::Def);
        let decorated = ComplexBox::new(def, DecoratedInterval::from(4.0));
        let abs = decorated.abs();
        assert_contains(abs.into(), 5.0);
        assert_eq!(crate::decoration_part(abs), Decoration::Def);

        let crossing =
            ComplexBox::<Interval>::new(Interval::new(-2.0, -1.0), Interval::new(-1.0, 1.0));
        let arg = crossing.arg();
        assert!(arg.inf() <= -core::f64::consts::PI);
        assert!(arg.sup() >= core::f64::consts::PI);
    }

    #[test]
    fn complex_conversion_uses_real_and_imaginary_components() {
        let complex = ComplexBox::<Interval>::from(Complex64::new(2.0, 3.0));
        assert!(complex.contains(Complex64::new(2.0, 3.0)));
        assert!(!complex.contains(Complex64::new(f64::NAN, 3.0)));
    }

    #[test]
    fn geometry_distinguishes_widths_radius_and_diameter() {
        let value = ComplexBox::<Interval>::new(Interval::new(0.0, 6.0), Interval::new(0.0, 8.0));
        assert_eq!(value.wid_box(), Complex64::new(6.0, 8.0));
        assert!(value.diameter() >= 10.0);
        assert!(value.rad() >= 5.0);
        assert_eq!(value.corners()[0], Complex64::new(0.0, 0.0));
        assert_eq!(value.corners()[2], Complex64::new(6.0, 8.0));
    }

    #[test]
    fn rectangular_set_relations_are_componentwise() {
        let inner = ComplexBox::<Interval>::new(Interval::new(1.0, 2.0), Interval::new(1.0, 2.0));
        let outer = ComplexBox::<Interval>::new(Interval::new(0.0, 3.0), Interval::new(0.0, 3.0));
        let apart = ComplexBox::<Interval>::new(Interval::new(4.0, 5.0), Interval::new(1.0, 2.0));
        assert!(inner.subset(&outer));
        assert!(inner.interior(&outer));
        assert!(inner.disjoint(&apart));
        assert!(!inner.intersects(&apart));
        assert_eq!(inner.intersection(&outer), inner);
        assert_eq!(inner.convex_hull(&outer), outer);

        let empty = inner.intersection(&apart);
        assert_eq!(empty, ComplexBox::<Interval>::EMPTY);
        assert!(empty.subset(&outer));
        assert_eq!(empty.convex_hull(&outer), outer);
    }

    #[test]
    fn constants_polar_helpers_and_scalar_operators_are_coherent() {
        assert!(ComplexBox::<Interval>::ZERO.is_zero());
        assert!(ComplexBox::<Interval>::ONE.is_real());
        assert_eq!(ComplexBox::<Interval>::I, ComplexBox::i());
        assert!(ComplexBox::<Interval>::cis(Interval::ZERO).contains(Complex64::new(1.0, 0.0)));
        let scaled = ComplexBox::<Interval>::from(Complex64::new(1.0, 2.0)) * 3.0;
        assert!(scaled.contains(Complex64::new(3.0, 6.0)));
    }
}
