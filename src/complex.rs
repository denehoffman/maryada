use core::{fmt, ops};
use impl_ops::*;
use num_complex::Complex64;

use crate::{DecoratedInterval, Decoration, Interval, IntervalDatum, SignalSink};

#[derive(Copy, Clone, Debug)]
/// A rectangular complex interval with independent real and imaginary components.
///
/// `I` is normally [`Interval`] or [`DecoratedInterval`]. The represented set
/// is the Cartesian product `re × im` in the complex plane.
pub struct ComplexBox<I> {
    /// Interval enclosing the real component.
    pub re: I,
    /// Interval enclosing the imaginary component.
    pub im: I,
}

impl<I: IntervalDatum> ComplexBox<I> {
    /// Constructs a rectangular complex interval from its components.
    pub const fn new(re: I, im: I) -> Self {
        Self { re, im }
    }

    /// Returns the singleton imaginary unit `i`.
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

    /// Adds a real interval to the real component.
    pub fn add_real(&self, value: I) -> Self {
        Self::new(crate::add(self.re, value), self.im)
    }

    /// Subtracts a real interval from the real component.
    pub fn sub_real(&self, value: I) -> Self {
        Self::new(crate::sub(self.re, value), self.im)
    }

    /// Multiplies both components by a real interval.
    pub fn scale(&self, value: I) -> Self {
        Self::new(crate::mul(self.re, value), crate::mul(self.im, value))
    }

    /// Divides both components by a real interval.
    pub fn div_real(&self, value: I) -> Self {
        Self::new(crate::div(self.re, value), crate::div(self.im, value))
    }

    /// Returns the complex conjugate.
    pub fn conj(&self) -> Self {
        Self::new(self.re, crate::neg(self.im))
    }

    /// Encloses the squared modulus `re² + im²`.
    pub fn norm_sqr(&self) -> I {
        crate::add(crate::sqr(self.re), crate::sqr(self.im))
    }

    /// Encloses the complex modulus.
    pub fn abs(&self) -> I {
        crate::hypot(self.re, self.im)
    }

    /// Alias for [`ComplexBox::abs`].
    pub fn norm(&self) -> I {
        self.abs()
    }

    /// Returns the componentwise midpoint as a complex scalar.
    pub fn mid(&self) -> Complex64 {
        Complex64::new(crate::mid(self.re), crate::mid(self.im))
    }

    /// Returns the componentwise interval widths.
    pub fn wid_box(&self) -> Complex64 {
        Complex64::new(crate::wid(self.re), crate::wid(self.im))
    }

    /// Alias for [`ComplexBox::wid_box`].
    pub fn wid(&self) -> Complex64 {
        self.wid_box()
    }

    /// Returns the componentwise interval radii.
    pub fn rad_box(&self) -> Complex64 {
        Complex64::new(crate::rad(self.re), crate::rad(self.im))
    }

    /// Returns the Euclidean radius of the rectangular box.
    pub fn rad(&self) -> f64 {
        let re = crate::rad(self.re);
        let im = crate::rad(self.im);
        if re.is_nan() || im.is_nan() {
            return f64::NAN;
        }
        crate::sup(crate::hypot::<Interval>(re.into(), im.into()))
    }

    /// Returns the Euclidean length of the box diagonal.
    pub fn diameter(&self) -> f64 {
        let widths = self.wid_box();
        if widths.re.is_nan() || widths.im.is_nan() {
            return f64::NAN;
        }
        crate::sup(crate::hypot::<Interval>(widths.re.into(), widths.im.into()))
    }

    /// Returns an upper bound on the modulus of every represented value.
    pub fn mag(&self) -> f64 {
        crate::sup(self.abs())
    }

    /// Returns a lower bound on the modulus of every represented value.
    pub fn mig(&self) -> f64 {
        crate::inf(self.abs())
    }

    /// Encloses the complex argument in radians.
    pub fn arg(&self) -> I {
        crate::atan2(self.im, self.re)
    }

    /// Returns modulus and argument interval enclosures.
    pub fn to_polar(&self) -> (I, I) {
        (self.abs(), self.arg())
    }

    /// Encloses `cos(theta) + i sin(theta)`.
    pub fn cis(theta: I) -> Self {
        Self::new(crate::cos(theta), crate::sin(theta))
    }

    /// Constructs a rectangular enclosure from polar coordinates.
    pub fn from_polar(radius: I, theta: I) -> Self {
        Self::new(
            crate::mul(radius, crate::cos(theta)),
            crate::mul(radius, crate::sin(theta)),
        )
    }

    /// Returns the corner formed by both lower component endpoints.
    pub fn lower_corner(&self) -> Complex64 {
        Complex64::new(crate::inf(self.re), crate::inf(self.im))
    }

    /// Returns the corner formed by both upper component endpoints.
    pub fn upper_corner(&self) -> Complex64 {
        Complex64::new(crate::sup(self.re), crate::sup(self.im))
    }

    /// Returns all four corners of the rectangular box.
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

    /// Returns componentwise midpoints and radii.
    pub fn mid_rad(&self) -> (Complex64, Complex64) {
        (self.mid(), self.rad_box())
    }

    /// Returns whether either component is empty.
    pub fn is_empty(&self) -> bool {
        self.is_empty_raw()
    }

    /// Returns whether both components are entire intervals.
    pub fn is_entire(&self) -> bool {
        crate::is_entire(self.re) && crate::is_entire(self.im)
    }

    /// Returns whether either component is NaI.
    pub fn is_nai(&self) -> bool {
        self.re.__is_nai() || self.im.__is_nai()
    }

    /// Returns whether the box contains exactly one complex value.
    pub fn is_singleton(&self) -> bool {
        !self.is_nai()
            && !self.is_empty()
            && crate::inf(self.re) == crate::sup(self.re)
            && crate::inf(self.im) == crate::sup(self.im)
    }

    /// Returns whether all component endpoints are finite.
    pub fn is_bounded(&self) -> bool {
        self.lower_corner().re.is_finite()
            && self.lower_corner().im.is_finite()
            && self.upper_corner().re.is_finite()
            && self.upper_corner().im.is_finite()
    }

    /// Returns whether this is the singleton complex zero.
    pub fn is_zero(&self) -> bool {
        self.is_singleton() && crate::inf(self.re) == 0.0 && crate::inf(self.im) == 0.0
    }

    /// Returns whether every represented value is real.
    pub fn is_real(&self) -> bool {
        !self.is_empty()
            && !self.is_nai()
            && crate::inf(self.im) == 0.0
            && crate::sup(self.im) == 0.0
    }

    /// Returns whether the finite complex scalar belongs to this box.
    pub fn contains(&self, value: Complex64) -> bool {
        !value.re.is_nan()
            && !value.im.is_nan()
            && crate::subset(crate::singleton(value.re), self.re)
            && crate::subset(crate::singleton(value.im), self.im)
    }

    /// Returns whether this rectangular set is a subset of `other`.
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

    /// Returns whether this box lies in the interior of `other`.
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

    /// Returns whether the two boxes have no common complex value.
    pub fn disjoint(&self, other: &Self) -> bool {
        crate::disjoint(self.re, other.re) || crate::disjoint(self.im, other.im)
    }

    /// Returns whether the two boxes share at least one complex value.
    pub fn intersects(&self, other: &Self) -> bool {
        !self.disjoint(other)
    }

    /// Returns the rectangular intersection of two boxes.
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

    /// Returns the smallest rectangular box containing both operands.
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

    /// Encloses the complex reciprocal.
    pub fn recip(&self) -> Self {
        let denom = crate::add(crate::sqr(self.re), crate::sqr(self.im));
        Self::new(
            crate::div(self.re, denom),
            crate::div(crate::neg(self.im), denom),
        )
    }

    /// Encloses the complex square.
    pub fn sqr(&self) -> Self {
        Self::new(
            crate::sub(crate::sqr(self.re), crate::sqr(self.im)),
            crate::mul(crate::singleton(2.0), crate::mul(self.re, self.im)),
        )
    }

    /// Encloses the principal complex square root.
    pub fn sqrt(&self) -> Self {
        Self::from(0.5).mul_raw(&self.log()).exp()
    }

    /// Encloses `self * y + z` componentwise using fused operations.
    pub fn mul_add(&self, y: &Self, z: &Self) -> Self {
        Self::new(
            crate::fma(self.re, y.re, crate::fma(crate::neg(self.im), y.im, z.re)),
            crate::fma(self.re, y.im, crate::fma(self.im, y.re, z.im)),
        )
    }

    /// Raises this box to an integer power by exponentiation by squaring.
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

    /// Alias for [`ComplexBox::pown`].
    pub fn powi(&self, i: i32) -> Self {
        self.pown(i)
    }

    /// Encloses the principal complex power with exponents in `other`.
    pub fn pow(&self, other: &Self) -> Self {
        other.mul_raw(&self.log()).exp()
    }

    /// Encloses the complex exponential.
    pub fn exp(&self) -> Self {
        Self::new(
            crate::mul(crate::exp(self.re), crate::cos(self.im)),
            crate::mul(crate::exp(self.re), crate::sin(self.im)),
        )
    }

    /// Encloses the base-two complex exponential.
    pub fn exp2(&self) -> Self {
        self.mul_raw(&Self::from(crate::log(crate::singleton::<I>(2.0))))
            .exp()
    }

    /// Encloses the base-ten complex exponential.
    pub fn exp10(&self) -> Self {
        self.mul_raw(&Self::from(crate::log(crate::singleton::<I>(10.0))))
            .exp()
    }

    /// Encloses the principal complex logarithm.
    pub fn log(&self) -> Self {
        Self::new(crate::log(self.abs()), self.arg())
    }

    /// Encloses the principal base-two complex logarithm.
    pub fn log2(&self) -> Self {
        self.log()
            .div_raw(&Self::from(crate::log(crate::singleton::<I>(2.0))))
    }

    /// Encloses the principal base-ten complex logarithm.
    pub fn log10(&self) -> Self {
        self.log()
            .div_raw(&Self::from(crate::log(crate::singleton::<I>(10.0))))
    }

    /// Encloses the complex sine.
    pub fn sin(&self) -> Self {
        Self::new(
            crate::mul(crate::sin(self.re), crate::cosh(self.im)),
            crate::mul(crate::cos(self.re), crate::sinh(self.im)),
        )
    }

    /// Encloses the complex cosine.
    pub fn cos(&self) -> Self {
        Self::new(
            crate::mul(crate::cos(self.re), crate::cosh(self.im)),
            crate::neg(crate::mul(crate::sin(self.re), crate::sinh(self.im))),
        )
    }

    /// Encloses the complex tangent.
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

    /// Encloses the principal complex inverse sine.
    pub fn asin(&self) -> Self {
        Self::i().neg_raw().mul_raw(&Self::log(
            &Self::i()
                .mul_raw(self)
                .add_raw(&((Self::from(1.0).sub_raw(&self.sqr())).sqrt())),
        ))
    }

    /// Encloses the principal complex inverse cosine.
    pub fn acos(&self) -> Self {
        Self::i().neg_raw().mul_raw(&Self::log(
            &self.add_raw(&Self::i().mul_raw(&Self::from(1.0).sub_raw(&self.sqr()).sqrt())),
        ))
    }

    /// Encloses the principal complex inverse tangent.
    pub fn atan(&self) -> Self {
        let iz = Self::i().mul_raw(self);
        Self::i().div_raw(&Self::from(2.0)).mul_raw(
            &Self::from(1.0)
                .sub_raw(&iz)
                .log()
                .sub_raw(&Self::from(1.0).add_raw(&iz).log()),
        )
    }

    /// Encloses the complex hyperbolic sine.
    pub fn sinh(&self) -> Self {
        Self::new(
            crate::mul(crate::sinh(self.re), crate::cos(self.im)),
            crate::mul(crate::cosh(self.re), crate::sin(self.im)),
        )
    }

    /// Encloses the complex hyperbolic cosine.
    pub fn cosh(&self) -> Self {
        Self::new(
            crate::mul(crate::cosh(self.re), crate::cos(self.im)),
            crate::mul(crate::sinh(self.re), crate::sin(self.im)),
        )
    }

    /// Encloses the complex hyperbolic tangent.
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

    /// Encloses the principal complex inverse hyperbolic sine.
    pub fn asinh(&self) -> Self {
        self.add_raw(&(self.sqr().add_raw(&Self::from(1.0))).sqrt())
            .log()
    }

    /// Encloses the principal complex inverse hyperbolic cosine.
    pub fn acosh(&self) -> Self {
        self.add_raw(
            &self
                .sub_raw(&Self::from(1.0))
                .sqrt()
                .mul_raw(&self.add_raw(&Self::from(1.0)).sqrt()),
        )
        .log()
    }

    /// Encloses the principal complex inverse hyperbolic tangent.
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
    /// The empty bare complex box.
    pub const EMPTY: Self = Self {
        re: Interval::EMPTY,
        im: Interval::EMPTY,
    };
    /// The box containing the entire complex plane.
    pub const ENTIRE: Self = Self {
        re: Interval::ENTIRE,
        im: Interval::ENTIRE,
    };
    /// The singleton complex zero.
    pub const ZERO: Self = Self {
        re: Interval::ZERO,
        im: Interval::ZERO,
    };
    /// The singleton complex one.
    pub const ONE: Self = Self {
        re: Interval::ONE,
        im: Interval::ZERO,
    };
    /// The singleton imaginary unit.
    pub const I: Self = Self {
        re: Interval::ZERO,
        im: Interval::ONE,
    };
}

impl ComplexBox<DecoratedInterval> {
    /// The empty decorated complex box.
    pub const EMPTY: Self = Self {
        re: DecoratedInterval::EMPTY,
        im: DecoratedInterval::EMPTY,
    };
    /// The decorated box containing the entire complex plane.
    pub const ENTIRE: Self = Self {
        re: DecoratedInterval::ENTIRE,
        im: DecoratedInterval::ENTIRE,
    };
    /// The decorated singleton complex zero.
    pub const ZERO: Self = Self {
        re: DecoratedInterval::ZERO,
        im: DecoratedInterval::ZERO,
    };
    /// The decorated singleton complex one.
    pub const ONE: Self = Self {
        re: DecoratedInterval::ONE,
        im: DecoratedInterval::ZERO,
    };
    /// The decorated singleton imaginary unit.
    pub const I: Self = Self {
        re: DecoratedInterval::ZERO,
        im: DecoratedInterval::ONE,
    };

    /// Returns the weaker decoration of the real and imaginary components.
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
    /// Decorates both components, reporting construction signals to `signals`.
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
