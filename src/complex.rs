use core::{
    fmt,
    ops::{self, Add, Div, Mul, Neg, Sub},
};
#[cfg(feature = "num-complex")]
use num_complex::Complex64;

use crate::{DecoratedInterval, Decoration, EnclosureOps, Interval, IntervalOps, SignalSink};

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

#[allow(clippy::arithmetic_side_effects, clippy::use_self)]
impl<I: IntervalOps> EnclosureOps for ComplexBox<I> {
    #[cfg(feature = "num-complex")]
    type Midpoint = Complex64;
    #[cfg(not(feature = "num-complex"))]
    type Midpoint = (f64, f64);

    const ZERO: Self = Self::ZERO;
    const ONE: Self = Self::ONE;
    const EMPTY: Self = Self::EMPTY;
    const ENTIRE: Self = Self::ENTIRE;

    fn singleton(value: f64) -> Self {
        Self::from(value)
    }

    #[cfg(feature = "num-complex")]
    fn from_midpoint(value: Self::Midpoint) -> Self {
        Self::from(value)
    }

    #[cfg(not(feature = "num-complex"))]
    fn from_midpoint(value: Self::Midpoint) -> Self {
        Self::new(I::from(value.0), I::from(value.1))
    }

    fn inflate(self, relative: f64, absolute: f64) -> Self {
        let scale = I::new(1.0 - relative, 1.0 + relative);
        let offset = I::new(-absolute, absolute);
        Self::new(self.re * scale + offset, self.im * scale + offset)
    }

    fn mul_add(self, rhs: Self, addend: Self) -> Self {
        Self::new(
            self.re
                .mul_add(rhs.re, self.im.mul_add(Neg::neg(rhs.im), addend.re)),
            self.re.mul_add(rhs.im, self.im.mul_add(rhs.re, addend.im)),
        )
    }

    fn recip(self) -> Self {
        let denominator = Add::add(self.re.sqr(), self.im.sqr());
        Self::new(
            Div::div(self.re, denominator),
            Div::div(Neg::neg(self.im), denominator),
        )
    }

    fn sqr(self) -> Self {
        Self::new(
            Sub::sub(self.re.sqr(), self.im.sqr()),
            Mul::mul(Mul::mul(self.re, self.im), 2.0),
        )
    }

    fn sqrt(self) -> Self {
        Mul::mul(0.5, self.log()).exp()
    }

    fn pown(self, exponent: i32) -> Self {
        if self.is_empty_raw() {
            return Self::empty_raw();
        }
        if exponent == 0 {
            return Self::from(1.0);
        }
        let mut base = if exponent < 0 { self.recip() } else { self };
        let mut exponent = exponent.unsigned_abs();
        let mut result = Self::from(1.0);
        while exponent != 0 {
            if exponent & 1 != 0 {
                result = Mul::mul(result, base);
            }
            exponent >>= 1;
            if exponent != 0 {
                base = base.sqr();
            }
        }
        result
    }

    fn pow(self, other: Self) -> Self {
        Mul::mul(other, self.log()).exp()
    }

    fn exp(self) -> Self {
        let exp_re = self.re.exp();
        Self::new(
            Mul::mul(exp_re, self.im.cos()),
            Mul::mul(exp_re, self.im.sin()),
        )
    }

    fn exp2(self) -> Self {
        Mul::mul(self, Self::from(I::from(2.0).log())).exp()
    }

    fn exp10(self) -> Self {
        Mul::mul(self, Self::from(I::from(10.0).log())).exp()
    }

    fn log(self) -> Self {
        Self::new(self.abs().log(), self.arg())
    }

    fn log2(self) -> Self {
        Div::div(self.log(), Self::from(I::from(2.0).log()))
    }

    fn log10(self) -> Self {
        Div::div(self.log(), Self::from(I::from(10.0).log()))
    }

    fn sin(self) -> Self {
        Self::new(
            Mul::mul(self.re.sin(), self.im.cosh()),
            Mul::mul(self.re.cos(), self.im.sinh()),
        )
    }

    fn cos(self) -> Self {
        Self::new(
            Mul::mul(self.re.cos(), self.im.cosh()),
            Neg::neg(Mul::mul(self.re.sin(), self.im.sinh())),
        )
    }

    fn tan(self) -> Self {
        let twice_re = Mul::mul(self.re, 2.0);
        let twice_im = Mul::mul(self.im, 2.0);
        let denominator = Add::add(twice_re.cos(), twice_im.cosh());
        Self::new(
            Div::div(twice_re.sin(), denominator),
            Div::div(twice_im.sinh(), denominator),
        )
    }

    fn asin(self) -> Self {
        let argument = Add::add(Mul::mul(Self::I, self), Sub::sub(1.0, self.sqr()).sqrt());
        Mul::mul(Neg::neg(Self::I), argument.log())
    }

    fn acos(self) -> Self {
        let argument = Add::add(self, Mul::mul(Self::I, Sub::sub(1.0, self.sqr()).sqrt()));
        Mul::mul(Neg::neg(Self::I), argument.log())
    }

    fn atan(self) -> Self {
        let iz = Mul::mul(Self::I, self);
        let logarithms = Sub::sub(Sub::sub(1.0, iz).log(), Add::add(1.0, iz).log());
        Mul::mul(Div::div(Self::I, 2.0), logarithms)
    }

    fn sinh(self) -> Self {
        Self::new(
            Mul::mul(self.re.sinh(), self.im.cos()),
            Mul::mul(self.re.cosh(), self.im.sin()),
        )
    }

    fn cosh(self) -> Self {
        Self::new(
            Mul::mul(self.re.cosh(), self.im.cos()),
            Mul::mul(self.re.sinh(), self.im.sin()),
        )
    }

    fn tanh(self) -> Self {
        let twice_re = Mul::mul(self.re, 2.0);
        let twice_im = Mul::mul(self.im, 2.0);
        let denominator = Add::add(twice_re.cosh(), twice_im.cos());
        Self::new(
            Div::div(twice_re.sinh(), denominator),
            Div::div(twice_im.sin(), denominator),
        )
    }

    fn asinh(self) -> Self {
        Add::add(self, Add::add(self.sqr(), 1.0).sqrt()).log()
    }

    fn acosh(self) -> Self {
        Add::add(
            self,
            Mul::mul(Sub::sub(self, 1.0).sqrt(), Add::add(self, 1.0).sqrt()),
        )
        .log()
    }

    fn atanh(self) -> Self {
        Mul::mul(
            0.5,
            Sub::sub(Add::add(1.0, self).log(), Sub::sub(1.0, self).log()),
        )
    }

    #[cfg(feature = "num-complex")]
    fn mid(self) -> Self::Midpoint {
        Complex64::new(self.re.mid(), self.im.mid())
    }

    #[cfg(not(feature = "num-complex"))]
    fn mid(self) -> Self::Midpoint {
        (self.re.mid(), self.im.mid())
    }

    #[cfg(feature = "num-complex")]
    fn contains(self, value: Self::Midpoint) -> bool {
        value.re.is_finite()
            && value.im.is_finite()
            && I::from(value.re).subset(self.re)
            && I::from(value.im).subset(self.im)
    }

    #[cfg(not(feature = "num-complex"))]
    fn contains(self, value: Self::Midpoint) -> bool {
        value.0.is_finite()
            && value.1.is_finite()
            && I::from(value.0).subset(self.re)
            && I::from(value.1).subset(self.im)
    }

    #[cfg(feature = "num-complex")]
    fn wid(self) -> Self::Midpoint {
        Complex64::new(self.re.wid(), self.im.wid())
    }

    #[cfg(not(feature = "num-complex"))]
    fn wid(self) -> Self::Midpoint {
        (self.re.wid(), self.im.wid())
    }

    fn rad(self) -> f64 {
        let re = self.re.rad();
        let im = self.im.rad();
        if re.is_nan() || im.is_nan() {
            return f64::NAN;
        }
        I::from(re).hypot(I::from(im)).sup()
    }

    fn inner_rad(self) -> f64 {
        let re = self.re.inner_rad();
        let im = self.im.inner_rad();
        if re.is_nan() || im.is_nan() {
            return f64::NAN;
        }
        I::from(re).hypot(I::from(im)).sup()
    }

    #[cfg(feature = "num-complex")]
    fn mid_rad(self) -> (Self::Midpoint, Self::Midpoint) {
        (self.mid(), Complex64::new(self.re.rad(), self.im.rad()))
    }

    #[cfg(not(feature = "num-complex"))]
    fn mid_rad(self) -> (Self::Midpoint, Self::Midpoint) {
        (self.mid(), (self.re.rad(), self.im.rad()))
    }

    fn is_empty(self) -> bool {
        self.is_empty_raw()
    }

    fn is_nai(self) -> bool {
        self.re.is_nai() || self.im.is_nai()
    }

    fn is_entire(self) -> bool {
        self.re.is_entire() && self.im.is_entire()
    }

    fn is_singleton(self) -> bool {
        !self.is_nai() && !self.is_empty() && self.re.is_singleton() && self.im.is_singleton()
    }

    fn is_bounded(self) -> bool {
        self.re.is_bounded() && self.im.is_bounded()
    }

    fn equal(self, other: Self) -> bool {
        self.re.equal(other.re) && self.im.equal(other.im)
    }

    fn subset(self, other: Self) -> bool {
        if self.is_nai() || other.is_nai() {
            return false;
        }
        if self.is_empty() {
            return true;
        }
        !other.is_empty() && self.re.subset(other.re) && self.im.subset(other.im)
    }

    fn interior(self, other: Self) -> bool {
        if self.is_nai() || other.is_nai() {
            return false;
        }
        if self.is_empty() {
            return true;
        }
        !other.is_empty() && self.re.interior(other.re) && self.im.interior(other.im)
    }

    fn disjoint(self, other: Self) -> bool {
        self.re.disjoint(other.re) || self.im.disjoint(other.im)
    }

    fn intersection(self, other: Self) -> Self {
        let result = Self::new(
            self.re.intersection(other.re),
            self.im.intersection(other.im),
        );
        if result.is_empty() {
            Self::empty_raw()
        } else {
            result
        }
    }

    fn convex_hull(self, other: Self) -> Self {
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
            return self;
        }
        Self::new(self.re.convex_hull(other.re), self.im.convex_hull(other.im))
    }

    fn mag(self) -> f64 {
        self.abs().sup()
    }

    fn mig(self) -> f64 {
        self.abs().inf()
    }
}

impl<I: IntervalOps> ComplexBox<I> {
    /// The empty bare complex box.
    pub const EMPTY: Self = Self {
        re: I::EMPTY,
        im: I::EMPTY,
    };
    /// The box containing the entire complex plane.
    pub const ENTIRE: Self = Self {
        re: I::ENTIRE,
        im: I::ENTIRE,
    };
    /// The singleton complex zero.
    pub const ZERO: Self = Self {
        re: I::ZERO,
        im: I::ZERO,
    };
    /// The singleton complex one.
    pub const ONE: Self = Self {
        re: I::ONE,
        im: I::ZERO,
    };
    /// The singleton imaginary unit.
    pub const I: Self = Self {
        re: I::ZERO,
        im: I::ONE,
    };

    /// Constructs a rectangular complex interval from its components.
    pub const fn new(re: I, im: I) -> Self {
        Self { re, im }
    }

    fn is_empty_raw(self) -> bool {
        self.re.is_empty() || self.im.is_empty()
    }

    const fn empty_raw() -> Self {
        Self::new(I::EMPTY, I::EMPTY)
    }

    /// Returns the complex conjugate.
    #[must_use]
    pub fn conj(self) -> Self {
        Self::new(self.re, Neg::neg(self.im))
    }

    /// Encloses the squared modulus `re² + im²`.
    pub fn norm_sqr(self) -> I {
        Add::add(self.re.sqr(), self.im.sqr())
    }

    /// Encloses the complex modulus.
    pub fn abs(self) -> I {
        self.re.hypot(self.im)
    }

    /// Alias for [`ComplexBox::abs`].
    pub fn norm(self) -> I {
        self.abs()
    }

    /// Returns the componentwise interval widths.
    #[cfg(feature = "num-complex")]
    pub fn wid_box(self) -> Complex64 {
        Complex64::new(self.re.wid(), self.im.wid())
    }

    /// Returns the componentwise interval radii.
    #[cfg(feature = "num-complex")]
    pub fn rad_box(self) -> Complex64 {
        Complex64::new(self.re.rad(), self.im.rad())
    }

    /// Returns the Euclidean length of the box diagonal.
    pub fn diameter(self) -> f64 {
        let re = self.re.wid();
        let im = self.im.wid();
        if re.is_nan() || im.is_nan() {
            return f64::NAN;
        }
        I::from(re).hypot(I::from(im)).sup()
    }

    /// Encloses the complex argument in radians.
    pub fn arg(self) -> I {
        self.im.atan2(self.re)
    }

    /// Returns modulus and argument interval enclosures.
    pub fn to_polar(self) -> (I, I) {
        (self.abs(), self.arg())
    }

    /// Encloses `cos(theta) + i sin(theta)`.
    pub fn cis(theta: I) -> Self {
        Self::new(theta.cos(), theta.sin())
    }

    /// Constructs a rectangular enclosure from polar coordinates.
    pub fn from_polar(radius: I, theta: I) -> Self {
        Self::new(Mul::mul(radius, theta.cos()), Mul::mul(radius, theta.sin()))
    }

    /// Returns the corner formed by both lower component endpoints.
    #[cfg(feature = "num-complex")]
    pub fn lower_corner(self) -> Complex64 {
        Complex64::new(self.re.inf(), self.im.inf())
    }

    /// Returns the corner formed by both upper component endpoints.
    #[cfg(feature = "num-complex")]
    pub fn upper_corner(self) -> Complex64 {
        Complex64::new(self.re.sup(), self.im.sup())
    }

    /// Returns all four corners of the rectangular box.
    #[cfg(feature = "num-complex")]
    pub fn corners(self) -> [Complex64; 4] {
        let lower = self.lower_corner();
        let upper = self.upper_corner();
        [
            lower,
            Complex64::new(upper.re, lower.im),
            upper,
            Complex64::new(lower.re, upper.im),
        ]
    }

    /// Returns whether this is the singleton complex zero.
    pub fn is_zero(self) -> bool {
        self.is_singleton() && self.re.inf() == 0.0 && self.im.inf() == 0.0
    }

    /// Returns whether every represented value is real.
    pub fn is_real(self) -> bool {
        !self.is_empty() && !self.is_nai() && self.im.inf() == 0.0 && self.im.sup() == 0.0
    }
}

impl<I: IntervalOps> Neg for ComplexBox<I> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(Neg::neg(self.re), Neg::neg(self.im))
    }
}

fn add_complex_boxes<I: IntervalOps>(lhs: ComplexBox<I>, rhs: ComplexBox<I>) -> ComplexBox<I> {
    ComplexBox::new(Add::add(lhs.re, rhs.re), Add::add(lhs.im, rhs.im))
}

fn sub_complex_boxes<I: IntervalOps>(lhs: ComplexBox<I>, rhs: ComplexBox<I>) -> ComplexBox<I> {
    ComplexBox::new(Sub::sub(lhs.re, rhs.re), Sub::sub(lhs.im, rhs.im))
}

fn mul_complex_boxes<I: IntervalOps>(lhs: ComplexBox<I>, rhs: ComplexBox<I>) -> ComplexBox<I> {
    ComplexBox::new(
        Sub::sub(Mul::mul(lhs.re, rhs.re), Mul::mul(lhs.im, rhs.im)),
        Add::add(Mul::mul(lhs.re, rhs.im), Mul::mul(lhs.im, rhs.re)),
    )
}

fn div_complex_boxes<I: IntervalOps>(lhs: ComplexBox<I>, rhs: ComplexBox<I>) -> ComplexBox<I> {
    let denominator = Add::add(rhs.re.sqr(), rhs.im.sqr());
    ComplexBox::new(
        Div::div(
            Add::add(Mul::mul(lhs.re, rhs.re), Mul::mul(lhs.im, rhs.im)),
            denominator,
        ),
        Div::div(
            Sub::sub(Mul::mul(lhs.im, rhs.re), Mul::mul(lhs.re, rhs.im)),
            denominator,
        ),
    )
}

fn add_complex_real<I: IntervalOps>(lhs: ComplexBox<I>, rhs: I) -> ComplexBox<I> {
    ComplexBox::new(Add::add(lhs.re, rhs), lhs.im)
}

fn sub_complex_real<I: IntervalOps>(lhs: ComplexBox<I>, rhs: I) -> ComplexBox<I> {
    ComplexBox::new(Sub::sub(lhs.re, rhs), lhs.im)
}

fn mul_complex_real<I: IntervalOps>(lhs: ComplexBox<I>, rhs: I) -> ComplexBox<I> {
    ComplexBox::new(Mul::mul(lhs.re, rhs), Mul::mul(lhs.im, rhs))
}

fn div_complex_real<I: IntervalOps>(lhs: ComplexBox<I>, rhs: I) -> ComplexBox<I> {
    ComplexBox::new(Div::div(lhs.re, rhs), Div::div(lhs.im, rhs))
}

macro_rules! impl_complex_box_op {
    (
        $op:ident,
        $method:ident,
        $assign_op:ident,
        $assign_method:ident,
        $function:path,
        $real_function:path
    ) => {
        impl<I: IntervalOps> ops::$op for ComplexBox<I> {
            type Output = Self;

            fn $method(self, rhs: Self) -> Self::Output {
                $function(self, rhs)
            }
        }

        impl<I: IntervalOps> ops::$op<I> for ComplexBox<I> {
            type Output = Self;

            fn $method(self, rhs: I) -> Self::Output {
                $real_function(self, rhs)
            }
        }

        impl<I: IntervalOps> ops::$assign_op for ComplexBox<I> {
            fn $assign_method(&mut self, rhs: Self) {
                *self = $function(*self, rhs);
            }
        }

        impl<I: IntervalOps> ops::$assign_op<I> for ComplexBox<I> {
            fn $assign_method(&mut self, rhs: I) {
                *self = $real_function(*self, rhs);
            }
        }
    };
}

macro_rules! impl_complex_real_scalar_op {
    (
        $scalar:ty,
        $op:ident,
        $method:ident,
        $assign_op:ident,
        $assign_method:ident,
        $function:path,
        $real_function:path
    ) => {
        impl<I: IntervalOps> ops::$op<$scalar> for ComplexBox<I> {
            type Output = Self;

            fn $method(self, rhs: $scalar) -> Self::Output {
                $real_function(self, I::from(rhs))
            }
        }

        impl<I: IntervalOps> ops::$op<ComplexBox<I>> for $scalar {
            type Output = ComplexBox<I>;

            fn $method(self, rhs: ComplexBox<I>) -> Self::Output {
                $function(ComplexBox::from(self), rhs)
            }
        }

        impl<I: IntervalOps> ops::$assign_op<$scalar> for ComplexBox<I> {
            fn $assign_method(&mut self, rhs: $scalar) {
                *self = $real_function(*self, I::from(rhs));
            }
        }
    };
}

#[cfg(feature = "num-complex")]
macro_rules! impl_complex_scalar_op {
    ($scalar:ty, $op:ident, $method:ident, $assign_op:ident, $assign_method:ident, $function:path) => {
        impl<I: IntervalOps> ops::$op<$scalar> for ComplexBox<I> {
            type Output = Self;

            fn $method(self, rhs: $scalar) -> Self::Output {
                $function(self, ComplexBox::from(rhs))
            }
        }

        impl<I: IntervalOps> ops::$op<ComplexBox<I>> for $scalar {
            type Output = ComplexBox<I>;

            fn $method(self, rhs: ComplexBox<I>) -> Self::Output {
                $function(ComplexBox::from(self), rhs)
            }
        }

        impl<I: IntervalOps> ops::$assign_op<$scalar> for ComplexBox<I> {
            fn $assign_method(&mut self, rhs: $scalar) {
                *self = $function(*self, ComplexBox::from(rhs));
            }
        }
    };
}

macro_rules! impl_promoting_complex_op {
    (
        $op:ident,
        $method:ident,
        $assign_op:ident,
        $assign_method:ident,
        $function:path,
        $real_function:path
    ) => {
        impl ops::$op<ComplexBox<DecoratedInterval>> for ComplexBox<Interval> {
            type Output = ComplexBox<DecoratedInterval>;

            fn $method(self, rhs: ComplexBox<DecoratedInterval>) -> Self::Output {
                $function(self.into(), rhs)
            }
        }

        impl ops::$op<ComplexBox<Interval>> for ComplexBox<DecoratedInterval> {
            type Output = ComplexBox<DecoratedInterval>;

            fn $method(self, rhs: ComplexBox<Interval>) -> Self::Output {
                $function(self, rhs.into())
            }
        }

        impl ops::$op<ComplexBox<Interval>> for Interval {
            type Output = ComplexBox<Interval>;

            fn $method(self, rhs: ComplexBox<Interval>) -> Self::Output {
                $function(ComplexBox::from(self), rhs)
            }
        }

        impl ops::$op<ComplexBox<DecoratedInterval>> for DecoratedInterval {
            type Output = ComplexBox<DecoratedInterval>;

            fn $method(self, rhs: ComplexBox<DecoratedInterval>) -> Self::Output {
                $function(ComplexBox::from(self), rhs)
            }
        }

        impl ops::$op<DecoratedInterval> for ComplexBox<Interval> {
            type Output = ComplexBox<DecoratedInterval>;

            fn $method(self, rhs: DecoratedInterval) -> Self::Output {
                $real_function(self.into(), rhs)
            }
        }

        impl ops::$op<ComplexBox<Interval>> for DecoratedInterval {
            type Output = ComplexBox<DecoratedInterval>;

            fn $method(self, rhs: ComplexBox<Interval>) -> Self::Output {
                $function(ComplexBox::from(self), rhs.into())
            }
        }

        impl ops::$op<Interval> for ComplexBox<DecoratedInterval> {
            type Output = ComplexBox<DecoratedInterval>;

            fn $method(self, rhs: Interval) -> Self::Output {
                $real_function(self, DecoratedInterval::from(rhs))
            }
        }

        impl ops::$op<ComplexBox<DecoratedInterval>> for Interval {
            type Output = ComplexBox<DecoratedInterval>;

            fn $method(self, rhs: ComplexBox<DecoratedInterval>) -> Self::Output {
                $function(ComplexBox::from(DecoratedInterval::from(self)), rhs)
            }
        }

        impl ops::$assign_op<ComplexBox<Interval>> for ComplexBox<DecoratedInterval> {
            fn $assign_method(&mut self, rhs: ComplexBox<Interval>) {
                *self = $function(*self, rhs.into());
            }
        }

        impl ops::$assign_op<Interval> for ComplexBox<DecoratedInterval> {
            fn $assign_method(&mut self, rhs: Interval) {
                *self = $real_function(*self, DecoratedInterval::from(rhs));
            }
        }
    };
}

#[cfg(feature = "num-complex")]
macro_rules! impl_num_complex_real_op {
    ($complex:ty, $op:ident, $method:ident, $function:path) => {
        impl ops::$op<$complex> for Interval {
            type Output = ComplexBox<Interval>;

            fn $method(self, rhs: $complex) -> Self::Output {
                $function(ComplexBox::from(self), ComplexBox::from(rhs))
            }
        }

        impl ops::$op<Interval> for $complex {
            type Output = ComplexBox<Interval>;

            fn $method(self, rhs: Interval) -> Self::Output {
                $function(ComplexBox::from(self), ComplexBox::from(rhs))
            }
        }

        impl ops::$op<$complex> for DecoratedInterval {
            type Output = ComplexBox<DecoratedInterval>;

            fn $method(self, rhs: $complex) -> Self::Output {
                $function(ComplexBox::from(self), ComplexBox::from(rhs))
            }
        }

        impl ops::$op<DecoratedInterval> for $complex {
            type Output = ComplexBox<DecoratedInterval>;

            fn $method(self, rhs: DecoratedInterval) -> Self::Output {
                $function(ComplexBox::from(self), ComplexBox::from(rhs))
            }
        }
    };
}

impl_complex_box_op!(
    Add,
    add,
    AddAssign,
    add_assign,
    add_complex_boxes,
    add_complex_real
);
impl_complex_box_op!(
    Sub,
    sub,
    SubAssign,
    sub_assign,
    sub_complex_boxes,
    sub_complex_real
);
impl_complex_box_op!(
    Mul,
    mul,
    MulAssign,
    mul_assign,
    mul_complex_boxes,
    mul_complex_real
);
impl_complex_box_op!(
    Div,
    div,
    DivAssign,
    div_assign,
    div_complex_boxes,
    div_complex_real
);

impl_complex_real_scalar_op!(
    f64,
    Add,
    add,
    AddAssign,
    add_assign,
    add_complex_boxes,
    add_complex_real
);
impl_complex_real_scalar_op!(
    f64,
    Sub,
    sub,
    SubAssign,
    sub_assign,
    sub_complex_boxes,
    sub_complex_real
);
impl_complex_real_scalar_op!(
    f64,
    Mul,
    mul,
    MulAssign,
    mul_assign,
    mul_complex_boxes,
    mul_complex_real
);
impl_complex_real_scalar_op!(
    f64,
    Div,
    div,
    DivAssign,
    div_assign,
    div_complex_boxes,
    div_complex_real
);

impl_promoting_complex_op!(
    Add,
    add,
    AddAssign,
    add_assign,
    add_complex_boxes,
    add_complex_real
);
impl_promoting_complex_op!(
    Sub,
    sub,
    SubAssign,
    sub_assign,
    sub_complex_boxes,
    sub_complex_real
);
impl_promoting_complex_op!(
    Mul,
    mul,
    MulAssign,
    mul_assign,
    mul_complex_boxes,
    mul_complex_real
);
impl_promoting_complex_op!(
    Div,
    div,
    DivAssign,
    div_assign,
    div_complex_boxes,
    div_complex_real
);

#[cfg(feature = "num-complex")]
impl_complex_scalar_op!(
    Complex64,
    Add,
    add,
    AddAssign,
    add_assign,
    add_complex_boxes
);
#[cfg(feature = "num-complex")]
impl_complex_scalar_op!(
    Complex64,
    Sub,
    sub,
    SubAssign,
    sub_assign,
    sub_complex_boxes
);
#[cfg(feature = "num-complex")]
impl_complex_scalar_op!(
    Complex64,
    Mul,
    mul,
    MulAssign,
    mul_assign,
    mul_complex_boxes
);
#[cfg(feature = "num-complex")]
impl_complex_scalar_op!(
    Complex64,
    Div,
    div,
    DivAssign,
    div_assign,
    div_complex_boxes
);

#[cfg(feature = "num-complex")]
impl_num_complex_real_op!(Complex64, Add, add, add_complex_boxes);
#[cfg(feature = "num-complex")]
impl_num_complex_real_op!(Complex64, Sub, sub, sub_complex_boxes);
#[cfg(feature = "num-complex")]
impl_num_complex_real_op!(Complex64, Mul, mul, mul_complex_boxes);
#[cfg(feature = "num-complex")]
impl_num_complex_real_op!(Complex64, Div, div, div_complex_boxes);

impl<I: IntervalOps> From<I> for ComplexBox<I> {
    fn from(value: I) -> Self {
        Self::new(value, I::ZERO)
    }
}

impl<I: IntervalOps> From<&I> for ComplexBox<I> {
    fn from(value: &I) -> Self {
        (*value).into()
    }
}

impl<I: IntervalOps> From<f64> for ComplexBox<I> {
    fn from(value: f64) -> Self {
        Self::new(I::singleton(value), I::ZERO)
    }
}

impl<I: IntervalOps> From<&f64> for ComplexBox<I> {
    fn from(value: &f64) -> Self {
        (*value).into()
    }
}

#[cfg(feature = "num-complex")]
impl<I: IntervalOps> From<Complex64> for ComplexBox<I> {
    fn from(value: Complex64) -> Self {
        Self::new(I::singleton(value.re), I::singleton(value.im))
    }
}

#[cfg(feature = "num-complex")]
impl<I: IntervalOps> From<&Complex64> for ComplexBox<I> {
    fn from(value: &Complex64) -> Self {
        (*value).into()
    }
}

impl<I: IntervalOps> Default for ComplexBox<I> {
    fn default() -> Self {
        Self::new(I::ZERO, I::ZERO)
    }
}

impl<I: IntervalOps + PartialEq> PartialEq for ComplexBox<I> {
    fn eq(&self, other: &Self) -> bool {
        self.re == other.re && self.im == other.im
    }
}

impl<I: IntervalOps + Eq> Eq for ComplexBox<I> {}

impl<I: IntervalOps + fmt::Display> fmt::Display for ComplexBox<I> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "({}) + ({})i", self.re, self.im)
    }
}

impl ComplexBox<DecoratedInterval> {
    /// Returns the weaker decoration of the real and imaginary components.
    #[must_use]
    pub fn decoration(self) -> Decoration {
        core::cmp::min(self.re.decoration_raw(), self.im.decoration_raw())
    }
}

impl ComplexBox<Interval> {
    /// Decorates both components, reporting construction signals to `signals`.
    pub fn decorate<S: SignalSink>(self, signals: &mut S) -> ComplexBox<DecoratedInterval> {
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
