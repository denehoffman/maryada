//! Operations shared by real and complex interval enclosures.

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::{DecoratedInterval, Decoration, Interval};

/// Chaining-friendly operations shared by real and complex interval enclosures.
pub trait EnclosureOps:
    Copy
    + From<f64>
    + Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
{
    /// Point scalar used by midpoint-based algorithms.
    type Midpoint: Copy;

    /// Singleton zero enclosure.
    const ZERO: Self;
    /// Singleton one enclosure.
    const ONE: Self;
    /// The empty enclosure.
    const EMPTY: Self;
    /// The enclosure containing the whole scalar space.
    const ENTIRE: Self;

    /// Constructs a singleton real enclosure.
    #[must_use]
    fn singleton(value: f64) -> Self;

    /// Embeds a point midpoint as a singleton enclosure.
    #[must_use]
    fn from_midpoint(value: Self::Midpoint) -> Self;

    /// Inflates an enclosure by relative and absolute amounts.
    #[must_use]
    fn inflate(self, relative: f64, absolute: f64) -> Self;

    /// Encloses `self * rhs + addend`.
    #[must_use]
    fn mul_add(self, rhs: Self, addend: Self) -> Self;

    /// Returns the reciprocal enclosure.
    #[must_use]
    fn recip(self) -> Self;
    /// Returns the square enclosure.
    #[must_use]
    fn sqr(self) -> Self;
    /// Returns the square-root enclosure.
    #[must_use]
    fn sqrt(self) -> Self;
    /// Raises this enclosure to an integer power.
    #[must_use]
    fn pown(self, exponent: i32) -> Self;
    /// Alias for [`EnclosureOps::pown`].
    #[must_use]
    fn powi(self, exponent: i32) -> Self {
        self.pown(exponent)
    }
    /// Encloses powers with bases in `self` and exponents in `other`.
    #[must_use]
    fn pow(self, other: Self) -> Self;
    /// Applies the natural exponential function.
    #[must_use]
    fn exp(self) -> Self;
    /// Applies the base-two exponential function.
    #[must_use]
    fn exp2(self) -> Self;
    /// Applies the base-ten exponential function.
    #[must_use]
    fn exp10(self) -> Self;
    /// Applies the natural logarithm.
    #[must_use]
    fn log(self) -> Self;
    /// Applies the base-two logarithm.
    #[must_use]
    fn log2(self) -> Self;
    /// Applies the base-ten logarithm.
    #[must_use]
    fn log10(self) -> Self;
    /// Returns the sine enclosure.
    #[must_use]
    fn sin(self) -> Self;
    /// Returns the cosine enclosure.
    #[must_use]
    fn cos(self) -> Self;
    /// Returns the tangent enclosure.
    #[must_use]
    fn tan(self) -> Self;
    /// Returns the inverse-sine enclosure.
    #[must_use]
    fn asin(self) -> Self;
    /// Returns the inverse-cosine enclosure.
    #[must_use]
    fn acos(self) -> Self;
    /// Returns the inverse-tangent enclosure.
    #[must_use]
    fn atan(self) -> Self;
    /// Returns the hyperbolic-sine enclosure.
    #[must_use]
    fn sinh(self) -> Self;
    /// Returns the hyperbolic-cosine enclosure.
    #[must_use]
    fn cosh(self) -> Self;
    /// Returns the hyperbolic-tangent enclosure.
    #[must_use]
    fn tanh(self) -> Self;
    /// Returns the inverse-hyperbolic-sine enclosure.
    #[must_use]
    fn asinh(self) -> Self;
    /// Returns the inverse-hyperbolic-cosine enclosure.
    #[must_use]
    fn acosh(self) -> Self;
    /// Returns the inverse-hyperbolic-tangent enclosure.
    #[must_use]
    fn atanh(self) -> Self;

    /// Returns a representative midpoint.
    #[must_use]
    fn mid(self) -> Self::Midpoint;
    /// Returns whether a finite point belongs to this enclosure.
    #[must_use]
    fn contains(self, value: Self::Midpoint) -> bool;
    /// Returns the componentwise width.
    #[must_use]
    fn wid(self) -> Self::Midpoint;
    /// Returns an enclosing radius.
    #[must_use]
    fn rad(self) -> f64;
    /// Returns an inner radius.
    #[must_use]
    fn inner_rad(self) -> f64;
    /// Returns componentwise midpoints and radii.
    #[must_use]
    fn mid_rad(self) -> (Self::Midpoint, Self::Midpoint);

    /// Returns whether this enclosure is empty.
    #[must_use]
    fn is_empty(self) -> bool;
    /// Returns whether this enclosure is `NaI`.
    #[must_use]
    fn is_nai(self) -> bool;
    /// Returns whether this enclosure contains its whole scalar space.
    #[must_use]
    fn is_entire(self) -> bool;
    /// Returns whether this enclosure contains one scalar value.
    #[must_use]
    fn is_singleton(self) -> bool;
    /// Returns whether this enclosure is bounded by finite endpoints.
    #[must_use]
    fn is_bounded(self) -> bool;
    /// Returns whether two enclosures describe the same set.
    #[must_use]
    fn equal(self, other: Self) -> bool;
    /// Returns whether this enclosure is a subset of `other`.
    #[must_use]
    fn subset(self, other: Self) -> bool;
    /// Returns whether this enclosure lies strictly inside `other`.
    #[must_use]
    fn interior(self, other: Self) -> bool;
    /// Returns whether this enclosure and `other` are disjoint.
    #[must_use]
    fn disjoint(self, other: Self) -> bool;
    /// Returns whether this enclosure and `other` intersect.
    #[must_use]
    fn intersects(self, other: Self) -> bool {
        !self.is_nai() && !other.is_nai() && !self.disjoint(other)
    }
    /// Intersects two enclosures.
    #[must_use]
    fn intersection(self, other: Self) -> Self;
    /// Computes the smallest representable hull containing both enclosures.
    #[must_use]
    fn convex_hull(self, other: Self) -> Self;
    /// Returns an upper bound for the absolute value over this enclosure.
    #[must_use]
    fn mag(self) -> f64;
    /// Returns a lower bound for the absolute value over this enclosure.
    #[must_use]
    fn mig(self) -> f64;
}

#[allow(clippy::arithmetic_side_effects)]
impl EnclosureOps for Interval {
    type Midpoint = f64;

    const ZERO: Self = Self::ZERO;
    const ONE: Self = Self::ONE;
    const EMPTY: Self = Self::EMPTY;
    const ENTIRE: Self = Self::ENTIRE;

    fn singleton(value: f64) -> Self {
        Self::nums_to_interval(value, value, &mut ())
    }

    fn from_midpoint(value: Self::Midpoint) -> Self {
        Self::singleton(value)
    }

    fn inflate(self, relative: f64, absolute: f64) -> Self {
        self * Self::nums_to_interval(1.0 - relative, 1.0 + relative, &mut ())
            + Self::nums_to_interval(-absolute, absolute, &mut ())
    }

    fn mul_add(self, rhs: Self, addend: Self) -> Self {
        crate::fma(self, rhs, addend)
    }

    fn recip(self) -> Self {
        crate::recip(self)
    }
    fn sqr(self) -> Self {
        crate::sqr(self)
    }
    fn sqrt(self) -> Self {
        crate::sqrt(self)
    }
    fn pown(self, exponent: i32) -> Self {
        crate::pown(self, exponent)
    }
    fn pow(self, other: Self) -> Self {
        crate::pow(self, other)
    }
    fn exp(self) -> Self {
        crate::exp(self)
    }
    fn exp2(self) -> Self {
        crate::exp2(self)
    }
    fn exp10(self) -> Self {
        crate::exp10(self)
    }
    fn log(self) -> Self {
        crate::log(self)
    }
    fn log2(self) -> Self {
        crate::log2(self)
    }
    fn log10(self) -> Self {
        crate::log10(self)
    }
    fn sin(self) -> Self {
        crate::sin(self)
    }
    fn cos(self) -> Self {
        crate::cos(self)
    }
    fn tan(self) -> Self {
        crate::tan(self)
    }
    fn asin(self) -> Self {
        crate::asin(self)
    }
    fn acos(self) -> Self {
        crate::acos(self)
    }
    fn atan(self) -> Self {
        crate::atan(self)
    }
    fn sinh(self) -> Self {
        crate::sinh(self)
    }
    fn cosh(self) -> Self {
        crate::cosh(self)
    }
    fn tanh(self) -> Self {
        crate::tanh(self)
    }
    fn asinh(self) -> Self {
        crate::asinh(self)
    }
    fn acosh(self) -> Self {
        crate::acosh(self)
    }
    fn atanh(self) -> Self {
        crate::atanh(self)
    }

    fn mid(self) -> Self::Midpoint {
        crate::mid(self)
    }

    fn contains(self, value: Self::Midpoint) -> bool {
        value.is_finite() && Self::from(value).subset(self)
    }

    fn wid(self) -> Self::Midpoint {
        crate::wid(self)
    }
    fn rad(self) -> f64 {
        crate::rad(self)
    }
    fn inner_rad(self) -> f64 {
        if self.is_empty() {
            return f64::NAN;
        }
        crate::rounding::inner_radius(crate::inf(self), crate::sup(self), self.mid())
    }
    fn mid_rad(self) -> (Self::Midpoint, Self::Midpoint) {
        crate::mid_rad(self)
    }

    fn is_empty(self) -> bool {
        crate::is_empty(self)
    }

    fn is_nai(self) -> bool {
        false
    }

    fn is_entire(self) -> bool {
        crate::is_entire(self)
    }

    fn is_singleton(self) -> bool {
        !self.is_empty() && crate::inf(self) == crate::sup(self)
    }

    fn is_bounded(self) -> bool {
        !self.is_empty() && crate::inf(self).is_finite() && crate::sup(self).is_finite()
    }

    fn equal(self, other: Self) -> bool {
        crate::equal(self, other)
    }

    fn subset(self, other: Self) -> bool {
        crate::subset(self, other)
    }

    fn interior(self, other: Self) -> bool {
        crate::interior(self, other)
    }

    fn disjoint(self, other: Self) -> bool {
        crate::disjoint(self, other)
    }

    fn intersection(self, other: Self) -> Self {
        crate::intersection(self, other)
    }

    fn convex_hull(self, other: Self) -> Self {
        crate::convex_hull(self, other)
    }

    fn mag(self) -> f64 {
        crate::mag(self)
    }

    fn mig(self) -> f64 {
        crate::mig(self)
    }
}

#[allow(clippy::arithmetic_side_effects)]
impl EnclosureOps for DecoratedInterval {
    type Midpoint = f64;

    const ZERO: Self = Self::ZERO;
    const ONE: Self = Self::ONE;
    const EMPTY: Self = Self::EMPTY;
    const ENTIRE: Self = Self::ENTIRE;

    fn singleton(value: f64) -> Self {
        Self::nums_to_interval(value, value, &mut ())
    }

    fn from_midpoint(value: Self::Midpoint) -> Self {
        Self::singleton(value)
    }

    fn inflate(self, relative: f64, absolute: f64) -> Self {
        self * Self::nums_to_interval(1.0 - relative, 1.0 + relative, &mut ())
            + Self::nums_to_interval(-absolute, absolute, &mut ())
    }

    fn mul_add(self, rhs: Self, addend: Self) -> Self {
        crate::fma(self, rhs, addend)
    }

    fn recip(self) -> Self {
        crate::recip(self)
    }
    fn sqr(self) -> Self {
        crate::sqr(self)
    }
    fn sqrt(self) -> Self {
        crate::sqrt(self)
    }
    fn pown(self, exponent: i32) -> Self {
        crate::pown(self, exponent)
    }
    fn pow(self, other: Self) -> Self {
        crate::pow(self, other)
    }
    fn exp(self) -> Self {
        crate::exp(self)
    }
    fn exp2(self) -> Self {
        crate::exp2(self)
    }
    fn exp10(self) -> Self {
        crate::exp10(self)
    }
    fn log(self) -> Self {
        crate::log(self)
    }
    fn log2(self) -> Self {
        crate::log2(self)
    }
    fn log10(self) -> Self {
        crate::log10(self)
    }
    fn sin(self) -> Self {
        crate::sin(self)
    }
    fn cos(self) -> Self {
        crate::cos(self)
    }
    fn tan(self) -> Self {
        crate::tan(self)
    }
    fn asin(self) -> Self {
        crate::asin(self)
    }
    fn acos(self) -> Self {
        crate::acos(self)
    }
    fn atan(self) -> Self {
        crate::atan(self)
    }
    fn sinh(self) -> Self {
        crate::sinh(self)
    }
    fn cosh(self) -> Self {
        crate::cosh(self)
    }
    fn tanh(self) -> Self {
        crate::tanh(self)
    }
    fn asinh(self) -> Self {
        crate::asinh(self)
    }
    fn acosh(self) -> Self {
        crate::acosh(self)
    }
    fn atanh(self) -> Self {
        crate::atanh(self)
    }

    fn mid(self) -> Self::Midpoint {
        crate::mid(self)
    }

    fn contains(self, value: Self::Midpoint) -> bool {
        value.is_finite() && Self::from(value).subset(self)
    }

    fn wid(self) -> Self::Midpoint {
        crate::wid(self)
    }
    fn rad(self) -> f64 {
        crate::rad(self)
    }
    fn inner_rad(self) -> f64 {
        if self.is_nai() || self.is_empty() {
            return f64::NAN;
        }
        crate::rounding::inner_radius(crate::inf(self), crate::sup(self), self.mid())
    }
    fn mid_rad(self) -> (Self::Midpoint, Self::Midpoint) {
        crate::mid_rad(self)
    }

    fn is_empty(self) -> bool {
        crate::is_empty(self)
    }

    fn is_nai(self) -> bool {
        self.decoration_raw() == Decoration::Ill
    }

    fn is_entire(self) -> bool {
        crate::is_entire(self)
    }

    fn is_singleton(self) -> bool {
        !self.is_nai() && !self.is_empty() && crate::inf(self) == crate::sup(self)
    }

    fn is_bounded(self) -> bool {
        !self.is_nai() && crate::inf(self).is_finite() && crate::sup(self).is_finite()
    }

    fn equal(self, other: Self) -> bool {
        crate::equal(self, other)
    }

    fn subset(self, other: Self) -> bool {
        crate::subset(self, other)
    }

    fn interior(self, other: Self) -> bool {
        crate::interior(self, other)
    }

    fn disjoint(self, other: Self) -> bool {
        crate::disjoint(self, other)
    }

    fn intersection(self, other: Self) -> Self {
        crate::intersection(self, other)
    }

    fn convex_hull(self, other: Self) -> Self {
        crate::convex_hull(self, other)
    }

    fn mag(self) -> f64 {
        crate::mag(self)
    }

    fn mig(self) -> f64 {
        crate::mig(self)
    }
}
