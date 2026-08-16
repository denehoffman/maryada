//! Shared scalar capabilities for interval enclosures.

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::{DecoratedInterval, Decoration, Interval};

macro_rules! impl_enclosure_methods {
    ($type:ty) => {
        #[allow(missing_docs)]
        impl $type {
            #[must_use]
            pub fn singleton(value: f64) -> Self {
                <Self as EnclosureScalar>::singleton(value)
            }

            #[must_use]
            pub fn from_midpoint(value: f64) -> Self {
                <Self as EnclosureScalar>::from_midpoint(value)
            }

            #[must_use]
            pub fn mul_add(self, rhs: Self, addend: Self) -> Self {
                <Self as EnclosureScalar>::mul_add(self, rhs, addend)
            }

            #[must_use]
            pub fn mid(self) -> f64 {
                <Self as EnclosureScalar>::mid(self)
            }

            #[must_use]
            pub fn is_empty(self) -> bool {
                <Self as EnclosureScalar>::is_empty(self)
            }

            #[must_use]
            pub fn is_nai(self) -> bool {
                <Self as EnclosureScalar>::is_nai(self)
            }

            #[must_use]
            pub fn is_entire(self) -> bool {
                <Self as EnclosureScalar>::is_entire(self)
            }

            #[must_use]
            pub fn is_singleton(self) -> bool {
                <Self as EnclosureScalar>::is_singleton(self)
            }

            #[must_use]
            pub fn is_bounded(self) -> bool {
                <Self as EnclosureScalar>::is_bounded(self)
            }

            #[must_use]
            pub fn equal(self, other: Self) -> bool {
                <Self as EnclosureScalar>::equal(self, other)
            }

            #[must_use]
            pub fn subset(self, other: Self) -> bool {
                <Self as EnclosureScalar>::subset(self, other)
            }

            #[must_use]
            pub fn interior(self, other: Self) -> bool {
                <Self as EnclosureScalar>::interior(self, other)
            }

            #[must_use]
            pub fn intersection(self, other: Self) -> Self {
                <Self as EnclosureScalar>::intersection(self, other)
            }

            #[must_use]
            pub fn convex_hull(self, other: Self) -> Self {
                <Self as EnclosureScalar>::convex_hull(self, other)
            }

            #[must_use]
            pub fn mag(self) -> f64 {
                <Self as EnclosureScalar>::mag(self)
            }

            #[must_use]
            pub fn mig(self) -> f64 {
                <Self as EnclosureScalar>::mig(self)
            }
        }
    };
}

/// Scalar operations shared by real and complex interval enclosures.
pub trait EnclosureScalar:
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

    /// Returns a representative midpoint.
    #[must_use]
    fn mid(self) -> Self::Midpoint;

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
impl EnclosureScalar for Interval {
    type Midpoint = f64;

    const ZERO: Self = Self::ZERO;
    const ONE: Self = Self::ONE;

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

    fn mid(self) -> Self::Midpoint {
        crate::mid(self)
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
impl EnclosureScalar for DecoratedInterval {
    type Midpoint = f64;

    const ZERO: Self = Self::ZERO;
    const ONE: Self = Self::ONE;

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

    fn mid(self) -> Self::Midpoint {
        crate::mid(self)
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

impl_enclosure_methods!(Interval);
impl_enclosure_methods!(DecoratedInterval);
