//! Scalar capabilities shared by real and complex interval enclosures.

use crate::{DecoratedInterval, Interval, IntervalDatum};

#[cfg(feature = "complex")]
use crate::ComplexBox;

mod sealed {
    pub trait Sealed {}
}

impl sealed::Sealed for Interval {}
impl sealed::Sealed for DecoratedInterval {}

#[cfg(feature = "complex")]
impl<I: IntervalDatum> sealed::Sealed for ComplexBox<I> {}

/// A scalar enclosure supported by Maryada's generic numerical algorithms.
///
/// This trait is sealed because its implementations must uphold containment,
/// empty-set, and `NaI` semantics relied upon by verified algorithms.
pub trait Enclosure: sealed::Sealed + Copy {
    /// Returns the singleton zero enclosure.
    fn zero() -> Self;

    /// Returns the singleton one enclosure.
    fn one() -> Self;

    /// Returns whether this enclosure represents the empty set.
    fn is_empty(self) -> bool;

    /// Returns whether this enclosure contains a Not-an-Interval component.
    ///
    /// Bare real intervals never contain `NaI`.
    fn is_nai(self) -> bool;

    /// Convert an f64 into an enclosure.
    fn from_f64(value: f64) -> Self;
}

/// Arithmetic operations that preserve the enclosure guarantee.
///
/// This trait describes operations rather than algebraic laws: interval
/// enclosures are not fields or rings in the usual sense.
pub trait EnclosureArithmetic: Enclosure {
    /// Returns the additive inverse enclosure.
    #[must_use]
    fn neg(self) -> Self;

    /// Encloses every pairwise sum of values in the operands.
    #[must_use]
    fn add(self, rhs: Self) -> Self;

    /// Encloses every pairwise difference of values in the operands.
    #[must_use]
    fn sub(self, rhs: Self) -> Self;

    /// Encloses every pairwise product of values in the operands.
    #[must_use]
    fn mul(self, rhs: Self) -> Self;

    /// Encloses every defined pairwise quotient of values in the operands.
    #[must_use]
    fn div(self, rhs: Self) -> Self;

    /// Encloses `self * rhs + addend`, using fused interval operations where
    /// the scalar implementation provides them.
    #[must_use]
    fn mul_add(self, rhs: Self, addend: Self) -> Self;
}

/// An enclosure for which complex conjugation is defined.
///
/// Conjugation is the identity operation for real intervals.
pub trait Conjugate: Enclosure {
    /// Returns the complex-conjugate enclosure.
    #[must_use]
    fn conj(self) -> Self;
}

/// An enclosure with a representative point midpoint.
pub trait Midpoint: Enclosure {
    /// Point scalar produced by midpoint extraction.
    type Point: Copy;

    /// Returns a representative point midpoint.
    fn mid(self) -> Self::Point;
}

/// An enclosure with scalar bounds on the modulus of its values.
pub trait Magnitude: Enclosure {
    /// Real enclosure containing every modulus represented by `Self`.
    type Absolute: Enclosure;

    /// Returns an enclosure of the modulus of every enclosed value.
    fn abs(self) -> Self::Absolute;

    /// Returns an upper bound on the modulus of every enclosed value.
    fn mag(self) -> f64;

    /// Returns a lower bound on the modulus of every enclosed value.
    fn mig(self) -> f64;
}

/// An enclosure with radius and width measures.
pub trait Radius: Enclosure {
    /// Width representation produced by this enclosure.
    ///
    /// Real intervals use `f64`. Rectangular complex boxes use `Complex64`
    /// containing the componentwise real and imaginary widths.
    type Width: Copy;

    /// Returns an upward-rounded radius of the enclosure.
    fn rad(self) -> f64;

    /// Returns an downward-rounded radius of the enclosure.
    fn inner_rad(self) -> f64;

    /// Returns the width of the enclosure.
    fn wid(self) -> Self::Width;
}

macro_rules! impl_real_enclosure {
    ($interval:ty) => {
        impl Enclosure for $interval {
            fn zero() -> Self {
                crate::zero()
            }

            fn one() -> Self {
                crate::one()
            }

            fn is_empty(self) -> bool {
                crate::is_empty(self)
            }

            fn is_nai(self) -> bool {
                self.__is_nai()
            }

            fn from_f64(value: f64) -> Self {
                value.into()
            }
        }

        impl EnclosureArithmetic for $interval {
            fn neg(self) -> Self {
                crate::neg(self)
            }

            fn add(self, rhs: Self) -> Self {
                crate::add(self, rhs)
            }

            fn sub(self, rhs: Self) -> Self {
                crate::sub(self, rhs)
            }

            fn mul(self, rhs: Self) -> Self {
                crate::mul(self, rhs)
            }

            fn div(self, rhs: Self) -> Self {
                crate::div(self, rhs)
            }

            fn mul_add(self, rhs: Self, addend: Self) -> Self {
                crate::fma(self, rhs, addend)
            }
        }

        impl Conjugate for $interval {
            fn conj(self) -> Self {
                self
            }
        }

        impl Midpoint for $interval {
            type Point = f64;

            fn mid(self) -> Self::Point {
                crate::mid(self)
            }
        }

        impl Magnitude for $interval {
            type Absolute = Self;

            fn abs(self) -> Self::Absolute {
                crate::abs(self)
            }

            fn mag(self) -> f64 {
                crate::mag(self)
            }

            fn mig(self) -> f64 {
                crate::mig(self)
            }
        }

        impl Radius for $interval {
            type Width = f64;

            fn rad(self) -> f64 {
                crate::rad(self)
            }

            fn inner_rad(self) -> f64 {
                crate::inner_rad(self)
            }

            fn wid(self) -> Self::Width {
                crate::wid(self)
            }
        }
    };
}

impl_real_enclosure!(Interval);
impl_real_enclosure!(DecoratedInterval);

#[cfg(feature = "complex")]
impl<I: IntervalDatum> Enclosure for ComplexBox<I> {
    fn zero() -> Self {
        Self::new(crate::zero(), crate::zero())
    }

    fn one() -> Self {
        Self::new(crate::one(), crate::zero())
    }

    fn is_empty(self) -> bool {
        self.is_empty()
    }

    fn is_nai(self) -> bool {
        self.is_nai()
    }

    fn from_f64(value: f64) -> Self {
        value.into()
    }
}

#[cfg(feature = "complex")]
macro_rules! impl_complex_enclosure_arithmetic {
    ($interval:ty) => {
        impl EnclosureArithmetic for ComplexBox<$interval> {
            fn neg(self) -> Self {
                core::ops::Neg::neg(self)
            }

            fn add(self, rhs: Self) -> Self {
                core::ops::Add::add(self, rhs)
            }

            fn sub(self, rhs: Self) -> Self {
                core::ops::Sub::sub(self, rhs)
            }

            fn mul(self, rhs: Self) -> Self {
                core::ops::Mul::mul(self, rhs)
            }

            fn div(self, rhs: Self) -> Self {
                core::ops::Div::div(self, rhs)
            }

            fn mul_add(self, rhs: Self, addend: Self) -> Self {
                self.mul_add(rhs, addend)
            }
        }
    };
}

#[cfg(feature = "complex")]
impl_complex_enclosure_arithmetic!(Interval);
#[cfg(feature = "complex")]
impl_complex_enclosure_arithmetic!(DecoratedInterval);

#[cfg(feature = "complex")]
impl<I: IntervalDatum> Conjugate for ComplexBox<I> {
    fn conj(self) -> Self {
        self.conj()
    }
}

#[cfg(feature = "num-complex")]
impl<I: IntervalDatum> Midpoint for ComplexBox<I> {
    type Point = num_complex::Complex64;

    fn mid(self) -> Self::Point {
        self.mid()
    }
}

#[cfg(feature = "complex")]
impl<I: Enclosure + IntervalDatum> Magnitude for ComplexBox<I> {
    type Absolute = I;

    fn abs(self) -> Self::Absolute {
        self.abs()
    }

    fn mag(self) -> f64 {
        self.mag()
    }

    fn mig(self) -> f64 {
        self.mig()
    }
}

#[cfg(feature = "num-complex")]
impl<I: IntervalDatum> Radius for ComplexBox<I> {
    type Width = num_complex::Complex64;

    fn rad(self) -> f64 {
        self.rad()
    }

    fn inner_rad(self) -> f64 {
        self.inner_rad()
    }

    fn wid(self) -> Self::Width {
        self.wid()
    }
}
