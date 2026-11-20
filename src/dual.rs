use core::ops::{Add, Div, Mul, Neg, Sub};

use crate::{EnclosureOps, Interval, IntervalOps};

/// A real interval value paired with a forward-mode derivative enclosure.
///
/// Seed the independent variable with [`Self::variable`] and use
/// [`Self::constant`] for parameters that do not vary with it. Evaluate the
/// same expression to enclose its value and derivative throughout the input.
/// A missing derivative propagates through later operations. For example,
/// a logarithm over an input that can be zero still encloses its defined
/// values, but carries no derivative certificate over the full input.
///
/// ```
/// use maryada::{Interval, IntervalDual, IntervalOps};
///
/// let x = IntervalDual::variable(Interval::new(1.0, 2.0));
/// let result = x.sqr() - 2.0;
/// assert_eq!(result.value, Interval::new(-1.0, 2.0));
/// assert_eq!(result.derivative, Some(Interval::new(2.0, 4.0)));
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IntervalDual {
    /// Enclosure of the expression's defined values.
    pub value: Interval,
    /// Enclosure of the derivative over the whole input, if certified.
    pub derivative: Option<Interval>,
}

#[allow(clippy::arithmetic_side_effects)]
impl IntervalDual {
    /// Seeds the derivative with one on a nonempty independent-variable range.
    #[must_use]
    pub const fn variable(value: Interval) -> Self {
        Self {
            value,
            derivative: if value.is_empty_raw() {
                None
            } else {
                Some(Interval::ONE)
            },
        }
    }

    /// Seeds the derivative with zero on a nonempty constant range.
    #[must_use]
    pub const fn constant(value: Interval) -> Self {
        Self {
            value,
            derivative: if value.is_empty_raw() {
                None
            } else {
                Some(Interval::ZERO)
            },
        }
    }

    fn chain(self, value: Interval, slope: impl FnOnce() -> Option<Interval>) -> Self {
        let derivative = self
            .derivative
            .and_then(|derivative| slope().map(|slope| derivative * slope));
        Self::from_parts(value, derivative)
    }

    fn from_parts(value: Interval, derivative: Option<Interval>) -> Self {
        Self {
            value,
            derivative: derivative.filter(|derivative| !value.is_empty() && !derivative.is_empty()),
        }
    }

    /// Encloses the reciprocal. The derivative requires an operand that excludes zero.
    #[must_use]
    pub fn recip(self) -> Self {
        let value = self.value.recip();
        self.chain(value, || {
            (!self.value.contains(0.0)).then(|| -(value / self.value))
        })
    }

    /// Encloses the square and its derivative.
    #[must_use]
    pub fn sqr(self) -> Self {
        let value = self.value.sqr();
        let derivative = self
            .derivative
            .map(|derivative| 2.0 * (self.value * derivative));
        Self::from_parts(value, derivative)
    }

    /// Encloses an integer power and its derivative.
    /// A negative exponent requires an operand that excludes zero.
    #[must_use]
    pub fn pown(self, exponent: i32) -> Self {
        let value = self.value.pown(exponent);
        self.chain(value, || {
            let factor = f64::from(exponent);
            match exponent.cmp(&0) {
                core::cmp::Ordering::Less if self.value.contains(0.0) => None,
                // Reuse the power to avoid decrementing i32::MIN.
                core::cmp::Ordering::Less => Some(factor * (value / self.value)),
                core::cmp::Ordering::Equal => Some(Interval::ZERO),
                core::cmp::Ordering::Greater => Some(factor * self.value.pown(exponent - 1)),
            }
        })
    }

    /// Encloses the square root. The derivative requires a strictly positive operand.
    #[must_use]
    pub fn sqrt(self) -> Self {
        let value = self.value.sqrt();
        self.chain(value, || {
            (self.value.inf() > 0.0).then(|| (2.0 * value).recip())
        })
    }

    /// Encloses the natural exponential and its derivative.
    #[must_use]
    pub fn exp(self) -> Self {
        let value = self.value.exp();
        self.chain(value, || Some(value))
    }

    /// Encloses the natural logarithm. The derivative requires a strictly positive operand.
    #[must_use]
    pub fn log(self) -> Self {
        self.chain(self.value.log(), || {
            (self.value.inf() > 0.0).then(|| self.value.recip())
        })
    }

    /// Encloses sine and its derivative.
    #[must_use]
    pub fn sin(self) -> Self {
        self.chain(self.value.sin(), || Some(self.value.cos()))
    }

    /// Encloses cosine and its derivative.
    #[must_use]
    pub fn cos(self) -> Self {
        self.chain(self.value.cos(), || Some(-self.value.sin()))
    }

    /// Encloses tangent. The derivative is available when the tangent range is bounded.
    #[must_use]
    pub fn tan(self) -> Self {
        let value = self.value.tan();
        // The tangent kernel returns Entire whenever it cannot rule out a pole.
        self.chain(value, || {
            value.is_bounded().then(|| Interval::ONE + value.sqr())
        })
    }

    /// Encloses absolute value. The derivative requires an operand that excludes zero.
    #[must_use]
    pub fn abs(self) -> Self {
        let value = self.value.abs();
        let derivative = if self.value.inf() > 0.0 {
            self.derivative
        } else if self.value.sup() < 0.0 {
            self.derivative.map(Neg::neg)
        } else {
            None
        };
        Self::from_parts(value, derivative)
    }
}

impl Neg for IntervalDual {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::from_parts(Neg::neg(self.value), self.derivative.map(Neg::neg))
    }
}

#[allow(clippy::arithmetic_side_effects)]
impl Add for IntervalDual {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let derivative = self
            .derivative
            .zip(rhs.derivative)
            .map(|(left, right)| left + right);
        Self::from_parts(self.value + rhs.value, derivative)
    }
}

#[allow(clippy::arithmetic_side_effects)]
impl Sub for IntervalDual {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let derivative = self
            .derivative
            .zip(rhs.derivative)
            .map(|(left, right)| left - right);
        Self::from_parts(self.value - rhs.value, derivative)
    }
}

#[allow(clippy::arithmetic_side_effects)]
impl Mul for IntervalDual {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let derivative = self
            .derivative
            .zip(rhs.derivative)
            .map(|(left, right)| left.mul_add(rhs.value, self.value * right));
        Self::from_parts(self.value * rhs.value, derivative)
    }
}

#[allow(clippy::arithmetic_side_effects)]
impl Div for IntervalDual {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let value = self.value / rhs.value;
        let derivative = if rhs.value.contains(0.0) {
            None
        } else {
            // Reuse the quotient to avoid squaring the denominator.
            self.derivative
                .zip(rhs.derivative)
                .map(|(left, right)| (-value).mul_add(right, left) / rhs.value)
        };
        Self::from_parts(value, derivative)
    }
}

macro_rules! scalar_operator {
    ($trait:ident, $method:ident, $operator:tt) => {
        impl $trait<f64> for IntervalDual {
            type Output = Self;

            #[allow(clippy::arithmetic_side_effects)]
            fn $method(self, rhs: f64) -> Self::Output {
                self $operator Self::constant(rhs.into())
            }
        }

        impl $trait<IntervalDual> for f64 {
            type Output = IntervalDual;

            #[allow(clippy::arithmetic_side_effects)]
            fn $method(self, rhs: IntervalDual) -> Self::Output {
                IntervalDual::constant(self.into()) $operator rhs
            }
        }
    };
}

scalar_operator!(Add, add, +);
scalar_operator!(Sub, sub, -);
scalar_operator!(Mul, mul, *);
scalar_operator!(Div, div, /);
