use core::{
    cmp::Ordering,
    fmt,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

#[cfg(feature = "num-complex")]
use num_complex::Complex64;

use crate::{
    ComplexBox, ComplexEnclosure, EnclosureOps, Interval, IntervalOps,
    complex::{division_exponent, scale_box},
    ops::div_interval,
    rounding::{self, Direction, ExactSum, WideFloatClass},
};

/// An invalid disk input or an unavailable finite disk hull.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComplexDiskError {
    /// A center coordinate is infinite or NaN.
    NonFiniteCenter,
    /// The radius is negative, infinite, or NaN.
    InvalidRadius,
    /// The box is unbounded and is a proper subset of the complex plane.
    UnboundedBox,
    /// The enclosing radius would overflow binary64 at the selected center.
    RadiusOverflow,
}

impl fmt::Display for ComplexDiskError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::NonFiniteCenter => "the disk center must be finite",
            Self::InvalidRadius => "the disk radius must be finite and nonnegative",
            Self::UnboundedBox => "an unbounded box cannot have a finite disk hull",
            Self::RadiusOverflow => "the enclosing disk radius overflows binary64",
        })
    }
}

impl core::error::Error for ComplexDiskError {}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Disk {
    Empty,
    Entire,
    Bounded { center: (f64, f64), radius: f64 },
}

/// A closed disk in the complex plane, with binary64 center coordinates and radius.
///
/// Disk arithmetic encloses the results of varying both operands independently.
/// Convert to [`ComplexBox<Interval>`] for the rectangular hull, or use
/// [`TryFrom`] to enclose a box in a disk. Both hull conversions can widen the set.
/// Operations whose enclosure cannot be represented as a finite disk return
/// [`Self::ENTIRE`]. Elementary functions use the principal branches.
///
/// ```
/// use maryada::{ComplexDisk, ComplexDiskError, IntervalOps};
///
/// # fn main() -> Result<(), ComplexDiskError> {
/// let disk = ComplexDisk::try_new((3.0, 4.0), 1.0)?;
/// assert_eq!(disk.abs().bounds(), (4.0, 6.0));
/// let scale = ComplexDisk::try_new((0.0, 2.0), 0.0)?;
/// assert_eq!((disk * scale).center(), Some((-8.0, 6.0)));
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ComplexDisk(Disk);

impl ComplexDisk {
    /// The empty complex set.
    pub const EMPTY: Self = Self(Disk::Empty);
    /// The entire complex plane.
    pub const ENTIRE: Self = Self(Disk::Entire);
    /// The singleton disk at the origin.
    pub const ZERO: Self = Self(Disk::Bounded {
        center: (0.0, 0.0),
        radius: 0.0,
    });
    /// The singleton complex one.
    pub const ONE: Self = Self(Disk::Bounded {
        center: (1.0, 0.0),
        radius: 0.0,
    });
    /// The singleton imaginary unit.
    pub const I: Self = Self(Disk::Bounded {
        center: (0.0, 1.0),
        radius: 0.0,
    });

    /// Constructs a closed disk around the given real and imaginary coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`ComplexDiskError::NonFiniteCenter`] if either coordinate is
    /// nonfinite, or [`ComplexDiskError::InvalidRadius`] if the radius is
    /// negative or nonfinite. A radius of negative zero is treated as zero.
    pub fn try_new(center: (f64, f64), radius: f64) -> Result<Self, ComplexDiskError> {
        if !center.0.is_finite() || !center.1.is_finite() {
            return Err(ComplexDiskError::NonFiniteCenter);
        }
        if !radius.is_finite() || radius < 0.0 {
            return Err(ComplexDiskError::InvalidRadius);
        }
        Ok(Self(Disk::Bounded {
            center,
            radius: radius.abs(),
        }))
    }

    /// Returns the real and imaginary coordinates of a finite disk's center.
    #[must_use]
    pub const fn center(self) -> Option<(f64, f64)> {
        match self.0 {
            Disk::Bounded { center, .. } => Some(center),
            Disk::Empty | Disk::Entire => None,
        }
    }

    /// Returns the radius, with infinity for Entire and `None` for Empty.
    #[must_use]
    pub const fn radius(self) -> Option<f64> {
        match self.0 {
            Disk::Bounded { radius, .. } => Some(radius),
            Disk::Empty => None,
            Disk::Entire => Some(f64::INFINITY),
        }
    }

    /// Returns whether the disk is Empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        matches!(self.0, Disk::Empty)
    }

    /// Returns whether the disk is Entire.
    #[must_use]
    pub const fn is_entire(self) -> bool {
        matches!(self.0, Disk::Entire)
    }

    /// Returns whether the disk has a finite center and radius.
    #[must_use]
    pub const fn is_bounded(self) -> bool {
        matches!(self.0, Disk::Bounded { .. })
    }

    /// Returns whether the disk contains exactly one point.
    #[must_use]
    pub const fn is_singleton(self) -> bool {
        matches!(self.0, Disk::Bounded { radius: 0.0, .. })
    }

    /// Encloses the modulus of every point in the disk.
    #[must_use]
    pub fn abs(self) -> Interval {
        let (center, radius) = match self.0 {
            Disk::Empty => return Interval::EMPTY,
            Disk::Entire => return Interval::from_valid_bounds(0.0, f64::INFINITY),
            Disk::Bounded { center, radius } => (center, radius),
        };
        let upper_norm = rounding::hypot(center.0, center.1, Direction::Up);
        let lower = if upper_norm.is_infinite() {
            // The norm can overflow even when its difference with the radius
            // fits. Compute the lower endpoint at half scale in that case.
            let half = |value: f64| rounding::scale_pow2(value.abs(), -1, Direction::Down);
            let norm = rounding::hypot(half(center.0), half(center.1), Direction::Down);
            let radius = rounding::scale_pow2(radius, -1, Direction::Up);
            let difference = rounding::sub(norm, radius, Direction::Down);
            rounding::scale_pow2(difference, 1, Direction::Down)
        } else {
            let norm = rounding::hypot(center.0, center.1, Direction::Down);
            rounding::sub(norm, radius, Direction::Down)
        };
        Interval::from_valid_bounds(
            lower.max(0.0),
            rounding::add(upper_norm, radius, Direction::Up),
        )
    }

    /// Returns whether the disk is the singleton zero.
    #[must_use]
    pub const fn is_zero(self) -> bool {
        matches!(
            self.0,
            Disk::Bounded {
                center: (0.0, 0.0),
                radius: 0.0
            }
        )
    }

    /// Returns whether the disk is a singleton on the real axis.
    #[must_use]
    pub const fn is_real(self) -> bool {
        matches!(
            self.0,
            Disk::Bounded {
                center: (_, 0.0),
                radius: 0.0
            }
        )
    }

    /// Returns the complex conjugate.
    #[must_use]
    pub const fn conj(self) -> Self {
        match self.0 {
            Disk::Bounded {
                center: (re, im),
                radius,
            } => Self(Disk::Bounded {
                center: (re, -im),
                radius,
            }),
            Disk::Empty | Disk::Entire => self,
        }
    }

    /// Encloses the squared modulus.
    #[must_use]
    pub fn norm_sqr(self) -> Interval {
        self.abs().sqr()
    }

    /// Alias for [`Self::abs`].
    #[must_use]
    pub fn norm(self) -> Interval {
        self.abs()
    }

    /// Encloses the principal argument in radians, using the rectangular hull.
    #[must_use]
    pub fn arg(self) -> Interval {
        ComplexBox::from(self).arg()
    }

    /// Returns modulus and principal argument enclosures.
    #[must_use]
    pub fn to_polar(self) -> (Interval, Interval) {
        (self.abs(), self.arg())
    }

    /// Encloses the points `cos(phase) + i sin(phase)`.
    #[must_use]
    pub fn cis(phase: Interval) -> Self {
        Self::try_from(ComplexBox::cis(phase)).unwrap_or(Self::ENTIRE)
    }

    /// Encloses points given by the radius and phase intervals.
    #[must_use]
    pub fn from_polar(radius: Interval, phase: Interval) -> Self {
        Self::try_from(ComplexBox::from_polar(radius, phase)).unwrap_or(Self::ENTIRE)
    }

    /// Encloses the product, trying an outward-shifted center for a smaller radius.
    ///
    /// Uses the recentered result only when it certifies a smaller radius than
    /// [`Mul::mul`]. A smaller radius can still give a wider real, imaginary,
    /// or modulus bound. This method costs more than fixed-center multiplication.
    #[must_use]
    #[allow(clippy::arithmetic_side_effects)]
    pub fn mul_recentered(self, rhs: Self) -> Self {
        let fixed = self * rhs;
        let (
            Disk::Bounded {
                center: left_center,
                radius: left_radius,
            },
            Disk::Bounded {
                center: right_center,
                radius: right_radius,
            },
        ) = (self.0, rhs.0)
        else {
            return fixed;
        };
        if left_radius == 0.0
            || right_radius == 0.0
            || left_center == (0.0, 0.0)
            || right_center == (0.0, 0.0)
        {
            return fixed;
        }
        let a_norm = point_box(left_center).abs();
        let b_norm = point_box(right_center).abs();
        if a_norm.inf_raw() <= 0.0
            || b_norm.inf_raw() <= 0.0
            || !a_norm.is_bounded_raw()
            || !b_norm.is_bounded_raw()
        {
            return fixed;
        }
        let Some(u) = recenter_parameter(
            left_radius,
            right_radius,
            a_norm.sup_raw(),
            b_norm.sup_raw(),
        ) else {
            return fixed;
        };

        // Rotate and scale the centers to 1. Write alpha = r/|a| and
        // beta = s/|b|. For t = cos(theta), the farthest product from
        // normalized center 1 + alpha*u has distance at most
        // alpha*sqrt(1 + u² - 2*u*t) + beta*sqrt(1 + alpha² + 2*alpha*t).
        // Cauchy–Schwarz cancels t for any positive u, giving the bound below.
        let alpha = Interval::from(left_radius) / a_norm;
        let beta = Interval::from(right_radius) / b_norm;
        let u = Interval::from(u);
        let shift = alpha * u;
        let bound =
            ((shift + beta.sqr()) * (Interval::ONE + alpha.sqr() + shift + alpha / u)).sqrt();
        let center = product_center(left_center, right_center) * (Interval::ONE + shift);
        let candidate = bounded_with_center(center, (a_norm * b_norm * bound).sup_raw());
        match candidate.0 {
            Disk::Bounded { radius, .. } if radius < fixed.radius().unwrap_or(f64::INFINITY) => {
                candidate
            }
            _ => fixed,
        }
    }

    /// Returns whether the disk contains the given finite complex point.
    #[must_use]
    pub fn contains(self, point: <ComplexBox<Interval> as EnclosureOps>::Midpoint) -> bool {
        #[cfg(feature = "num-complex")]
        let point = (point.re, point.im);
        if !point.0.is_finite() || !point.1.is_finite() {
            return false;
        }
        match self.0 {
            Disk::Empty => false,
            Disk::Entire => true,
            Disk::Bounded { center, radius } => !disk_gap(center, point, radius, 0.0).is_gt(),
        }
    }

    /// Returns whether every point in this disk belongs to `other`.
    #[must_use]
    pub fn subset(self, other: Self) -> bool {
        match (self.0, other.0) {
            (Disk::Empty, _) | (_, Disk::Entire) => true,
            (Disk::Entire, _) | (_, Disk::Empty) => false,
            (
                Disk::Bounded {
                    center: a,
                    radius: r,
                },
                Disk::Bounded {
                    center: b,
                    radius: s,
                },
            ) => r <= s && !disk_gap(a, b, s, -r).is_gt(),
        }
    }

    /// Returns whether this disk lies strictly inside `other`.
    #[must_use]
    pub fn interior(self, other: Self) -> bool {
        match (self.0, other.0) {
            (Disk::Empty, _) | (_, Disk::Entire) => true,
            (Disk::Entire, _) | (_, Disk::Empty) => false,
            (
                Disk::Bounded {
                    center: a,
                    radius: r,
                },
                Disk::Bounded {
                    center: b,
                    radius: s,
                },
            ) => r < s && disk_gap(a, b, s, -r).is_lt(),
        }
    }

    /// Returns whether the disks have no points in common.
    #[must_use]
    pub fn disjoint(self, other: Self) -> bool {
        match (self.0, other.0) {
            (Disk::Empty, _) | (_, Disk::Empty) => true,
            (Disk::Entire, _) | (_, Disk::Entire) => false,
            (
                Disk::Bounded {
                    center: a,
                    radius: r,
                },
                Disk::Bounded {
                    center: b,
                    radius: s,
                },
            ) => disk_gap(a, b, r, s).is_gt(),
        }
    }

    /// Returns whether the disks have at least one point in common.
    #[must_use]
    pub fn intersects(self, other: Self) -> bool {
        !self.disjoint(other)
    }

    /// Encloses the reciprocal of every nonzero point in the disk.
    ///
    /// A disk reaching zero with positive radius has unbounded reciprocals.
    /// A singleton zero gives Empty.
    #[must_use]
    #[allow(clippy::arithmetic_side_effects)]
    pub fn recip(self) -> Self {
        let Disk::Bounded {
            center: (re, im),
            radius,
        } = self.0
        else {
            return self;
        };
        if self.is_zero() {
            return Self::EMPTY;
        }
        let exponent = division_exponent(re.abs().max(im.abs()).max(radius));
        let scaled = scale_box(point_box((re, -im)), -exponent);
        let r = Interval::from_valid_bounds(
            rounding::scale_pow2(radius, -exponent, Direction::Down),
            rounding::scale_pow2(radius, -exponent, Direction::Up),
        );
        // Inversion takes a circle avoiding zero to a circle with center
        // conj(c)/(|c|²-r²) and radius r/(|c|²-r²). Scale that denominator
        // before squaring to preserve tiny and huge finite denominators.
        let mut denominator = scaled.norm_sqr() - r.sqr();
        if denominator.inf_raw() <= 0.0 {
            // Factor the dominant square difference to preserve gaps such as
            // (1 + i*y, radius 1), where the entire denominator is y².
            let (major, minor) = if re.abs() >= im.abs() {
                (scaled.re.abs(), scaled.im.abs())
            } else {
                (scaled.im.abs(), scaled.re.abs())
            };
            denominator = (major - r) * (major + r) + minor.sqr();
        }
        if denominator.inf_raw() > 0.0 {
            let divide = |value| {
                div_interval(value, denominator, |x, y, d| {
                    rounding::div_scaled(x, y, -exponent, d)
                })
            };
            return bounded_with_center(
                ComplexBox::new(divide(scaled.re), divide(scaled.im)),
                divide(r).sup_raw(),
            );
        }
        let distance = self.abs().inf_raw();
        if distance <= 0.0 {
            return Self::ENTIRE;
        }
        // When cancellation prevents certifying the circle formula, bound
        // the reciprocals in a disk centered at zero.
        Self::try_new((0.0, 0.0), rounding::recip(distance, Direction::Up)).unwrap_or(Self::ENTIRE)
    }

    /// Encloses the sum of the disk product and the addend.
    #[must_use]
    pub fn mul_add(self, rhs: Self, addend: Self) -> Self {
        Add::add(Mul::mul(self, rhs), addend)
    }

    /// Encloses the square of each point in the disk.
    #[must_use]
    pub fn sqr(self) -> Self {
        Mul::mul(self, self)
    }

    /// Encloses integer powers of points in the disk.
    #[must_use]
    pub fn pown(self, exponent: i32) -> Self {
        if self.is_empty() {
            return Self::EMPTY;
        }
        let mut base = if exponent < 0 { self.recip() } else { self };
        let mut exponent = exponent.unsigned_abs();
        let mut result = Self::ONE;
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

    /// Alias for [`Self::pown`].
    #[must_use]
    pub fn powi(self, exponent: i32) -> Self {
        self.pown(exponent)
    }

    /// Encloses powers with bases in this disk and exponents in `other`.
    #[must_use]
    pub fn pow(self, other: Self) -> Self {
        Self::try_from(ComplexBox::from(self).pow(ComplexBox::from(other))).unwrap_or(Self::ENTIRE)
    }
}

// Use the same elementary-function enclosures as ComplexBox. Returning a disk
// requires a second hull, with Entire as the fallback for unbounded images.
macro_rules! disk_elementary_functions {
    ($($(#[$doc:meta])* $method:ident),+ $(,)?) => {
        impl ComplexDisk {
            $(
                $(#[$doc])*
                #[must_use]
                pub fn $method(self) -> Self {
                    Self::try_from(ComplexBox::from(self).$method()).unwrap_or(Self::ENTIRE)
                }
            )+
        }
    };
}

disk_elementary_functions! {
    /// Encloses the principal square root.
    sqrt,
    /// Encloses the natural exponential.
    exp,
    /// Encloses the base-two exponential.
    exp2,
    /// Encloses the base-ten exponential.
    exp10,
    /// Encloses the principal natural logarithm.
    log,
    /// Encloses the principal base-two logarithm.
    log2,
    /// Encloses the principal base-ten logarithm.
    log10,
    /// Encloses the sine.
    sin,
    /// Encloses the cosine.
    cos,
    /// Encloses the tangent.
    tan,
    /// Encloses the principal inverse sine.
    asin,
    /// Encloses the principal inverse cosine.
    acos,
    /// Encloses the principal inverse tangent.
    atan,
    /// Encloses the hyperbolic sine.
    sinh,
    /// Encloses the hyperbolic cosine.
    cosh,
    /// Encloses the hyperbolic tangent.
    tanh,
    /// Encloses the principal inverse hyperbolic sine.
    asinh,
    /// Encloses the principal inverse hyperbolic cosine.
    acosh,
    /// Encloses the principal inverse hyperbolic tangent.
    atanh,
}

impl TryFrom<ComplexBox<Interval>> for ComplexDisk {
    type Error = ComplexDiskError;

    /// Encloses a box in a disk centered at its rounded midpoint.
    ///
    /// The radius bounds the distance to the farthest box corner. Empty and
    /// Entire retain their meanings. A proper unbounded box or overflowing
    /// enclosing radius returns an error.
    fn try_from(value: ComplexBox<Interval>) -> Result<Self, Self::Error> {
        if value.is_empty() {
            return Ok(Self::EMPTY);
        }
        if value.is_entire() {
            return Ok(Self::ENTIRE);
        }
        if !value.is_bounded() {
            return Err(ComplexDiskError::UnboundedBox);
        }
        let (center, radius) = midpoint_and_radius(value);
        if !radius.is_finite() {
            return Err(ComplexDiskError::RadiusOverflow);
        }
        Ok(Self(Disk::Bounded { center, radius }))
    }
}

impl TryFrom<Interval> for ComplexDisk {
    type Error = ComplexDiskError;

    /// Encloses a real interval in a disk. Unbounded real intervals return an error.
    fn try_from(value: Interval) -> Result<Self, Self::Error> {
        Self::try_from(ComplexBox::from(value))
    }
}

impl From<ComplexDisk> for ComplexBox<Interval> {
    /// Returns the tightest binary64 rectangular enclosure of the disk.
    fn from(disk: ComplexDisk) -> Self {
        match disk.0 {
            Disk::Empty => Self::EMPTY,
            Disk::Entire => Self::ENTIRE,
            Disk::Bounded { center, radius } => {
                let projection = |coordinate| {
                    Interval::from_valid_bounds(
                        rounding::sub(coordinate, radius, Direction::Down),
                        rounding::add(coordinate, radius, Direction::Up),
                    )
                };
                Self::new(projection(center.0), projection(center.1))
            }
        }
    }
}

impl TryFrom<(f64, f64)> for ComplexDisk {
    type Error = ComplexDiskError;

    fn try_from(point: (f64, f64)) -> Result<Self, Self::Error> {
        Self::try_new(point, 0.0)
    }
}

impl From<f64> for ComplexDisk {
    /// Embeds a finite real point. A nonfinite value gives Empty.
    fn from(value: f64) -> Self {
        Self::try_from((value, 0.0)).unwrap_or(Self::EMPTY)
    }
}

#[cfg(feature = "num-complex")]
impl From<Complex64> for ComplexDisk {
    /// Embeds a finite complex point. A nonfinite coordinate gives Empty.
    fn from(value: Complex64) -> Self {
        Self::try_from((value.re, value.im)).unwrap_or(Self::EMPTY)
    }
}

impl Default for ComplexDisk {
    fn default() -> Self {
        Self::ZERO
    }
}

impl ComplexEnclosure for ComplexDisk {
    type Real = Interval;

    fn abs(self) -> Interval {
        Self::abs(self)
    }

    fn arg(self) -> Interval {
        Self::arg(self)
    }

    fn conj(self) -> Self {
        Self::conj(self)
    }
}

#[allow(clippy::arithmetic_side_effects)]
fn recenter_parameter(r: f64, s: f64, a_norm: f64, b_norm: f64) -> Option<f64> {
    let alpha = r / a_norm;
    let beta = s / b_norm;
    let base = 1.0 + alpha * alpha + beta * beta;
    // One refinement of the cubic for the radius-minimizing shift.
    let seed = beta / libm::sqrt(base);
    let parameter = beta / libm::sqrt(base + 2.0 * alpha * seed);
    (parameter.is_finite() && parameter > 0.0).then_some(parameter)
}

fn point_box(point: (f64, f64)) -> ComplexBox<Interval> {
    ComplexBox::new(Interval::from(point.0), Interval::from(point.1))
}

#[inline]
fn product_center((ar, ai): (f64, f64), (br, bi): (f64, f64)) -> ComplexBox<Interval> {
    let (re_lo, re_hi) = rounding::two_product_bounds((ar, br), (-ai, bi));
    let (im_lo, im_hi) = rounding::two_product_bounds((ar, bi), (ai, br));
    ComplexBox::new(
        Interval::from_valid_bounds(re_lo, re_hi),
        Interval::from_valid_bounds(im_lo, im_hi),
    )
}

fn bounded_with_center(center: ComplexBox<Interval>, radius: f64) -> ComplexDisk {
    if !center.is_bounded() {
        return ComplexDisk::ENTIRE;
    }
    let (center, displacement) = midpoint_and_radius(center);
    let radius = rounding::add(radius, displacement, Direction::Up);
    if radius.is_finite() {
        ComplexDisk(Disk::Bounded { center, radius })
    } else {
        ComplexDisk::ENTIRE
    }
}

fn disk_gap((ax, ay): (f64, f64), (bx, by): (f64, f64), r: f64, s: f64) -> Ordering {
    // Expand |a-b|² - (r+s)² into exact products. The sign test
    // works at tangency and across the full binary64 exponent range.
    let mut sum = ExactSum::new();
    for (lhs, rhs) in [
        (ax, ax),
        (ay, ay),
        (bx, bx),
        (by, by),
        (ax, -bx),
        (ax, -bx),
        (ay, -by),
        (ay, -by),
        (r, -r),
        (s, -s),
        (r, -s),
        (r, -s),
    ] {
        sum.add_product(WideFloatClass::exact_product(lhs, rhs));
    }
    sum.sign()
}

fn midpoint_and_radius(value: ComplexBox<Interval>) -> ((f64, f64), f64) {
    let coordinate = |interval: Interval| {
        let lower = interval.inf_raw();
        let upper = interval.sup_raw();
        if lower == upper {
            return (lower, 0.0);
        }
        let midpoint = lower.midpoint(upper);
        let deviation = rounding::sub(midpoint, lower, Direction::Up).max(rounding::sub(
            upper,
            midpoint,
            Direction::Up,
        ));
        (midpoint, deviation)
    };
    let (re, dx) = coordinate(value.re);
    let (im, dy) = coordinate(value.im);
    ((re, im), rounding::hypot(dx, dy, Direction::Up))
}

impl Neg for ComplexDisk {
    type Output = Self;

    fn neg(self) -> Self {
        match self.0 {
            Disk::Bounded {
                center: (re, im),
                radius,
            } => Self(Disk::Bounded {
                center: (-re, -im),
                radius,
            }),
            Disk::Empty | Disk::Entire => self,
        }
    }
}

impl Add for ComplexDisk {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        if self.is_empty() || rhs.is_empty() {
            return Self::EMPTY;
        }
        let (
            Disk::Bounded {
                center: a,
                radius: r,
            },
            Disk::Bounded {
                center: b,
                radius: s,
            },
        ) = (self.0, rhs.0)
        else {
            return Self::ENTIRE;
        };
        bounded_with_center(
            Add::add(point_box(a), point_box(b)),
            rounding::add(r, s, Direction::Up),
        )
    }
}

impl Sub for ComplexDisk {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Add::add(self, Neg::neg(rhs))
    }
}

impl Mul for ComplexDisk {
    type Output = Self;

    /// Encloses the product in a disk centered at the rounded product of the centers.
    ///
    /// For centers `a` and `b` and radii `r` and `s`, the radius bounds
    /// `|a|s + |b|r + rs` plus the displacement due to rounding the center.
    /// Returns Entire when this construction yields an infinite bound.
    fn mul(self, rhs: Self) -> Self {
        if self.is_empty() || rhs.is_empty() {
            return Self::EMPTY;
        }
        if self.is_zero() || rhs.is_zero() {
            return Self::ZERO;
        }
        let (
            Disk::Bounded {
                center: (ar, ai),
                radius: r,
            },
            Disk::Bounded {
                center: (br, bi),
                radius: s,
            },
        ) = (self.0, rhs.0)
        else {
            return Self::ENTIRE;
        };
        let product = product_center((ar, ai), (br, bi));
        if !product.re.is_bounded_raw() || !product.im.is_bounded_raw() {
            return Self::ENTIRE;
        }
        let (center, displacement) = midpoint_and_radius(product);

        // Distribute each radius before taking the norm. The norm of
        // the raw center can overflow even when this term fits.
        let weighted_norm = |re: f64, im: f64, weight| {
            if weight == 0.0 {
                return 0.0;
            }
            rounding::hypot(
                rounding::mul(re.abs(), weight, Direction::Up),
                rounding::mul(im.abs(), weight, Direction::Up),
                Direction::Up,
            )
        };
        let linear = rounding::add(
            weighted_norm(ar, ai, s),
            weighted_norm(br, bi, r),
            Direction::Up,
        );
        let quadratic = rounding::mul(r, s, Direction::Up);
        let radius = rounding::add(
            rounding::add(linear, quadratic, Direction::Up),
            displacement,
            Direction::Up,
        );
        if radius.is_finite() {
            Self(Disk::Bounded { center, radius })
        } else {
            Self::ENTIRE
        }
    }
}

impl Div for ComplexDisk {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        if self.is_empty() || rhs.is_empty() || rhs.is_zero() {
            return Self::EMPTY;
        }
        let inverse = rhs.recip();
        if self.is_bounded() && rhs.is_bounded() && inverse.is_entire() {
            // The reciprocal can overflow while the quotient still fits.
            let quotient = Div::div(ComplexBox::from(self), ComplexBox::from(rhs));
            return Self::try_from(quotient).unwrap_or(Self::ENTIRE);
        }
        Mul::mul(self, inverse)
    }
}

macro_rules! disk_assignment_ops {
    ($rhs:ty; $($operator:ident, $assign:ident, $op:ident, $set:ident);+ $(;)?) => {
        $(
            impl $assign<$rhs> for ComplexDisk {
                fn $set(&mut self, rhs: $rhs) {
                    *self = $operator::$op(*self, rhs);
                }
            }
        )+
    };
}

macro_rules! disk_point_ops {
    ($point:ty; $($operator:ident, $assign:ident, $op:ident, $set:ident);+ $(;)?) => {
        $(
            impl $operator<$point> for ComplexDisk {
                type Output = Self;

                fn $op(self, rhs: $point) -> Self {
                    $operator::$op(self, Self::from(rhs))
                }
            }

            impl $operator<ComplexDisk> for $point {
                type Output = ComplexDisk;

                fn $op(self, rhs: ComplexDisk) -> ComplexDisk {
                    $operator::$op(ComplexDisk::from(self), rhs)
                }
            }
        )+
        disk_assignment_ops!($point; $($operator, $assign, $op, $set);+);
    };
}

disk_assignment_ops! {
    ComplexDisk;
    Add, AddAssign, add, add_assign;
    Sub, SubAssign, sub, sub_assign;
    Mul, MulAssign, mul, mul_assign;
    Div, DivAssign, div, div_assign;
}

disk_point_ops! {
    f64;
    Add, AddAssign, add, add_assign;
    Sub, SubAssign, sub, sub_assign;
    Mul, MulAssign, mul, mul_assign;
    Div, DivAssign, div, div_assign;
}

#[cfg(feature = "num-complex")]
disk_point_ops! {
    Complex64;
    Add, AddAssign, add, add_assign;
    Sub, SubAssign, sub, sub_assign;
    Mul, MulAssign, mul, mul_assign;
    Div, DivAssign, div, div_assign;
}
