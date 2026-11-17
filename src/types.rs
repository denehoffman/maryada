use crate::{
    rounding::{self, Direction},
    signals::{Signal, SignalSink},
    text,
};

/// An invalid input to a checked scalar interval constructor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IntervalConstructionError {
    /// An endpoint, midpoint, or radius is NaN.
    NaN,
    /// The lower endpoint is greater than the upper endpoint.
    ReversedBounds,
    /// The lower endpoint is positive infinity or the upper endpoint is negative infinity.
    InvalidInfinity,
    /// The radius is strictly negative.
    NegativeRadius,
    /// The midpoint is infinite.
    InfiniteMidpoint,
}

impl core::fmt::Display for IntervalConstructionError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(match self {
            Self::NaN => "interval constructor input is NaN",
            Self::ReversedBounds => "lower endpoint exceeds upper endpoint",
            Self::InvalidInfinity => "lower endpoint is +infinity or upper endpoint is -infinity",
            Self::NegativeRadius => "radius is negative",
            Self::InfiniteMidpoint => "midpoint is infinite",
        })
    }
}

impl core::error::Error for IntervalConstructionError {}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
/// A closed binary64 interval, or the distinguished empty interval.
///
/// Finite and infinite endpoints are canonicalized so that lower zero is
/// `-0.0` and upper zero is `+0.0`.
pub struct Interval {
    inf: f64,
    sup: f64,
}

impl Interval {
    /// The empty interval.
    pub const EMPTY: Self = Self {
        inf: f64::INFINITY,
        sup: f64::NEG_INFINITY,
    };

    /// The interval containing every real number.
    pub const ENTIRE: Self = Self {
        inf: f64::NEG_INFINITY,
        sup: f64::INFINITY,
    };

    /// The singleton interval containing zero.
    pub const ZERO: Self = Self {
        inf: -0.0,
        sup: 0.0,
    };

    /// The singleton interval containing one.
    pub const ONE: Self = Self { inf: 1.0, sup: 1.0 };

    /// Constructs an interval from checked binary64 endpoints.
    ///
    /// Infinite outer endpoints are allowed. Zero endpoints are canonicalized.
    ///
    /// # Errors
    ///
    /// Returns [`IntervalConstructionError::NaN`] for NaN inputs,
    /// [`IntervalConstructionError::ReversedBounds`] for reversed endpoints,
    /// or [`IntervalConstructionError::InvalidInfinity`] for a lower endpoint
    /// of positive infinity or an upper endpoint of negative infinity.
    /// Checks are performed in that order.
    ///
    /// ```
    /// use maryada::{Interval, IntervalConstructionError};
    ///
    /// let endpoints = Interval::try_new(0.0, 1.0)?;
    /// let midpoint_radius = Interval::try_from_mid_rad(0.5, 0.5)?;
    /// assert_eq!(endpoints, midpoint_radius);
    /// assert_eq!(Interval::try_new(2.0, 1.0), Err(IntervalConstructionError::ReversedBounds));
    /// # Ok::<(), IntervalConstructionError>(())
    /// ```
    pub fn try_new(inf: f64, sup: f64) -> Result<Self, IntervalConstructionError> {
        if inf.is_nan() || sup.is_nan() {
            return Err(IntervalConstructionError::NaN);
        }
        if inf > sup {
            return Err(IntervalConstructionError::ReversedBounds);
        }
        if inf == f64::INFINITY || sup == f64::NEG_INFINITY {
            return Err(IntervalConstructionError::InvalidInfinity);
        }
        Ok(Self::from_valid_bounds(inf, sup))
    }

    /// Constructs the tight binary64 enclosure of `midpoint ± radius`.
    ///
    /// The midpoint must be finite and the radius nonnegative. A positive
    /// infinite radius produces [`Self::ENTIRE`]. Bounds are rounded outward,
    /// including when the exact endpoints overflow binary64.
    ///
    /// # Errors
    ///
    /// Returns [`IntervalConstructionError::NaN`] for NaN inputs,
    /// [`IntervalConstructionError::InfiniteMidpoint`] for an infinite midpoint,
    /// or [`IntervalConstructionError::NegativeRadius`] for a negative radius.
    /// Checks are performed in that order. Negative zero is accepted as a radius.
    pub fn try_from_mid_rad(midpoint: f64, radius: f64) -> Result<Self, IntervalConstructionError> {
        if midpoint.is_nan() || radius.is_nan() {
            return Err(IntervalConstructionError::NaN);
        }
        if midpoint.is_infinite() {
            return Err(IntervalConstructionError::InfiniteMidpoint);
        }
        if radius < 0.0 {
            return Err(IntervalConstructionError::NegativeRadius);
        }
        let inf = rounding::sub(midpoint, radius, Direction::Down);
        let sup = rounding::add(midpoint, radius, Direction::Up);
        Ok(Self::from_valid_bounds(inf, sup))
    }

    /// Constructs a bare interval from binary64 endpoints.
    ///
    /// Invalid bounds raise [`Signal::UndefinedOperation`] and return
    /// [`Interval::EMPTY`].
    pub fn nums_to_interval<S: SignalSink>(l: f64, u: f64, signals: &mut S) -> Self {
        Self::try_new(l, u).unwrap_or_else(|_| {
            signals.raise(Signal::UndefinedOperation);
            Self::EMPTY
        })
    }

    /// Parses an IEEE 1788 interval literal.
    ///
    /// Invalid text raises the corresponding signal and returns
    /// [`Interval::EMPTY`].
    ///
    /// ```
    /// use maryada::{EnclosureOps, Interval, IntervalOps, SignalFlags};
    ///
    /// let mut signals = SignalFlags::NONE;
    /// let x = Interval::text_to_interval("[0.1, 0.2]", &mut signals);
    /// assert!(x.contains(0.15));
    /// assert!(signals.is_empty());
    /// ```
    pub fn text_to_interval<S: SignalSink>(s: &str, signals: &mut S) -> Self {
        text::text_to_interval(s, signals)
    }

    pub(crate) const fn from_valid_bounds(inf: f64, sup: f64) -> Self {
        Self {
            inf: canonical_inf(inf),
            sup: canonical_sup(sup),
        }
    }

    pub(crate) const fn inf_raw(self) -> f64 {
        self.inf
    }

    pub(crate) const fn sup_raw(self) -> f64 {
        self.sup
    }

    pub(crate) const fn is_empty_raw(self) -> bool {
        self.inf == f64::INFINITY && self.sup == f64::NEG_INFINITY
    }

    pub(crate) const fn is_entire_raw(self) -> bool {
        self.inf == f64::NEG_INFINITY && self.sup == f64::INFINITY
    }

    pub(crate) const fn is_bounded_raw(self) -> bool {
        !self.is_empty_raw() && self.inf.is_finite() && self.sup.is_finite()
    }

    pub(crate) const fn contains_raw(self, value: f64) -> bool {
        !self.is_empty_raw() && !value.is_nan() && self.inf <= value && value <= self.sup
    }
}

const fn canonical_inf(value: f64) -> f64 {
    if value == 0.0 { -0.0 } else { value }
}

const fn canonical_sup(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

/// The definedness and continuity information attached to an interval result.
///
/// Declaration order and discriminants follow `ill < trv < def < dac < com`,
/// so taking the minimum selects the weaker decoration.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum Decoration {
    /// Ill-formed: used only for Not an Interval (`NaI`).
    Ill = 0,
    /// Trivial: no continuity or definedness guarantee.
    Trv = 4,
    /// Defined throughout the input box.
    Def = 8,
    /// Defined and continuous throughout the input box.
    Dac = 12,
    /// Common: defined, continuous, bounded, and nonempty.
    Com = 16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Error returned when a byte or string is not a valid [`Decoration`].
pub struct InvalidDecoration;

impl TryFrom<u8> for Decoration {
    type Error = InvalidDecoration;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Ill),
            4 => Ok(Self::Trv),
            8 => Ok(Self::Def),
            12 => Ok(Self::Dac),
            16 => Ok(Self::Com),
            _ => Err(InvalidDecoration),
        }
    }
}

impl From<Decoration> for u8 {
    fn from(value: Decoration) -> Self {
        match value {
            Decoration::Ill => 0,
            Decoration::Trv => 4,
            Decoration::Def => 8,
            Decoration::Dac => 12,
            Decoration::Com => 16,
        }
    }
}

impl core::str::FromStr for Decoration {
    type Err = InvalidDecoration;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.eq_ignore_ascii_case("trv") {
            Ok(Self::Trv)
        } else if value.eq_ignore_ascii_case("def") {
            Ok(Self::Def)
        } else if value.eq_ignore_ascii_case("dac") {
            Ok(Self::Dac)
        } else if value.eq_ignore_ascii_case("com") {
            Ok(Self::Com)
        } else if value.eq_ignore_ascii_case("ill") {
            Ok(Self::Ill)
        } else {
            Err(InvalidDecoration)
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
/// A binary64 interval paired with an IEEE 1788 decoration.
///
/// The type includes the distinguished [`DecoratedInterval::NAI`] value.
pub struct DecoratedInterval {
    interval: Interval,
    decoration: Decoration,
}

impl DecoratedInterval {
    /// Not an Interval, representing an invalid decorated result.
    pub const NAI: Self = Self {
        interval: Interval::EMPTY,
        decoration: Decoration::Ill,
    };

    /// The empty interval with the trivial decoration.
    pub const EMPTY: Self = Self {
        interval: Interval::EMPTY,
        decoration: Decoration::Trv,
    };

    /// The entire interval with the defined-and-continuous decoration.
    pub const ENTIRE: Self = Self {
        interval: Interval::ENTIRE,
        decoration: Decoration::Dac,
    };

    /// The decorated singleton interval containing zero.
    pub const ZERO: Self = Self {
        interval: Interval::ZERO,
        decoration: Decoration::Com,
    };

    /// The decorated singleton interval containing one.
    pub const ONE: Self = Self {
        interval: Interval::ONE,
        decoration: Decoration::Com,
    };

    /// Constructs a decorated interval from checked binary64 endpoints.
    ///
    /// The decoration is [`Decoration::Com`] for finite bounds and
    /// [`Decoration::Dac`] for unbounded intervals.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Interval::try_new`].
    pub fn try_new(inf: f64, sup: f64) -> Result<Self, IntervalConstructionError> {
        let interval = Interval::try_new(inf, sup)?;
        Ok(Self::new_dec_raw(interval))
    }

    /// Constructs a decorated enclosure of `midpoint ± radius`.
    ///
    /// Uses outward rounding and assigns [`Decoration::Com`] to finite bounds
    /// and [`Decoration::Dac`] to unbounded intervals.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Interval::try_from_mid_rad`].
    pub fn try_from_mid_rad(midpoint: f64, radius: f64) -> Result<Self, IntervalConstructionError> {
        let interval = Interval::try_from_mid_rad(midpoint, radius)?;
        Ok(Self::new_dec_raw(interval))
    }

    /// Constructs a decorated interval from binary64 endpoints.
    ///
    /// Invalid bounds raise [`Signal::UndefinedOperation`] and return `NaI`.
    pub fn nums_to_interval<S: SignalSink>(l: f64, u: f64, signals: &mut S) -> Self {
        Interval::try_new(l, u).map_or_else(
            |_| {
                signals.raise(Signal::UndefinedOperation);
                Self::NAI
            },
            Self::new_dec_raw,
        )
    }

    /// Parses an IEEE 1788 decorated interval literal.
    ///
    /// Invalid text raises the corresponding signal and returns `NaI`.
    pub fn text_to_interval<S: SignalSink>(s: &str, signals: &mut S) -> Self {
        text::text_to_decorated_interval(s, signals)
    }

    pub(crate) const fn new_dec_raw(interval: Interval) -> Self {
        let decoration = if interval.is_empty_raw() {
            Decoration::Trv
        } else if interval.inf_raw().is_finite() && interval.sup_raw().is_finite() {
            Decoration::Com
        } else {
            Decoration::Dac
        };

        Self {
            interval,
            decoration,
        }
    }

    pub(crate) fn set_dec_raw(interval: Interval, decoration: Decoration) -> Self {
        if decoration == Decoration::Ill {
            return Self::NAI;
        }

        if interval.is_empty_raw() {
            return Self::EMPTY;
        }

        let decoration = if decoration == Decoration::Com
            && !(interval.inf_raw().is_finite() && interval.sup_raw().is_finite())
        {
            Decoration::Dac
        } else {
            decoration
        };

        Self {
            interval,
            decoration,
        }
    }

    pub(crate) const fn interval_raw(self) -> Interval {
        self.interval
    }

    pub(crate) const fn decoration_raw(self) -> Decoration {
        self.decoration
    }

    pub(crate) fn is_nai_raw(self) -> bool {
        self.decoration == Decoration::Ill
    }
}

mod sealed {
    pub trait Sealed {}
}

/// Common sealed representation surface used by all generic operations.
///
/// External implementations are intentionally prohibited. IEEE 1788.1
/// defines exactly the bare and decorated binary64 interval types.
pub trait IntervalDatum: sealed::Sealed + Copy {
    #[doc(hidden)]
    fn __interval(self) -> Interval;

    #[doc(hidden)]
    fn __decoration(self) -> Option<Decoration>;

    #[doc(hidden)]
    fn __is_nai(self) -> bool;

    #[doc(hidden)]
    fn __empty() -> Self;

    #[doc(hidden)]
    fn __entire() -> Self;

    #[doc(hidden)]
    #[must_use]
    fn __unary_result(self, interval: Interval, local: Decoration) -> Self;

    #[doc(hidden)]
    #[must_use]
    fn __binary_result(self, rhs: Self, interval: Interval, local: Decoration) -> Self;

    #[doc(hidden)]
    #[must_use]
    fn __ternary_result(
        self,
        second: Self,
        third: Self,
        interval: Interval,
        local: Decoration,
    ) -> Self;
}

impl sealed::Sealed for Interval {}

impl IntervalDatum for Interval {
    fn __interval(self) -> Interval {
        self
    }

    fn __decoration(self) -> Option<Decoration> {
        None
    }

    fn __is_nai(self) -> bool {
        false
    }

    fn __empty() -> Self {
        Self::EMPTY
    }

    fn __entire() -> Self {
        Self::ENTIRE
    }

    fn __unary_result(self, interval: Interval, _: Decoration) -> Self {
        interval
    }

    fn __binary_result(self, _: Self, interval: Interval, _: Decoration) -> Self {
        interval
    }

    fn __ternary_result(self, _: Self, _: Self, interval: Interval, _: Decoration) -> Self {
        interval
    }
}

impl sealed::Sealed for DecoratedInterval {}

impl IntervalDatum for DecoratedInterval {
    fn __interval(self) -> Interval {
        self.interval
    }

    fn __decoration(self) -> Option<Decoration> {
        Some(self.decoration)
    }

    fn __is_nai(self) -> bool {
        self.is_nai_raw()
    }

    fn __empty() -> Self {
        Self::EMPTY
    }

    fn __entire() -> Self {
        Self::ENTIRE
    }

    fn __unary_result(self, interval: Interval, local: Decoration) -> Self {
        if self.is_nai_raw() {
            return Self::NAI;
        }

        Self::set_dec_raw(interval, self.decoration.min(local))
    }

    fn __binary_result(self, rhs: Self, interval: Interval, local: Decoration) -> Self {
        if self.is_nai_raw() || rhs.is_nai_raw() {
            return Self::NAI;
        }

        Self::set_dec_raw(interval, self.decoration.min(rhs.decoration).min(local))
    }

    fn __ternary_result(
        self,
        second: Self,
        third: Self,
        interval: Interval,
        local: Decoration,
    ) -> Self {
        if self.is_nai_raw() || second.is_nai_raw() || third.is_nai_raw() {
            return Self::NAI;
        }

        Self::set_dec_raw(
            interval,
            self.decoration
                .min(second.decoration)
                .min(third.decoration)
                .min(local),
        )
    }
}

/// An allocation-free union of up to two closed interval enclosures.
///
/// [`crate::div_split`] and [`crate::recip_split`] return sorted, disjoint
/// components. Overlapping or touching components are merged. Empty has no
/// components. A decorated `NaI` result has a single `NaI` component.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IntervalSet<T: IntervalDatum = Interval> {
    pub(crate) components: [T; 2],
}

impl<T: IntervalDatum> IntervalSet<T> {
    /// Returns the zero, one, or two components in increasing order.
    #[must_use]
    #[allow(clippy::indexing_slicing)]
    pub fn components(&self) -> &[T] {
        let [first, second] = self.components;
        if first.__interval().is_empty_raw() && !first.__is_nai() {
            &[]
        } else if second.__interval().is_empty_raw() {
            &self.components[..1]
        } else {
            &self.components
        }
    }

    /// Returns the smallest single interval enclosing all components.
    ///
    /// Preserves Empty, `NaI`, and the weakest component decoration.
    #[must_use]
    pub fn hull(self) -> T {
        let [first, second] = self.components;
        let right = second.__interval();
        if right.is_empty_raw() {
            return first;
        }
        let left = first.__interval();
        let interval = Interval::from_valid_bounds(left.inf_raw(), right.sup_raw());
        first.__binary_result(second, interval, Decoration::Com)
    }
}
