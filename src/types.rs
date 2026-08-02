use crate::{
    signals::{Signal, SignalSink},
    text,
};

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Interval {
    inf: f64,
    sup: f64,
}

impl Interval {
    pub const EMPTY: Self = Self {
        inf: f64::INFINITY,
        sup: f64::NEG_INFINITY,
    };

    pub const ENTIRE: Self = Self {
        inf: f64::NEG_INFINITY,
        sup: f64::INFINITY,
    };

    pub const ZERO: Self = Self {
        inf: -0.0,
        sup: 0.0,
    };

    pub const ONE: Self = Self { inf: 1.0, sup: 1.0 };

    /// IEEE numsToInterval for the bare type.
    pub fn nums_to_interval<S: SignalSink>(l: f64, u: f64, signals: &mut S) -> Self {
        match Self::from_nums(l, u) {
            Some(value) => value,
            None => {
                signals.raise(Signal::UndefinedOperation);
                Self::EMPTY
            }
        }
    }

    /// IEEE textToInterval for the bare type.
    pub fn text_to_interval<S: SignalSink>(s: &str, signals: &mut S) -> Self {
        text::text_to_interval(s, signals)
    }

    pub(crate) fn from_nums(l: f64, u: f64) -> Option<Self> {
        if l.is_nan() || u.is_nan() || l > u || l == f64::INFINITY || u == f64::NEG_INFINITY {
            return None;
        }

        Some(Self::from_valid_bounds(l, u))
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

    pub(crate) fn is_bounded_raw(self) -> bool {
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

/// Declaration order and discriminants both follow:
///
/// ill < trv < def < dac < com
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum Decoration {
    Ill = 0,
    Trv = 4,
    Def = 8,
    Dac = 12,
    Com = 16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
        value as u8
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
pub struct DecoratedInterval {
    interval: Interval,
    decoration: Decoration,
}

impl DecoratedInterval {
    pub const NAI: Self = Self {
        interval: Interval::EMPTY,
        decoration: Decoration::Ill,
    };

    pub const EMPTY: Self = Self {
        interval: Interval::EMPTY,
        decoration: Decoration::Trv,
    };

    pub const ENTIRE: Self = Self {
        interval: Interval::ENTIRE,
        decoration: Decoration::Dac,
    };

    pub const ZERO: Self = Self {
        interval: Interval::ZERO,
        decoration: Decoration::Com,
    };

    pub const ONE: Self = Self {
        interval: Interval::ONE,
        decoration: Decoration::Com,
    };

    /// IEEE numsToInterval for the decorated type.
    pub fn nums_to_interval<S: SignalSink>(l: f64, u: f64, signals: &mut S) -> Self {
        match Interval::from_nums(l, u) {
            Some(value) => Self::new_dec_raw(value),
            None => {
                signals.raise(Signal::UndefinedOperation);
                Self::NAI
            }
        }
    }

    /// IEEE textToInterval for the decorated type.
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
    fn __zero() -> Self;

    #[doc(hidden)]
    fn __from_nums(inf: f64, sup: f64) -> Self;

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
    fn __unary_result(self, interval: Interval, local: Decoration) -> Self;

    #[doc(hidden)]
    fn __binary_result(self, rhs: Self, interval: Interval, local: Decoration) -> Self;

    #[doc(hidden)]
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
    fn __zero() -> Self {
        Self::ZERO
    }

    fn __from_nums(inf: f64, sup: f64) -> Self {
        Self::nums_to_interval(inf, sup, &mut ())
    }

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
    fn __zero() -> Self {
        Self::ZERO
    }

    fn __from_nums(inf: f64, sup: f64) -> Self {
        Self::nums_to_interval(inf, sup, &mut ())
    }

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
