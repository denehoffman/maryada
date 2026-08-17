use core::{cmp::Ordering, fmt, ops, str::FromStr};

use crate::{
    DecoratedInterval, Decoration, Interval, IntervalDatum, ParseIntervalError, Signal,
    SignalFlags, SignalSink, enclosure::EnclosureOps,
};

impl Interval {
    /// Converts this bare interval to a decorated interval.
    pub fn decorate<S: SignalSink>(self, signals: &mut S) -> DecoratedInterval {
        Self::from_nums(self.inf_raw(), self.sup_raw()).map_or_else(
            || {
                signals.raise(Signal::UndefinedOperation);
                DecoratedInterval::NAI
            },
            DecoratedInterval::new_dec_raw,
        )
    }

    /// Encodes this interval in the big-endian interchange format.
    #[must_use]
    pub fn to_be_bytes(self) -> [u8; crate::INTERVAL_ENCODED_LEN] {
        crate::interval_to_be_bytes(self)
    }

    /// Encodes this interval in the little-endian interchange format.
    #[must_use]
    pub fn to_le_bytes(self) -> [u8; crate::INTERVAL_ENCODED_LEN] {
        crate::interval_to_le_bytes(self)
    }

    /// Decodes a big-endian interval, reporting invalid input to `signals`.
    pub fn from_be_bytes<S: SignalSink>(bytes: &[u8], signals: &mut S) -> Self {
        crate::interval_from_be_bytes(bytes, signals)
    }

    /// Decodes a little-endian interval, reporting invalid input to `signals`.
    pub fn from_le_bytes<S: SignalSink>(bytes: &[u8], signals: &mut S) -> Self {
        crate::interval_from_le_bytes(bytes, signals)
    }
}

impl From<f64> for Interval {
    fn from(value: f64) -> Self {
        Self::nums_to_interval(value, value, &mut ())
    }
}

impl From<&f64> for Interval {
    fn from(value: &f64) -> Self {
        (*value).into()
    }
}

impl From<&Interval> for DecoratedInterval {
    fn from(value: &Interval) -> Self {
        value.decorate(&mut ())
    }
}

impl From<Interval> for DecoratedInterval {
    fn from(value: Interval) -> Self {
        value.decorate(&mut ())
    }
}

impl From<&DecoratedInterval> for Interval {
    fn from(value: &DecoratedInterval) -> Self {
        value.interval_raw()
    }
}

impl From<DecoratedInterval> for Interval {
    fn from(value: DecoratedInterval) -> Self {
        value.interval_raw()
    }
}

impl DecoratedInterval {
    /// Encodes this decorated interval in the big-endian interchange format.
    #[must_use]
    pub fn to_be_bytes(self) -> [u8; crate::DECORATED_INTERVAL_ENCODED_LEN] {
        crate::decorated_interval_to_be_bytes(self)
    }

    /// Encodes this decorated interval in the little-endian interchange format.
    #[must_use]
    pub fn to_le_bytes(self) -> [u8; crate::DECORATED_INTERVAL_ENCODED_LEN] {
        crate::decorated_interval_to_le_bytes(self)
    }

    /// Decodes a big-endian decorated interval, reporting invalid input.
    pub fn from_be_bytes<S: SignalSink>(bytes: &[u8], signals: &mut S) -> Self {
        crate::decorated_interval_from_be_bytes(bytes, signals)
    }

    /// Decodes a little-endian decorated interval, reporting invalid input.
    pub fn from_le_bytes<S: SignalSink>(bytes: &[u8], signals: &mut S) -> Self {
        crate::decorated_interval_from_le_bytes(bytes, signals)
    }

    /// Returns this interval's decoration.
    #[must_use]
    pub const fn decoration(self) -> crate::Decoration {
        crate::decoration_part(self)
    }
}

impl From<f64> for DecoratedInterval {
    fn from(value: f64) -> Self {
        Self::nums_to_interval(value, value, &mut ())
    }
}

impl From<&f64> for DecoratedInterval {
    fn from(value: &f64) -> Self {
        (*value).into()
    }
}

#[derive(Clone, Copy)]
enum DecimalNotation {
    Display,
    LowerExp,
    UpperExp,
}

struct CharCounter(usize);

impl fmt::Write for CharCounter {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        self.0 = self.0.saturating_add(text.chars().count());
        Ok(())
    }
}

fn write_endpoint<W: fmt::Write>(
    output: &mut W,
    value: f64,
    precision: Option<usize>,
    notation: DecimalNotation,
) -> fmt::Result {
    match (notation, precision) {
        (DecimalNotation::Display, Some(precision)) => write!(output, "{value:.precision$}"),
        (DecimalNotation::Display, None) => write!(output, "{value}"),
        (DecimalNotation::LowerExp, Some(precision)) => write!(output, "{value:.precision$e}"),
        (DecimalNotation::LowerExp, None) => write!(output, "{value:e}"),
        (DecimalNotation::UpperExp, Some(precision)) => write!(output, "{value:.precision$E}"),
        (DecimalNotation::UpperExp, None) => write!(output, "{value:E}"),
    }
}

fn write_decoration<W: fmt::Write>(output: &mut W, decoration: Decoration) -> fmt::Result {
    output.write_str(match decoration {
        Decoration::Ill => "ill",
        Decoration::Trv => "trv",
        Decoration::Def => "def",
        Decoration::Dac => "dac",
        Decoration::Com => "com",
    })
}

fn write_decimal_interval<W: fmt::Write, T: IntervalDatum>(
    output: &mut W,
    value: T,
    precision: Option<usize>,
    notation: DecimalNotation,
) -> fmt::Result {
    if value.__is_nai() {
        return output.write_str("[nai]");
    }
    let interval = value.__interval();
    if interval.is_empty_raw() {
        output.write_str("[empty]")?;
    } else if interval.is_entire_raw() {
        output.write_str("[entire]")?;
    } else {
        output.write_char('[')?;
        write_endpoint(output, interval.inf_raw(), precision, notation)?;
        output.write_char(',')?;
        write_endpoint(output, interval.sup_raw(), precision, notation)?;
        output.write_char(']')?;
    }
    if let Some(decoration) = value.__decoration() {
        output.write_char('_')?;
        write_decoration(output, decoration)?;
    }
    Ok(())
}

fn write_fill(formatter: &mut fmt::Formatter<'_>, fill: char, count: usize) -> fmt::Result {
    for _ in 0..count {
        fmt::Write::write_char(formatter, fill)?;
    }
    Ok(())
}

fn format_decimal_interval<T: IntervalDatum>(
    value: T,
    formatter: &mut fmt::Formatter<'_>,
    notation: DecimalNotation,
) -> fmt::Result {
    let mut counter = CharCounter(0);
    write_decimal_interval(&mut counter, value, formatter.precision(), notation)?;
    let padding = formatter.width().unwrap_or(0).saturating_sub(counter.0);
    let (left, right) = match formatter.align().unwrap_or(fmt::Alignment::Left) {
        fmt::Alignment::Left => (0, padding),
        fmt::Alignment::Right => (padding, 0),
        fmt::Alignment::Center => {
            let left = padding / 2;
            (left, padding.saturating_sub(left))
        }
    };
    write_fill(formatter, formatter.fill(), left)?;
    write_decimal_interval(formatter, value, formatter.precision(), notation)?;
    write_fill(formatter, formatter.fill(), right)
}

#[allow(clippy::indexing_slicing)]
fn format_hex_interval<T: IntervalDatum>(
    value: T,
    formatter: &mut fmt::Formatter<'_>,
    uppercase: bool,
) -> fmt::Result {
    let mut output = [0u8; 128];
    let len = crate::interval_to_text(value, Some("hex"), &mut output).map_err(|_| fmt::Error)?;
    if uppercase {
        output[..len].make_ascii_uppercase();
    }
    let text = core::str::from_utf8(&output[..len]).map_err(|_| fmt::Error)?;
    let padding = formatter
        .width()
        .unwrap_or(0)
        .saturating_sub(text.chars().count());
    let (left, right) = match formatter.align().unwrap_or(fmt::Alignment::Left) {
        fmt::Alignment::Left => (0, padding),
        fmt::Alignment::Right => (padding, 0),
        fmt::Alignment::Center => {
            let left = padding / 2;
            (left, padding.saturating_sub(left))
        }
    };
    write_fill(formatter, formatter.fill(), left)?;
    formatter.write_str(text)?;
    write_fill(formatter, formatter.fill(), right)
}

macro_rules! impl_decimal_format {
    ($type:ty, $trait:path, $notation:expr) => {
        impl $trait for $type {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                format_decimal_interval(*self, formatter, $notation)
            }
        }
    };
}

impl_decimal_format!(Interval, fmt::Display, DecimalNotation::Display);
impl_decimal_format!(Interval, fmt::LowerExp, DecimalNotation::LowerExp);
impl_decimal_format!(Interval, fmt::UpperExp, DecimalNotation::UpperExp);
impl_decimal_format!(DecoratedInterval, fmt::Display, DecimalNotation::Display);
impl_decimal_format!(DecoratedInterval, fmt::LowerExp, DecimalNotation::LowerExp);
impl_decimal_format!(DecoratedInterval, fmt::UpperExp, DecimalNotation::UpperExp);

impl fmt::LowerHex for Interval {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_hex_interval(*self, formatter, false)
    }
}

impl fmt::UpperHex for Interval {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_hex_interval(*self, formatter, true)
    }
}

impl fmt::LowerHex for DecoratedInterval {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_hex_interval(*self, formatter, false)
    }
}

impl fmt::UpperHex for DecoratedInterval {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_hex_interval(*self, formatter, true)
    }
}

impl FromStr for Interval {
    type Err = ParseIntervalError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let mut signals = SignalFlags::NONE;
        let result = Self::text_to_interval(value, &mut signals);
        if signals.contains(Signal::UndefinedOperation) || signals.contains(Signal::InvalidOperand)
        {
            Err(ParseIntervalError)
        } else {
            Ok(result)
        }
    }
}

impl FromStr for DecoratedInterval {
    type Err = ParseIntervalError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let mut signals = SignalFlags::NONE;
        let result = Self::text_to_interval(value, &mut signals);
        if signals.contains(Signal::UndefinedOperation) || signals.contains(Signal::InvalidOperand)
        {
            Err(ParseIntervalError)
        } else {
            Ok(result)
        }
    }
}

macro_rules! impl_interval_binary_op {
    ($op:ident, $method:ident, $assign_op:ident, $assign_method:ident, $function:path) => {
        impl ops::$op<f64> for Interval {
            type Output = Interval;

            fn $method(self, rhs: f64) -> Self::Output {
                $function(self, rhs.into())
            }
        }

        impl ops::$op<Interval> for f64 {
            type Output = Interval;

            fn $method(self, rhs: Interval) -> Self::Output {
                $function(self.into(), rhs)
            }
        }

        impl ops::$op<f64> for DecoratedInterval {
            type Output = DecoratedInterval;

            fn $method(self, rhs: f64) -> Self::Output {
                $function(self, rhs.into())
            }
        }

        impl ops::$op<DecoratedInterval> for f64 {
            type Output = DecoratedInterval;

            fn $method(self, rhs: DecoratedInterval) -> Self::Output {
                $function(self.into(), rhs)
            }
        }

        impl ops::$op for Interval {
            type Output = Interval;

            fn $method(self, rhs: Self) -> Self::Output {
                $function(self, rhs)
            }
        }

        impl ops::$op<DecoratedInterval> for Interval {
            type Output = DecoratedInterval;

            fn $method(self, rhs: DecoratedInterval) -> Self::Output {
                $function(self.into(), rhs)
            }
        }

        impl ops::$op<Interval> for DecoratedInterval {
            type Output = DecoratedInterval;

            fn $method(self, rhs: Interval) -> Self::Output {
                $function(self, rhs.into())
            }
        }

        impl ops::$op for DecoratedInterval {
            type Output = DecoratedInterval;

            fn $method(self, rhs: Self) -> Self::Output {
                $function(self, rhs)
            }
        }

        impl ops::$assign_op for Interval {
            fn $assign_method(&mut self, rhs: Self) {
                *self = $function(*self, rhs);
            }
        }

        impl ops::$assign_op<f64> for Interval {
            fn $assign_method(&mut self, rhs: f64) {
                *self = $function(*self, rhs.into());
            }
        }

        impl ops::$assign_op for DecoratedInterval {
            fn $assign_method(&mut self, rhs: Self) {
                *self = $function(*self, rhs);
            }
        }

        impl ops::$assign_op<Interval> for DecoratedInterval {
            fn $assign_method(&mut self, rhs: Interval) {
                *self = $function(*self, rhs.into());
            }
        }

        impl ops::$assign_op<f64> for DecoratedInterval {
            fn $assign_method(&mut self, rhs: f64) {
                *self = $function(*self, rhs.into());
            }
        }
    };
}

impl_interval_binary_op!(Add, add, AddAssign, add_assign, crate::add);
impl_interval_binary_op!(Sub, sub, SubAssign, sub_assign, crate::sub);
impl_interval_binary_op!(Mul, mul, MulAssign, mul_assign, crate::mul);
impl_interval_binary_op!(Div, div, DivAssign, div_assign, crate::div);

macro_rules! impl_interval_neg {
    ($($interval:ty),+ $(,)?) => {$(
        impl ops::Neg for $interval {
            type Output = Self;

            fn neg(self) -> Self::Output {
                crate::neg(self)
            }
        }
    )+};
}

impl_interval_neg!(Interval, DecoratedInterval);

/// Chaining-friendly operations specific to bare and decorated real intervals.
///
/// This trait is sealed transitively through [`IntervalDatum`]. It can be used
/// as a public generic bound, but external crates cannot implement it.
pub trait IntervalOps:
    EnclosureOps<Midpoint = f64>
    + IntervalDatum
    + From<f64>
    + ops::Neg<Output = Self>
    + ops::Add<Output = Self>
    + ops::Sub<Output = Self>
    + ops::Mul<Output = Self>
    + ops::Div<Output = Self>
    + ops::Add<f64, Output = Self>
    + ops::Sub<f64, Output = Self>
    + ops::Mul<f64, Output = Self>
    + ops::Div<f64, Output = Self>
    + ops::AddAssign
    + ops::SubAssign
    + ops::MulAssign
    + ops::DivAssign
    + ops::AddAssign<f64>
    + ops::SubAssign<f64>
    + ops::MulAssign<f64>
    + ops::DivAssign<f64>
{
    /// Constructs an interval from binary64 endpoints without reporting signals.
    #[must_use]
    fn new(inf: f64, sup: f64) -> Self;

    /// Returns the two-argument angle enclosure `atan2(self, x)`.
    #[must_use]
    fn atan2(self, x: Self) -> Self {
        crate::atan2(self, x)
    }

    /// Maps values to their signs.
    #[must_use]
    fn sign(self) -> Self {
        crate::sign(self)
    }

    /// Applies the ceiling function pointwise.
    #[must_use]
    fn ceil(self) -> Self {
        crate::ceil(self)
    }

    /// Applies the floor function pointwise.
    #[must_use]
    fn floor(self) -> Self {
        crate::floor(self)
    }

    /// Applies truncation toward zero pointwise.
    #[must_use]
    fn trunc(self) -> Self {
        crate::trunc(self)
    }

    /// Rounds pointwise to nearest integers with ties to even.
    #[must_use]
    fn round_ties_to_even(self) -> Self {
        crate::round_ties_to_even(self)
    }

    /// Rounds pointwise to nearest integers with ties away from zero.
    #[must_use]
    fn round_ties_to_away(self) -> Self {
        crate::round_ties_to_away(self)
    }

    /// Returns the absolute-value enclosure.
    #[must_use]
    fn abs(self) -> Self {
        crate::abs(self)
    }

    /// Returns the pointwise-minimum enclosure.
    #[must_use]
    fn min(self, other: Self) -> Self {
        crate::min(self, other)
    }

    /// Returns the pointwise-maximum enclosure.
    #[must_use]
    fn max(self, other: Self) -> Self {
        crate::max(self, other)
    }

    /// Encloses `sqrt(self² + other²)`.
    #[must_use]
    fn hypot(self, other: Self) -> Self {
        crate::sqrt(crate::add(crate::sqr(self), crate::sqr(other)))
    }

    /// Computes cancellative subtraction `self ⊖ other`.
    #[must_use]
    fn cancel_minus(self, other: Self) -> Self {
        crate::cancel_minus(self, other)
    }

    /// Computes cancellative addition `self ⊕ other`.
    #[must_use]
    fn cancel_plus(self, other: Self) -> Self {
        crate::cancel_plus(self, other)
    }

    /// Extends this interval's hull to include `value`.
    #[must_use]
    fn hull_value(self, value: f64) -> Self {
        self.convex_hull(Self::from(value))
    }

    /// Splits the interval at its midpoint into two covering intervals.
    #[must_use]
    fn bisect(self) -> (Self, Self) {
        if self.is_nai() || self.is_empty() {
            return (self, self);
        }
        let midpoint = self.mid();
        let left = Self::new(self.inf(), midpoint);
        let right = Self::new(midpoint, self.sup());
        (
            self.__unary_result(left.__interval(), Decoration::Com),
            self.__unary_result(right.__interval(), Decoration::Com),
        )
    }

    /// Returns the lower endpoint, or `NaN` for `NaI`.
    #[must_use]
    fn inf(self) -> f64 {
        crate::inf(self)
    }

    /// Returns the upper endpoint, or `NaN` for `NaI`.
    #[must_use]
    fn sup(self) -> f64 {
        crate::sup(self)
    }

    /// Returns `(lower, upper)` endpoints.
    #[must_use]
    fn bounds(self) -> (f64, f64) {
        (self.inf(), self.sup())
    }
}

impl IntervalOps for Interval {
    fn new(inf: f64, sup: f64) -> Self {
        Self::nums_to_interval(inf, sup, &mut ())
    }
}

impl IntervalOps for DecoratedInterval {
    fn new(inf: f64, sup: f64) -> Self {
        Self::nums_to_interval(inf, sup, &mut ())
    }
}

impl PartialEq for Interval {
    fn eq(&self, other: &Self) -> bool {
        self.inf_raw() == other.inf_raw() && self.sup_raw() == other.sup_raw()
    }
}

impl Eq for Interval {}

impl PartialOrd for Interval {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self == other {
            Some(Ordering::Equal)
        } else if crate::subset(*self, *other) {
            Some(Ordering::Less)
        } else if crate::subset(*other, *self) {
            Some(Ordering::Greater)
        } else {
            None
        }
    }
}

impl PartialEq for DecoratedInterval {
    fn eq(&self, other: &Self) -> bool {
        self.interval_raw() == other.interval_raw()
            && self.decoration_raw() == other.decoration_raw()
    }
}

impl Eq for DecoratedInterval {}

impl PartialOrd for DecoratedInterval {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self == other {
            return Some(Ordering::Equal);
        }

        let left = self.interval_raw();
        let right = other.interval_raw();
        if left == right {
            None
        } else if crate::subset(left, right) {
            Some(Ordering::Less)
        } else if crate::subset(right, left) {
            Some(Ordering::Greater)
        } else {
            None
        }
    }
}
