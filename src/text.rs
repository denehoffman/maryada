use core::cmp::Ordering;

use crate::{
    DecoratedInterval, Decoration, Interval, IntervalDatum,
    rounding::{self, Direction},
    signals::{Signal, SignalFlags, SignalSink},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Error returned when formatting an interval into a caller-provided buffer.
pub enum TextError {
    /// The output slice cannot hold the complete representation.
    BufferTooSmall {
        /// Minimum number of bytes required for this representation.
        required: usize,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Error returned by [`core::str::FromStr`] for an invalid interval literal.
pub struct ParseIntervalError;

impl core::fmt::Display for ParseIntervalError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("invalid interval literal")
    }
}

impl core::error::Error for ParseIntervalError {}

/// The cause of a checked interval parse failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParseIntervalErrorKind {
    /// The input does not have the required interval syntax.
    InvalidSyntax,
    /// An endpoint is not a valid decimal, hexadecimal, rational, or infinity literal.
    InvalidNumber,
    /// A decoration is unknown or incompatible with the source interval.
    InvalidDecoration,
    /// The exact lower endpoint exceeds the exact upper endpoint.
    ReversedBounds,
    /// An infinity literal is used as a singleton or on the wrong side.
    InvalidInfinity,
    /// The uncertain mantissa, radius, or exponent is malformed.
    InvalidUncertainty,
}

/// A checked parse failure with a byte offset in the original input.
///
/// The offset identifies the start of the offending token. A missing closing
/// bracket is reported at the end of the trimmed literal, where the bracket
/// was expected. Every failure corresponds to [`Signal::UndefinedOperation`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParseIntervalDiagnostic {
    /// The cause of the failure.
    pub kind: ParseIntervalErrorKind,
    /// Zero-based UTF-8 byte offset in the original input.
    pub byte_offset: usize,
}

impl core::fmt::Display for ParseIntervalDiagnostic {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let message = match self.kind {
            ParseIntervalErrorKind::InvalidSyntax => "invalid interval syntax",
            ParseIntervalErrorKind::InvalidNumber => "invalid endpoint literal",
            ParseIntervalErrorKind::InvalidDecoration => "invalid interval decoration",
            ParseIntervalErrorKind::ReversedBounds => "lower endpoint exceeds upper endpoint",
            ParseIntervalErrorKind::InvalidInfinity => "invalid infinity endpoint",
            ParseIntervalErrorKind::InvalidUncertainty => "invalid uncertain literal",
        };
        write!(formatter, "{message} at byte {}", self.byte_offset)
    }
}

impl core::error::Error for ParseIntervalDiagnostic {}

impl core::fmt::Display for TextError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BufferTooSmall { required } => {
                write!(formatter, "interval text requires {required} bytes")
            }
        }
    }
}

impl core::error::Error for TextError {}

struct ParsedBare {
    interval: Interval,

    /// Whether the Level 1 source interval was bounded before
    /// conversion to binary64.
    source_bounded: bool,

    /// Whether an accuracy-relaxed pair could have reversed endpoints.
    possibly_undefined: bool,
}

enum ParsedDecorated {
    NaI,
    Interval {
        bare: ParsedBare,
        decoration: Option<Decoration>,
    },
}

/// Bare IEEE textToInterval.
#[allow(clippy::arithmetic_side_effects)]
pub fn text_to_interval<S: SignalSink>(s: &str, signals: &mut S) -> Interval {
    let Ok((interval, flags)) = Interval::parse_checked(s) else {
        signals.raise(Signal::UndefinedOperation);
        return Interval::EMPTY;
    };
    if flags.contains(Signal::PossiblyUndefinedOperation) {
        signals.raise(Signal::PossiblyUndefinedOperation);
    }
    interval
}

/// Decorated IEEE textToInterval.
#[allow(clippy::arithmetic_side_effects)]
pub fn text_to_decorated_interval<S: SignalSink>(s: &str, signals: &mut S) -> DecoratedInterval {
    let Ok((interval, flags)) = DecoratedInterval::parse_checked(s) else {
        signals.raise(Signal::UndefinedOperation);
        return DecoratedInterval::NAI;
    };
    if flags.contains(Signal::PossiblyUndefinedOperation) {
        signals.raise(Signal::PossiblyUndefinedOperation);
    }
    interval
}

impl Interval {
    /// Parses a bare literal and returns its IEEE exception flags.
    ///
    /// Successful accuracy-relaxed parses may report
    /// [`Signal::PossiblyUndefinedOperation`]. Explicit Empty is a successful parse.
    ///
    /// # Errors
    ///
    /// Returns a typed diagnostic and byte offset for an invalid literal.
    ///
    /// ```
    /// use maryada::{Interval, ParseIntervalErrorKind, Signal};
    ///
    /// let (_, flags) = Interval::parse_checked("[0.100000000000000001,1/10]")?;
    /// assert!(flags.contains(Signal::PossiblyUndefinedOperation));
    /// let error = Interval::parse_checked("[2,1]").unwrap_err();
    /// assert_eq!(error.kind, ParseIntervalErrorKind::ReversedBounds);
    /// assert_eq!(error.byte_offset, 3);
    /// # Ok::<(), maryada::ParseIntervalDiagnostic>(())
    /// ```
    pub fn parse_checked(s: &str) -> Result<(Self, SignalFlags), ParseIntervalDiagnostic> {
        let parsed = parse_bare_literal(s)?;
        let mut signals = SignalFlags::NONE;
        if parsed.possibly_undefined {
            signals.insert(Signal::PossiblyUndefinedOperation);
        }
        Ok((parsed.interval, signals))
    }
}

impl DecoratedInterval {
    /// Parses a decorated literal and returns its IEEE exception flags.
    ///
    /// Successful accuracy-relaxed parses may report
    /// [`Signal::PossiblyUndefinedOperation`]. Explicit Empty and `[nai]` are
    /// successful parses. Bounded source intervals that overflow binary64 have
    /// their requested `com` decoration reduced to `dac`.
    ///
    /// # Errors
    ///
    /// Returns a typed diagnostic and byte offset for an invalid literal or decoration.
    pub fn parse_checked(s: &str) -> Result<(Self, SignalFlags), ParseIntervalDiagnostic> {
        let mut signals = SignalFlags::NONE;
        match parse_decorated_literal(s)? {
            ParsedDecorated::NaI => Ok((Self::NAI, signals)),
            ParsedDecorated::Interval { bare, decoration } => {
                if bare.possibly_undefined {
                    signals.insert(Signal::PossiblyUndefinedOperation);
                }
                let Some(decoration) = decoration else {
                    return Ok((Self::new_dec_raw(bare.interval), signals));
                };
                // The parser must reject combinations forbidden at
                // Level 1. setDec normalization is only applied after
                // that validation.
                //
                // If the source was bounded, but conversion overflowed
                // and produced an unbounded interval, com becomes dac.
                let decoration = if decoration == Decoration::Com
                    && bare.source_bounded
                    && !bare.interval.is_bounded_raw()
                {
                    Decoration::Dac
                } else {
                    decoration
                };

                Ok((Self::set_dec_raw(bare.interval, decoration), signals))
            }
        }
    }
}

/// IEEE intervalToText.
///
/// The recognized conversion specifiers are:
///
/// - absent;
/// - "";
/// - "hex".
///
/// Every other specifier falls back to the same exact hexadecimal
/// representation, as permitted by the standard.
///
/// Returns the number of UTF-8/ASCII bytes written.
///
/// # Errors
///
/// Returns [`TextError::BufferTooSmall`] when `output` cannot hold the
/// complete representation.
///
/// # Example
///
/// ```
/// use maryada::{Interval, IntervalOps, interval_to_text};
///
/// let mut output = [0_u8; 64];
/// let length = interval_to_text(Interval::new(1.0, 2.0), None, &mut output)?;
/// assert_eq!(core::str::from_utf8(&output[..length]).unwrap(), "[0x1p+0,0x1p+1]");
/// # Ok::<(), maryada::TextError>(())
/// ```
#[allow(clippy::arithmetic_side_effects, clippy::indexing_slicing)]
pub fn interval_to_text<T: IntervalDatum>(
    x: T,
    _cs: Option<&str>,
    output: &mut [u8],
) -> Result<usize, TextError> {
    let mut text = [0u8; 64];
    if x.__is_nai() {
        return write_ascii(output, b"[nai]");
    }
    let interval = x.__interval();
    let decoration = x.__decoration();
    let mut cursor = 0;
    cursor += write_bare_literal(interval, &mut text[cursor..])?;
    if let Some(decoration) = decoration {
        cursor += write_ascii(&mut text[cursor..], b"_")?;
        cursor += write_decoration(decoration, &mut text[cursor..])?;
    }
    write_ascii(output, &text[..cursor])
}

// Required grammar:
//
// Number literals:
// - decimal;
// - C99 hexadecimal floating constant;
// - rational p/q;
// - inf/infinity.
//
// Bare interval literals:
// - [l,u], including omitted bounds;
// - [x];
// - uncertain forms m?r, m?ru, m??d, and exponent forms;
// - [empty], [];
// - [entire];
//
// Decorated literals:
// - bare_literal_trv;
// - bare_literal_def;
// - bare_literal_dac;
// - bare_literal_com;
// - [nai].
//
// Alphabetic matching is case-insensitive.

#[allow(clippy::arithmetic_side_effects, clippy::string_slice)]
fn parse_bare_literal(s: &str) -> Result<ParsedBare, ParseIntervalDiagnostic> {
    let offset = s.len() - s.trim_start_matches(char::is_whitespace).len();
    let s = trim_ascii_space(s);
    if s.starts_with('[') {
        let Some(closing) = s.find(']') else {
            return Err(ParseIntervalDiagnostic {
                kind: ParseIntervalErrorKind::InvalidSyntax,
                byte_offset: offset + s.len(),
            });
        };
        if closing + 1 != s.len() {
            let trailing = s[closing + 1..].trim_start_matches(char::is_whitespace);
            return Err(ParseIntervalDiagnostic {
                kind: ParseIntervalErrorKind::InvalidSyntax,
                byte_offset: offset + s.len() - trailing.len(),
            });
        }
        return parse_bracket_literal(&s[1..closing], offset + 1);
    }
    parse_uncertain_literal(s, offset)
}

#[allow(clippy::arithmetic_side_effects, clippy::string_slice)]
fn parse_decorated_literal(s: &str) -> Result<ParsedDecorated, ParseIntervalDiagnostic> {
    let offset = s.len() - s.trim_start_matches(char::is_whitespace).len();
    let s = trim_ascii_space(s);
    if eq_ascii_case(s, "[nai]") {
        return Ok(ParsedDecorated::NaI);
    }

    let (bare_text, decoration) = if let Some(index) = s.rfind('_') {
        let invalid_decoration = ParseIntervalDiagnostic {
            kind: ParseIntervalErrorKind::InvalidDecoration,
            byte_offset: offset + index + 1,
        };
        let decoration: Decoration = s[index + 1..].parse().map_err(|_| invalid_decoration)?;
        if decoration == Decoration::Ill {
            return Err(invalid_decoration);
        }
        (&s[..index], Some(decoration))
    } else {
        (s, None)
    };
    let bare = parse_bare_literal(bare_text).map_err(|mut error| {
        error.byte_offset += offset;
        error
    })?;
    if let Some(decoration) = decoration
        && (bare.interval.is_empty_raw() && decoration != Decoration::Trv
            || decoration == Decoration::Com && !bare.source_bounded)
    {
        return Err(ParseIntervalDiagnostic {
            kind: ParseIntervalErrorKind::InvalidDecoration,
            byte_offset: offset + bare_text.len() + 1,
        });
    }
    Ok(ParsedDecorated::Interval { bare, decoration })
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum NumberKind {
    Decimal,
    Hexadecimal,
    Rational,
    Infinity,
}

fn trim_ascii_space(value: &str) -> &str {
    value.trim_matches(char::is_whitespace)
}

const fn eq_ascii_case(left: &str, right: &str) -> bool {
    left.eq_ignore_ascii_case(right)
}

fn number_kind(value: &str) -> NumberKind {
    let unsigned = value.strip_prefix(['+', '-']).unwrap_or(value);
    if eq_ascii_case(unsigned, "inf") || eq_ascii_case(unsigned, "infinity") {
        NumberKind::Infinity
    } else if value.as_bytes().contains(&b'/') {
        NumberKind::Rational
    } else if unsigned.len() >= 2
        && unsigned.as_bytes().first() == Some(&b'0')
        && unsigned
            .as_bytes()
            .get(1)
            .is_some_and(|byte| byte.eq_ignore_ascii_case(&b'x'))
    {
        NumberKind::Hexadecimal
    } else {
        NumberKind::Decimal
    }
}

struct RadixLiteral<'a> {
    negative: bool,
    mantissa: &'a str,
    exponent: &'a str,
    point_offset: i128,
}

impl<'a> RadixLiteral<'a> {
    // The caller has already validated the token and found a nonzero endpoint.
    #[allow(clippy::arithmetic_side_effects)]
    fn from_nonzero(value: &'a str, hexadecimal: bool) -> Option<Self> {
        let negative = value.starts_with('-');
        let unsigned = value.strip_prefix(['+', '-']).unwrap_or(value);
        let (mantissa, exponent) = if hexadecimal {
            unsigned.get(2..)?.split_once(['p', 'P'])?
        } else {
            unsigned.split_once(['e', 'E']).unwrap_or((unsigned, ""))
        };

        let point = mantissa.find('.').unwrap_or(mantissa.len());
        let first = mantissa
            .bytes()
            .position(|byte| byte != b'0' && byte != b'.')?;
        let leading_zeros = first - usize::from(first > point);
        let mantissa = mantissa.get(first..)?;
        let mut point_offset = i128::try_from(point).ok()? - i128::try_from(leading_zeros).ok()?;
        if hexadecimal {
            let first_digit = mantissa.chars().next()?.to_digit(16)?;
            let first_digit_bits = u32::BITS - first_digit.leading_zeros();
            point_offset = 4 * (point_offset - 1) + i128::from(first_digit_bits);
        }
        Some(Self {
            negative,
            mantissa,
            exponent,
            point_offset,
        })
    }
}

#[allow(clippy::arithmetic_side_effects)]
fn compare_shifted_exponents(left: &str, right: &str, mut carry: i128) -> Ordering {
    fn signed_digits(exponent: &str) -> impl Iterator<Item = i128> + '_ {
        let sign = if exponent.starts_with('-') { -1 } else { 1 };
        let digits = exponent.strip_prefix(['+', '-']).unwrap_or(exponent);
        digits
            .bytes()
            .rev()
            .map(move |digit| i128::from(digit - b'0') * sign)
    }

    // Subtract from the least significant digit upward, starting with the
    // difference between the mantissa offsets. This bounds the carry even
    // when an exponent has more decimal digits than any machine integer.
    let mut left = signed_digits(left);
    let mut right = signed_digits(right);
    let mut nonzero = false;
    loop {
        let (left, right) = (left.next(), right.next());
        if left.is_none() && right.is_none() {
            break;
        }
        carry += left.unwrap_or(0) - right.unwrap_or(0);
        nonzero |= carry.rem_euclid(10) != 0;
        carry = carry.div_euclid(10);
    }
    if carry == 0 && nonzero {
        Ordering::Greater
    } else {
        carry.cmp(&0)
    }
}

fn compare_padded_digits(
    mut left: impl Iterator<Item = u32>,
    mut right: impl Iterator<Item = u32>,
) -> Ordering {
    loop {
        let (left, right) = (left.next(), right.next());
        if left.is_none() && right.is_none() {
            return Ordering::Equal;
        }
        let order = left.unwrap_or(0).cmp(&right.unwrap_or(0));
        if order != Ordering::Equal {
            return order;
        }
    }
}

#[allow(clippy::arithmetic_side_effects)]
fn compare_same_radix_literals(left: &str, right: &str, hexadecimal: bool) -> Option<Ordering> {
    fn hex_bits(mantissa: &str) -> impl Iterator<Item = u32> + '_ {
        mantissa
            .chars()
            .filter_map(|digit| digit.to_digit(16))
            .flat_map(|digit| (0..4).rev().map(move |shift| (digit >> shift) & 1))
            .skip_while(|&bit| bit == 0)
    }

    let left = RadixLiteral::from_nonzero(left, hexadecimal)?;
    let right = RadixLiteral::from_nonzero(right, hexadecimal)?;
    debug_assert_eq!(left.negative, right.negative);
    let order = compare_shifted_exponents(
        left.exponent,
        right.exponent,
        left.point_offset - right.point_offset,
    )
    .then_with(|| {
        if hexadecimal {
            compare_padded_digits(hex_bits(left.mantissa), hex_bits(right.mantissa))
        } else {
            compare_padded_digits(
                left.mantissa.chars().filter_map(|digit| digit.to_digit(10)),
                right
                    .mantissa
                    .chars()
                    .filter_map(|digit| digit.to_digit(10)),
            )
        }
    });
    Some(if left.negative {
        order.reverse()
    } else {
        order
    })
}

#[allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::string_slice,
    clippy::too_many_lines
)]
fn parse_bracket_literal(
    inner: &str,
    offset: usize,
) -> Result<ParsedBare, ParseIntervalDiagnostic> {
    let offset = offset + inner.len() - inner.trim_start_matches(char::is_whitespace).len();
    let inner = trim_ascii_space(inner);
    if inner.is_empty() || eq_ascii_case(inner, "empty") {
        return Ok(ParsedBare {
            interval: Interval::EMPTY,
            source_bounded: true,
            possibly_undefined: false,
        });
    }
    if eq_ascii_case(inner, "entire") {
        return Ok(ParsedBare {
            interval: Interval::ENTIRE,
            source_bounded: false,
            possibly_undefined: false,
        });
    }

    let comma = inner.find(',');
    if comma.is_none() {
        let value = trim_ascii_space(inner);
        let kind = number_kind(value);
        let invalid_number = ParseIntervalDiagnostic {
            kind: ParseIntervalErrorKind::InvalidNumber,
            byte_offset: offset,
        };
        let lower = parse_number_lower(value).ok_or(invalid_number)?;
        let upper = parse_number_upper(value).ok_or(invalid_number)?;
        if kind == NumberKind::Infinity {
            return Err(ParseIntervalDiagnostic {
                kind: ParseIntervalErrorKind::InvalidInfinity,
                byte_offset: offset,
            });
        }
        return Ok(ParsedBare {
            interval: Interval::from_valid_bounds(lower, upper),
            source_bounded: true,
            possibly_undefined: false,
        });
    }

    let comma = comma.ok_or(ParseIntervalDiagnostic {
        kind: ParseIntervalErrorKind::InvalidSyntax,
        byte_offset: offset,
    })?;
    if let Some(extra_comma) = inner[comma + 1..].find(',') {
        return Err(ParseIntervalDiagnostic {
            kind: ParseIntervalErrorKind::InvalidSyntax,
            byte_offset: offset + comma + 1 + extra_comma,
        });
    }
    let lower_text = trim_ascii_space(&inner[..comma]);
    let upper_text = trim_ascii_space(&inner[comma + 1..]);
    let lower_kind = (!lower_text.is_empty()).then(|| number_kind(lower_text));
    let upper_kind = (!upper_text.is_empty()).then(|| number_kind(upper_text));
    let lower_error = ParseIntervalDiagnostic {
        kind: ParseIntervalErrorKind::InvalidNumber,
        byte_offset: offset,
    };
    let upper_error = ParseIntervalDiagnostic {
        kind: ParseIntervalErrorKind::InvalidNumber,
        byte_offset: offset + inner.len()
            - inner[comma + 1..]
                .trim_start_matches(char::is_whitespace)
                .len(),
    };

    let (lower_down, lower_up) = if lower_text.is_empty() {
        (f64::NEG_INFINITY, f64::NEG_INFINITY)
    } else {
        (
            parse_number_lower(lower_text).ok_or(lower_error)?,
            parse_number_upper(lower_text).ok_or(lower_error)?,
        )
    };
    let (upper_down, upper_up) = if upper_text.is_empty() {
        (f64::INFINITY, f64::INFINITY)
    } else {
        (
            parse_number_lower(upper_text).ok_or(upper_error)?,
            parse_number_upper(upper_text).ok_or(upper_error)?,
        )
    };
    if lower_kind == Some(NumberKind::Infinity) && lower_down == f64::INFINITY {
        return Err(ParseIntervalDiagnostic {
            kind: ParseIntervalErrorKind::InvalidInfinity,
            ..lower_error
        });
    }
    if upper_kind == Some(NumberKind::Infinity) && upper_up == f64::NEG_INFINITY {
        return Err(ParseIntervalDiagnostic {
            kind: ParseIntervalErrorKind::InvalidInfinity,
            ..upper_error
        });
    }
    let mut possibly_undefined = false;
    if lower_up > upper_down {
        if lower_down >= upper_up {
            return Err(ParseIntervalDiagnostic {
                kind: ParseIntervalErrorKind::ReversedBounds,
                ..upper_error
            });
        }
        // Both exact endpoints are inside the same gap between binary64 values.
        if lower_kind == upper_kind
            && matches!(
                lower_kind,
                Some(NumberKind::Decimal | NumberKind::Hexadecimal)
            )
        {
            let hexadecimal = lower_kind == Some(NumberKind::Hexadecimal);
            if compare_same_radix_literals(lower_text, upper_text, hexadecimal)
                .ok_or(upper_error)?
                == Ordering::Greater
            {
                return Err(ParseIntervalDiagnostic {
                    kind: ParseIntervalErrorKind::ReversedBounds,
                    ..upper_error
                });
            }
        } else {
            // Rational and mixed-radix pairs may use the hull of this gap.
            possibly_undefined = lower_text != upper_text;
        }
    }

    let source_bounded = !lower_text.is_empty()
        && !upper_text.is_empty()
        && lower_kind != Some(NumberKind::Infinity)
        && upper_kind != Some(NumberKind::Infinity);
    Ok(ParsedBare {
        interval: Interval::from_valid_bounds(lower_down, upper_up),
        source_bounded,
        possibly_undefined,
    })
}

#[derive(Clone, Copy)]
struct DecimalDigits<'a> {
    written: &'a str,
    append_zero: bool,
}

impl<'a> DecimalDigits<'a> {
    const ZERO: Self = Self {
        written: "",
        append_zero: false,
    };

    #[allow(clippy::arithmetic_side_effects)]
    fn digits(self) -> impl DoubleEndedIterator<Item = u8> + Clone + 'a {
        self.written
            .bytes()
            .filter(|&byte| byte != b'.')
            .map(|byte| byte - b'0')
            .chain(self.append_zero.then_some(0))
    }

    fn cmp_magnitude(self, other: Self) -> Ordering {
        let left = self.digits().skip_while(|&digit| digit == 0);
        let right = other.digits().skip_while(|&digit| digit == 0);
        left.clone()
            .count()
            .cmp(&right.clone().count())
            .then_with(|| left.cmp(right))
    }
}

#[allow(clippy::arithmetic_side_effects, clippy::indexing_slicing)]
fn round_uncertain_endpoint(
    center: DecimalDigits<'_>,
    negative: bool,
    radius: DecimalDigits<'_>,
    exponent: i128,
    direction: Direction,
) -> Option<f64> {
    // Every finite binary64 value has fewer than 800 significant decimal
    // digits (an integer significand of at most 53 bits times at most 5^1074,
    // or an integer below 2^1024). An omitted nonzero tail can therefore be
    // replaced by one digit just beyond the prefix without changing either
    // directed bound.
    const PREFIX_DIGITS: usize = 800;

    let subtract = negative != (direction == Direction::Down);
    let (left, right, negative) = if subtract && center.cmp_magnitude(radius) == Ordering::Less {
        (radius, center, !negative)
    } else {
        (center, radius, negative)
    };
    let digits = rounding::sum_decimal_digits(left.digits().rev(), right.digits().rev(), subtract);
    // Locate the highest nonzero digit after all carries and borrows.
    let Some(highest) = digits
        .clone()
        .enumerate()
        .filter_map(|(place, digit)| (digit != 0).then_some(place))
        .last()
    else {
        return Some(0.0);
    };

    let mut literal = [0_u8; PREFIX_DIGITS + 24];
    let start = usize::from(negative);
    if negative {
        literal[0] = b'-';
    }
    let length = highest + 1;
    let omitted = length.saturating_sub(PREFIX_DIGITS);
    let mut sticky = false;
    for (place, digit) in digits.take(length).enumerate() {
        if place < omitted {
            sticky |= digit != 0;
        } else {
            literal[start + highest - place] = b'0' + digit;
        }
    }
    let mut cursor = start + length.min(PREFIX_DIGITS);
    if sticky {
        literal[cursor] = b'1';
        cursor += 1;
    }
    // At these extremes any nonzero leading digit already fixes the
    // directed bounds to the underflow or overflow pair.
    let scientific_exponent = exponent
        .saturating_add(i128::try_from(highest).ok()?)
        .clamp(-325, 309);
    let exponent = scientific_exponent - i128::try_from(cursor - start - 1).ok()?;
    literal[cursor] = b'e';
    cursor += 1;
    cursor += write_i64_decimal(i64::try_from(exponent).ok()?, &mut literal[cursor..])?;
    rounding::number_literal(core::str::from_utf8(&literal[..cursor]).ok()?, direction)
}

#[allow(clippy::arithmetic_side_effects, clippy::indexing_slicing)]
fn write_i64_decimal(value: i64, output: &mut [u8]) -> Option<usize> {
    let negative = value < 0;
    let mut magnitude = value.unsigned_abs();
    let mut reversed = [0u8; 20];
    let mut count = 0;
    loop {
        reversed[count] = b'0' + u8::try_from(magnitude % 10).ok()?;
        count += 1;
        magnitude /= 10;
        if magnitude == 0 {
            break;
        }
    }
    let required = count + usize::from(negative);
    if output.len() < required {
        return None;
    }
    let mut cursor = 0;
    if negative {
        output[cursor] = b'-';
        cursor += 1;
    }
    for index in (0..count).rev() {
        output[cursor] = reversed[index];
        cursor += 1;
    }
    Some(cursor)
}

#[allow(clippy::arithmetic_side_effects, clippy::string_slice)]
fn parse_uncertain_suffix(
    suffix: &str,
    offset: usize,
) -> Result<(&str, i128, Option<u8>), ParseIntervalDiagnostic> {
    let invalid_suffix = ParseIntervalDiagnostic {
        kind: ParseIntervalErrorKind::InvalidUncertainty,
        byte_offset: offset,
    };
    let mut suffix_without_exponent = suffix;
    let exponent_index = suffix
        .bytes()
        .position(|byte| byte.eq_ignore_ascii_case(&b'e'));
    let exponent = if let Some(index) = exponent_index {
        let exponent = suffix.get(index + 1..).ok_or(invalid_suffix)?;
        suffix_without_exponent = suffix.get(..index).ok_or(invalid_suffix)?;
        parse_uncertain_exponent(exponent).ok_or(ParseIntervalDiagnostic {
            byte_offset: offset + index + 1,
            ..invalid_suffix
        })?
    } else {
        0
    };
    let direction = match suffix_without_exponent.as_bytes().last().copied() {
        Some(byte) if byte.eq_ignore_ascii_case(&b'u') => {
            suffix_without_exponent = suffix_without_exponent
                .get(..suffix_without_exponent.len() - 1)
                .ok_or(invalid_suffix)?;
            Some(b'u')
        }
        Some(byte) if byte.eq_ignore_ascii_case(&b'd') => {
            suffix_without_exponent = suffix_without_exponent
                .get(..suffix_without_exponent.len() - 1)
                .ok_or(invalid_suffix)?;
            Some(b'd')
        }
        _ => None,
    };
    Ok((suffix_without_exponent, exponent, direction))
}

#[allow(clippy::arithmetic_side_effects)]
fn parse_uncertain_literal(s: &str, offset: usize) -> Result<ParsedBare, ParseIntervalDiagnostic> {
    let (mantissa, suffix) = s.split_once('?').ok_or(ParseIntervalDiagnostic {
        kind: ParseIntervalErrorKind::InvalidSyntax,
        byte_offset: offset,
    })?;
    let invalid_mantissa = ParseIntervalDiagnostic {
        kind: ParseIntervalErrorKind::InvalidUncertainty,
        byte_offset: offset,
    };
    let suffix_offset = offset + mantissa.len() + 1;
    let invalid_radius = ParseIntervalDiagnostic {
        byte_offset: suffix_offset,
        ..invalid_mantissa
    };
    let negative = mantissa.starts_with('-');
    let mantissa = mantissa.strip_prefix(['+', '-']).unwrap_or(mantissa);
    let (integer, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    if integer.is_empty() && fraction.is_empty()
        || !integer
            .bytes()
            .chain(fraction.bytes())
            .all(|byte| byte.is_ascii_digit())
    {
        return Err(invalid_mantissa);
    }
    let (suffix, exponent, direction) = parse_uncertain_suffix(suffix, suffix_offset)?;

    let unbounded = suffix == "?";
    let half_unit = suffix.is_empty();
    let radius = DecimalDigits {
        written: match suffix {
            "" => "5",
            "?" => "",
            digits if digits.bytes().all(|byte| byte.is_ascii_digit()) => digits,
            _ => return Err(invalid_radius),
        },
        append_zero: false,
    };
    let center = DecimalDigits {
        written: mantissa,
        append_zero: half_unit,
    };
    // For the default half-unit radius, add one decimal place so that both
    // the center and the radius are exact integers in the same units.
    let places =
        i128::try_from(fraction.len()).map_err(|_| invalid_mantissa)? + i128::from(half_unit);
    let exponent = exponent.saturating_sub(places);

    let lower = if unbounded && direction != Some(b'u') {
        f64::NEG_INFINITY
    } else {
        let radius = if direction == Some(b'u') {
            DecimalDigits::ZERO
        } else {
            radius
        };
        round_uncertain_endpoint(center, negative, radius, exponent, Direction::Down)
            .ok_or(invalid_radius)?
    };
    let upper = if unbounded && direction != Some(b'd') {
        f64::INFINITY
    } else {
        let radius = if direction == Some(b'd') {
            DecimalDigits::ZERO
        } else {
            radius
        };
        round_uncertain_endpoint(center, negative, radius, exponent, Direction::Up)
            .ok_or(invalid_radius)?
    };

    Ok(ParsedBare {
        interval: Interval::from_valid_bounds(lower, upper),
        source_bounded: !unbounded,
        possibly_undefined: false,
    })
}

#[allow(clippy::arithmetic_side_effects)]
fn parse_uncertain_exponent(value: &str) -> Option<i128> {
    let negative = value.starts_with('-');
    let digits = value.strip_prefix(['+', '-']).unwrap_or(value);
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let mut magnitude = 0_i128;
    for digit in digits.bytes() {
        magnitude = magnitude
            .saturating_mul(10)
            .saturating_add(i128::from(digit - b'0'));
    }
    Some(if negative { -magnitude } else { magnitude })
}

fn parse_number_lower(s: &str) -> Option<f64> {
    rounding::number_literal(s, Direction::Down)
}

fn parse_number_upper(s: &str) -> Option<f64> {
    rounding::number_literal(s, Direction::Up)
}

#[allow(clippy::arithmetic_side_effects, clippy::indexing_slicing)]
fn write_bare_literal(interval: Interval, output: &mut [u8]) -> Result<usize, TextError> {
    if interval.is_empty_raw() {
        return write_ascii(output, b"[empty]");
    }
    if interval.is_entire_raw() {
        return write_ascii(output, b"[entire]");
    }
    let mut cursor = 0;
    cursor += write_ascii(&mut output[cursor..], b"[")?;
    cursor += write_f64_hex(interval.inf_raw(), &mut output[cursor..])?;
    cursor += write_ascii(&mut output[cursor..], b",")?;
    cursor += write_f64_hex(interval.sup_raw(), &mut output[cursor..])?;
    cursor += write_ascii(&mut output[cursor..], b"]")?;
    Ok(cursor)
}

/// Write an exact C99 hexadecimal floating constant.
///
/// Required examples of the intended form:
///
/// - `-0x0p+0`
/// - `0x1.8p+1`
/// - `-inf`
/// - `inf`
#[allow(
    clippy::arithmetic_side_effects,
    clippy::expect_used,
    clippy::indexing_slicing
)]
fn write_f64_hex(value: f64, output: &mut [u8]) -> Result<usize, TextError> {
    debug_assert!(!value.is_nan());
    let mut text = [0u8; 32];
    let mut cursor = 0;
    if value.is_sign_negative() {
        text[cursor] = b'-';
        cursor += 1;
    }
    if value.is_infinite() {
        text[cursor..cursor + 3].copy_from_slice(b"inf");
        cursor += 3;
        return write_ascii(output, &text[..cursor]);
    }
    if value == 0.0 {
        text[cursor..cursor + 6].copy_from_slice(b"0x0p+0");
        cursor += 6;
        return write_ascii(output, &text[..cursor]);
    }

    let bits = value.to_bits();
    let raw_exponent = i32::try_from((bits >> 52) & 0x7ff).expect("binary64 exponent fits in i32");
    let raw_fraction = bits & ((1u64 << 52) - 1);
    let (significand, exponent) = if raw_exponent == 0 {
        let highest = raw_fraction.ilog2();
        let shift = 52 - highest;
        (
            raw_fraction << shift,
            -1022 - i32::try_from(shift).expect("subnormal shift fits in i32"),
        )
    } else {
        (raw_fraction | (1u64 << 52), raw_exponent - 1023)
    };
    text[cursor..cursor + 3].copy_from_slice(b"0x1");
    cursor += 3;
    let fraction = significand & ((1u64 << 52) - 1);
    if fraction != 0 {
        text[cursor] = b'.';
        cursor += 1;
        let mut digits = 13usize;
        while digits > 0 && (fraction >> ((13 - digits) * 4)).trailing_zeros() >= 4 {
            digits -= 1;
        }
        for index in 0..digits {
            let shift = (12 - index) * 4;
            let digit = u8::try_from((fraction >> shift) & 0xf).expect("hex digit fits in u8");
            text[cursor] = if digit < 10 {
                b'0' + digit
            } else {
                b'a' + digit - 10
            };
            cursor += 1;
        }
    }
    text[cursor] = b'p';
    cursor += 1;
    if exponent >= 0 {
        text[cursor] = b'+';
        cursor += 1;
    }
    cursor += write_i64_decimal(i64::from(exponent), &mut text[cursor..])
        .expect("buffer is large enough");
    write_ascii(output, &text[..cursor])
}

fn write_decoration(decoration: Decoration, output: &mut [u8]) -> Result<usize, TextError> {
    let text = match decoration {
        Decoration::Ill => b"ill",
        Decoration::Trv => b"trv",
        Decoration::Def => b"def",
        Decoration::Dac => b"dac",
        Decoration::Com => b"com",
    };
    write_ascii(output, text)
}

#[allow(clippy::indexing_slicing)]
fn write_ascii(output: &mut [u8], value: &[u8]) -> Result<usize, TextError> {
    if output.len() < value.len() {
        return Err(TextError::BufferTooSmall {
            required: value.len(),
        });
    }
    output[..value.len()].copy_from_slice(value);
    Ok(value.len())
}
