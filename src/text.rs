use crate::{
    DecoratedInterval, Decoration, Interval, IntervalDatum,
    rounding::{self, Direction},
    signals::{Signal, SignalSink},
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

struct ParsedBare {
    interval: Interval,

    /// Whether the Level 1 source interval was bounded before
    /// conversion to binary64.
    source_bounded: bool,
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
    parse_bare_literal(s).map_or_else(
        |()| {
            signals.raise(Signal::UndefinedOperation);
            Interval::EMPTY
        },
        |parsed| {
            // This implementation chooses the most accurate permitted
            // behavior for accuracy-relaxed input:
            //
            // - return the exact binary64 hull when l <= u;
            // - fail when l > u.
            //
            // Therefore PossiblyUndefinedOperation need not be raised.
            parsed.interval
        },
    )
}

/// Decorated IEEE textToInterval.
#[allow(clippy::arithmetic_side_effects)]
pub fn text_to_decorated_interval<S: SignalSink>(s: &str, signals: &mut S) -> DecoratedInterval {
    match parse_decorated_literal(s) {
        Ok(ParsedDecorated::NaI) => DecoratedInterval::NAI,
        Ok(ParsedDecorated::Interval {
            bare,
            decoration: None,
        }) => DecoratedInterval::new_dec_raw(bare.interval),
        Ok(ParsedDecorated::Interval {
            bare,
            decoration: Some(decoration),
        }) => {
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

            DecoratedInterval::set_dec_raw(bare.interval, decoration)
        }
        Err(()) => {
            signals.raise(Signal::UndefinedOperation);
            DecoratedInterval::NAI
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
/// use maryada::{Interval, interval_to_text};
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
fn parse_bare_literal(s: &str) -> Result<ParsedBare, ()> {
    let s = trim_ascii_space(s);
    if s.starts_with('[') && s.ends_with(']') {
        return parse_bracket_literal(&s[1..s.len() - 1]);
    }
    parse_uncertain_literal(s)
}

#[allow(clippy::arithmetic_side_effects, clippy::string_slice)]
fn parse_decorated_literal(s: &str) -> Result<ParsedDecorated, ()> {
    let s = trim_ascii_space(s);
    if eq_ascii_case(s, "[nai]") {
        return Ok(ParsedDecorated::NaI);
    }

    let (bare_text, decoration) = if let Some(index) = s.rfind('_') {
        let decoration: Decoration = s[index + 1..].parse().map_err(|_| ())?;
        if decoration == Decoration::Ill {
            return Err(());
        }
        (trim_ascii_space(&s[..index]), Some(decoration))
    } else {
        (s, None)
    };
    let bare = parse_bare_literal(bare_text)?;
    if let Some(decoration) = decoration {
        if bare.interval.is_empty_raw() && decoration != Decoration::Trv {
            return Err(());
        }
        if decoration == Decoration::Com && !bare.source_bounded {
            return Err(());
        }
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

fn eq_ascii_case(left: &str, right: &str) -> bool {
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

#[allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::string_slice
)]
fn parse_bracket_literal(inner: &str) -> Result<ParsedBare, ()> {
    let inner = trim_ascii_space(inner);
    if inner.is_empty() || eq_ascii_case(inner, "empty") {
        return Ok(ParsedBare {
            interval: Interval::EMPTY,
            source_bounded: true,
        });
    }
    if eq_ascii_case(inner, "entire") {
        return Ok(ParsedBare {
            interval: Interval::ENTIRE,
            source_bounded: false,
        });
    }

    let comma = inner.find(',');
    if comma.is_none() {
        let value = trim_ascii_space(inner);
        let kind = number_kind(value);
        let lower = parse_number_lower(value).ok_or(())?;
        let upper = parse_number_upper(value).ok_or(())?;
        if kind == NumberKind::Infinity {
            return Err(());
        }
        return Ok(ParsedBare {
            interval: Interval::from_valid_bounds(lower, upper),
            source_bounded: true,
        });
    }

    let comma = comma.ok_or(())?;
    if inner[comma + 1..].contains(',') {
        return Err(());
    }
    let lower_text = trim_ascii_space(&inner[..comma]);
    let upper_text = trim_ascii_space(&inner[comma + 1..]);
    let lower_kind = (!lower_text.is_empty()).then(|| number_kind(lower_text));
    let upper_kind = (!upper_text.is_empty()).then(|| number_kind(upper_text));

    let (lower_down, lower_up) = if lower_text.is_empty() {
        (f64::NEG_INFINITY, f64::NEG_INFINITY)
    } else {
        (
            parse_number_lower(lower_text).ok_or(())?,
            parse_number_upper(lower_text).ok_or(())?,
        )
    };
    let (upper_down, upper_up) = if upper_text.is_empty() {
        (f64::INFINITY, f64::INFINITY)
    } else {
        (
            parse_number_lower(upper_text).ok_or(())?,
            parse_number_upper(upper_text).ok_or(())?,
        )
    };
    if lower_kind == Some(NumberKind::Infinity) && lower_down == f64::INFINITY {
        return Err(());
    }
    if upper_kind == Some(NumberKind::Infinity) && upper_up == f64::NEG_INFINITY {
        return Err(());
    }
    // The directed enclosures normally decide the exact ordering. When two
    // nonrepresentable values occupy the same adjacent-float gap, accepting
    // the order is the permitted accurate-relaxed choice.
    if lower_down > upper_up || (lower_up > upper_down && lower_down == lower_up) {
        return Err(());
    }

    let source_bounded = !lower_text.is_empty()
        && !upper_text.is_empty()
        && lower_kind != Some(NumberKind::Infinity)
        && upper_kind != Some(NumberKind::Infinity);
    Ok(ParsedBare {
        interval: Interval::from_valid_bounds(lower_down, upper_up),
        source_bounded,
    })
}

#[allow(clippy::arithmetic_side_effects)]
fn parse_u128_digits(value: &str) -> Option<u128> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    value.bytes().try_fold(0u128, |result, digit| {
        result
            .checked_mul(10)?
            .checked_add(u128::from(digit - b'0'))
    })
}

#[allow(clippy::arithmetic_side_effects)]
fn parse_uncertain_mantissa(value: &str) -> Option<(i128, usize)> {
    let (negative, unsigned) = value.strip_prefix('-').map_or_else(
        || {
            value
                .strip_prefix('+')
                .map_or((false, value), |value| (false, value))
        },
        |value| (true, value),
    );
    let mut point_seen = false;
    let mut fractional_digits = 0usize;
    let mut coefficient = 0u128;
    let mut digit_count = 0usize;
    for byte in unsigned.bytes() {
        if byte == b'.' {
            if point_seen {
                return None;
            }
            point_seen = true;
        } else if byte.is_ascii_digit() {
            coefficient = coefficient
                .checked_mul(10)?
                .checked_add(u128::from(byte - b'0'))?;
            digit_count += 1;
            if point_seen {
                fractional_digits += 1;
            }
        } else {
            return None;
        }
    }
    if digit_count == 0 || coefficient > u128::MAX / 2 {
        return None;
    }
    let coefficient = i128::try_from(coefficient).ok()?;
    Some((
        if negative { -coefficient } else { coefficient },
        fractional_digits,
    ))
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

#[allow(clippy::arithmetic_side_effects, clippy::indexing_slicing)]
fn parse_scaled_half_integer(
    value_twice: i128,
    exponent: i64,
    direction: Direction,
) -> Option<f64> {
    let mut literal = [0u8; 96];
    let mut cursor = 0;
    if value_twice < 0 {
        literal[cursor] = b'-';
        cursor += 1;
    }
    let magnitude = value_twice.unsigned_abs();
    let integer = magnitude / 2;
    cursor += write_u128_decimal(integer, &mut literal[cursor..])?;
    if !magnitude.is_multiple_of(2) {
        literal[cursor..cursor + 2].copy_from_slice(b".5");
        cursor += 2;
    }
    if exponent != 0 {
        literal[cursor] = b'e';
        cursor += 1;
        cursor += write_i64_decimal(exponent, &mut literal[cursor..])?;
    }
    let literal = core::str::from_utf8(&literal[..cursor]).ok()?;
    rounding::number_literal(literal, direction)
}

#[allow(clippy::arithmetic_side_effects, clippy::indexing_slicing)]
fn write_u128_decimal(mut value: u128, output: &mut [u8]) -> Option<usize> {
    let mut reversed = [0u8; 39];
    let mut count = 0;
    loop {
        reversed[count] = b'0' + u8::try_from(value % 10).ok()?;
        count += 1;
        value /= 10;
        if value == 0 {
            break;
        }
    }
    if output.len() < count {
        return None;
    }
    for (output_digit, input_index) in output.iter_mut().zip((0..count).rev()) {
        *output_digit = reversed[input_index];
    }
    Some(count)
}

// The exponent is deliberately rounded to binary64 before directed scaling.
#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_precision_loss
)]
fn scaled_decimal_interval(value: &str, exponent: i64) -> Option<Interval> {
    let value_inf = parse_number_lower(value)?;
    let value_sup = parse_number_upper(value)?;
    if value_inf == 0.0 && value_sup == 0.0 {
        return Some(Interval::ZERO);
    }
    let exponent = exponent as f64;
    let scale_inf = rounding::exp10(exponent, Direction::Down);
    let scale_sup = rounding::exp10(exponent, Direction::Up);
    let lower_candidates = [
        rounding::mul(value_inf, scale_inf, Direction::Down),
        rounding::mul(value_inf, scale_sup, Direction::Down),
        rounding::mul(value_sup, scale_inf, Direction::Down),
        rounding::mul(value_sup, scale_sup, Direction::Down),
    ];
    let upper_candidates = [
        rounding::mul(value_inf, scale_inf, Direction::Up),
        rounding::mul(value_inf, scale_sup, Direction::Up),
        rounding::mul(value_sup, scale_inf, Direction::Up),
        rounding::mul(value_sup, scale_sup, Direction::Up),
    ];
    if lower_candidates.iter().any(|value| value.is_nan())
        || upper_candidates.iter().any(|value| value.is_nan())
    {
        return Some(Interval::ENTIRE);
    }
    let lower = lower_candidates.into_iter().fold(f64::INFINITY, f64::min);
    let upper = upper_candidates
        .into_iter()
        .fold(f64::NEG_INFINITY, f64::max);
    Some(Interval::from_valid_bounds(lower, upper))
}

// Input lengths are bounded by the representable string slice and fit in
// the exponent width used by the parser.
#[allow(
    clippy::arithmetic_side_effects,
    clippy::expect_used,
    clippy::unreachable
)]
fn fallback_uncertain_interval(
    mantissa: &str,
    radius: &str,
    direction: Option<u8>,
    exponent: i64,
    fractional_digits: usize,
) -> Result<Interval, ()> {
    let center = scaled_decimal_interval(mantissa, exponent).ok_or(())?;
    if radius == "?" {
        return Ok(match direction {
            Some(b'd') => Interval::from_valid_bounds(f64::NEG_INFINITY, center.sup_raw()),
            Some(b'u') => Interval::from_valid_bounds(center.inf_raw(), f64::INFINITY),
            None => Interval::ENTIRE,
            _ => unreachable!(),
        });
    }
    let radius = if radius.is_empty() { "0.5" } else { radius };
    let radius_exponent = exponent.saturating_sub(
        i64::try_from(fractional_digits).expect("fractional digit count fits in i64"),
    );
    let radius = scaled_decimal_interval(radius, radius_exponent).ok_or(())?;
    if center.is_entire_raw() || radius.is_entire_raw() {
        return Ok(Interval::ENTIRE);
    }
    let lower = if direction == Some(b'u') {
        center.inf_raw()
    } else {
        rounding::sub(center.inf_raw(), radius.sup_raw(), Direction::Down)
    };
    let upper = if direction == Some(b'd') {
        center.sup_raw()
    } else {
        rounding::add(center.sup_raw(), radius.sup_raw(), Direction::Up)
    };
    if lower.is_nan() || upper.is_nan() {
        Ok(Interval::ENTIRE)
    } else {
        Ok(Interval::from_valid_bounds(lower, upper))
    }
}

fn fallback_uncertain_literal(
    mantissa: &str,
    suffix: &str,
    direction: Option<u8>,
    exponent: i64,
    fractional_digits: usize,
) -> Result<ParsedBare, ()> {
    // Validate the decimal mantissa and the radius before using the
    // conservative, allocation-free fallback for very long literals.
    parse_number_lower(mantissa).ok_or(())?;
    if suffix != "?" && !suffix.is_empty() && !suffix.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(());
    }
    let interval =
        fallback_uncertain_interval(mantissa, suffix, direction, exponent, fractional_digits)?;
    Ok(ParsedBare {
        interval,
        source_bounded: suffix != "?",
    })
}

#[allow(clippy::arithmetic_side_effects, clippy::string_slice)]
fn parse_uncertain_suffix(suffix: &str) -> Result<(&str, i64, Option<u8>), ()> {
    let mut suffix_without_exponent = suffix;
    let exponent_index = suffix
        .bytes()
        .position(|byte| byte.eq_ignore_ascii_case(&b'e'));
    let exponent = if let Some(index) = exponent_index {
        let exponent = suffix.get(index + 1..).ok_or(())?;
        suffix_without_exponent = suffix.get(..index).ok_or(())?;
        parse_i64(exponent).ok_or(())?
    } else {
        0
    };
    let direction = match suffix_without_exponent.as_bytes().last().copied() {
        Some(byte) if byte.eq_ignore_ascii_case(&b'u') => {
            suffix_without_exponent = suffix_without_exponent
                .get(..suffix_without_exponent.len() - 1)
                .ok_or(())?;
            Some(b'u')
        }
        Some(byte) if byte.eq_ignore_ascii_case(&b'd') => {
            suffix_without_exponent = suffix_without_exponent
                .get(..suffix_without_exponent.len() - 1)
                .ok_or(())?;
            Some(b'd')
        }
        _ => None,
    };
    Ok((suffix_without_exponent, exponent, direction))
}

#[allow(
    clippy::arithmetic_side_effects,
    clippy::expect_used,
    clippy::string_slice,
    clippy::unreachable
)]
fn parse_uncertain_literal(s: &str) -> Result<ParsedBare, ()> {
    if s.bytes().any(|byte| byte.is_ascii_whitespace()) {
        return Err(());
    }
    let question = s.find('?').ok_or(())?;
    let mantissa = &s[..question];
    if mantissa
        .bytes()
        .any(|byte| byte.eq_ignore_ascii_case(&b'e'))
    {
        return Err(());
    }
    let (suffix, exponent, direction) = parse_uncertain_suffix(s.get(question + 1..).ok_or(())?)?;

    let fractional_digits = mantissa
        .split_once('.')
        .map_or(0, |(_, fraction)| fraction.len());
    let Some((center, fractional_digits)) = parse_uncertain_mantissa(mantissa) else {
        return fallback_uncertain_literal(
            mantissa,
            suffix,
            direction,
            exponent,
            fractional_digits,
        );
    };
    let decimal_exponent = exponent.saturating_sub(
        i64::try_from(fractional_digits).expect("fractional digit count fits in i64"),
    );
    if center.checked_mul(2).is_none() {
        let interval =
            fallback_uncertain_interval(mantissa, suffix, direction, exponent, fractional_digits)?;
        return Ok(ParsedBare {
            interval,
            source_bounded: suffix != "?",
        });
    }
    if suffix == "?" {
        let center_twice = center * 2;
        let center_lower =
            parse_scaled_half_integer(center_twice, decimal_exponent, Direction::Down)
                .unwrap_or(f64::NEG_INFINITY);
        let center_upper = parse_scaled_half_integer(center_twice, decimal_exponent, Direction::Up)
            .unwrap_or(f64::INFINITY);
        let interval = match direction {
            Some(b'd') => Interval::from_valid_bounds(f64::NEG_INFINITY, center_upper),
            Some(b'u') => Interval::from_valid_bounds(center_lower, f64::INFINITY),
            None => Interval::ENTIRE,
            _ => unreachable!(),
        };
        return Ok(ParsedBare {
            interval,
            source_bounded: false,
        });
    }

    let radius_twice = if suffix.is_empty() {
        1u128
    } else {
        parse_u128_digits(suffix)
            .ok_or(())?
            .checked_mul(2)
            .ok_or(())?
    };
    if radius_twice > u128::MAX / 2 {
        return Ok(ParsedBare {
            interval: Interval::ENTIRE,
            source_bounded: true,
        });
    }
    let center_twice = center * 2;
    let radius_twice = i128::try_from(radius_twice).map_err(|_| ())?;
    let lower_twice = if direction == Some(b'u') {
        Some(center_twice)
    } else {
        center_twice.checked_sub(radius_twice)
    };
    let upper_twice = if direction == Some(b'd') {
        Some(center_twice)
    } else {
        center_twice.checked_add(radius_twice)
    };
    let (Some(lower_twice), Some(upper_twice)) = (lower_twice, upper_twice) else {
        return Ok(ParsedBare {
            interval: Interval::ENTIRE,
            source_bounded: true,
        });
    };
    let lower = parse_scaled_half_integer(lower_twice, decimal_exponent, Direction::Down)
        .unwrap_or(f64::NEG_INFINITY);
    let upper = parse_scaled_half_integer(upper_twice, decimal_exponent, Direction::Up)
        .unwrap_or(f64::INFINITY);
    Ok(ParsedBare {
        interval: Interval::from_valid_bounds(lower, upper),
        source_bounded: true,
    })
}

#[allow(clippy::arithmetic_side_effects)]
fn parse_i64(value: &str) -> Option<i64> {
    let (negative, digits) = value.strip_prefix('-').map_or_else(
        || {
            value
                .strip_prefix('+')
                .map_or((false, value), |value| (false, value))
        },
        |value| (true, value),
    );
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let mut result = 0i64;
    for digit in digits.bytes() {
        result = result
            .saturating_mul(10)
            .saturating_add(i64::from(digit - b'0'));
    }
    Some(if negative {
        result.saturating_neg()
    } else {
        result
    })
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
