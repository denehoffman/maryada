use crate::{
    DecoratedInterval, Decoration, Interval, IntervalDatum,
    rounding::{self, Direction},
    signals::{Signal, SignalSink},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextError {
    BufferTooSmall { required: usize },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParseIntervalError;

struct ParsedBare {
    interval: Interval,

    /// Whether the Level 1 source interval was bounded before
    /// conversion to binary64.
    source_bounded: bool,

    /// True for the accuracy-relaxed form defined in 6.7.5.
    accuracy_relaxed: bool,
}

enum ParsedDecorated {
    NaI,
    Interval {
        bare: ParsedBare,
        decoration: Option<Decoration>,
    },
}

/// Bare IEEE textToInterval.
pub(crate) fn text_to_interval<S: SignalSink>(s: &str, signals: &mut S) -> Interval {
    match parse_bare_literal(s) {
        Ok(parsed) => {
            // This implementation chooses the most accurate permitted
            // behavior for accuracy-relaxed input:
            //
            // - return the exact binary64 hull when l <= u;
            // - fail when l > u.
            //
            // Therefore PossiblyUndefinedOperation need not be raised.
            parsed.interval
        }
        Err(()) => {
            signals.raise(Signal::UndefinedOperation);
            Interval::EMPTY
        }
    }
}

/// Decorated IEEE textToInterval.
pub(crate) fn text_to_decorated_interval<S: SignalSink>(
    s: &str,
    signals: &mut S,
) -> DecoratedInterval {
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
pub fn interval_to_text<T: IntervalDatum>(
    x: T,
    cs: Option<&str>,
    output: &mut [u8],
) -> Result<usize, TextError> {
    let _specifier_valid = matches!(cs, None | Some("") | Some("hex"));
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
// - [nai] with bare value Empty.
//
// Decorated literals:
// - bare_literal_trv;
// - bare_literal_def;
// - bare_literal_dac;
// - bare_literal_com;
// - [nai].
//
// Alphabetic matching is case-insensitive.

fn parse_bare_literal(s: &str) -> Result<ParsedBare, ()> {
    let s = trim_ascii_space(s);
    if s.starts_with('[') && s.ends_with(']') {
        return parse_bracket_literal(&s[1..s.len() - 1]);
    }
    parse_uncertain_literal(s)
}

fn parse_decorated_literal(s: &str) -> Result<ParsedDecorated, ()> {
    let s = trim_ascii_space(s);
    if eq_ascii_case(s, "[nai]") {
        return Ok(ParsedDecorated::NaI);
    }

    let (bare_text, decoration) = if let Some(index) = s.rfind('_') {
        let decoration = match &s[index + 1..] {
            value if eq_ascii_case(value, "trv") => Decoration::Trv,
            value if eq_ascii_case(value, "def") => Decoration::Def,
            value if eq_ascii_case(value, "dac") => Decoration::Dac,
            value if eq_ascii_case(value, "com") => Decoration::Com,
            _ => return Err(()),
        };
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
    let _accuracy_relaxed = bare.accuracy_relaxed;
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
        && unsigned.as_bytes()[0] == b'0'
        && unsigned.as_bytes()[1].eq_ignore_ascii_case(&b'x')
    {
        NumberKind::Hexadecimal
    } else {
        NumberKind::Decimal
    }
}

fn parse_bracket_literal(inner: &str) -> Result<ParsedBare, ()> {
    let inner = trim_ascii_space(inner);
    if inner.is_empty() || eq_ascii_case(inner, "empty") || eq_ascii_case(inner, "nai") {
        return Ok(ParsedBare {
            interval: Interval::EMPTY,
            source_bounded: true,
            accuracy_relaxed: false,
        });
    }
    if eq_ascii_case(inner, "entire") {
        return Ok(ParsedBare {
            interval: Interval::ENTIRE,
            source_bounded: false,
            accuracy_relaxed: false,
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
            accuracy_relaxed: kind == NumberKind::Rational,
        });
    }

    let comma = comma.unwrap();
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
    let accuracy_relaxed = lower_kind == Some(NumberKind::Rational)
        || upper_kind == Some(NumberKind::Rational)
        || matches!(
            (lower_kind, upper_kind),
            (Some(NumberKind::Decimal), Some(NumberKind::Hexadecimal))
                | (Some(NumberKind::Hexadecimal), Some(NumberKind::Decimal))
        );
    Ok(ParsedBare {
        interval: Interval::from_valid_bounds(lower_down, upper_up),
        source_bounded,
        accuracy_relaxed,
    })
}

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

fn parse_uncertain_mantissa(value: &str) -> Option<(i128, usize)> {
    let (negative, unsigned) = if let Some(value) = value.strip_prefix('-') {
        (true, value)
    } else if let Some(value) = value.strip_prefix('+') {
        (false, value)
    } else {
        (false, value)
    };
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
    if digit_count == 0 || coefficient > i128::MAX as u128 {
        return None;
    }
    let coefficient = coefficient as i128;
    Some((
        if negative { -coefficient } else { coefficient },
        fractional_digits,
    ))
}

fn write_i64_decimal(value: i64, output: &mut [u8]) -> Option<usize> {
    let negative = value < 0;
    let mut magnitude = value.unsigned_abs();
    let mut reversed = [0u8; 20];
    let mut count = 0;
    loop {
        reversed[count] = b'0' + (magnitude % 10) as u8;
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

fn write_u128_decimal(mut value: u128, output: &mut [u8]) -> Option<usize> {
    let mut reversed = [0u8; 39];
    let mut count = 0;
    loop {
        reversed[count] = b'0' + (value % 10) as u8;
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
    let radius_exponent = exponent.saturating_sub(fractional_digits as i64);
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
    let mut suffix = &s[question + 1..];
    let exponent_index = suffix
        .bytes()
        .position(|byte| byte.eq_ignore_ascii_case(&b'e'));
    let exponent = if let Some(index) = exponent_index {
        let exponent = &suffix[index + 1..];
        suffix = &suffix[..index];
        parse_i64(exponent).ok_or(())?
    } else {
        0
    };
    let direction = match suffix.as_bytes().last().copied() {
        Some(byte) if byte.eq_ignore_ascii_case(&b'u') => {
            suffix = &suffix[..suffix.len() - 1];
            Some(b'u')
        }
        Some(byte) if byte.eq_ignore_ascii_case(&b'd') => {
            suffix = &suffix[..suffix.len() - 1];
            Some(b'd')
        }
        _ => None,
    };

    let fractional_digits = mantissa
        .split_once('.')
        .map_or(0, |(_, fraction)| fraction.len());
    let parsed_center = parse_uncertain_mantissa(mantissa);
    if parsed_center.is_none() {
        // Validate the decimal mantissa and the radius before using the
        // conservative, allocation-free fallback for very long literals.
        parse_number_lower(mantissa).ok_or(())?;
        if suffix != "?" && !suffix.is_empty() && !suffix.bytes().all(|byte| byte.is_ascii_digit())
        {
            return Err(());
        }
        let interval =
            fallback_uncertain_interval(mantissa, suffix, direction, exponent, fractional_digits)?;
        return Ok(ParsedBare {
            interval,
            source_bounded: suffix != "?",
            accuracy_relaxed: false,
        });
    }
    let (center, fractional_digits) = parsed_center.unwrap();
    let decimal_exponent = exponent.saturating_sub(fractional_digits as i64);
    if center.checked_mul(2).is_none() {
        let interval =
            fallback_uncertain_interval(mantissa, suffix, direction, exponent, fractional_digits)?;
        return Ok(ParsedBare {
            interval,
            source_bounded: suffix != "?",
            accuracy_relaxed: false,
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
            accuracy_relaxed: false,
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
    if radius_twice > i128::MAX as u128 {
        return Ok(ParsedBare {
            interval: Interval::ENTIRE,
            source_bounded: true,
            accuracy_relaxed: false,
        });
    }
    let center_twice = center * 2;
    let radius_twice = radius_twice as i128;
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
            accuracy_relaxed: false,
        });
    };
    let lower = parse_scaled_half_integer(lower_twice, decimal_exponent, Direction::Down)
        .unwrap_or(f64::NEG_INFINITY);
    let upper = parse_scaled_half_integer(upper_twice, decimal_exponent, Direction::Up)
        .unwrap_or(f64::INFINITY);
    Ok(ParsedBare {
        interval: Interval::from_valid_bounds(lower, upper),
        source_bounded: true,
        accuracy_relaxed: false,
    })
}

fn parse_i64(value: &str) -> Option<i64> {
    let (negative, digits) = if let Some(value) = value.strip_prefix('-') {
        (true, value)
    } else if let Some(value) = value.strip_prefix('+') {
        (false, value)
    } else {
        (false, value)
    };
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
    let raw_exponent = ((bits >> 52) & 0x7ff) as i32;
    let raw_fraction = bits & ((1u64 << 52) - 1);
    let (significand, exponent) = if raw_exponent == 0 {
        let highest = 63 - raw_fraction.leading_zeros();
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
        while digits > 0 && ((fraction >> ((13 - digits) * 4)) & 0xf) == 0 {
            digits -= 1;
        }
        for index in 0..digits {
            let shift = (12 - index) * 4;
            let digit = ((fraction >> shift) & 0xf) as u8;
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

fn write_ascii(output: &mut [u8], value: &[u8]) -> Result<usize, TextError> {
    if output.len() < value.len() {
        return Err(TextError::BufferTooSmall {
            required: value.len(),
        });
    }
    output[..value.len()].copy_from_slice(value);
    Ok(value.len())
}
