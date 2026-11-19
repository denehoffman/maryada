use core::{fmt, num::NonZeroU16};

use crate::{
    Interval, IntervalDatum,
    rounding::{Direction, ExactDecimal, exact_decimal_from_float, sum_decimal_digits},
    ux::write_decoration,
};

/// Decimal notation and precision for an enclosing interval literal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecimalIntervalFormat {
    /// Bracketed endpoints with a fixed number of digits after the point.
    Fixed {
        /// Number of digits after the point in each finite endpoint.
        fractional_digits: u16,
    },
    /// Bracketed scientific endpoints with a fixed number of significant digits.
    Scientific {
        /// Total digits in each finite endpoint's significand.
        significant_digits: NonZeroU16,
    },
    /// An uncertain literal `m?r` giving a decimal center and radius.
    ///
    /// The center and radius describe the same enclosure as [`Self::Fixed`].
    /// The radius counts units of the center's last decimal place, so `1.3?1`
    /// represents `[1.2,1.4]`.
    /// The center uses an extra decimal place when the rounded endpoints
    /// lie an odd number of units apart. Unbounded intervals use brackets.
    CenterRadius {
        /// Number of fractional digits used to round the endpoints.
        fractional_digits: u16,
    },
}

/// Writes an outward-rounded decimal literal for a bare or decorated interval.
///
/// The lower endpoint rounds toward negative infinity and the upper endpoint
/// toward positive infinity. The literal encloses the input before parsing.
/// Empty, Entire, `NaI`, and decoration suffixes use the usual interval syntax.
/// Parsing a rounded endpoint beyond the binary64 range produces an unbounded
/// result and can reduce a `com` decoration to `dac`.
///
/// # Errors
///
/// Returns a formatting error if the destination fails to accept the output.
///
/// # Examples
///
/// ```
/// use maryada::{DecimalIntervalFormat, Interval, IntervalOps, write_interval_decimal};
///
/// let interval = Interval::new(1.26, 1.34);
/// let mut output = String::new();
/// let format = DecimalIntervalFormat::Fixed { fractional_digits: 1 };
/// write_interval_decimal(interval, format, &mut output)?;
/// assert_eq!(output, "[1.2,1.4]");
/// # Ok::<(), core::fmt::Error>(())
/// ```
pub fn write_interval_decimal<T: IntervalDatum, W: fmt::Write>(
    value: T,
    format: DecimalIntervalFormat,
    output: &mut W,
) -> fmt::Result {
    if value.__is_nai() {
        return output.write_str("[nai]");
    }

    let interval = value.__interval();
    match format {
        _ if interval.is_empty_raw() => output.write_str("[empty]")?,
        _ if interval.is_entire_raw() => output.write_str("[entire]")?,
        DecimalIntervalFormat::CenterRadius { fractional_digits } if interval.is_bounded_raw() => {
            write_center_radius(output, interval, fractional_digits)?;
        }
        _ => {
            output.write_char('[')?;
            write_endpoint(output, interval.inf_raw(), format, Direction::Down)?;
            output.write_char(',')?;
            write_endpoint(output, interval.sup_raw(), format, Direction::Up)?;
            output.write_char(']')?;
        }
    }
    if let Some(decoration) = value.__decoration() {
        output.write_char('_')?;
        write_decoration(output, decoration)?;
    }
    Ok(())
}

impl ExactDecimal {
    #[allow(clippy::arithmetic_side_effects, clippy::indexing_slicing)]
    fn round_at(&mut self, power: i32, away_from_zero: bool) {
        if power <= self.shift || self.len == 0 {
            return;
        }

        let discarded = usize::try_from(power - self.shift)
            .unwrap_or(self.len)
            .min(self.len);
        let increment = away_from_zero && self.digits[..discarded].iter().any(|&digit| digit != 0);
        self.digits.copy_within(discarded..self.len, 0);
        self.len -= discarded;
        self.shift = power;
        if increment {
            for digit in &mut self.digits[..self.len] {
                if *digit < 9 {
                    *digit += 1;
                    return;
                }
                *digit = 0;
            }
            // Rounding discarded at least one digit, so there is room for the carry.
            self.digits[self.len] = 1;
            self.len += 1;
        }
    }
}

#[allow(clippy::arithmetic_side_effects, clippy::expect_used)]
fn leading_power(digits: &[u8], shift: i32) -> i32 {
    if digits.is_empty() {
        0
    } else {
        shift + i32::try_from(digits.len()).expect("bounded decimal digit count") - 1
    }
}

#[allow(clippy::arithmetic_side_effects)]
fn digit_at(digits: &[u8], shift: i32, power: i32) -> u8 {
    usize::try_from(power - shift)
        .ok()
        .and_then(|place| digits.get(place))
        .copied()
        .unwrap_or(0)
}

#[allow(clippy::arithmetic_side_effects)]
fn write_fixed<W: fmt::Write>(
    output: &mut W,
    digits: &[u8],
    shift: i32,
    fractional_digits: i32,
) -> fmt::Result {
    let leading = leading_power(digits, shift).max(0);
    for power in (-fractional_digits..=leading).rev() {
        output.write_char(char::from(b'0' + digit_at(digits, shift, power)))?;
        if power == 0 && fractional_digits != 0 {
            output.write_char('.')?;
        }
    }
    Ok(())
}

#[allow(clippy::arithmetic_side_effects, clippy::indexing_slicing)]
fn write_endpoint<W: fmt::Write>(
    output: &mut W,
    value: f64,
    format: DecimalIntervalFormat,
    direction: Direction,
) -> fmt::Result {
    if !value.is_finite() {
        return write!(output, "{value}");
    }

    let negative = value.is_sign_negative();
    let away_from_zero = negative == (direction == Direction::Down);
    let mut decimal = exact_decimal_from_float(value);
    if negative {
        output.write_char('-')?;
    }
    match format {
        DecimalIntervalFormat::Fixed { fractional_digits }
        | DecimalIntervalFormat::CenterRadius { fractional_digits } => {
            let places = i32::from(fractional_digits);
            decimal.round_at(-places, away_from_zero);
            write_fixed(
                output,
                &decimal.digits[..decimal.len],
                decimal.shift,
                places,
            )
        }
        DecimalIntervalFormat::Scientific { significant_digits } => {
            let places = i32::from(significant_digits.get()) - 1;
            let leading = leading_power(&decimal.digits[..decimal.len], decimal.shift);
            decimal.round_at(leading - places, away_from_zero);
            let digits = &decimal.digits[..decimal.len];
            let exponent = leading_power(digits, decimal.shift);
            for place in 0..=places {
                if place == 1 {
                    output.write_char('.')?;
                }
                output.write_char(char::from(
                    b'0' + digit_at(digits, decimal.shift, exponent - place),
                ))?;
            }
            write!(output, "e{exponent}")
        }
    }
}

// Binary64 has at most 309 integer digits and at most 1074 fractional digits.
// Two extra places cover a carry in the sum and the half-grid digit.
const MAX_FRACTIONAL_DIGITS: u16 = 1074;
const HALF_SUM_DIGITS: usize = 309 + 1074 + 2;

#[allow(clippy::arithmetic_side_effects, clippy::indexing_slicing)]
fn write_center_radius<W: fmt::Write>(
    output: &mut W,
    interval: Interval,
    fractional_digits: u16,
) -> fmt::Result {
    let lower = interval.inf_raw();
    let upper = interval.sup_raw();
    let shift = -i32::from(fractional_digits.min(MAX_FRACTIONAL_DIGITS));
    let mut lower_decimal = exact_decimal_from_float(lower);
    let mut upper_decimal = exact_decimal_from_float(upper);
    lower_decimal.round_at(shift, lower.is_sign_negative());
    upper_decimal.round_at(shift, upper.is_sign_positive());

    // Outward rounding on the common grid preserves the magnitude ordering.
    let lower_magnitude = lower.abs();
    let upper_magnitude = upper.abs();
    let (large, small, negative_center) = if lower_magnitude > upper_magnitude
        || (lower_magnitude == upper_magnitude && lower.is_sign_negative())
    {
        (&lower_decimal, &upper_decimal, lower.is_sign_negative())
    } else {
        (&upper_decimal, &lower_decimal, upper.is_sign_negative())
    };
    let large_digits = &large.digits[..large.len];
    let small_digits = &small.digits[..small.len];
    let leading =
        leading_power(large_digits, large.shift).max(leading_power(small_digits, small.shift));
    let large = (shift..=leading).map(|power| digit_at(large_digits, large.shift, power));
    let small = (shift..=leading).map(|power| digit_at(small_digits, small.shift, power));
    let opposite_signs = lower.is_sign_negative() != upper.is_sign_negative();

    let mut buffer = [0; HALF_SUM_DIGITS];
    let center = half_sum_digits(large.clone(), small.clone(), opposite_signs, &mut buffer);
    let extra_place = i32::from(center.first().is_some_and(|&digit| digit != 0));
    let places = i32::from(fractional_digits) + extra_place;
    if negative_center && !center.is_empty() {
        output.write_char('-')?;
    }
    write_fixed(output, center, shift - 1, places)?;
    output.write_char('?')?;

    let radius = half_sum_digits(large, small, !opposite_signs, &mut buffer);
    if radius.is_empty() {
        return output.write_char('0');
    }
    let leading = leading_power(radius, shift - 1);
    // The uncertain literal records the radius in units of the center's last place.
    for power in (-places..=leading).rev() {
        output.write_char(char::from(b'0' + digit_at(radius, shift - 1, power)))?;
    }
    Ok(())
}

#[allow(clippy::arithmetic_side_effects, clippy::indexing_slicing)]
fn half_sum_digits(
    large: impl Iterator<Item = u8> + Clone,
    small: impl Iterator<Item = u8> + Clone,
    subtract: bool,
    buffer: &mut [u8; HALF_SUM_DIGITS],
) -> &[u8] {
    // Dividing by two is multiplying by five and shifting down one place.
    let mut carry = 0;
    let mut len = 0;
    for (place, digit) in sum_decimal_digits(large, small, subtract)
        .chain(Some(0))
        .enumerate()
    {
        let product = 5 * digit + carry;
        carry = product / 10;
        let digit = product % 10;
        buffer[place] = digit;
        if digit != 0 {
            len = place + 1;
        }
    }
    &buffer[..len]
}
