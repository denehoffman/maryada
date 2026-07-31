use crate::{
    DecoratedInterval, Decoration, Interval, IntervalDatum,
    rounding::{self, Direction},
    signals::{Signal, SignalSink},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextError {
    BufferTooSmall { required: usize },
}

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
    if x.__is_nai() {
        return write_ascii(output, b"[nai]");
    }
    let interval = x.__interval();
    let decoration = x.__decoration();
    let mut cursor = 0;
    cursor += write_bare_literal(interval, &mut output[cursor..])?;
    if let Some(decoration) = decoration {
        cursor += write_ascii(&mut output[cursor..], b"_")?;
        cursor += write_decoration(decoration, &mut output[cursor..])?;
    }
    Ok(cursor)
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
    // Must:
    //
    // 1. recognize every form in 6.6.1 and 6.6.2;
    // 2. compute the tightest binary64 hull;
    // 3. identify accuracy-relaxed bracket forms;
    // 4. reject invalid non-relaxed literals;
    // 5. choose failure for reversed relaxed bounds;
    // 6. preserve whether the Level 1 source was bounded.
    todo!()
}

fn parse_decorated_literal(s: &str) -> Result<ParsedDecorated, ()> {
    // Must:
    //
    // 1. recognize [nai];
    // 2. recognize a bare literal with no suffix;
    // 3. recognize _trv, _def, _dac and _com;
    // 4. reject _ill;
    // 5. reject Empty_def, Empty_dac and Empty_com;
    // 6. reject an originally unbounded interval with _com;
    // 7. apply the decorated-constructor rules from 6.7.5.
    todo!()
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
    // Implement directly from `to_bits()`:
    //
    // - preserve signed zero;
    // - emit exact normal and subnormal values;
    // - emit inf/-inf;
    // - NaN is never valid here.
    todo!()
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
