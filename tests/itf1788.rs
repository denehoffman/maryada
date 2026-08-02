// Data-driven port of compatible tests shipped by IntervalArithmetic.jl.
//
// The test vectors in tests/itf1788 retain their original Apache-2.0 notices.
// This harness is a Rust adaptation and intentionally exercises only operations
// that are part of maryada's current public API.

use maryada::{DecoratedInterval, Decoration, Interval};

#[derive(Clone, Copy, Debug)]
enum AnyInterval {
    Bare(Interval),
    Decorated(DecoratedInterval),
}

// ITL numeric tokens denote binary64 operands. This differs from IEEE
// textToInterval: decimal text such as 17.1 is a rounded binary64 endpoint in
// ITL, while "[17.1]" asks textToInterval to enclose the exact decimal value.
fn parse_itl_binary64(text: &str) -> f64 {
    let text = text.trim();
    match text.to_ascii_lowercase().as_str() {
        "infinity" | "+infinity" => f64::INFINITY,
        "-infinity" => f64::NEG_INFINITY,
        "nan" | "+nan" | "-nan" => f64::NAN,
        _ if text.to_ascii_lowercase().contains("0x") => {
            let interval: Interval = format!("[{text}]")
                .parse()
                .unwrap_or_else(|_| panic!("invalid hexadecimal number {text:?}"));
            assert!(
                interval.is_singleton(),
                "non-singleton hexadecimal number {text:?}"
            );
            interval.inf()
        }
        _ => text
            .parse()
            .unwrap_or_else(|_| panic!("invalid number {text:?}")),
    }
}

fn parse_itl_decoration(text: &str) -> Decoration {
    text.parse()
        .unwrap_or_else(|_| panic!("unknown ITL decoration {text:?}"))
}

fn parse_itl_interval_operand(text: &str) -> (AnyInterval, &str) {
    let text = text.trim_start();
    assert!(text.starts_with('['), "expected interval in {text:?}");
    let close = text.find(']').expect("closing interval bracket");
    let body = text[1..close].trim();
    let mut rest = &text[close + 1..];
    let decoration = if let Some(suffix) = rest.strip_prefix('_') {
        let end = suffix
            .find(|c: char| !c.is_ascii_alphabetic())
            .unwrap_or(suffix.len());
        let decoration = parse_itl_decoration(&suffix[..end]);
        rest = &suffix[end..];
        Some(decoration)
    } else {
        None
    };

    if body.eq_ignore_ascii_case("nai") {
        return (AnyInterval::Decorated(DecoratedInterval::NAI), rest);
    }

    let bare = if body.eq_ignore_ascii_case("empty") {
        Interval::EMPTY
    } else if body.eq_ignore_ascii_case("entire") {
        Interval::ENTIRE
    } else if let Some((inf, sup)) = body.split_once(',') {
        Interval::nums_to_interval(parse_itl_binary64(inf), parse_itl_binary64(sup), &mut ())
    } else {
        Interval::nums_to_interval(parse_itl_binary64(body), parse_itl_binary64(body), &mut ())
    };

    match decoration {
        Some(decoration) => (
            AnyInterval::Decorated(maryada::set_dec(bare, decoration)),
            rest,
        ),
        None => (AnyInterval::Bare(bare), rest),
    }
}

fn same_number(actual: f64, expected: f64) -> bool {
    (actual.is_nan() && expected.is_nan()) || actual.to_bits() == expected.to_bits()
}

fn assert_interval(actual: AnyInterval, expected: AnyInterval, context: &str) {
    match (actual, expected) {
        (AnyInterval::Bare(actual), AnyInterval::Bare(expected)) => {
            assert!(
                (maryada::is_empty(expected) && maryada::is_empty(actual))
                    || maryada::subset(expected, actual),
                "{context}: expected {expected:?}, got {actual:?}"
            );
        }
        (AnyInterval::Decorated(actual), AnyInterval::Decorated(expected)) => {
            let same_bounds = if maryada::is_nai(actual) && maryada::is_nai(expected) {
                true
            } else {
                (maryada::is_empty(expected) && maryada::is_empty(actual))
                    || maryada::subset(expected, actual)
            };
            assert!(
                same_bounds
                    && maryada::decoration_part(actual) <= maryada::decoration_part(expected),
                "{context}: expected {expected:?}, got {actual:?}"
            );
        }
        _ => panic!("{context}: bare/decorated result mismatch"),
    }
}

macro_rules! unary {
    ($value:expr, $function:path) => {
        match $value {
            AnyInterval::Bare(x) => AnyInterval::Bare($function(x)),
            AnyInterval::Decorated(x) => AnyInterval::Decorated($function(x)),
        }
    };
}

macro_rules! binary {
    ($left:expr, $right:expr, $function:path) => {
        match ($left, $right) {
            (AnyInterval::Bare(x), AnyInterval::Bare(y)) => AnyInterval::Bare($function(x, y)),
            (AnyInterval::Decorated(x), AnyInterval::Decorated(y)) => {
                AnyInterval::Decorated($function(x, y))
            }
            _ => panic!("mixed bare/decorated operands"),
        }
    };
}

fn apply_unary(op: &str, value: AnyInterval) -> Option<AnyInterval> {
    Some(match op {
        "pos" => value,
        "neg" => unary!(value, maryada::neg),
        "recip" => unary!(value, maryada::recip),
        "sqr" => unary!(value, maryada::sqr),
        "sqrt" => unary!(value, maryada::sqrt),
        "exp" => unary!(value, maryada::exp),
        "exp2" => unary!(value, maryada::exp2),
        "exp10" => unary!(value, maryada::exp10),
        "log" => unary!(value, maryada::log),
        "log2" => unary!(value, maryada::log2),
        "log10" => unary!(value, maryada::log10),
        "sin" => unary!(value, maryada::sin),
        "cos" => unary!(value, maryada::cos),
        "tan" => unary!(value, maryada::tan),
        "asin" => unary!(value, maryada::asin),
        "acos" => unary!(value, maryada::acos),
        "atan" => unary!(value, maryada::atan),
        "sinh" => unary!(value, maryada::sinh),
        "cosh" => unary!(value, maryada::cosh),
        "tanh" => unary!(value, maryada::tanh),
        "asinh" => unary!(value, maryada::asinh),
        "acosh" => unary!(value, maryada::acosh),
        "atanh" => unary!(value, maryada::atanh),
        "sign" => unary!(value, maryada::sign),
        "ceil" => unary!(value, maryada::ceil),
        "floor" => unary!(value, maryada::floor),
        "trunc" => unary!(value, maryada::trunc),
        "roundTiesToEven" => unary!(value, maryada::round_ties_to_even),
        "roundTiesToAway" => unary!(value, maryada::round_ties_to_away),
        "abs" => unary!(value, maryada::abs),
        _ => return None,
    })
}

fn apply_binary(op: &str, left: AnyInterval, right: AnyInterval) -> Option<AnyInterval> {
    Some(match op {
        "add" => binary!(left, right, maryada::add),
        "sub" => binary!(left, right, maryada::sub),
        "mul" => binary!(left, right, maryada::mul),
        "div" => binary!(left, right, maryada::div),
        "pow" => binary!(left, right, maryada::pow),
        "atan2" => binary!(left, right, maryada::atan2),
        "min" => binary!(left, right, maryada::min),
        "max" => binary!(left, right, maryada::max),
        "cancelMinus" => binary!(left, right, maryada::cancel_minus),
        "cancelPlus" => binary!(left, right, maryada::cancel_plus),
        "intersection" => binary!(left, right, maryada::intersection),
        "convexHull" => binary!(left, right, maryada::convex_hull),
        _ => return None,
    })
}

fn apply_pown(value: AnyInterval, exponent: i32) -> AnyInterval {
    match value {
        AnyInterval::Bare(x) => AnyInterval::Bare(maryada::pown(x, exponent)),
        AnyInterval::Decorated(x) => AnyInterval::Decorated(maryada::pown(x, exponent)),
    }
}

fn apply_fma(x: AnyInterval, y: AnyInterval, z: AnyInterval) -> AnyInterval {
    match (x, y, z) {
        (AnyInterval::Bare(x), AnyInterval::Bare(y), AnyInterval::Bare(z)) => {
            AnyInterval::Bare(maryada::fma(x, y, z))
        }
        (AnyInterval::Decorated(x), AnyInterval::Decorated(y), AnyInterval::Decorated(z)) => {
            AnyInterval::Decorated(maryada::fma(x, y, z))
        }
        _ => panic!("mixed bare/decorated operands"),
    }
}

fn boolean_unary(op: &str, value: AnyInterval) -> Option<bool> {
    Some(match (op, value) {
        ("isEmpty", AnyInterval::Bare(x)) => maryada::is_empty(x),
        ("isEmpty", AnyInterval::Decorated(x)) => maryada::is_empty(x),
        ("isEntire", AnyInterval::Bare(x)) => maryada::is_entire(x),
        ("isEntire", AnyInterval::Decorated(x)) => maryada::is_entire(x),
        ("isNaI", AnyInterval::Decorated(x)) => maryada::is_nai(x),
        ("isSingleton", AnyInterval::Bare(x)) => x.is_singleton(),
        ("isSingleton", AnyInterval::Decorated(x)) => x.is_singleton(),
        ("isCommonInterval", AnyInterval::Bare(x)) => !x.is_empty() && x.is_bounded(),
        ("isCommonInterval", AnyInterval::Decorated(x)) => {
            !maryada::is_nai(x) && !x.is_empty() && x.is_bounded()
        }
        _ => return None,
    })
}

fn boolean_binary(op: &str, left: AnyInterval, right: AnyInterval) -> Option<bool> {
    macro_rules! call {
        ($function:path) => {
            match (left, right) {
                (AnyInterval::Bare(x), AnyInterval::Bare(y)) => $function(x, y),
                (AnyInterval::Decorated(x), AnyInterval::Decorated(y)) => $function(x, y),
                _ => panic!("mixed bare/decorated operands"),
            }
        };
    }
    Some(match op {
        "equal" => call!(maryada::equal),
        "subset" => call!(maryada::subset),
        "interior" => call!(maryada::interior),
        "disjoint" => call!(maryada::disjoint),
        _ => return None,
    })
}

fn numeric_unary(op: &str, value: AnyInterval) -> Option<f64> {
    macro_rules! call {
        ($function:path) => {
            match value {
                AnyInterval::Bare(x) => $function(x),
                AnyInterval::Decorated(x) => $function(x),
            }
        };
    }
    Some(match op {
        "inf" => call!(maryada::inf),
        "sup" => call!(maryada::sup),
        "mid" => call!(maryada::mid),
        "rad" => call!(maryada::rad),
        "wid" => call!(maryada::wid),
        "mag" => call!(maryada::mag),
        "mig" => call!(maryada::mig),
        _ => return None,
    })
}

fn parse_bool(text: &str) -> bool {
    match text.trim() {
        "true" => true,
        "false" => false,
        other => panic!("invalid boolean {other:?}"),
    }
}

fn run_statement(statement: &str) -> bool {
    let statement = statement.trim().trim_end_matches(';').trim();
    let Some((left, expected_text)) = statement.split_once(" = ") else {
        return false;
    };
    let mut words = left.splitn(2, char::is_whitespace);
    let op = words.next().unwrap();
    let args = words.next().unwrap_or("").trim();

    if matches!(
        op,
        "isEmpty" | "isEntire" | "isNaI" | "isSingleton" | "isCommonInterval"
    ) {
        let (value, rest) = parse_itl_interval_operand(args);
        assert!(rest.trim().is_empty(), "{statement}: unexpected arguments");
        let Some(actual) = boolean_unary(op, value) else {
            return false;
        };
        assert_eq!(actual, parse_bool(expected_text), "{statement}");
        return true;
    }

    if op == "isMember" {
        let split = args
            .find(char::is_whitespace)
            .expect("number followed by interval");
        let value = parse_itl_binary64(&args[..split]);
        let (interval, rest) = parse_itl_interval_operand(&args[split..]);
        assert!(rest.trim().is_empty(), "{statement}: unexpected arguments");
        let actual = match interval {
            AnyInterval::Bare(interval) => interval.contains(value),
            AnyInterval::Decorated(interval) => interval.contains(value),
        };
        assert_eq!(actual, parse_bool(expected_text), "{statement}");
        return true;
    }

    if matches!(op, "equal" | "subset" | "interior" | "disjoint") {
        let (left, rest) = parse_itl_interval_operand(args);
        let (right, rest) = parse_itl_interval_operand(rest);
        assert!(rest.trim().is_empty(), "{statement}: unexpected arguments");
        let Some(actual) = boolean_binary(op, left, right) else {
            return false;
        };
        assert_eq!(actual, parse_bool(expected_text), "{statement}");
        return true;
    }

    if matches!(op, "inf" | "sup" | "mid" | "rad" | "wid" | "mag" | "mig") {
        let (value, rest) = parse_itl_interval_operand(args);
        assert!(rest.trim().is_empty(), "{statement}: unexpected arguments");
        let actual = numeric_unary(op, value).unwrap();
        let expected = parse_itl_binary64(expected_text);
        assert!(
            same_number(actual, expected),
            "{statement}: expected {expected:?}, got {actual:?}"
        );
        return true;
    }

    if !matches!(
        op,
        "pos"
            | "neg"
            | "recip"
            | "sqr"
            | "sqrt"
            | "exp"
            | "exp2"
            | "exp10"
            | "log"
            | "log2"
            | "log10"
            | "sin"
            | "cos"
            | "tan"
            | "asin"
            | "acos"
            | "atan"
            | "sinh"
            | "cosh"
            | "tanh"
            | "asinh"
            | "acosh"
            | "atanh"
            | "sign"
            | "ceil"
            | "floor"
            | "trunc"
            | "roundTiesToEven"
            | "roundTiesToAway"
            | "abs"
            | "add"
            | "sub"
            | "mul"
            | "div"
            | "pow"
            | "atan2"
            | "min"
            | "max"
            | "cancelMinus"
            | "cancelPlus"
            | "intersection"
            | "convexHull"
            | "pown"
            | "fma"
    ) {
        return false;
    }

    let (first, rest) = parse_itl_interval_operand(args);
    let actual = if op == "pown" {
        apply_pown(first, rest.trim().parse().expect("integer pown exponent"))
    } else if op == "fma" {
        let (second, rest) = parse_itl_interval_operand(rest);
        let (third, rest) = parse_itl_interval_operand(rest);
        assert!(rest.trim().is_empty(), "{statement}: unexpected arguments");
        apply_fma(first, second, third)
    } else if let Some(result) = apply_unary(op, first) {
        assert!(rest.trim().is_empty(), "{statement}: unexpected arguments");
        result
    } else {
        let (second, rest) = parse_itl_interval_operand(rest);
        assert!(rest.trim().is_empty(), "{statement}: unexpected arguments");
        let Some(result) = apply_binary(op, first, second) else {
            return false;
        };
        result
    };
    let (expected, rest) = parse_itl_interval_operand(expected_text);
    assert!(
        rest.trim().is_empty(),
        "{statement}: unexpected expected-result suffix"
    );
    assert_interval(actual, expected, statement);
    true
}

fn run_itl(source: &str) -> usize {
    let mut in_comment = false;
    let mut executed = 0;
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("/*") {
            in_comment = true;
        }
        if !in_comment && trimmed.ends_with(';') && run_statement(trimmed) {
            executed += 1;
        }
        if trimmed.ends_with("*/") {
            in_comment = false;
        }
    }
    executed
}

#[test]
fn interval_arithmetic_itf1788_port() {
    let sources = [
        include_str!("itf1788/atan2.itl"),
        include_str!("itf1788/libieeep1788_bool.itl"),
        include_str!("itf1788/libieeep1788_cancel.itl"),
        include_str!("itf1788/libieeep1788_elem.itl"),
        include_str!("itf1788/libieeep1788_num.itl"),
        include_str!("itf1788/libieeep1788_rec_bool.itl"),
        include_str!("itf1788/libieeep1788_set.itl"),
    ];
    let executed: usize = sources.into_iter().map(run_itl).sum();
    assert_eq!(executed, 4_642, "unexpected number of ITF1788 cases");
}
