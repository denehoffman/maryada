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

#[derive(Clone, Copy, Debug)]
enum Api {
    Intrinsic,
    Ux,
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

fn assert_interval(actual: AnyInterval, expected: AnyInterval, exact: bool, context: &str) {
    match (actual, expected) {
        (AnyInterval::Bare(actual), AnyInterval::Bare(expected)) => {
            assert!(
                if exact {
                    actual == expected
                } else {
                    (maryada::is_empty(expected) && maryada::is_empty(actual))
                        || maryada::subset(expected, actual)
                },
                "{context}: expected {expected:?}, got {actual:?}"
            );
        }
        (AnyInterval::Decorated(actual), AnyInterval::Decorated(expected)) => {
            let acceptable_bounds = if maryada::is_nai(actual) && maryada::is_nai(expected) {
                true
            } else if exact {
                maryada::interval_part(actual, &mut ()) == maryada::interval_part(expected, &mut ())
            } else {
                (maryada::is_empty(expected) && maryada::is_empty(actual))
                    || maryada::subset(expected, actual)
            };
            assert!(
                acceptable_bounds
                    && if exact {
                        maryada::decoration_part(actual) == maryada::decoration_part(expected)
                    } else {
                        maryada::decoration_part(actual) <= maryada::decoration_part(expected)
                    },
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

macro_rules! unary_method {
    ($value:expr, $method:ident) => {
        match $value {
            AnyInterval::Bare(x) => AnyInterval::Bare(x.$method()),
            AnyInterval::Decorated(x) => AnyInterval::Decorated(x.$method()),
        }
    };
}

macro_rules! binary_method {
    ($left:expr, $right:expr, $method:ident) => {
        match ($left, $right) {
            (AnyInterval::Bare(x), AnyInterval::Bare(y)) => AnyInterval::Bare(x.$method(y)),
            (AnyInterval::Decorated(x), AnyInterval::Decorated(y)) => {
                AnyInterval::Decorated(x.$method(y))
            }
            _ => panic!("mixed bare/decorated operands"),
        }
    };
}

fn apply_unary_intrinsic(op: &str, value: AnyInterval) -> Option<AnyInterval> {
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

fn apply_unary_ux(op: &str, value: AnyInterval) -> Option<AnyInterval> {
    Some(match op {
        "pos" => value,
        "neg" => match value {
            AnyInterval::Bare(x) => AnyInterval::Bare(-x),
            AnyInterval::Decorated(x) => AnyInterval::Decorated(-x),
        },
        "recip" => unary_method!(value, recip),
        "sqr" => unary_method!(value, sqr),
        "sqrt" => unary_method!(value, sqrt),
        "exp" => unary_method!(value, exp),
        "exp2" => unary_method!(value, exp2),
        "exp10" => unary_method!(value, exp10),
        "log" => unary_method!(value, log),
        "log2" => unary_method!(value, log2),
        "log10" => unary_method!(value, log10),
        "sin" => unary_method!(value, sin),
        "cos" => unary_method!(value, cos),
        "tan" => unary_method!(value, tan),
        "asin" => unary_method!(value, asin),
        "acos" => unary_method!(value, acos),
        "atan" => unary_method!(value, atan),
        "sinh" => unary_method!(value, sinh),
        "cosh" => unary_method!(value, cosh),
        "tanh" => unary_method!(value, tanh),
        "asinh" => unary_method!(value, asinh),
        "acosh" => unary_method!(value, acosh),
        "atanh" => unary_method!(value, atanh),
        "sign" => unary_method!(value, sign),
        "ceil" => unary_method!(value, ceil),
        "floor" => unary_method!(value, floor),
        "trunc" => unary_method!(value, trunc),
        "roundTiesToEven" => unary_method!(value, round_ties_to_even),
        "roundTiesToAway" => unary_method!(value, round_ties_to_away),
        "abs" => unary_method!(value, abs),
        _ => return None,
    })
}

fn apply_unary(api: Api, op: &str, value: AnyInterval) -> Option<AnyInterval> {
    match api {
        Api::Intrinsic => apply_unary_intrinsic(op, value),
        Api::Ux => apply_unary_ux(op, value),
    }
}

fn apply_binary_intrinsic(op: &str, left: AnyInterval, right: AnyInterval) -> Option<AnyInterval> {
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

fn apply_binary_ux(op: &str, left: AnyInterval, right: AnyInterval) -> Option<AnyInterval> {
    Some(match op {
        "add" => match (left, right) {
            (AnyInterval::Bare(x), AnyInterval::Bare(y)) => AnyInterval::Bare(x + y),
            (AnyInterval::Decorated(x), AnyInterval::Decorated(y)) => AnyInterval::Decorated(x + y),
            _ => panic!("mixed bare/decorated operands"),
        },
        "sub" => match (left, right) {
            (AnyInterval::Bare(x), AnyInterval::Bare(y)) => AnyInterval::Bare(x - y),
            (AnyInterval::Decorated(x), AnyInterval::Decorated(y)) => AnyInterval::Decorated(x - y),
            _ => panic!("mixed bare/decorated operands"),
        },
        "mul" => match (left, right) {
            (AnyInterval::Bare(x), AnyInterval::Bare(y)) => AnyInterval::Bare(x * y),
            (AnyInterval::Decorated(x), AnyInterval::Decorated(y)) => AnyInterval::Decorated(x * y),
            _ => panic!("mixed bare/decorated operands"),
        },
        "div" => match (left, right) {
            (AnyInterval::Bare(x), AnyInterval::Bare(y)) => AnyInterval::Bare(x / y),
            (AnyInterval::Decorated(x), AnyInterval::Decorated(y)) => AnyInterval::Decorated(x / y),
            _ => panic!("mixed bare/decorated operands"),
        },
        "pow" => binary_method!(left, right, pow),
        "atan2" => binary_method!(left, right, atan2),
        "min" => binary_method!(left, right, min),
        "max" => binary_method!(left, right, max),
        // The cancellation operations have no ergonomic method equivalents.
        "cancelMinus" => binary!(left, right, maryada::cancel_minus),
        "cancelPlus" => binary!(left, right, maryada::cancel_plus),
        "intersection" => binary_method!(left, right, intersection),
        "convexHull" => binary_method!(left, right, convex_hull),
        _ => return None,
    })
}

fn apply_binary(api: Api, op: &str, left: AnyInterval, right: AnyInterval) -> Option<AnyInterval> {
    match api {
        Api::Intrinsic => apply_binary_intrinsic(op, left, right),
        Api::Ux => apply_binary_ux(op, left, right),
    }
}

fn apply_pown(api: Api, value: AnyInterval, exponent: i32) -> AnyInterval {
    match value {
        AnyInterval::Bare(x) => AnyInterval::Bare(match api {
            Api::Intrinsic => maryada::pown(x, exponent),
            Api::Ux => x.pown(exponent),
        }),
        AnyInterval::Decorated(x) => AnyInterval::Decorated(match api {
            Api::Intrinsic => maryada::pown(x, exponent),
            Api::Ux => x.pown(exponent),
        }),
    }
}

fn apply_fma(api: Api, x: AnyInterval, y: AnyInterval, z: AnyInterval) -> AnyInterval {
    match (x, y, z) {
        (AnyInterval::Bare(x), AnyInterval::Bare(y), AnyInterval::Bare(z)) => {
            AnyInterval::Bare(match api {
                Api::Intrinsic => maryada::fma(x, y, z),
                Api::Ux => x.mul_add(y, z),
            })
        }
        (AnyInterval::Decorated(x), AnyInterval::Decorated(y), AnyInterval::Decorated(z)) => {
            AnyInterval::Decorated(match api {
                Api::Intrinsic => maryada::fma(x, y, z),
                Api::Ux => x.mul_add(y, z),
            })
        }
        _ => panic!("mixed bare/decorated operands"),
    }
}

fn boolean_unary_intrinsic(op: &str, value: AnyInterval) -> Option<bool> {
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

fn boolean_unary_ux(op: &str, value: AnyInterval) -> Option<bool> {
    Some(match (op, value) {
        ("isEmpty", AnyInterval::Bare(x)) => x.is_empty(),
        ("isEmpty", AnyInterval::Decorated(x)) => x.is_empty(),
        ("isEntire", AnyInterval::Bare(x)) => x.is_entire(),
        ("isEntire", AnyInterval::Decorated(x)) => x.is_entire(),
        ("isNaI", AnyInterval::Decorated(x)) => x.is_nai(),
        ("isSingleton", AnyInterval::Bare(x)) => x.is_singleton(),
        ("isSingleton", AnyInterval::Decorated(x)) => x.is_singleton(),
        ("isCommonInterval", AnyInterval::Bare(x)) => !x.is_empty() && x.is_bounded(),
        ("isCommonInterval", AnyInterval::Decorated(x)) => {
            !x.is_nai() && !x.is_empty() && x.is_bounded()
        }
        _ => return None,
    })
}

fn boolean_unary(api: Api, op: &str, value: AnyInterval) -> Option<bool> {
    match api {
        Api::Intrinsic => boolean_unary_intrinsic(op, value),
        Api::Ux => boolean_unary_ux(op, value),
    }
}

fn boolean_binary_intrinsic(op: &str, left: AnyInterval, right: AnyInterval) -> Option<bool> {
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

fn boolean_binary_ux(op: &str, left: AnyInterval, right: AnyInterval) -> Option<bool> {
    Some(match (op, left, right) {
        // ITF equality ignores decorations, unlike Rust's `PartialEq` implementation.
        ("equal", AnyInterval::Bare(x), AnyInterval::Bare(y)) => maryada::equal(x, y),
        ("equal", AnyInterval::Decorated(x), AnyInterval::Decorated(y)) => maryada::equal(x, y),
        ("subset", AnyInterval::Bare(x), AnyInterval::Bare(y)) => x.subset(y),
        ("subset", AnyInterval::Decorated(x), AnyInterval::Decorated(y)) => x.subset(y),
        ("interior", AnyInterval::Bare(x), AnyInterval::Bare(y)) => x.interior(y),
        ("interior", AnyInterval::Decorated(x), AnyInterval::Decorated(y)) => x.interior(y),
        ("disjoint", AnyInterval::Bare(x), AnyInterval::Bare(y)) => x.disjoint(y),
        ("disjoint", AnyInterval::Decorated(x), AnyInterval::Decorated(y)) => x.disjoint(y),
        ("equal" | "subset" | "interior" | "disjoint", _, _) => {
            panic!("mixed bare/decorated operands")
        }
        _ => return None,
    })
}

fn boolean_binary(api: Api, op: &str, left: AnyInterval, right: AnyInterval) -> Option<bool> {
    match api {
        Api::Intrinsic => boolean_binary_intrinsic(op, left, right),
        Api::Ux => boolean_binary_ux(op, left, right),
    }
}

fn numeric_unary_intrinsic(op: &str, value: AnyInterval) -> Option<f64> {
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

fn numeric_unary_ux(op: &str, value: AnyInterval) -> Option<f64> {
    Some(match (op, value) {
        ("inf", AnyInterval::Bare(x)) => x.inf(),
        ("inf", AnyInterval::Decorated(x)) => x.inf(),
        ("sup", AnyInterval::Bare(x)) => x.sup(),
        ("sup", AnyInterval::Decorated(x)) => x.sup(),
        ("mid", AnyInterval::Bare(x)) => x.mid(),
        ("mid", AnyInterval::Decorated(x)) => x.mid(),
        ("rad", AnyInterval::Bare(x)) => x.rad(),
        ("rad", AnyInterval::Decorated(x)) => x.rad(),
        ("wid", AnyInterval::Bare(x)) => x.wid(),
        ("wid", AnyInterval::Decorated(x)) => x.wid(),
        ("mag", AnyInterval::Bare(x)) => x.mag(),
        ("mag", AnyInterval::Decorated(x)) => x.mag(),
        ("mig", AnyInterval::Bare(x)) => x.mig(),
        ("mig", AnyInterval::Decorated(x)) => x.mig(),
        _ => return None,
    })
}

fn numeric_unary(api: Api, op: &str, value: AnyInterval) -> Option<f64> {
    match api {
        Api::Intrinsic => numeric_unary_intrinsic(op, value),
        Api::Ux => numeric_unary_ux(op, value),
    }
}

fn parse_bool(text: &str) -> bool {
    match text.trim() {
        "true" => true,
        "false" => false,
        other => panic!("invalid boolean {other:?}"),
    }
}

fn run_statement(api: Api, statement: &str) -> bool {
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
        let Some(actual) = boolean_unary(api, op, value) else {
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
        let Some(actual) = boolean_binary(api, op, left, right) else {
            return false;
        };
        assert_eq!(actual, parse_bool(expected_text), "{statement}");
        return true;
    }

    if matches!(op, "inf" | "sup" | "mid" | "rad" | "wid" | "mag" | "mig") {
        let (value, rest) = parse_itl_interval_operand(args);
        assert!(rest.trim().is_empty(), "{statement}: unexpected arguments");
        let actual = numeric_unary(api, op, value).unwrap();
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
        apply_pown(
            api,
            first,
            rest.trim().parse().expect("integer pown exponent"),
        )
    } else if op == "fma" {
        let (second, rest) = parse_itl_interval_operand(rest);
        let (third, rest) = parse_itl_interval_operand(rest);
        assert!(rest.trim().is_empty(), "{statement}: unexpected arguments");
        apply_fma(api, first, second, third)
    } else if let Some(result) = apply_unary(api, op, first) {
        assert!(rest.trim().is_empty(), "{statement}: unexpected arguments");
        result
    } else {
        let (second, rest) = parse_itl_interval_operand(rest);
        assert!(rest.trim().is_empty(), "{statement}: unexpected arguments");
        let Some(result) = apply_binary(api, op, first, second) else {
            return false;
        };
        result
    };
    let (expected, rest) = parse_itl_interval_operand(expected_text);
    assert!(
        rest.trim().is_empty(),
        "{statement}: unexpected expected-result suffix"
    );
    let tightest_required = matches!(
        op,
        "neg"
            | "add"
            | "sub"
            | "mul"
            | "div"
            | "recip"
            | "sqr"
            | "sqrt"
            | "fma"
            | "sign"
            | "ceil"
            | "floor"
            | "trunc"
            | "roundTiesToEven"
            | "roundTiesToAway"
            | "abs"
            | "min"
            | "max"
    );
    assert_interval(actual, expected, tightest_required, statement);
    true
}

fn run_itl(api: Api, source: &str) -> usize {
    let mut in_comment = false;
    let mut executed = 0;
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("/*") {
            in_comment = true;
        }
        if !in_comment && trimmed.ends_with(';') && run_statement(api, trimmed) {
            executed += 1;
        }
        if trimmed.ends_with("*/") {
            in_comment = false;
        }
    }
    executed
}

const ITF1788_SOURCES: [&str; 7] = [
    include_str!("itf1788/atan2.itl"),
    include_str!("itf1788/libieeep1788_bool.itl"),
    include_str!("itf1788/libieeep1788_cancel.itl"),
    include_str!("itf1788/libieeep1788_elem.itl"),
    include_str!("itf1788/libieeep1788_num.itl"),
    include_str!("itf1788/libieeep1788_rec_bool.itl"),
    include_str!("itf1788/libieeep1788_set.itl"),
];

fn run_port(api: Api) {
    let executed: usize = ITF1788_SOURCES
        .into_iter()
        .map(|source| run_itl(api, source))
        .sum();
    assert_eq!(
        executed, 4_642,
        "unexpected number of ITF1788 cases for {api:?} API"
    );
}

#[test]
fn interval_arithmetic_intrinsic_itf1788_port() {
    run_port(Api::Intrinsic);
}

#[test]
fn interval_arithmetic_ux_itf1788_port() {
    run_port(Api::Ux);
}
