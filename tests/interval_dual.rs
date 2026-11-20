#![allow(missing_docs, clippy::arithmetic_side_effects)]

use maryada::{EnclosureOps, Interval, IntervalDual, IntervalOps};

#[test]
fn seeds_and_arithmetic_enclose_value_and_derivative_together() {
    let x = IntervalDual::variable(Interval::new(1.0, 2.0));
    let parameter = IntervalDual::constant(Interval::new(-2.0, -1.0));
    assert_eq!(x.derivative, Some(Interval::ONE));
    assert_eq!(parameter.derivative, Some(Interval::ZERO));

    let polynomial = x.sqr() + parameter * x;
    assert_eq!(polynomial.value, Interval::new(-3.0, 3.0));
    assert_eq!(polynomial.derivative, Some(Interval::new(0.0, 3.0)));
    assert_eq!((-x).derivative, Some(Interval::from(-1.0)));
    assert_eq!((3.0 - x).derivative, Some(Interval::from(-1.0)));
    assert_eq!((x - x).derivative, Some(Interval::ZERO));
    assert_eq!((3.0 + x).derivative, Some(Interval::ONE));
    assert_eq!((x * 3.0).derivative, Some(Interval::from(3.0)));
    assert_eq!((x / 2.0).derivative, Some(Interval::from(0.5)));
    assert_eq!((2.0 / x).derivative, Some(Interval::new(-2.0, -0.5)));
}

#[test]
fn integer_powers_and_reciprocals_cover_extreme_exponents() {
    let x = IntervalDual::variable(Interval::new(-1.0, 2.0));
    assert_eq!(x.pown(3).value, Interval::new(-1.0, 8.0));
    assert_eq!(x.pown(3).derivative, Some(Interval::new(0.0, 12.0)));
    assert_eq!(x.pown(0), IntervalDual::constant(Interval::ONE));

    let x = IntervalDual::variable(Interval::ONE);
    assert_eq!(x.pown(i32::MIN).value, Interval::ONE);
    assert_eq!(
        x.pown(i32::MIN).derivative,
        Some(Interval::from(f64::from(i32::MIN)))
    );

    let x = IntervalDual::variable(Interval::new(2.0, 4.0));
    assert_eq!(x.pown(-2).derivative, Some(Interval::new(-0.25, -0.03125)));
    assert_eq!(x.recip(), 1.0 / x);

    let large = IntervalDual::variable(Interval::from(libm::scalbn(1.0, 600)));
    let expected = Some(Interval::new(-f64::from_bits(1), 0.0));
    assert_eq!(large.recip().derivative, expected);
    assert_eq!((1.0 / large).derivative, expected);
}

#[test]
fn elementary_functions_apply_the_chain_rule() {
    let zero = IntervalDual::variable(Interval::ZERO);
    assert_eq!(zero.exp().value, Interval::ONE);
    assert_eq!(zero.exp().derivative, Some(Interval::ONE));
    assert_eq!(zero.sin(), zero);
    assert_eq!(zero.cos(), IntervalDual::constant(Interval::ONE));
    assert_eq!(zero.tan(), zero);

    let x = IntervalDual::variable(Interval::new(1.0, 4.0));
    assert_eq!(x.sqrt().value, Interval::new(1.0, 2.0));
    assert_eq!(x.sqrt().derivative, Some(Interval::new(0.25, 0.5)));
    assert_eq!(x.log().derivative, Some(Interval::new(0.25, 1.0)));
    let composition = IntervalDual::variable(Interval::from(4.0)).sqrt().log();
    assert_eq!(composition.derivative, Some(Interval::from(0.125)));
    assert!(composition.value.contains(2.0_f64.ln()));

    let point = IntervalDual::variable(Interval::from(2.0));
    assert_eq!(
        (point.sqr() + 5.0).sqrt().derivative,
        Some(Interval::ONE / Interval::from(1.5))
    );
    assert_eq!((-point).abs(), point);
}

#[test]
fn partial_domains_and_corners_keep_values_without_claiming_a_derivative() {
    let crossing = IntervalDual::variable(Interval::new(-1.0, 4.0));
    assert_eq!(crossing.sqrt().value, Interval::new(0.0, 2.0));
    assert_eq!(crossing.abs().value, Interval::new(0.0, 4.0));
    assert_eq!(crossing.log().value, crossing.value.log());
    for result in [
        crossing.sqrt(),
        crossing.abs(),
        crossing.log(),
        crossing.recip(),
        crossing.pown(-2),
        1.0 / crossing,
        0.0 * crossing.log(),
        crossing.log().pown(0),
    ] {
        assert!(!result.value.is_empty());
        assert_eq!(result.derivative, None);
    }
    let touches_zero = IntervalDual::variable(Interval::new(0.0, 4.0));
    assert_eq!(touches_zero.sqrt().derivative, None);
    assert_eq!(touches_zero.abs().derivative, None);

    let pole = IntervalDual::variable(Interval::new(1.0, 2.0)).tan();
    assert!(pole.value.is_entire());
    assert_eq!(pole.derivative, None);

    let empty = IntervalDual::variable(Interval::EMPTY);
    assert_eq!(empty.derivative, None);
    assert_eq!(empty.exp(), empty);
    assert_eq!(crossing + empty, empty);
}
