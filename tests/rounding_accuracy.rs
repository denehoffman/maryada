#![allow(missing_docs)]

use maryada::{EnclosureOps, Interval, IntervalOps, exp, exp2, log};

#[test]
fn addition_and_subtraction_enclose_finite_sums_near_overflow() {
    let small = Interval::from(libm::scalbn(3.0, 970));
    let negative_max = -f64::MAX;
    let large = Interval::from(negative_max);
    // The exact sum is one and a half representable steps toward zero.
    let expected = Interval::new(negative_max.next_up(), negative_max.next_up().next_up());

    for (x, y, result) in [(small, large, expected), (-small, -large, -expected)] {
        assert_eq!(x + y, result);
        assert_eq!(y + x, result);
        assert_eq!(x - (-y), result);
        assert_eq!(y - (-x), result);
    }
}

#[test]
fn addition_and_subtraction_round_outward_past_finite_range() {
    let max = Interval::from(f64::MAX);
    let overflow = Interval::new(f64::MAX, f64::INFINITY);

    // These offsets push the exact sum past MAX with a finite or infinite rounded sum.
    for offset in [f64::from_bits(1), libm::scalbn(1.0, 970)] {
        let small = Interval::from(offset);
        for (x, y, result) in [(max, small, overflow), (-max, -small, -overflow)] {
            assert_eq!(x + y, result);
            assert_eq!(y + x, result);
            assert_eq!(x - (-y), result);
        }
    }
}

#[test]
fn cancellation_recognizes_equal_widths_near_overflow() {
    let x = Interval::new(-f64::MAX, -libm::scalbn(3.0, 970));

    assert_eq!(x.cancel_minus(x), Interval::ZERO);
    assert_eq!(x.cancel_plus(-x), Interval::ZERO);
}

#[test]
fn cancellation_distinguishes_widths_a_half_step_apart() {
    let half_step = libm::scalbn(1.0, -53);
    let x = Interval::new(-half_step, 1.0);
    let y = Interval::new(0.0, 1.0);

    assert_eq!(x.cancel_minus(y), Interval::new(-half_step, 0.0));
    assert_eq!(y.cancel_minus(x), Interval::ENTIRE);
    assert_eq!(x.cancel_plus(-y), Interval::new(-half_step, 0.0));
    assert_eq!(y.cancel_plus(-x), Interval::ENTIRE);
}

#[test]
fn cancellation_compares_widths_beyond_the_finite_range() {
    // Their halved widths round to the same finite value.
    let x = Interval::new(-f64::MAX, f64::MAX.next_down());
    let y = Interval::new(-f64::MAX, f64::MAX.next_down().next_down());
    let finite_width = Interval::new(0.0, f64::MAX);
    let step = libm::scalbn(1.0, 971);

    assert_eq!(x.wid(), f64::INFINITY);
    assert_eq!(y.wid(), f64::INFINITY);
    assert_eq!(x.cancel_minus(x), Interval::ZERO);
    assert_eq!(x.cancel_minus(y), Interval::new(0.0, step));
    assert_eq!(y.cancel_minus(x), Interval::ENTIRE);
    assert_eq!(
        x.cancel_minus(finite_width),
        Interval::new(-f64::MAX, -step)
    );
    assert_eq!(finite_width.cancel_minus(x), Interval::ENTIRE);
}

#[test]
fn exp_and_log_take_one_outward_step_from_libm() {
    let exp_lower = libm::exp(0.5);
    let exp_upper = libm::exp(1.5);
    assert_eq!(
        exp(Interval::new(0.5, 1.5)).bounds(),
        (exp_lower.next_down(), exp_upper.next_up())
    );

    let log_lower = libm::log(0.5);
    let log_upper = libm::log(2.0);
    assert_eq!(
        log(Interval::new(0.5, 2.0)).bounds(),
        (log_lower.next_down(), log_upper.next_up())
    );
}

#[test]
fn exp_and_log_preserve_exact_and_limiting_values() {
    assert_eq!(exp(Interval::from(0.0)).bounds(), (1.0, 1.0));
    assert_eq!(log(Interval::from(1.0)).bounds(), (-0.0, 0.0));
    assert_eq!(
        exp(Interval::new(f64::NEG_INFINITY, 0.0)).bounds(),
        (-0.0, 1.0)
    );
    assert_eq!(
        log(Interval::new(1.0, f64::INFINITY)).bounds(),
        (-0.0, f64::INFINITY)
    );
}

#[test]
fn exp2_uses_the_documented_normal_result_bound() {
    let approximation = libm::exp2(0.5);
    assert!(approximation.is_normal());
    assert_eq!(
        exp2(Interval::from(0.5)).bounds(),
        (approximation.next_down(), approximation.next_up())
    );

    let exact_subnormal = f64::from_bits(1);
    assert_eq!(
        exp2(Interval::from(-1074.0)).bounds(),
        (exact_subnormal, exact_subnormal)
    );
}
