#![allow(missing_docs)]

use maryada::{Interval, IntervalOps, exp, exp2, log};

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
