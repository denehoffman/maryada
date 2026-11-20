#![allow(missing_docs)]

use maryada::{
    DecoratedInterval, Decoration, EnclosureOps, Interval, IntervalOps, asinh, atanh,
    decoration_part, exp, exp2, expm1, log, log1p, set_dec, subset, tanh,
};

#[test]
fn trig_encloses_near_the_series_limit_and_at_the_smallest_subnormal() {
    let tiny = f64::from_bits(1);
    for (point, sine, cosine, tangent) in [
        // Directed binary64 reference bounds from MPFR.
        (
            0.78,
            Interval::new(
                f64::from_bits(0x3fe6_8143_d72d_4ce4),
                f64::from_bits(0x3fe6_8143_d72d_4ce5),
            ),
            Interval::new(
                f64::from_bits(0x3fe6_bfcd_bf81_7bfa),
                f64::from_bits(0x3fe6_bfcd_bf81_7bfb),
            ),
            Interval::new(
                f64::from_bits(0x3fef_a807_cf82_6c44),
                f64::from_bits(0x3fef_a807_cf82_6c45),
            ),
        ),
        (
            tiny,
            Interval::new(0.0, tiny),
            Interval::new(1.0_f64.next_down(), 1.0),
            Interval::new(tiny, tiny.next_up()),
        ),
    ] {
        let x = Interval::from(point);
        assert!(subset(sine, x.sin()));
        assert!(subset(-sine, (-x).sin()));
        assert!(subset(cosine, x.cos()));
        assert!(subset(cosine, (-x).cos()));
        assert!(subset(tangent, x.tan()));
        assert!(subset(-tangent, (-x).tan()));
    }
}

#[test]
fn hypot_keeps_exact_norms_across_binary64_exponents() {
    for exponent in [-1074, -1024, -665, 0, 664, 1021] {
        let x = Interval::from(libm::scalbn(3.0, exponent));
        let y = Interval::from(libm::scalbn(4.0, exponent));
        let norm = Interval::from(libm::scalbn(5.0, exponent));
        assert_eq!(x.hypot(y), norm);
        assert_eq!((-y).hypot(x), norm);
    }

    for magnitude in [1e-200, 1e200, f64::MAX] {
        let x = Interval::from(magnitude);
        assert_eq!(x.hypot(Interval::ZERO), x);
        assert_eq!(Interval::ZERO.hypot(-x), x);
    }
}

#[test]
fn hypot_rounds_subnormals_and_tiny_contributions_outward() {
    let tiny = f64::from_bits(1);
    let x = Interval::from(tiny);
    let largest = Interval::from(f64::MAX);
    assert_eq!(x.hypot(x), Interval::new(tiny, tiny.next_up()));
    assert_eq!(
        Interval::ONE.hypot(x),
        Interval::new(1.0, 1.0_f64.next_up())
    );
    assert_eq!(largest.hypot(x), Interval::new(f64::MAX, f64::INFINITY));
    assert_eq!(
        largest.hypot(largest),
        Interval::new(f64::MAX, f64::INFINITY)
    );
}

#[test]
fn hypot_uses_magnitude_extrema_and_preserves_interval_decorations() {
    assert_eq!(
        Interval::new(-3.0, 1.0).hypot(Interval::from(4.0)),
        Interval::new(4.0, 5.0)
    );
    assert_eq!(
        Interval::new(f64::NEG_INFINITY, -3.0).hypot(Interval::new(4.0, f64::INFINITY)),
        Interval::new(5.0, f64::INFINITY)
    );
    let x = DecoratedInterval::from(1e200);
    assert_eq!(x.hypot(DecoratedInterval::ZERO), x);
    assert_eq!(x.hypot(DecoratedInterval::EMPTY), DecoratedInterval::EMPTY);
    assert!(
        DecoratedInterval::NAI
            .hypot(DecoratedInterval::EMPTY)
            .is_nai()
    );
    let y = DecoratedInterval::from(f64::MAX);
    assert_eq!(decoration_part(y.hypot(y)), Decoration::Dac);
}

#[test]
fn hyperbolic_enclosures_keep_binary64_spacing_at_tiny_inputs() {
    let limit = f64::from_bits(0x3e40_0000_0000_0000);
    for value in [f64::from_bits(1), f64::MIN_POSITIVE, limit] {
        let x = Interval::from(value);
        let lower = Interval::new(value.next_down(), value);
        let upper = Interval::new(value, value.next_up());
        assert_eq!(tanh(x), lower);
        assert_eq!(asinh(x), lower);
        assert_eq!(atanh(x), upper);
        assert_eq!(tanh(-x), -lower);
        assert_eq!(asinh(-x), -lower);
        assert_eq!(atanh(-x), -upper);
    }
    assert_eq!(tanh(Interval::ZERO), Interval::ZERO);
    assert_eq!(asinh(Interval::ZERO), Interval::ZERO);
    assert_eq!(atanh(Interval::ZERO), Interval::ZERO);
}

#[test]
fn tanh_stays_tight_even_when_sinh_and_cosh_would_overflow() {
    let saturated = Interval::new(1.0_f64.next_down(), 1.0);
    for magnitude in [20.0, 1000.0, f64::MAX] {
        let point = Interval::from(magnitude);
        assert_eq!(tanh(point), saturated);
        assert_eq!(tanh(-point), -saturated);
    }
    assert_eq!(tanh(Interval::new(20.0, f64::INFINITY)), saturated);
    assert_eq!(tanh(Interval::ENTIRE), Interval::new(-1.0, 1.0));
    let at_pole = atanh(DecoratedInterval::new(0.0, 1.0));
    assert_eq!(
        at_pole,
        set_dec(Interval::new(0.0, f64::INFINITY), Decoration::Trv)
    );
    assert_eq!(atanh(Interval::ONE), Interval::EMPTY);
}

#[test]
fn log1p_and_expm1_keep_tight_enclosures_for_tiny_increments() {
    let tiny = f64::from_bits(1);
    let limit = f64::from_bits(0x3c90_0000_0000_0000);
    for x in [
        -limit,
        -f64::MIN_POSITIVE,
        -tiny,
        tiny,
        f64::MIN_POSITIVE,
        limit,
    ] {
        assert_eq!(expm1(Interval::from(x)), Interval::new(x, x.next_up()));
        assert_eq!(log1p(Interval::from(x)), Interval::new(x.next_down(), x));
    }
    assert_eq!(expm1(Interval::ZERO), Interval::ZERO);
    assert_eq!(log1p(Interval::ZERO), Interval::ZERO);
    assert_eq!(
        log1p(Interval::new(-tiny, tiny)),
        Interval::new((-tiny).next_down(), tiny)
    );
    assert_eq!(
        expm1(Interval::new(-tiny, tiny)),
        Interval::new(-tiny, tiny.next_up())
    );
}

#[test]
fn log1p_and_expm1_preserve_domain_limits_and_decorations() {
    assert!(subset(Interval::LN_2, log1p(Interval::ONE)));
    assert_eq!(log1p(Interval::new(-2.0, -1.0)), Interval::EMPTY);
    assert_eq!(log1p(Interval::ENTIRE), Interval::ENTIRE);
    assert_eq!(log1p(Interval::EMPTY), Interval::EMPTY);
    assert_eq!(expm1(Interval::EMPTY), Interval::EMPTY);
    assert_eq!(expm1(Interval::ENTIRE), Interval::new(-1.0, f64::INFINITY));
    assert_eq!(
        expm1(Interval::from(-f64::MAX)),
        Interval::new(-1.0, (-1.0_f64).next_up())
    );

    let partial = log1p(DecoratedInterval::new(-1.0, 0.0));
    assert_eq!(
        partial,
        set_dec(Interval::new(f64::NEG_INFINITY, 0.0), Decoration::Trv)
    );
    assert_eq!(
        log1p(DecoratedInterval::new(-2.0, -1.0)),
        DecoratedInterval::EMPTY
    );
    assert!(log1p(DecoratedInterval::NAI).is_nai());
    assert!(expm1(DecoratedInterval::NAI).is_nai());
    assert_eq!(
        decoration_part(log1p(DecoratedInterval::ONE)),
        Decoration::Com
    );
    assert_eq!(
        decoration_part(expm1(DecoratedInterval::ENTIRE)),
        Decoration::Dac
    );
    assert_eq!(
        decoration_part(expm1(set_dec(Interval::ONE, Decoration::Def))),
        Decoration::Def
    );
}

#[test]
fn product_extrema_and_fused_addends_cover_mixed_signs_and_unbounded_inputs() {
    let x = Interval::new(-5.0, 2.0);
    let y = Interval::new(-3.0, 7.0);
    assert_eq!(x * y, Interval::new(-35.0, 15.0));
    assert_eq!(
        x.mul_add(y, Interval::new(5.0, 6.0)),
        Interval::new(-30.0, 21.0)
    );

    let nonpositive = Interval::new(f64::NEG_INFINITY, -1.0);
    let nonnegative = Interval::new(0.0, f64::INFINITY);
    assert_eq!(
        nonpositive * nonnegative,
        Interval::new(f64::NEG_INFINITY, 0.0)
    );
    assert_eq!(
        nonpositive.mul_add(nonnegative, Interval::ONE),
        Interval::new(f64::NEG_INFINITY, 1.0)
    );
    assert_eq!(
        nonpositive.mul_add(nonpositive, Interval::new(f64::NEG_INFINITY, 0.0)),
        Interval::ENTIRE
    );
    assert_eq!(Interval::ZERO.mul_add(Interval::ENTIRE, x), x);

    let max = Interval::from(f64::MAX);
    assert_eq!(max.mul_add(Interval::from(2.0), -max), max);
}

#[test]
fn quotient_extrema_use_the_appropriate_denominator_limits() {
    let numerator = Interval::new(-2.0, 4.0);
    assert_eq!(
        numerator / Interval::new(-4.0, -2.0),
        Interval::new(-2.0, 1.0)
    );
    assert_eq!(
        numerator / Interval::new(0.0, f64::INFINITY),
        Interval::ENTIRE
    );
    assert_eq!(
        Interval::new(0.0, 4.0) / Interval::new(f64::NEG_INFINITY, 0.0),
        Interval::new(f64::NEG_INFINITY, 0.0)
    );
    assert_eq!(
        Interval::new(2.0, f64::INFINITY) / Interval::new(4.0, f64::INFINITY),
        Interval::new(0.0, f64::INFINITY)
    );
}

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
