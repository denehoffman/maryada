#![allow(missing_docs)]
#![cfg(feature = "complex")]

use maryada::{
    ComplexBox, DecoratedInterval, Decoration, EnclosureOps, Interval, IntervalOps, set_dec,
};

#[test]
fn complex_boxes_work_without_num_complex() {
    let x = ComplexBox::<Interval>::new(Interval::new(1.0, 2.0), Interval::new(3.0, 4.0));
    let y = ComplexBox::<Interval>::new(Interval::from(2.0), Interval::from(1.0));

    let sum = x + y;
    assert_eq!(sum.re.bounds(), (3.0, 4.0));
    assert_eq!(sum.im.bounds(), (4.0, 5.0));
    assert!(x.is_bounded());
    assert!(x.diameter().is_finite());
}

#[test]
fn complex_addition_encloses_both_components_near_overflow() {
    let small = Interval::from(libm::scalbn(3.0, 970));
    let negative_max = -f64::MAX;
    let large = Interval::from(negative_max);
    let expected = Interval::new(negative_max.next_up(), negative_max.next_up().next_up());

    assert_eq!(
        ComplexBox::new(small, -small) + ComplexBox::new(large, -large),
        ComplexBox::new(expected, -expected)
    );
}

#[test]
#[allow(clippy::arithmetic_side_effects)]
fn complex_quotients_cover_both_ends_of_the_binary64_range() {
    for magnitude in [f64::from_bits(1), 1e-200, 1e200, f64::MAX] {
        let real = ComplexBox::<Interval>::from(magnitude);
        let imaginary = ComplexBox::new(Interval::ZERO, Interval::from(magnitude));
        assert_eq!(real / real, ComplexBox::ONE);
        assert_eq!(imaginary / imaginary, ComplexBox::ONE);
        assert_eq!(imaginary / real, ComplexBox::I);
        let reciprocal = Interval::from(magnitude).recip();
        assert_eq!(real.recip(), ComplexBox::from(reciprocal));
        assert_eq!(
            imaginary.recip(),
            ComplexBox::new(Interval::ZERO, -reciprocal)
        );
    }
    for exponent in [-1074, -1000, -664, 0, 664, 1023] {
        let component = libm::scalbn(1.0, exponent);
        let point = ComplexBox::new(Interval::from(component), Interval::from(component));
        assert_eq!(point / point, ComplexBox::ONE);
        let reciprocal = libm::scalbn(0.5, -exponent);
        let expected = if reciprocal.is_finite() {
            Interval::from(reciprocal)
        } else {
            Interval::new(f64::MAX, f64::INFINITY)
        };
        assert_eq!(point.recip(), ComplexBox::new(expected, -expected));
    }
}

#[test]
fn complex_quotients_rescale_before_rounding_the_division() {
    let lhs = ComplexBox::<Interval>::from(libm::scalbn(1.0, -700));
    let lower = libm::scalbn(1.0, 70);
    let upper = libm::scalbn(1.0, 600);
    let rhs = ComplexBox::new(Interval::new(lower, upper), Interval::from(lower));
    let quotient = lhs / rhs;
    assert!(quotient.is_bounded());
    assert!(quotient.re.contains(libm::scalbn(1.0, -771)));
    assert!(quotient.im.contains(libm::scalbn(-1.0, -771)));
    assert_eq!(quotient.re.sup(), libm::scalbn(1.0, -241));
}

#[test]
fn complex_quotients_distinguish_underflow_from_an_actual_pole() {
    let tiny = f64::from_bits(1);
    let lhs = ComplexBox::<DecoratedInterval>::from(tiny);
    let rhs = ComplexBox::new(
        DecoratedInterval::new(tiny, f64::MAX),
        DecoratedInterval::new(0.0, tiny),
    );
    let quotient = lhs / rhs;
    assert_eq!(quotient.re, DecoratedInterval::new(0.0, 1.0));
    assert_eq!(quotient.im, DecoratedInterval::new(-1.0, 0.0));
    assert_eq!(quotient.decoration(), Decoration::Com);

    let crossing = ComplexBox::new(
        DecoratedInterval::new(0.0, 1.0),
        DecoratedInterval::new(0.0, 1.0),
    );
    let reciprocal = crossing.recip();
    assert_eq!(
        reciprocal.re,
        set_dec(Interval::new(0.0, f64::INFINITY), Decoration::Trv)
    );
    assert_eq!(
        reciprocal.im,
        set_dec(Interval::new(f64::NEG_INFINITY, 0.0), Decoration::Trv)
    );
    assert!((lhs / ComplexBox::<DecoratedInterval>::ZERO).is_empty());
    assert!(ComplexBox::<DecoratedInterval>::ZERO.recip().is_empty());
    assert!((lhs / ComplexBox::new(DecoratedInterval::NAI, DecoratedInterval::ZERO)).is_nai());
}

#[test]
fn complex_square_root_maps_zero_to_zero() {
    assert_eq!(
        ComplexBox::<Interval>::ZERO.sqrt(),
        ComplexBox::<Interval>::ZERO
    );

    let zero = ComplexBox::new(
        set_dec(Interval::ZERO, Decoration::Def),
        DecoratedInterval::ZERO,
    );
    let root = zero.sqrt();
    assert!(root.is_zero());
    assert_eq!(root.re.decoration(), Decoration::Def);
    assert_eq!(root.im.decoration(), Decoration::Def);
}

#[test]
fn complex_square_root_encloses_a_box_containing_the_origin() {
    let root = ComplexBox::new(Interval::new(-4.0, 4.0), Interval::new(-2.0, 2.0)).sqrt();

    for (re, im) in [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (1.0, 1.0), (1.0, -1.0)] {
        assert!(root.re.contains(re) && root.im.contains(im));
    }
}

#[test]
fn complex_square_root_uses_the_principal_branch_on_the_negative_real_axis() {
    let real = Interval::from(-4.0);
    let on_axis = ComplexBox::new(real, Interval::ZERO).sqrt();
    assert!(on_axis.re.contains(0.0));
    assert!(on_axis.im.contains(2.0));
    assert!(on_axis.im.inf() > 0.0);

    let above = ComplexBox::new(real, Interval::new(0.0, 7.5)).sqrt();
    assert!(above.re.contains(1.5));
    assert!(above.im.contains(2.0) && above.im.contains(2.5));
    assert!(above.im.inf() > 0.0);

    let below = ComplexBox::new(real, Interval::new(-7.5, 0.0)).sqrt();
    assert!(below.re.contains(1.5));
    assert!(below.im.contains(-2.5) && below.im.contains(2.0));
}

#[test]
fn complex_box_promotion_preserves_empty_components() {
    let bare = ComplexBox::new(Interval::EMPTY, Interval::ONE);
    let expected = ComplexBox::new(DecoratedInterval::EMPTY, DecoratedInterval::ONE);
    assert_eq!(ComplexBox::<DecoratedInterval>::from(bare), expected);
    assert_eq!(ComplexBox::<DecoratedInterval>::from(&bare), expected);
    assert_eq!(
        ComplexBox::<DecoratedInterval>::from(ComplexBox::<Interval>::EMPTY),
        ComplexBox::<DecoratedInterval>::EMPTY
    );
}

#[test]
fn mixed_complex_arithmetic_propagates_empty() {
    let empty_real = Interval::EMPTY;
    let empty_box = ComplexBox::new(Interval::EMPTY, Interval::ONE);
    let one_real = DecoratedInterval::ONE;
    let one_box = ComplexBox::<DecoratedInterval>::ONE;

    let mut assigned = one_box;
    assigned += empty_box;
    for result in [
        empty_box + one_box,
        one_box + empty_box,
        empty_box + one_real,
        one_real + empty_box,
        empty_real + one_box,
        one_box + empty_real,
        assigned,
    ] {
        assert!(result.is_empty());
        assert!(!result.is_nai());
        assert_eq!(result.decoration(), Decoration::Trv);
    }
}

#[test]
fn complex_operators_cover_real_and_box_combinations() {
    type BareBox = ComplexBox<Interval>;
    type DecoratedBox = ComplexBox<DecoratedInterval>;

    fn bare(_: BareBox) {}
    fn decorated(_: DecoratedBox) {}

    macro_rules! check_op {
        ($op:tt) => {{
            let bare_box = BareBox::from(2.0);
            let decorated_box = DecoratedBox::from(3.0);
            let bare_interval = Interval::from(4.0);
            let decorated_interval = DecoratedInterval::from(5.0);
            let scalar = 6.0;

            bare(bare_box $op bare_box);
            decorated(bare_box $op decorated_box);
            decorated(decorated_box $op bare_box);
            decorated(decorated_box $op decorated_box);

            bare(bare_box $op bare_interval);
            bare(bare_interval $op bare_box);
            decorated(bare_box $op decorated_interval);
            decorated(decorated_interval $op bare_box);
            decorated(decorated_box $op bare_interval);
            decorated(bare_interval $op decorated_box);
            decorated(decorated_box $op decorated_interval);
            decorated(decorated_interval $op decorated_box);

            bare(bare_box $op scalar);
            bare(scalar $op bare_box);
            decorated(decorated_box $op scalar);
            decorated(scalar $op decorated_box);
        }};
    }

    check_op!(+);
    check_op!(-);
    check_op!(*);
    check_op!(/);
}

#[test]
fn complex_assignment_operators_cover_non_promoting_combinations() {
    type BareBox = ComplexBox<Interval>;
    type DecoratedBox = ComplexBox<DecoratedInterval>;

    macro_rules! check_assign_op {
        ($op:tt) => {{
            let mut bare_box = BareBox::from(8.0);
            bare_box $op BareBox::from(2.0);
            bare_box $op Interval::from(2.0);
            bare_box $op 2.0;

            let mut decorated_box = DecoratedBox::from(8.0);
            decorated_box $op BareBox::from(2.0);
            decorated_box $op DecoratedBox::from(2.0);
            decorated_box $op Interval::from(2.0);
            decorated_box $op DecoratedInterval::from(2.0);
            decorated_box $op 2.0;
        }};
    }

    check_assign_op!(+=);
    check_assign_op!(-=);
    check_assign_op!(*=);
    check_assign_op!(/=);
}
