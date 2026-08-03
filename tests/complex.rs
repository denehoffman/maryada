use maryada::{ComplexBox, DecoratedInterval, Decoration, Interval, decoration_part, set_dec};
use num_complex::Complex64;

fn assert_contains(interval: Interval, value: f64) {
    assert!(interval.inf() <= value && value <= interval.sup());
}

#[test]
fn arithmetic_and_elementary_functions_enclose_singleton_references() {
    let x = ComplexBox::<Interval>::new(1.0.into(), 2.0.into());
    let y = ComplexBox::<Interval>::new(3.0.into(), 4.0.into());
    let z = ComplexBox::<Interval>::new(5.0.into(), 6.0.into());
    let fused = x.mul_add(&y, &z);
    assert_eq!(fused.re.bounds(), (0.0, 0.0));
    assert_eq!(fused.im.bounds(), (16.0, 16.0));

    let square = ComplexBox::<Interval>::new(3.0.into(), 4.0.into()).sqr();
    assert_eq!(square.re.bounds(), (-7.0, -7.0));
    assert_eq!(square.im.bounds(), (24.0, 24.0));

    for (result, reference) in [
        (ComplexBox::<Interval>::from(0.5).atan(), 0.5_f64.atan()),
        (ComplexBox::<Interval>::from(0.5).atanh(), 0.5_f64.atanh()),
    ] {
        assert_contains(result.re, reference);
        assert_contains(result.im, 0.0);
    }
    let acosh = ComplexBox::<Interval>::from(2.0).acosh();
    assert_contains(acosh.re, 2.0_f64.acosh());
    assert_contains(acosh.im, 0.0);
}

#[test]
fn rectangular_geometry_and_set_relations_are_componentwise() {
    let inner = ComplexBox::<Interval>::new(Interval::new(1.0, 2.0), Interval::new(1.0, 2.0));
    let outer = ComplexBox::<Interval>::new(Interval::new(0.0, 3.0), Interval::new(0.0, 3.0));
    let apart = ComplexBox::<Interval>::new(Interval::new(4.0, 5.0), Interval::new(1.0, 2.0));

    assert!(inner.subset(&outer));
    assert!(inner.interior(&outer));
    assert!(inner.disjoint(&apart));
    assert_eq!(inner.intersection(&outer), inner);
    assert_eq!(inner.convex_hull(&outer), outer);

    let value = ComplexBox::<Interval>::new(Interval::new(0.0, 6.0), Interval::new(0.0, 8.0));
    assert_eq!(value.wid_box(), Complex64::new(6.0, 8.0));
    assert!(value.diameter() >= 10.0);
    assert!(value.rad() >= 5.0);
    assert_eq!(value.lower_corner(), Complex64::new(0.0, 0.0));
    assert_eq!(value.upper_corner(), Complex64::new(6.0, 8.0));
}

#[test]
fn decorations_conversions_and_constants_preserve_semantics() {
    let decorated = ComplexBox::new(
        set_dec(Interval::from(3.0), Decoration::Def),
        DecoratedInterval::from(4.0),
    );
    let abs = decorated.abs();
    assert_contains(abs.into(), 5.0);
    assert_eq!(decoration_part(abs), Decoration::Def);

    let converted = ComplexBox::<Interval>::from(Complex64::new(2.0, 3.0));
    assert!(converted.contains(Complex64::new(2.0, 3.0)));
    assert!(!converted.contains(Complex64::new(f64::NAN, 3.0)));
    assert!(ComplexBox::<Interval>::ZERO.is_zero());
    assert!(ComplexBox::<Interval>::ONE.is_real());
    assert_eq!(ComplexBox::<Interval>::I, ComplexBox::i());
}
