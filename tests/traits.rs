//! Tests for the sealed scalar capability traits.

use maryada::{
    Conjugate, DecoratedInterval, Enclosure, EnclosureArithmetic, Interval, Magnitude, Midpoint,
};

#[test]
fn real_enclosure_capabilities_forward_to_interval_operations() {
    let x = Interval::new(1.0, 2.0);
    let y = Interval::new(3.0, 4.0);

    assert_eq!(<Interval as Enclosure>::zero(), Interval::from(0.0));
    assert!(!Enclosure::is_empty(x));
    assert!(!Enclosure::is_nai(x));
    assert_eq!(EnclosureArithmetic::add(x, y), x + y);
    assert_eq!(EnclosureArithmetic::mul_add(x, y, x), x.mul_add(y, x));
    assert_eq!(Conjugate::conj(x), x);
    assert_eq!(Midpoint::midpoint(x), x.mid());
    assert_eq!(Magnitude::mag(x), x.mag());
    assert_eq!(Magnitude::mig(x), x.mig());
}

#[test]
fn decorated_enclosure_capabilities_preserve_nai() {
    let nai = DecoratedInterval::NAI;

    assert!(Enclosure::is_nai(nai));
    assert!(Enclosure::is_nai(EnclosureArithmetic::add(
        nai,
        DecoratedInterval::from(1.0),
    )));
}

#[cfg(feature = "complex")]
#[test]
fn complex_enclosure_capabilities_forward_to_box_operations() {
    use maryada::ComplexBox;

    let x = ComplexBox::<Interval>::new(Interval::new(1.0, 2.0), Interval::from(1.0));
    let y = ComplexBox::<Interval>::new(Interval::from(2.0), Interval::new(-1.0, 1.0));

    assert!(!Enclosure::is_empty(x));
    assert!(!Enclosure::is_nai(x));
    assert_eq!(EnclosureArithmetic::mul(x, y), x * y);
    assert_eq!(Conjugate::conj(x), x.conj());
    assert_eq!(Magnitude::mag(x), x.mag());
    assert_eq!(Magnitude::mig(x), x.mig());
}

#[cfg(feature = "num-complex")]
#[test]
fn complex_midpoint_uses_complex64() {
    use maryada::ComplexBox;
    use num_complex::Complex64;

    let value = ComplexBox::<Interval>::new(Interval::new(1.0, 3.0), Interval::new(-2.0, 2.0));

    assert_eq!(Midpoint::midpoint(value), Complex64::new(2.0, 0.0));
}
