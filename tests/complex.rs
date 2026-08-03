#![cfg(feature = "num-complex")]

use maryada::{ComplexBox, DecoratedInterval, Decoration, Interval, decoration_part, set_dec};
use num_complex::Complex64;

fn assert_contains(interval: Interval, value: f64) {
    assert!(interval.inf() <= value && value <= interval.sup());
}

fn assert_contains_complex(value: ComplexBox<Interval>, reference: Complex64) {
    assert_contains(value.re, reference.re);
    assert_contains(value.im, reference.im);
}

#[test]
fn arithmetic_and_elementary_functions_enclose_singleton_references() {
    let x = ComplexBox::<Interval>::new(1.0.into(), 2.0.into());
    let y = ComplexBox::<Interval>::new(3.0.into(), 4.0.into());
    let z = ComplexBox::<Interval>::new(5.0.into(), 6.0.into());
    let fused = x.mul_add(y, z);
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

    assert!(inner.subset(outer));
    assert!(inner.interior(outer));
    assert!(inner.disjoint(apart));
    assert_eq!(inner.intersection(outer), inner);
    assert_eq!(inner.convex_hull(outer), outer);

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
    let point = Complex64::new(2.0, 3.0);
    assert_eq!(ComplexBox::<Interval>::from(&point), converted);
    let real = Interval::from(2.0);
    assert_eq!(ComplexBox::<Interval>::from(&real), ComplexBox::from(2.0));
    let scalar = 2.0;
    assert_eq!(ComplexBox::<Interval>::from(&scalar), ComplexBox::from(2.0));

    let decorated_owned = ComplexBox::<DecoratedInterval>::from(converted);
    assert_eq!(ComplexBox::<Interval>::from(decorated_owned), converted);
    assert!(ComplexBox::<Interval>::ZERO.is_zero());
    assert!(ComplexBox::<Interval>::ONE.is_real());
    assert_eq!(ComplexBox::<Interval>::I, ComplexBox::i());
}

#[test]
fn arithmetic_operators_polar_forms_and_powers_enclose_reference_values() {
    let z = ComplexBox::<Interval>::from(Complex64::new(2.0, 1.0));
    let w = ComplexBox::<Interval>::from(Complex64::new(-0.5, 0.25));

    for (result, reference) in [
        (z + w, Complex64::new(1.5, 1.25)),
        (z - w, Complex64::new(2.5, 0.75)),
        (z * w, Complex64::new(-1.25, 0.0)),
        (z / w, Complex64::new(-2.4, -3.2)),
        (-z, Complex64::new(-2.0, -1.0)),
        (z + 2.0, Complex64::new(4.0, 1.0)),
        (2.0 - z, Complex64::new(0.0, -1.0)),
        (z * 2.0, Complex64::new(4.0, 2.0)),
        (2.0 / z, Complex64::new(0.8, -0.4)),
        (z.add_real(1.0.into()), Complex64::new(3.0, 1.0)),
        (z.sub_real(1.0.into()), Complex64::new(1.0, 1.0)),
        (z.scale(2.0.into()), Complex64::new(4.0, 2.0)),
        (z.div_real(2.0.into()), Complex64::new(1.0, 0.5)),
        (z.conj(), Complex64::new(2.0, -1.0)),
        (z.recip(), Complex64::new(0.4, -0.2)),
        (z.pown(0), Complex64::new(1.0, 0.0)),
        (z.pown(2), Complex64::new(3.0, 4.0)),
        (z.powi(3), Complex64::new(2.0, 11.0)),
        (z.pown(-1), Complex64::new(0.4, -0.2)),
    ] {
        assert_contains_complex(result, reference);
    }

    assert_contains(z.norm_sqr(), 5.0);
    assert_contains(z.norm(), 5.0_f64.sqrt());
    let (magnitude, argument) = z.to_polar();
    assert_contains(magnitude, 5.0_f64.sqrt());
    assert_contains(argument, 1.0_f64.atan2(2.0));

    let theta = Interval::from(0.25);
    assert_contains_complex(
        ComplexBox::cis(theta),
        Complex64::new(0.25_f64.cos(), 0.25_f64.sin()),
    );
    assert_contains_complex(
        ComplexBox::from_polar(Interval::from(2.0), theta),
        Complex64::new(2.0 * 0.25_f64.cos(), 2.0 * 0.25_f64.sin()),
    );
    assert_contains_complex(
        ComplexBox::<Interval>::from(2.0).pow(ComplexBox::from(3.0)),
        Complex64::new(8.0, 0.0),
    );
}

#[test]
fn complex_scalar_operators_cover_interval_and_box_combinations() {
    type BareBox = ComplexBox<Interval>;
    type DecoratedBox = ComplexBox<DecoratedInterval>;

    fn bare(_: BareBox) {}
    fn decorated(_: DecoratedBox) {}

    macro_rules! check_op {
        ($op:tt) => {{
            let scalar = Complex64::new(2.0, 1.0);
            let bare_box = BareBox::from(3.0);
            let decorated_box = DecoratedBox::from(4.0);
            let bare_interval = Interval::from(5.0);
            let decorated_interval = DecoratedInterval::from(6.0);

            bare(bare_box $op scalar);
            bare(scalar $op bare_box);
            decorated(decorated_box $op scalar);
            decorated(scalar $op decorated_box);
            bare(bare_interval $op scalar);
            bare(scalar $op bare_interval);
            decorated(decorated_interval $op scalar);
            decorated(scalar $op decorated_interval);
        }};
    }

    check_op!(+);
    check_op!(-);
    check_op!(*);
    check_op!(/);
}

#[test]
fn analytic_families_enclose_real_and_nonreal_reference_formulas() {
    let real = ComplexBox::<Interval>::from(0.5);
    for (result, reference) in [
        (real.sqrt(), 0.5_f64.sqrt()),
        (real.exp(), 0.5_f64.exp()),
        (real.exp2(), 0.5_f64.exp2()),
        (real.exp10(), 10.0_f64.powf(0.5)),
        (real.sin(), 0.5_f64.sin()),
        (real.cos(), 0.5_f64.cos()),
        (real.tan(), 0.5_f64.tan()),
        (real.asin(), 0.5_f64.asin()),
        (real.acos(), 0.5_f64.acos()),
        (real.sinh(), 0.5_f64.sinh()),
        (real.cosh(), 0.5_f64.cosh()),
        (real.tanh(), 0.5_f64.tanh()),
        (real.asinh(), 0.5_f64.asinh()),
    ] {
        assert_contains_complex(result, Complex64::new(reference, 0.0));
    }

    let positive = ComplexBox::<Interval>::from(2.0);
    for (result, reference) in [
        (positive.log(), 2.0_f64.ln()),
        (positive.log2(), 2.0_f64.log2()),
        (positive.log10(), 2.0_f64.log10()),
        (positive.acosh(), 2.0_f64.acosh()),
    ] {
        assert_contains_complex(result, Complex64::new(reference, 0.0));
    }

    let a = 0.25_f64;
    let b = 0.5_f64;
    let z = ComplexBox::<Interval>::from(Complex64::new(a, b));
    assert_contains_complex(
        z.exp(),
        Complex64::new(a.exp() * b.cos(), a.exp() * b.sin()),
    );
    assert_contains_complex(
        z.sin(),
        Complex64::new(a.sin() * b.cosh(), a.cos() * b.sinh()),
    );
    assert_contains_complex(
        z.cos(),
        Complex64::new(a.cos() * b.cosh(), -a.sin() * b.sinh()),
    );
    assert_contains_complex(
        z.sinh(),
        Complex64::new(a.sinh() * b.cos(), a.cosh() * b.sin()),
    );
    assert_contains_complex(
        z.cosh(),
        Complex64::new(a.cosh() * b.cos(), a.sinh() * b.sin()),
    );

    for result in [
        z.tan(),
        z.tanh(),
        z.atan(),
        z.asin(),
        z.acos(),
        z.asinh(),
        z.atanh(),
    ] {
        assert!(!result.is_empty());
        assert!(result.is_bounded());
    }

    let decorated = ComplexBox::new(
        set_dec(Interval::from(0.5), Decoration::Def),
        DecoratedInterval::ZERO,
    );
    for result in [
        decorated.recip(),
        decorated.sqrt(),
        decorated.exp(),
        decorated.log(),
        decorated.sin(),
        decorated.cos(),
        decorated.tan(),
        decorated.sinh(),
        decorated.cosh(),
        decorated.tanh(),
        decorated.asin(),
        decorated.acos(),
        decorated.atan(),
        decorated.asinh(),
        decorated.atanh(),
    ] {
        assert!(!result.is_nai());
        assert_eq!(result.decoration(), Decoration::Def);
    }
}

#[test]
fn geometry_and_set_relations_handle_empty_entire_and_nai_boxes() {
    let value = ComplexBox::<Interval>::new(Interval::new(0.0, 2.0), Interval::new(-2.0, 4.0));
    assert_eq!(value.mid(), Complex64::new(1.0, 1.0));
    assert_eq!(value.wid(), value.wid_box());
    assert_eq!(value.rad_box(), Complex64::new(1.0, 3.0));
    assert_eq!(value.mid_rad(), (value.mid(), value.rad_box()));
    assert_eq!(
        value.corners(),
        [
            Complex64::new(0.0, -2.0),
            Complex64::new(2.0, -2.0),
            Complex64::new(2.0, 4.0),
            Complex64::new(0.0, 4.0),
        ]
    );
    assert!(value.mag() >= value.mig());
    assert!(value.is_bounded());
    assert!(!value.is_singleton());
    assert!(!value.is_real());
    assert!(value.intersects(ComplexBox::from(Complex64::new(1.0, 0.0))));

    let empty = ComplexBox::<Interval>::new(Interval::EMPTY, Interval::ONE);
    assert!(empty.is_empty());
    assert!(empty.rad().is_nan());
    assert!(empty.diameter().is_nan());
    assert!(empty.subset(value));
    assert!(empty.interior(value));
    assert!(!value.subset(empty));
    assert!(!value.interior(empty));
    assert_eq!(empty.convex_hull(value), value);
    assert_eq!(value.convex_hull(empty), value);
    assert!(value.intersection(ComplexBox::from(10.0)).is_empty());
    assert!(empty.pown(3).is_empty());

    assert!(ComplexBox::<Interval>::ENTIRE.is_entire());
    assert!(!ComplexBox::<Interval>::ENTIRE.is_bounded());
    assert_eq!(
        ComplexBox::<Interval>::default(),
        ComplexBox::<Interval>::ZERO
    );
    assert!(ComplexBox::<Interval>::ONE.is_singleton());
    assert!(ComplexBox::<Interval>::ONE.is_real());
    assert!(ComplexBox::<Interval>::ZERO.is_zero());
    assert!(ComplexBox::<Interval>::ONE.to_string().contains("i"));

    let nai = ComplexBox::new(DecoratedInterval::NAI, DecoratedInterval::ZERO);
    let ordinary = ComplexBox::<DecoratedInterval>::ONE;
    assert!(nai.is_nai());
    assert!(!nai.subset(ordinary));
    assert!(!ordinary.subset(nai));
    assert!(!nai.interior(ordinary));
    assert!(!ordinary.interior(nai));

    let decorated = value.decorate(&mut ());
    assert_eq!(ComplexBox::<Interval>::from(&decorated), value);
    assert_eq!(ComplexBox::<DecoratedInterval>::from(&value), decorated);
}
