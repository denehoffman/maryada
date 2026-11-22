#![cfg(feature = "complex")]
#![allow(missing_docs, clippy::arithmetic_side_effects, clippy::unwrap_used)]

use maryada::{
    ComplexBox, ComplexDisk, ComplexDiskError, ComplexEnclosure, Interval,
    IntervalConstructionError, IntervalOps,
};

fn contains(disk: ComplexDisk, re: f64, im: f64) -> bool {
    #[cfg(feature = "num-complex")]
    let point = num_complex::Complex64::new(re, im);
    #[cfg(not(feature = "num-complex"))]
    let point = (re, im);
    disk.contains(point)
}

#[test]
fn disk_construction_checks_center_and_radius() {
    use ComplexDiskError::{InvalidRadius, NonFiniteCenter};

    assert_eq!(
        ComplexDisk::try_new((0.0, f64::NAN), 1.0),
        Err(NonFiniteCenter)
    );
    assert_eq!(
        ComplexDisk::try_new((f64::INFINITY, 0.0), 1.0),
        Err(NonFiniteCenter)
    );
    for radius in [-1.0, f64::INFINITY, f64::NAN] {
        assert_eq!(ComplexDisk::try_new((0.0, 0.0), radius), Err(InvalidRadius));
    }
    assert_eq!(
        ComplexDisk::try_new((-0.0, 0.0), -0.0),
        Ok(ComplexDisk::ZERO)
    );
}

#[test]
fn box_hull_uses_the_stored_center_and_rounds_outward() {
    let wide = 1e16_f64;
    let source = ComplexBox::new(Interval::new(wide, wide.next_up()), Interval::ZERO);
    let disk = ComplexDisk::try_from(source).unwrap();
    assert_eq!(disk.radius(), Some(2.0));
    let hull = ComplexBox::from(disk);
    assert!(hull.re.inf() <= source.re.inf() && hull.re.sup() >= source.re.sup());
    assert_eq!(hull.im, Interval::new(-2.0, 2.0));

    let disk = ComplexDisk::try_new((f64::MAX, 0.0), 1.0).unwrap();
    assert_eq!(
        ComplexBox::from(disk).re,
        Interval::new(f64::MAX.next_down(), f64::INFINITY)
    );
}

#[test]
fn box_to_disk_distinguishes_empty_unbounded_and_overflowing_hulls() {
    let empty = ComplexBox::new(Interval::new(1.0, 2.0), Interval::EMPTY);
    assert_eq!(ComplexDisk::try_from(empty), Ok(ComplexDisk::EMPTY));
    assert_eq!(
        ComplexDisk::try_from(ComplexBox::ENTIRE),
        Ok(ComplexDisk::ENTIRE)
    );
    assert_eq!(ComplexBox::from(ComplexDisk::EMPTY), ComplexBox::EMPTY);
    assert_eq!(ComplexBox::from(ComplexDisk::ENTIRE), ComplexBox::ENTIRE);
    assert_eq!(
        ComplexDisk::try_from(ComplexBox::new(Interval::ENTIRE, Interval::ONE)),
        Err(ComplexDiskError::UnboundedBox),
    );
    let wide = Interval::new(-f64::MAX, f64::MAX);
    assert_eq!(
        ComplexDisk::try_from(ComplexBox::new(wide, wide)),
        Err(ComplexDiskError::RadiusOverflow),
    );
}

#[test]
fn modulus_retains_a_finite_lower_bound_when_the_center_norm_overflows() {
    assert_eq!(
        ComplexDisk::try_new((3.0, 4.0), 1.0).unwrap().abs(),
        Interval::new(4.0, 6.0)
    );
    assert_eq!(ComplexDisk::ZERO.abs(), Interval::ZERO);
    assert_eq!(ComplexDisk::EMPTY.abs(), Interval::EMPTY);
    assert_eq!(ComplexDisk::ENTIRE.abs(), Interval::new(0.0, f64::INFINITY));
    let modulus = ComplexDisk::try_new((f64::MAX, f64::MAX), f64::MAX)
        .unwrap()
        .abs();
    assert!(modulus.inf() > f64::MAX / 4.0);
    assert_eq!(modulus.sup(), f64::INFINITY);
}

#[test]
fn fixed_center_product_encloses_the_radius_and_the_rounded_center() {
    let left = ComplexDisk::try_new((3.0, 4.0), 2.0).unwrap();
    let right = ComplexDisk::try_new((-2.0, 0.0), 3.0).unwrap();
    assert_eq!(
        left * right,
        ComplexDisk::try_new((-6.0, -8.0), 25.0).unwrap()
    );
    let tiny = f64::from_bits(1);
    let left = ComplexDisk::try_new((tiny, 0.0), 0.0).unwrap();
    let right = ComplexDisk::try_new((0.5, 0.0), 0.0).unwrap();
    let product = left * right;
    assert_eq!(product.radius(), Some(tiny));
    assert!(ComplexBox::from(product).re.inf() <= 0.0);
    assert!(ComplexBox::from(product).re.sup() >= tiny);
}

#[test]
fn disk_product_scales_before_norm_and_handles_degenerate_sets() {
    let tiny = f64::from_bits(1);
    let huge = ComplexDisk::try_new((f64::MAX, f64::MAX), 0.0).unwrap();
    let centered = ComplexDisk::try_new((0.0, 0.0), tiny).unwrap();
    let product = huge * centered;
    assert_eq!(product.center(), Some((0.0, 0.0)));
    let radius = product.radius().unwrap();
    assert!(radius > 0.0 && radius.is_finite());

    let rotated = huge * ComplexDisk::try_new((0.5, -0.5), 0.0).unwrap();
    assert_eq!(rotated, ComplexDisk::try_new((f64::MAX, 0.0), 0.0).unwrap());
    let cancel = ComplexDisk::try_new((f64::MAX, f64::MAX / 4.0), 0.0).unwrap()
        * ComplexDisk::try_new((1.0625, 0.25), 0.0).unwrap();
    assert_eq!(cancel.center().unwrap().0, f64::MAX);
    assert!(cancel.radius().unwrap().is_finite());
    assert_eq!(
        huge * ComplexDisk::try_new((1.0, -1.0), 0.0).unwrap(),
        ComplexDisk::ENTIRE
    );
    assert_eq!(ComplexDisk::ZERO * ComplexDisk::ENTIRE, ComplexDisk::ZERO);
    assert_eq!(
        ComplexDisk::try_new((-0.0, -0.0), 0.0).unwrap() * ComplexDisk::ENTIRE,
        ComplexDisk::ZERO
    );
    assert_eq!(
        ComplexDisk::ENTIRE * ComplexDisk::try_new((0.0, 0.0), 1.0).unwrap(),
        ComplexDisk::ENTIRE
    );
    assert_eq!(ComplexDisk::EMPTY * ComplexDisk::ZERO, ComplexDisk::EMPTY);
}

#[test]
fn recentering_can_exclude_zero_and_bounds_the_product() {
    let disk = ComplexDisk::try_new((1.0, 0.0), 0.4375).unwrap();
    let fixed = disk * disk;
    let tight = disk.mul_recentered(disk);
    assert!(tight.radius().unwrap() < fixed.radius().unwrap());
    assert!(contains(fixed, 0.0, 0.0));
    assert!(!contains(tight, 0.0, 0.0));

    // Products at the outward tip, inward tip, and opposite angles.
    for (re, im) in [
        (2.066_406_25, 0.0),
        (0.316_406_25, 0.0),
        (0.808_593_75, 0.875),
        (0.808_593_75, -0.875),
        (1.191_406_25, 0.0),
    ] {
        assert!(contains(tight, re, im));
    }
    let rotated = disk * ComplexDisk::I;
    assert_eq!(disk.mul_recentered(ComplexDisk::I), rotated);
    assert_eq!(disk.mul_recentered(ComplexDisk::EMPTY), ComplexDisk::EMPTY);
    assert_eq!(
        disk.mul_recentered(ComplexDisk::ENTIRE),
        ComplexDisk::ENTIRE
    );
    let tiny = ComplexDisk::try_new((f64::from_bits(1), 0.0), 1.0).unwrap();
    assert_eq!(tiny.mul_recentered(tiny), tiny * tiny);
}

#[test]
fn checked_point_conversions_and_generic_arithmetic_work_for_both_shapes() {
    fn conjugate_product<T: ComplexEnclosure>(value: T) -> T::Real {
        (value * value.conj()).abs()
    }

    let point = (3.0, 4.0);
    let rectangular = ComplexBox::<Interval>::try_from(point).unwrap();
    let disk = ComplexDisk::try_from(point).unwrap();
    assert_eq!(ComplexDisk::try_from(rectangular), Ok(disk));
    assert_eq!(ComplexBox::from(disk), rectangular);
    assert_eq!(
        ComplexBox::<Interval>::try_from((f64::NAN, 0.0)),
        Err(IntervalConstructionError::NaN)
    );
    assert_eq!(
        ComplexDisk::try_from((f64::NAN, 0.0)),
        Err(ComplexDiskError::NonFiniteCenter)
    );
    assert_eq!(
        ComplexDisk::try_from(Interval::new(-1.0, 3.0)),
        ComplexDisk::try_new((1.0, 0.0), 2.0)
    );

    assert_eq!(conjugate_product(rectangular), Interval::from(25.0));
    assert_eq!(conjugate_product(disk), Interval::from(25.0));
}

#[test]
fn disk_addition_and_subtraction_keep_radius_and_center_rounding_error() {
    let left = ComplexDisk::try_new((2.0, -3.0), 0.5).unwrap();
    let right = ComplexDisk::try_new((-1.0, 2.0), 0.25).unwrap();
    assert_eq!(
        left + right,
        ComplexDisk::try_new((1.0, -1.0), 0.75).unwrap()
    );
    assert_eq!(
        left - right,
        ComplexDisk::try_new((3.0, -5.0), 0.75).unwrap()
    );
    let mut value = left;
    value -= 2.0;
    value *= ComplexDisk::I;
    assert_eq!(value, ComplexDisk::try_new((3.0, 0.0), 0.5).unwrap());
    let sum = ComplexDisk::from(1.0) + f64::from_bits(1);
    assert!(sum.radius().unwrap() > 0.0);
    assert!(ComplexBox::from(sum).re.sup() >= 1.0_f64.next_up());
    assert_eq!(left + ComplexDisk::EMPTY, ComplexDisk::EMPTY);
    assert_eq!(left - ComplexDisk::ENTIRE, ComplexDisk::ENTIRE);
}

#[test]
fn reciprocal_preserves_circles_away_from_zero_and_handles_singular_divisors() {
    let disk = ComplexDisk::try_new((2.0, 0.0), 1.0).unwrap();
    let reciprocal = disk.recip();
    assert!(reciprocal.is_bounded());
    assert!(contains(reciprocal, 1.0, 0.0));
    assert!(contains(reciprocal, 0.5, 0.0));
    assert!(!contains(reciprocal, 0.0, 0.0));
    assert_eq!(reciprocal, ComplexDisk::ONE / disk);
    let tangent = ComplexDisk::try_new((1.0, 0.0), 1.0).unwrap();
    assert_eq!(tangent.recip(), ComplexDisk::ENTIRE);
    let offset = 1.0 / 134_217_728.0;
    let almost_tangent = ComplexDisk::try_new((1.0, offset), 1.0).unwrap();
    let inverse = almost_tangent.recip();
    assert_eq!(
        inverse.center(),
        Some((18_014_398_509_481_984.0, -134_217_728.0))
    );
    assert_eq!(inverse.radius(), Some(18_014_398_509_481_984.0));
    assert_eq!(ComplexDisk::ZERO.recip(), ComplexDisk::EMPTY);
    assert_eq!(ComplexDisk::ZERO / tangent, ComplexDisk::ZERO);
    assert_eq!(disk / ComplexDisk::ZERO, ComplexDisk::EMPTY);

    let tiny = ComplexDisk::from(f64::from_bits(1));
    assert_eq!(tiny.recip(), ComplexDisk::ENTIRE);
    assert_eq!(tiny / tiny, ComplexDisk::ONE);
    let large = ComplexDisk::from(1.0e308);
    assert!(large.recip().is_bounded());
    assert!(large.recip().abs().inf() > 0.0);
}

#[test]
fn disk_set_queries_distinguish_tangency_on_binary64_boundaries() {
    let outer = ComplexDisk::try_new((0.0, 0.0), 5.0).unwrap();
    assert!(contains(outer, 3.0, 4.0));
    assert!(!contains(outer, 3.0, 4.0_f64.next_up()));
    let inside = ComplexDisk::try_new((4.0, 0.0), 1.0).unwrap();
    let tangent = ComplexDisk::try_new((6.0, 0.0), 1.0).unwrap();
    assert!(inside.subset(outer) && !inside.interior(outer));
    assert!(outer.intersects(tangent) && !outer.subset(tangent));
    assert!(outer.disjoint(ComplexDisk::from(5.0_f64.next_up())));
    assert!(outer.disjoint(ComplexDisk::EMPTY));
    let huge = ComplexDisk::try_new((f64::MAX, 0.0), f64::MAX).unwrap();
    assert!(contains(huge, 0.0, 0.0));
    assert!(!contains(huge, -f64::from_bits(1), 0.0));
}

#[test]
fn disk_elementary_functions_enclose_values_on_both_principal_branches() {
    let point = ComplexDisk::try_from((3.0, 4.0)).unwrap();
    assert!(contains(point.sqrt(), 2.0, 1.0));
    assert!(contains(point.conj().sqrt(), 2.0, -1.0));
    assert!(contains(ComplexDisk::from(-4.0).sqrt(), 0.0, 2.0));
    assert_eq!(point.pown(0), ComplexDisk::ONE);
    assert_eq!(point.powi(2), ComplexDisk::try_from((-7.0, 24.0)).unwrap());
    assert!(contains(
        ComplexDisk::from(2.0).pow(ComplexDisk::from(3.0)),
        8.0,
        0.0
    ));
    for (disk, reference) in [
        (ComplexDisk::ZERO.sqrt(), ComplexDisk::ZERO),
        (ComplexDisk::ZERO.exp(), ComplexDisk::ONE),
        (ComplexDisk::ZERO.exp2(), ComplexDisk::ONE),
        (ComplexDisk::ZERO.exp10(), ComplexDisk::ONE),
        (ComplexDisk::ONE.log(), ComplexDisk::ZERO),
        (ComplexDisk::ONE.log2(), ComplexDisk::ZERO),
        (ComplexDisk::ONE.log10(), ComplexDisk::ZERO),
        (ComplexDisk::ZERO.sin(), ComplexDisk::ZERO),
        (ComplexDisk::ZERO.cos(), ComplexDisk::ONE),
        (ComplexDisk::ZERO.tan(), ComplexDisk::ZERO),
        (ComplexDisk::ZERO.asin(), ComplexDisk::ZERO),
        (ComplexDisk::ONE.acos(), ComplexDisk::ZERO),
        (ComplexDisk::ZERO.atan(), ComplexDisk::ZERO),
        (ComplexDisk::ZERO.sinh(), ComplexDisk::ZERO),
        (ComplexDisk::ZERO.cosh(), ComplexDisk::ONE),
        (ComplexDisk::ZERO.tanh(), ComplexDisk::ZERO),
        (ComplexDisk::ZERO.asinh(), ComplexDisk::ZERO),
        (ComplexDisk::ONE.acosh(), ComplexDisk::ZERO),
        (ComplexDisk::ZERO.atanh(), ComplexDisk::ZERO),
        (ComplexDisk::cis(Interval::ZERO), ComplexDisk::ONE),
    ] {
        assert!(reference.subset(disk), "{reference:?} outside {disk:?}");
    }
    assert_eq!(ComplexDisk::EMPTY.sin(), ComplexDisk::EMPTY);
    assert_eq!(
        ComplexDisk::try_new((0.0, 0.0), 0.5).unwrap().log(),
        ComplexDisk::ENTIRE
    );
}
