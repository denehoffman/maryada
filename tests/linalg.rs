//! Compile-time coverage for the public nalgebra storage aliases.
#![cfg(feature = "linalg")]

use maryada::{Interval, IntervalMatrix, RegularityResult, empty, entire};
use nalgebra::SMatrix;

#[test]
fn beeck_inf_norm_certifies_regular_matrix() {
    let matrix = SMatrix::<Interval, 2, 2>::from_row_slice(&[
        Interval::new(1.9, 2.1),
        Interval::new(-0.01, 0.01),
        Interval::new(-0.01, 0.01),
        Interval::new(2.9, 3.1),
    ]);
    let matrix = IntervalMatrix::from_inner(matrix);

    assert!(matches!(
        matrix.is_regular(),
        RegularityResult::ProvenRegular
    ));
}

#[test]
fn beeck_inf_norm_is_inconclusive_for_singular_midpoint() {
    let matrix = SMatrix::<Interval, 1, 1>::from_element(Interval::new(-1.0, 1.0));
    let matrix = IntervalMatrix::from_inner(matrix);

    assert!(matches!(
        matrix.is_regular(),
        RegularityResult::Inconclusive
    ));
}

#[test]
fn beeck_inf_norm_requires_a_strict_bound() {
    // Midpoint = 1, R = 1, and I - R*[0, 2] = [-1, 1].
    // Its infinity norm is exactly 1, so the strict test must fail.
    let matrix = SMatrix::<Interval, 1, 1>::from_element(Interval::new(0.0, 2.0));
    let matrix = IntervalMatrix::from_inner(matrix);

    assert!(matches!(
        matrix.is_regular(),
        RegularityResult::Inconclusive
    ));
}

#[test]
fn beeck_inf_norm_rejects_empty_entries() {
    let matrix = SMatrix::<Interval, 1, 1>::from_element(empty::<Interval>());
    let matrix = IntervalMatrix::from_inner(matrix);

    assert!(matches!(
        matrix.is_regular(),
        RegularityResult::Inconclusive
    ));
}

#[test]
fn beeck_inf_norm_is_inconclusive_for_unbounded_entries() {
    let matrix = SMatrix::<Interval, 1, 1>::from_element(entire::<Interval>());
    let matrix = IntervalMatrix::from_inner(matrix);

    assert!(matches!(
        matrix.is_regular(),
        RegularityResult::Inconclusive
    ));
}
