#![cfg(feature = "complex-linalg")]
#![allow(missing_docs)]
#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::op_ref)]

use maryada::{
    ComplexBox, EnclosureOps, EpsilonInflationSolver, Interval, IntervalOps,
    SComplexIntervalMatrix, SComplexIntervalVector, SolveError,
};
use num_complex::Complex64;

fn point(value: Complex64) -> ComplexBox<Interval> {
    ComplexBox::from(value)
}

#[test]
fn verified_solver_encloses_a_thin_complex_system() {
    let lhs = SComplexIntervalMatrix::<Interval, 2, 2>::from_row_slice(&[
        point(Complex64::new(2.0, 1.0)),
        point(Complex64::new(0.5, -0.25)),
        point(Complex64::new(-0.75, 0.5)),
        point(Complex64::new(3.0, -1.0)),
    ]);
    let expected = SComplexIntervalVector::<Interval, 2>::from_column_slice(&[
        point(Complex64::new(1.0, 0.5)),
        point(Complex64::new(-0.25, 2.0)),
    ]);
    let rhs = &lhs * &expected;

    let solution = lhs
        .solve(&rhs)
        .expect("the well-conditioned complex system should verify");

    assert!(solution[0].contains(Complex64::new(1.0, 0.5)));
    assert!(solution[1].contains(Complex64::new(-0.25, 2.0)));
}

#[test]
fn complex_solve_reports_typed_failures() {
    let singular = SComplexIntervalMatrix::<Interval, 1, 1>::from_element(point(Complex64::ZERO));
    let rhs = SComplexIntervalVector::<Interval, 1>::from_element(point(Complex64::ONE));
    assert_eq!(singular.try_solve(&rhs), Err(SolveError::SingularMidpoint));

    let identity = SComplexIntervalMatrix::<Interval, 1, 1>::from_element(point(Complex64::ONE));
    let no_iterations = EpsilonInflationSolver::default().with_max_iterations(0);
    assert_eq!(
        identity.try_solve_with(&rhs, &no_iterations),
        Err(SolveError::CertificationFailed { iterations: 0 })
    );
}

#[test]
fn complex_aliases_preserve_matrix_structure() {
    let value = ComplexBox::new(Interval::new(1.0, 2.0), Interval::new(-0.5, 0.5));
    let matrix = SComplexIntervalMatrix::<Interval, 2, 2>::from_element(value);

    assert_eq!(matrix.shape(), (2, 2));
    assert!(matrix.iter().all(|entry| *entry == value));
    assert_eq!(matrix.mid()[(0, 0)], Complex64::new(1.5, 0.0));
}
