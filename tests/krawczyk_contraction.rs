#![cfg(feature = "linalg")]
#![allow(
    missing_docs,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::unwrap_used
)]

use core::convert::Infallible;

use maryada::{
    EnclosureOps, Interval, IntervalOps, KrawczykError, KrawczykOutcome, SIntervalMatrix,
    SIntervalVector, krawczyk_contract,
};

type Box2 = SIntervalVector<Interval, 2>;
type Jacobian2 = SIntervalMatrix<Interval, 2, 2>;

#[test]
fn krawczyk_certifies_and_refines_a_coupled_nonlinear_root() {
    let domain = Box2::from_element(Interval::new(1.5, 2.5));
    let outcome = krawczyk_contract(
        &domain,
        |x| {
            Ok::<_, Infallible>(Box2::from_column_slice(&[
                x[0].sqr() + x[1] - 6.0,
                x[0] + x[1].sqr() - 6.0,
            ]))
        },
        |x| {
            Ok::<_, Infallible>(Jacobian2::from_row_slice(&[
                2.0 * x[0],
                Interval::ONE,
                Interval::ONE,
                2.0 * x[1],
            ]))
        },
    )
    .unwrap();

    assert!(matches!(outcome, KrawczykOutcome::Unique(_)));
    if let KrawczykOutcome::Unique(contracted) = outcome {
        assert!(contracted.interior(&domain));
        assert!(Box2::from_element(Interval::from(2.0)).subset(&contracted));
    }
}

#[test]
fn krawczyk_can_exclude_a_box_that_passes_every_component_range_test() {
    let domain = Box2::from_element(Interval::new(0.0, 1.0));
    let function =
        |x: &Box2| Ok::<_, Infallible>(Box2::from_column_slice(&[x[0] + x[1], x[0] - x[1] - 1.0]));
    assert!(
        function(&domain)
            .unwrap()
            .iter()
            .all(|entry| entry.contains(0.0))
    );
    let outcome = krawczyk_contract(&domain, function, |_| {
        Ok::<_, Infallible>(Jacobian2::from_row_slice(&[
            Interval::ONE,
            Interval::ONE,
            Interval::ONE,
            Interval::from(-1.0),
        ]))
    })
    .unwrap();
    assert_eq!(outcome, KrawczykOutcome::Excluded);
}

#[test]
fn a_self_map_with_norm_one_keeps_the_contraction_without_certifying_uniqueness() {
    // f(x, y) = (max(0, x)^2, y) vanishes on the whole segment [-1, 0] × {0}.
    let x = Interval::new(-1.0, 1.0);
    let domain = Box2::from_element(x);
    let outcome = krawczyk_contract(
        &domain,
        |x| {
            Ok::<_, Infallible>(Box2::from_column_slice(&[
                x[0].max(Interval::ZERO).sqr(),
                x[1],
            ]))
        },
        |x| {
            Ok::<_, Infallible>(Jacobian2::from_diagonal(&Box2::from_column_slice(&[
                2.0 * x[0].max(Interval::ZERO),
                Interval::ONE,
            ])))
        },
    )
    .unwrap();
    assert_eq!(
        outcome,
        KrawczykOutcome::Unresolved(Box2::from_column_slice(&[x, Interval::ZERO]))
    );
}

#[test]
fn krawczyk_handles_unbounded_boxes_and_singular_midpoint_jacobians() {
    let domain = Box2::from_element(Interval::ENTIRE);
    let root = Box2::from_element(Interval::ONE);
    let linear = krawczyk_contract(
        &domain,
        |x| Ok::<_, Infallible>(x - root),
        |_| Ok::<_, Infallible>(Jacobian2::identity()),
    )
    .unwrap();
    assert_eq!(linear, KrawczykOutcome::Unique(root));

    let domain = Box2::from_element(Interval::new(-2.0, 2.0));
    let singular = krawczyk_contract(
        &domain,
        |x| Ok::<_, Infallible>(Box2::from_column_slice(&[x[0].sqr() - 1.0, x[1]])),
        |x| {
            Ok::<_, Infallible>(Jacobian2::from_diagonal(&Box2::from_column_slice(&[
                2.0 * x[0],
                Interval::ONE,
            ])))
        },
    )
    .unwrap();
    assert_eq!(singular, KrawczykOutcome::Unresolved(domain));
}

#[test]
fn empty_coordinates_and_exact_singletons_need_no_jacobian() {
    let domain = Box2::from_column_slice(&[Interval::EMPTY, Interval::ENTIRE]);
    assert_eq!(
        krawczyk_contract(
            &domain,
            |_| Err::<Box2, _>("function must be skipped"),
            |_| Err::<Jacobian2, _>("Jacobian must be skipped"),
        )
        .unwrap(),
        KrawczykOutcome::Excluded
    );

    let domain = Box2::zeros();
    assert_eq!(
        krawczyk_contract(
            &domain,
            |x| Ok::<_, Infallible>(Box2::from_column_slice(&[x[0].sqr(), x[1]])),
            |_| Err::<Jacobian2, _>("Jacobian must be skipped"),
        )
        .unwrap(),
        KrawczykOutcome::Unique(domain)
    );
}

#[test]
fn krawczyk_rejects_empty_enclosures_and_preserves_callback_errors() {
    let domain = Box2::from_element(Interval::new(-1.0, 1.0));
    assert_eq!(
        krawczyk_contract(
            &domain,
            |_| Ok::<_, Infallible>(Box2::from_element(Interval::EMPTY)),
            |_| Ok::<_, Infallible>(Jacobian2::identity()),
        ),
        Err(KrawczykError::EmptyFunctionEnclosure)
    );
    assert_eq!(
        krawczyk_contract(
            &domain,
            |x| Ok::<_, Infallible>(*x),
            |_| Ok::<_, Infallible>(Jacobian2::from_element(Interval::EMPTY)),
        ),
        Err(KrawczykError::EmptyJacobianEnclosure)
    );
    assert_eq!(
        krawczyk_contract(
            &domain,
            |x| Ok::<_, Infallible>(*x),
            |_| Err::<Jacobian2, _>("Jacobian callback failed"),
        ),
        Err(KrawczykError::Jacobian("Jacobian callback failed"))
    );
}

#[cfg(feature = "alloc")]
#[test]
fn dynamic_shapes_are_checked_for_each_callback_and_support_certification() {
    use maryada::{DIntervalMatrix, DIntervalVector};
    type Vector = DIntervalVector<Interval>;
    type Matrix = DIntervalMatrix<Interval>;

    let domain = Vector::from_element(2, Interval::new(-1.0, 1.0));
    let identity = |_: &Vector| Ok::<_, Infallible>(Matrix::identity(2));
    let invalid_range =
        krawczyk_contract(&domain, |_| Ok::<_, Infallible>(Vector::zeros(1)), identity);
    assert_eq!(
        invalid_range,
        Err(KrawczykError::FunctionShape {
            expected: 2,
            actual: 1
        })
    );

    let invalid_point = krawczyk_contract(
        &domain,
        |x| {
            Ok::<_, Infallible>(if x.iter().all(|entry| entry.is_singleton()) {
                Vector::zeros(1)
            } else {
                x.clone()
            })
        },
        identity,
    );
    assert_eq!(
        invalid_point,
        Err(KrawczykError::FunctionShape {
            expected: 2,
            actual: 1
        })
    );

    let invalid_derivative = krawczyk_contract(
        &domain,
        |x| Ok::<_, Infallible>(x.clone()),
        |_| Ok::<_, Infallible>(Matrix::zeros(2, 3)),
    );
    assert_eq!(
        invalid_derivative,
        Err(KrawczykError::JacobianShape {
            expected: 2,
            actual: (2, 3)
        })
    );
    assert_eq!(
        krawczyk_contract(&domain, |x| Ok::<_, Infallible>(x.clone()), identity).unwrap(),
        KrawczykOutcome::Unique(Vector::zeros(2)),
    );
    assert_eq!(
        krawczyk_contract(
            &Vector::zeros(0),
            |x| Ok::<_, Infallible>(x.clone()),
            identity
        ),
        Err(KrawczykError::ZeroDimensions),
    );
}
