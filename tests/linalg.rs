#![allow(missing_docs)]
#![cfg(feature = "linalg")]
#![allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::op_ref
)]
use maryada::{
    DecoratedInterval, EpsilonInflationSolver, GaussianEliminationSolver, HansenBliekRohnSolver,
    Interval, IntervalMatrix, IntervalOps, KrawczykSolver, Preconditioned, Preconditioner,
    SIntervalMatrix, SIntervalRowVector, SIntervalVector, Solver, StoppingTolerance,
};
use nalgebra::SMatrix;

#[test]
fn basic_access_matches_underlying_matrix() {
    let mut matrix: SIntervalMatrix<Interval, 2, 2> = SMatrix::<Interval, 2, 2>::from_row_slice(&[
        Interval::singleton(1.0),
        Interval::singleton(2.0),
        Interval::singleton(3.0),
        Interval::singleton(4.0),
    ])
    .into();

    assert_eq!(matrix.shape(), (2, 2));
    assert_eq!(matrix.nrows(), 2);
    assert_eq!(matrix.ncols(), 2);
    assert_eq!(matrix.len(), 4);
    assert!(!matrix.is_empty());
    assert!(matrix.is_square());
    assert_eq!(matrix[(0, 1)], Interval::singleton(2.0));
    assert_eq!(matrix.get(1, 0), Some(&Interval::singleton(3.0)));
    assert_eq!(matrix.get(2, 0), None);
    assert_eq!(matrix.iter().next(), Some(&Interval::singleton(1.0)));
    assert_eq!((&matrix).into_iter().count(), 4);
    assert_eq!(matrix.row_iter().count(), 2);
    assert_eq!(matrix.column_iter().count(), 2);

    let changed = matrix.get_mut(1, 1).is_some_and(|entry| {
        *entry = Interval::singleton(5.0);
        true
    });
    assert!(changed);
    matrix.iter_mut().for_each(|entry| *entry += 1.0);
    assert_eq!((&mut matrix).into_iter().count(), 4);
    assert_eq!(matrix.row_iter_mut().count(), 2);
    assert_eq!(matrix.column_iter_mut().count(), 2);

    let inner: &SMatrix<Interval, 2, 2> = matrix.as_ref();
    assert_eq!(inner[(1, 1)], Interval::singleton(6.0));
    let inner: &mut SMatrix<Interval, 2, 2> = matrix.as_mut();
    inner[(0, 0)] = Interval::ZERO;
    assert_eq!(matrix[(0, 0)], Interval::ZERO);
}

#[test]
fn static_constructors_preserve_layout_and_interval_semantics() {
    let one = Interval::singleton(1.0);
    let two = Interval::singleton(2.0);
    let three = Interval::singleton(3.0);
    let four = Interval::singleton(4.0);

    let filled = SIntervalMatrix::<Interval, 2, 2>::from_element(one);
    assert!(filled.iter().all(|entry| *entry == one));

    let from_fn = SIntervalMatrix::<Interval, 2, 2>::from_fn(|i, j| {
        Interval::singleton(match (i, j) {
            (0, 0) => 0.0,
            (1, 1) => 2.0,
            _ => 1.0,
        })
    });
    assert_eq!(from_fn[(1, 1)], Interval::singleton(2.0));

    let column_major = SIntervalMatrix::<Interval, 2, 2>::from_iterator([one, two, three, four]);
    let row_major = SIntervalMatrix::<Interval, 2, 2>::from_row_iterator([one, two, three, four]);
    assert_eq!(column_major[(0, 1)], three);
    assert_eq!(row_major[(0, 1)], two);

    assert_eq!(
        SIntervalMatrix::<Interval, 2, 2>::from_row_slice(&[one, two, three, four]),
        row_major,
    );
    assert_eq!(
        SIntervalMatrix::<Interval, 2, 2>::from_column_slice(&[one, two, three, four]),
        column_major,
    );

    let rows = [
        SIntervalRowVector::<Interval, 2>::from_row_slice(&[one, two]),
        SIntervalRowVector::<Interval, 2>::from_row_slice(&[three, four]),
    ];
    let columns = [
        SIntervalVector::<Interval, 2>::from_column_slice(&[one, three]),
        SIntervalVector::<Interval, 2>::from_column_slice(&[two, four]),
    ];
    assert_eq!(SIntervalMatrix::from_rows(&rows), row_major);
    assert_eq!(SIntervalMatrix::from_columns(&columns), row_major);

    let diagonal = SIntervalVector::<Interval, 2>::from_column_slice(&[two, three]);
    let diagonal_matrix = SIntervalMatrix::<Interval, 2, 2>::from_diagonal(&diagonal);
    assert_eq!(diagonal_matrix[(0, 0)], two);
    assert_eq!(diagonal_matrix[(0, 1)], Interval::ZERO);
    assert_eq!(
        SIntervalMatrix::<Interval, 2, 2>::zeros()[(1, 1)],
        Interval::ZERO
    );
    assert_eq!(
        SIntervalMatrix::<Interval, 2, 2>::identity()[(1, 1)],
        Interval::ONE
    );

    let points = SMatrix::<f64, 2, 2>::from_row_slice(&[1.0, 2.0, 3.0, 4.0]);
    let singletons = SIntervalMatrix::<Interval, 2, 2>::from_singletons(&points);
    assert_eq!(singletons, row_major);

    let lower = SMatrix::<f64, 1, 2>::from_row_slice(&[0.0, 1.0]);
    let upper = SMatrix::<f64, 1, 2>::from_row_slice(&[2.0, 3.0]);
    let bounds = SIntervalMatrix::<Interval, 1, 2>::from_bounds(&lower, &upper);
    assert_eq!(bounds[(0, 0)], Interval::new(0.0, 2.0));

    let midpoint = SMatrix::<f64, 1, 1>::from_element(1.0);
    let radius = SMatrix::<f64, 1, 1>::from_element(0.1);
    let mid_rad = SIntervalMatrix::<Interval, 1, 1>::from_mid_rad(&midpoint, &radius);
    assert!(mid_rad[(0, 0)].contains(0.9));
    assert!(mid_rad[(0, 0)].contains(1.1));
}

#[cfg(feature = "alloc")]
#[test]
fn dynamic_constructors_infer_dimensions_from_rows_and_columns() {
    use maryada::{DIntervalMatrix, DIntervalRowVector, DIntervalVector};

    let one = Interval::singleton(1.0);
    let two = Interval::singleton(2.0);
    let three = Interval::singleton(3.0);
    let four = Interval::singleton(4.0);
    let rows = [
        DIntervalRowVector::from_row_slice(&[one, two]),
        DIntervalRowVector::from_row_slice(&[three, four]),
    ];
    let columns = [
        DIntervalVector::from_column_slice(&[one, three]),
        DIntervalVector::from_column_slice(&[two, four]),
    ];

    let from_rows = DIntervalMatrix::from_rows(&rows);
    let from_columns = DIntervalMatrix::from_columns(&columns);

    assert_eq!(from_rows, from_columns);
    assert_eq!(from_rows.shape(), (2, 2));
    assert_eq!(DIntervalMatrix::<Interval>::zeros(2, 3).shape(), (2, 3));
    assert_eq!(
        DIntervalMatrix::<Interval>::identity(2)[(0, 0)],
        Interval::ONE
    );
}

#[test]
fn interval_predicates_and_set_operations_are_componentwise() {
    let narrow = SIntervalMatrix::<Interval, 1, 2>::from_row_slice(&[
        Interval::new(1.0, 2.0),
        Interval::new(3.0, 4.0),
    ]);
    let wide = SIntervalMatrix::<Interval, 1, 2>::from_row_slice(&[
        Interval::new(0.0, 3.0),
        Interval::new(2.0, 5.0),
    ]);

    assert!(narrow.subset(&wide));
    assert!(narrow.interior(&wide));
    assert!(narrow.equal(&narrow));
    assert_eq!(narrow.intersection(&wide), narrow);
    assert_eq!(narrow.convex_hull(&wide), wide);
    assert!(narrow.all(IntervalOps::is_bounded));
    assert!(narrow.any(|entry| entry.contains(1.5)));

    let exceptional =
        SIntervalMatrix::<Interval, 1, 2>::from_row_slice(&[Interval::EMPTY, Interval::ENTIRE]);
    assert!(exceptional.has_empty_entries());
    assert!(exceptional.has_entire_entries());
    assert!(!exceptional.has_nai_entries());

    let decorated =
        SIntervalMatrix::<DecoratedInterval, 1, 1>::from_element(DecoratedInterval::NAI);
    assert!(decorated.has_nai_entries());
}

#[test]
fn solvers_accept_borrowed_matrix_and_vector_views() {
    let lhs_storage = SIntervalMatrix::<Interval, 3, 3>::from_fn(|row, column| {
        if row == column {
            Interval::singleton(2.0)
        } else {
            Interval::ZERO
        }
    });
    let rhs_storage = SIntervalVector::<Interval, 3>::from_column_slice(&[
        Interval::singleton(4.0),
        Interval::singleton(6.0),
        Interval::singleton(8.0),
    ]);
    let lhs = IntervalMatrix::from_inner(lhs_storage.as_inner().fixed_view::<2, 2>(0, 0));
    let rhs = IntervalMatrix::from_inner(rhs_storage.as_inner().fixed_rows::<2>(0));

    let solution = lhs.solve_with(&rhs, &GaussianEliminationSolver);
    assert!(
        solution
            .as_ref()
            .is_some_and(|solution| solution[0].contains(2.0) && solution[1].contains(3.0))
    );

    let inverse = lhs.inverse_with(&GaussianEliminationSolver);
    assert!(
        inverse
            .as_ref()
            .is_some_and(|inverse| inverse[(0, 0)].contains(0.5) && inverse[(1, 1)].contains(0.5))
    );
}

#[test]
fn weighted_vector_norm_uses_positive_component_weights() {
    let vector = SIntervalVector::<Interval, 2>::from_column_slice(&[
        Interval::new(-2.0, 1.0),
        Interval::new(3.0, 4.0),
    ]);
    let weights = nalgebra::SVector::<f64, 2>::from_column_slice(&[4.0, 2.0]);

    assert_eq!(vector.v_norm(&weights), 2.0);
}

#[test]
fn matrix_display_forwards_precision_and_measures_interval_widths() {
    let matrix = SIntervalMatrix::<Interval, 2, 1>::from_column_slice(&[
        Interval::new(-1.25, 2.5),
        Interval::new(-100.0, 200.0),
    ]);

    let rendered = std::format!("{matrix:.2}");
    assert!(rendered.contains("[-1.25,2.50]"));
    assert!(rendered.contains("[-100.00,200.00]"));
    let row_widths: std::vec::Vec<_> = rendered
        .lines()
        .filter(|line| line.contains('│'))
        .map(str::len)
        .collect();
    assert!(row_widths.windows(2).all(|widths| widths[0] == widths[1]));
    assert!(std::format!("{matrix:.1e}").contains('e'));
    assert!(std::format!("{matrix:x}").contains("0x"));
}

#[test]
fn structural_operations_match_matrix_layout() {
    let one = Interval::singleton(1.0);
    let two = Interval::singleton(2.0);
    let three = Interval::singleton(3.0);
    let four = Interval::singleton(4.0);
    let matrix = SIntervalMatrix::<Interval, 2, 2>::from_row_slice(&[one, two, three, four]);

    let expected_transpose =
        SIntervalMatrix::<Interval, 2, 2>::from_row_slice(&[one, three, two, four]);
    assert_eq!(matrix.transpose(), expected_transpose);

    let doubled = matrix.zip_map(&matrix, |lhs, rhs| lhs + rhs);
    assert!(doubled.all(|entry| entry.inf() >= 2.0));
    assert!(doubled.any(|entry| entry.contains(8.0)));

    let mut transformed = matrix;
    transformed.apply(|entry| *entry += 1.0);
    transformed.swap_rows(0, 1);
    transformed.swap_columns(0, 1);
    transformed.transpose_mut();
    transformed.fill(Interval::ZERO);
    assert!(transformed.all(|entry| entry == Interval::ZERO));
}

#[test]
fn component_and_scalar_arithmetic_are_complete() {
    let one = Interval::singleton(1.0);
    let two = Interval::singleton(2.0);
    let three = Interval::singleton(3.0);
    let four = Interval::singleton(4.0);
    let lhs = SIntervalMatrix::<Interval, 2, 2>::from_row_slice(&[one, two, three, four]);
    let rhs = SIntervalMatrix::<Interval, 2, 2>::from_row_slice(&[four, three, two, one]);

    let sum = SIntervalMatrix::<Interval, 2, 2>::from_element(Interval::singleton(5.0));
    assert_eq!(&lhs + &rhs, sum);
    assert_eq!(lhs + &rhs, sum);
    assert_eq!(&lhs + rhs, sum);
    assert_eq!(lhs + rhs, sum);

    let difference = SIntervalMatrix::<Interval, 2, 2>::from_row_slice(&[
        Interval::singleton(-3.0),
        Interval::singleton(-1.0),
        Interval::singleton(1.0),
        Interval::singleton(3.0),
    ]);
    assert_eq!(&lhs - &rhs, difference);
    assert_eq!(lhs - &rhs, difference);
    assert_eq!(&lhs - rhs, difference);
    assert_eq!(lhs - rhs, difference);

    let product = SIntervalMatrix::<Interval, 2, 2>::from_row_slice(&[
        four,
        Interval::singleton(6.0),
        Interval::singleton(6.0),
        four,
    ]);
    assert_eq!(lhs.component_mul(&rhs), product);
    assert_eq!(product.component_div(&rhs), lhs);

    let scalar = Interval::singleton(2.0);
    assert_eq!(lhs.add_scalar(scalar)[(0, 0)], Interval::singleton(3.0));
    assert_eq!(lhs / scalar, lhs.unscale(2.0));
    assert_eq!(&lhs * scalar, lhs.scale(2.0));
    assert_eq!((-&lhs)[(0, 1)], Interval::singleton(-2.0));
    assert_eq!((-lhs)[(1, 0)], Interval::singleton(-3.0));

    let mut real_scalar = lhs;
    real_scalar.add_scalar_mut(Interval::singleton(1.0));
    real_scalar.add_scalar_mut(Interval::singleton(-1.0));
    real_scalar.scale_mut(2.0);
    real_scalar.unscale_mut(2.0);
    assert_eq!(real_scalar, lhs);
}

#[test]
fn assignment_and_owned_matrix_multiplication_match_borrowed_operations() {
    let one = Interval::singleton(1.0);
    let two = Interval::singleton(2.0);
    let three = Interval::singleton(3.0);
    let four = Interval::singleton(4.0);
    let lhs = SIntervalMatrix::<Interval, 2, 2>::from_row_slice(&[one, two, three, four]);
    let rhs = SIntervalMatrix::<Interval, 2, 2>::from_row_slice(&[four, three, two, one]);
    let identity = SIntervalMatrix::<Interval, 2, 2>::identity();

    let borrowed_product = &lhs * &rhs;
    assert_eq!(lhs * &rhs, borrowed_product);
    assert_eq!(&lhs * rhs, borrowed_product);
    assert_eq!(lhs * rhs, borrowed_product);

    let mut assigned = lhs;
    assigned += &rhs;
    assigned -= rhs;
    assert_eq!(assigned, lhs);
    assigned *= Interval::singleton(2.0);
    assigned /= Interval::singleton(2.0);
    assigned *= &identity;
    assert_eq!(assigned, lhs);

    assigned.component_mul_assign(&rhs);
    assigned.component_div_assign(&rhs);
    assert_eq!(assigned, lhs);
}

#[cfg(feature = "alloc")]
#[test]
#[should_panic(expected = "interval matrix addition dimension mismatch")]
fn component_arithmetic_rejects_dynamic_dimension_mismatches() {
    use maryada::DIntervalMatrix;

    let lhs = DIntervalMatrix::<Interval>::zeros(1, 2);
    let rhs = DIntervalMatrix::<Interval>::zeros(2, 1);
    let _ = &lhs + &rhs;
}

#[cfg(feature = "alloc")]
#[test]
fn component_arithmetic_accepts_compatible_static_and_dynamic_dimensions() {
    use maryada::DIntervalMatrix;

    let one = Interval::singleton(1.0);
    let static_matrix = SIntervalMatrix::<Interval, 1, 2>::from_element(one);
    let dynamic_matrix = DIntervalMatrix::<Interval>::from_element(1, 2, one);

    let sum = &static_matrix + &dynamic_matrix;
    assert!(sum.iter().all(|entry| *entry == Interval::singleton(2.0)));

    let mut assigned = static_matrix;
    assigned += &dynamic_matrix;
    assigned -= dynamic_matrix;
    assert_eq!(assigned, static_matrix);
}

#[test]
fn weighted_norm_enclosure_contains_the_solution() {
    let lhs = SIntervalMatrix::<Interval, 2, 2>::identity();
    let rhs = SIntervalVector::<Interval, 2>::from_column_slice(&[
        Interval::singleton(2.0),
        Interval::singleton(-3.0),
    ]);

    let enclosure = lhs.initial_enclosure_v_norm(&rhs);

    assert!(
        enclosure
            .as_ref()
            .is_some_and(|enclosure| enclosure[0].contains(2.0) && enclosure[1].contains(-3.0))
    );
}

#[test]
fn infinity_norm_enclosure_contains_the_solution() {
    let lhs =
        SIntervalMatrix::<Interval, 2, 2>::from_diagonal(&SIntervalVector::from_column_slice(&[
            Interval::new(0.9, 1.1),
            Interval::new(0.9, 1.1),
        ]));
    let rhs = SIntervalVector::<Interval, 2>::from_column_slice(&[
        Interval::singleton(2.0),
        Interval::singleton(-3.0),
    ]);

    let enclosure = lhs.initial_enclosure_inf_norm(&rhs);

    assert!(
        enclosure
            .as_ref()
            .is_some_and(|enclosure| enclosure[0].contains(2.0) && enclosure[1].contains(-3.0))
    );
}

#[test]
fn infinity_norm_enclosure_rejects_noncontraction() {
    let lhs = SIntervalMatrix::<Interval, 1, 1>::from_element(Interval::new(-1.0, 1.0));
    let rhs = SIntervalVector::<Interval, 1>::from_element(Interval::ONE);

    assert!(lhs.initial_enclosure_inf_norm(&rhs).is_none());
}

#[test]
fn stopping_tolerance_can_be_derived_from_the_system_matrix() {
    let lhs = SIntervalMatrix::<Interval, 2, 2>::from_row_slice(&[
        Interval::new(0.0, 2.0),
        Interval::new(-2.0, 2.0),
        Interval::new(-3.0, 3.0),
        Interval::new(1.0, 9.0),
    ]);

    let tolerance = StoppingTolerance::from_matrix(&lhs);

    assert_eq!(tolerance.epsilon(), 2e-5);
}

#[test]
fn krawczyk_solver_uses_derived_defaults() {
    let lhs =
        SIntervalMatrix::<Interval, 2, 2>::from_diagonal(&SIntervalVector::from_column_slice(&[
            Interval::new(0.99, 1.01),
            Interval::new(0.99, 1.01),
        ]));
    let rhs = SIntervalVector::<Interval, 2>::from_column_slice(&[
        Interval::singleton(2.0),
        Interval::singleton(-3.0),
    ]);
    let solver = Preconditioned::new(KrawczykSolver::new(&lhs), Preconditioner::default());

    let solution = solver.solve(&lhs, &rhs);

    assert!(
        solution
            .as_ref()
            .is_some_and(|solution| solution[0].contains(2.0) && solution[1].contains(-3.0))
    );
}

#[test]
fn encloses_solution_of_h_matrix_system() {
    let lhs = SIntervalMatrix::<Interval, 2, 2>::from_row_slice(&[
        Interval::new(1.9, 2.1),
        Interval::new(-0.01, 0.01),
        Interval::new(-0.01, 0.01),
        Interval::new(2.9, 3.1),
    ]);
    let rhs = SIntervalVector::<Interval, 2>::from_column_slice(&[
        Interval::singleton(4.0),
        Interval::singleton(9.0),
    ]);

    let solution = HansenBliekRohnSolver.solve(&lhs, &rhs);

    assert!(
        solution
            .as_ref()
            .is_some_and(|solution| { solution[0].contains(2.0) && solution[1].contains(3.0) })
    );
}

#[test]
fn rejects_system_without_h_matrix() {
    let lhs = SIntervalMatrix::<Interval, 1, 1>::from_element(Interval::new(-1.0, 1.0));
    let rhs = SIntervalVector::<Interval, 1>::from_element(Interval::singleton(1.0));

    assert!(HansenBliekRohnSolver.solve(&lhs, &rhs).is_none());
}

#[cfg(feature = "alloc")]
#[test]
fn supports_dynamic_dimensions() {
    use maryada::{DIntervalMatrix, DIntervalVector};

    let lhs = DIntervalMatrix::<Interval>::from_row_slice(
        2,
        2,
        &[
            Interval::singleton(2.0),
            Interval::ZERO,
            Interval::ZERO,
            Interval::singleton(3.0),
        ],
    );
    let rhs = DIntervalVector::<Interval>::from_column_slice(&[
        Interval::singleton(4.0),
        Interval::singleton(9.0),
    ]);

    let solution = HansenBliekRohnSolver.solve(&lhs, &rhs);

    assert!(
        solution
            .as_ref()
            .is_some_and(|solution| { solution[0].contains(2.0) && solution[1].contains(3.0) })
    );
}

#[test]
fn builder_methods_configure_the_solver() {
    let solver = EpsilonInflationSolver::new()
        .with_relative_inflation(0.05)
        .with_absolute_inflation(1e-16)
        .with_max_iterations(40);

    assert_eq!(solver.relative_inflation(), 0.05);
    assert_eq!(solver.absolute_inflation(), 1e-16);
    assert_eq!(solver.max_iterations(), 40);
}

#[test]
fn invalid_inflation_parameters_are_rejected() {
    let lhs = SIntervalMatrix::<Interval, 1, 1>::identity();
    let rhs = SIntervalVector::<Interval, 1>::from_element(Interval::ONE);

    assert!(
        EpsilonInflationSolver::new()
            .with_relative_inflation(-0.1)
            .solve(&lhs, &rhs)
            .is_none()
    );
    assert!(
        EpsilonInflationSolver::new()
            .with_absolute_inflation(f64::NAN)
            .solve(&lhs, &rhs)
            .is_none()
    );
}
