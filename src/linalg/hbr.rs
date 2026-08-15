use super::*;

pub struct HBR;
impl<T, D, SA, SB> Solver<T, D, SA, SB> for HBR
where
    T: IntervalOps + Scalar,
    D: Dim + DimMin<D, Output = D>,
    SA: Storage<T, D, D>,
    SB: Storage<T, D, Const<1>>,
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
    ShapeConstraint: AreMultipliable<D, D, D, D> + AreMultipliable<D, D, D, Const<1>>,
{
    fn solve(
        &self,
        lhs: &IntervalMatrix<Matrix<T, D, D, SA>>,
        rhs: &IntervalMatrix<Matrix<T, D, Const<1>, SB>>,
    ) -> Option<IntervalMatrix<OMatrix<T, D, Const<1>>>> {
        if !lhs.0.is_square()
            || lhs.0.nrows() == 0
            || lhs.0.nrows() != rhs.0.nrows()
            || lhs.has_invalid_entries()
            || rhs.has_invalid_entries()
            || !lhs.is_h_matrix()
        {
            return None;
        }

        let a_comp = lhs.comparison_matrix()?;
        let (dim, _) = rhs.0.shape_generic();
        let a_comp_interval: IntervalMatrix<OMatrix<T, D, D>> = a_comp.into();
        let identity = OMatrix::<f64, D, D>::identity_generic(dim, dim);
        let identity: IntervalMatrix<OMatrix<T, D, D>> = identity.into();
        let a_comp_inv =
            ei::EpsilonInflation::default().solve_matrix(&a_comp_interval, &identity)?;

        let b_mag = rhs.mag();
        let b_mag: IntervalMatrix<OMatrix<T, D, Const<1>>> = b_mag.into();
        let u = &a_comp_inv * &b_mag;
        let d = a_comp_inv.diagonal();
        if d.as_inner().iter().any(|entry| entry.inf() <= 0.0) {
            return None;
        }

        let a_diag = a_comp_interval.diagonal();
        let d_recip = d.map(IntervalOps::recip);
        let alpha_interval = a_diag.component_sub(&d_recip);
        let alpha = alpha_interval.mag();
        let scaled_u = u.component_div(&d);
        let beta_interval = scaled_u.component_sub(&b_mag);
        let beta = beta_interval.mag();
        if alpha
            .iter()
            .chain(beta.iter())
            .any(|entry| !entry.is_finite())
        {
            return None;
        }

        for i in 0..lhs.0.nrows() {
            let denominator = lhs.0[(i, i)] + T::new(-alpha[i], alpha[i]);
            if denominator.mig() == 0.0 {
                return None;
            }
        }

        let x = Matrix::from_fn_generic(dim, Const::<1>, |i, _| {
            (rhs.0[i] + T::new(-beta[i], beta[i])) / (lhs.0[(i, i)] + T::new(-alpha[i], alpha[i]))
        });
        Some(IntervalMatrix::from_inner(x))
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{SMatrix, SVector};

    use crate::{Interval, IntervalOps};

    use super::*;

    #[test]
    fn encloses_solution_of_h_matrix_system() {
        let lhs = IntervalMatrix::from_inner(SMatrix::<Interval, 2, 2>::from_row_slice(&[
            Interval::new(1.9, 2.1),
            Interval::new(-0.01, 0.01),
            Interval::new(-0.01, 0.01),
            Interval::new(2.9, 3.1),
        ]));
        let rhs = IntervalMatrix::from_inner(SVector::<Interval, 2>::from_row_slice(&[
            Interval::singleton(4.0),
            Interval::singleton(9.0),
        ]));

        let solution = HBR.solve(&lhs, &rhs);

        assert!(solution.as_ref().is_some_and(|solution| {
            solution.as_inner()[0].contains(2.0) && solution.as_inner()[1].contains(3.0)
        }));
    }

    #[test]
    fn rejects_system_without_h_matrix() {
        let lhs = IntervalMatrix::from_inner(SMatrix::<Interval, 1, 1>::from_element(
            Interval::new(-1.0, 1.0),
        ));
        let rhs = IntervalMatrix::from_inner(SVector::<Interval, 1>::from_element(
            Interval::singleton(1.0),
        ));

        assert!(HBR.solve(&lhs, &rhs).is_none());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn supports_dynamic_dimensions() {
        use nalgebra::{DMatrix, DVector};

        let lhs = IntervalMatrix::from_inner(DMatrix::<Interval>::from_row_slice(
            2,
            2,
            &[
                Interval::singleton(2.0),
                Interval::ZERO,
                Interval::ZERO,
                Interval::singleton(3.0),
            ],
        ));
        let rhs = IntervalMatrix::from_inner(DVector::<Interval>::from_row_slice(&[
            Interval::singleton(4.0),
            Interval::singleton(9.0),
        ]));

        let solution = HBR.solve(&lhs, &rhs);

        assert!(solution.as_ref().is_some_and(|solution| {
            solution.as_inner()[0].contains(2.0) && solution.as_inner()[1].contains(3.0)
        }));
    }
}
