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
        lhs: &IntervalMatrix<T, D, D, SA>,
        rhs: &IntervalMatrix<T, D, Const<1>, SB>,
    ) -> Option<OIntervalVector<T, D>> {
        if !lhs.is_square()
            || lhs.is_empty()
            || lhs.nrows() != rhs.nrows()
            || lhs.has_invalid_entries()
            || rhs.has_invalid_entries()
            || !lhs.is_h_matrix()
        {
            return None;
        }

        let a_comp = lhs.comparison_matrix()?;
        let (dim, _) = rhs.shape_generic();
        let a_comp_interval = OIntervalMatrix::from_singletons(&a_comp);
        let identity = OIntervalMatrix::<T, D, D>::identity_generic(dim);
        let a_comp_inv =
            ei::EpsilonInflation::default().solve_matrix(&a_comp_interval, &identity)?;

        let b_mag = rhs.mag();
        let b_mag = OIntervalVector::from_singletons(&b_mag);
        let u = &a_comp_inv * &b_mag;
        let d = a_comp_inv.diagonal();
        if d.iter().any(|entry| entry.inf() <= 0.0) {
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

        for i in 0..lhs.nrows() {
            let denominator = lhs[(i, i)] + T::new(-alpha[i], alpha[i]);
            if denominator.mig() == 0.0 {
                return None;
            }
        }

        let x = OIntervalVector::from_fn_generic(dim, Const::<1>, |i, _| {
            (rhs[i] + T::new(-beta[i], beta[i])) / (lhs[(i, i)] + T::new(-alpha[i], alpha[i]))
        });
        Some(x)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Interval, IntervalOps};

    use super::*;

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

        let solution = HBR.solve(&lhs, &rhs);

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

        assert!(HBR.solve(&lhs, &rhs).is_none());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn supports_dynamic_dimensions() {
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

        let solution = HBR.solve(&lhs, &rhs);

        assert!(
            solution
                .as_ref()
                .is_some_and(|solution| { solution[0].contains(2.0) && solution[1].contains(3.0) })
        );
    }
}
