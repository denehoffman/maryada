#![allow(clippy::arithmetic_side_effects, clippy::indexing_slicing)]

use nalgebra::{
    Const, DefaultAllocator, Dim, DimMin, Scalar, Storage,
    allocator::Allocator,
    constraint::{AreMultipliable, ShapeConstraint},
};

use crate::{EpsilonInflationSolver, IntervalOps};

use super::{IntervalMatrix, OIntervalMatrix, OIntervalVector, SolveError, Solver};

/// Hansen–Bliek–Rohn verified solver for interval systems with an H-matrix.
pub struct HansenBliekRohnSolver;
impl<T, D> Solver<T, D> for HansenBliekRohnSolver
where
    T: IntervalOps + Scalar,
    D: Dim + DimMin<D, Output = D>,
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
    ShapeConstraint: AreMultipliable<D, D, D, D> + AreMultipliable<D, D, D, Const<1>>,
{
    fn try_solve<SA, SB>(
        &self,
        lhs: &IntervalMatrix<T, D, D, SA>,
        rhs: &IntervalMatrix<T, D, Const<1>, SB>,
    ) -> Result<OIntervalVector<T, D>, SolveError>
    where
        SA: Storage<T, D, D>,
        SB: Storage<T, D, Const<1>>,
    {
        let result = (|| -> Option<OIntervalVector<T, D>> {
            if !lhs.is_square()
                || lhs.is_empty()
                || lhs.nrows() != rhs.nrows()
                || lhs.has_empty_entries()
                || lhs.has_nai_entries()
                || rhs.has_empty_entries()
                || rhs.has_nai_entries()
                || !lhs.is_h_matrix()
            {
                return None;
            }

            let a_comp = lhs.comparison_matrix()?;
            let (dim, _) = rhs.shape_generic();
            let a_comp_interval: OIntervalMatrix<T, D, D> =
                OIntervalMatrix::from_singletons(&a_comp);
            let a_comp_inv = EpsilonInflationSolver::default()
                .try_inverse(&a_comp_interval)
                .ok()?;

            let b_mag = rhs.mag();
            let b_mag = OIntervalVector::from_singletons(&b_mag);
            let u = &a_comp_inv * &b_mag;
            let d = a_comp_inv.diagonal();
            if d.any(|entry| entry.inf() <= 0.0) {
                return None;
            }

            let a_diag = a_comp_interval.diagonal();
            let d_recip = d.map(IntervalOps::recip);
            let alpha_interval = &a_diag - &d_recip;
            let alpha = alpha_interval.mag();
            let scaled_u = u.component_div(&d);
            let beta_interval = &scaled_u - &b_mag;
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
        })();
        result.ok_or(SolveError::CertificationFailed { iterations: 0 })
    }
}
