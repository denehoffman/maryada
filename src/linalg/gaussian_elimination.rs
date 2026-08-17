#![allow(clippy::arithmetic_side_effects, clippy::indexing_slicing)]

use nalgebra::{
    Const, DefaultAllocator, Dim, DimAdd, DimSum, Scalar, Storage, StorageMut,
    allocator::{Allocator, Reallocator},
};

use crate::IntervalOps;

use super::{IntervalMatrix, OIntervalVector, SolveError, Solver};

impl<T, R, C, S> IntervalMatrix<T, R, C, S>
where
    T: IntervalOps + Scalar,
    R: Dim,
    C: Dim,
    S: StorageMut<T, R, C> + Clone,
{
    /// Computes a verified forward elimination with diagnostics.
    ///
    /// # Errors
    ///
    /// Returns [`SolveError::PivotNotCertified`] when a pivot interval cannot
    /// be certified as nonzero, or [`SolveError::InvalidInput`] for invalid
    /// entries.
    pub fn try_forward_elimination(&self) -> Result<Self, SolveError> {
        if self.has_empty_entries()
            || self.has_nai_entries()
            || self.iter().any(|entry| !entry.is_bounded())
        {
            return Err(SolveError::InvalidInput);
        }
        // Algorithm 5.9
        let mut a = self.as_inner().clone();
        let (n, m) = a.shape();
        for i in 0..n.min(m) {
            let j = (i..n)
                .max_by(|&k, &l| a[(k, i)].mig().total_cmp(&a[(l, i)].mig()))
                .ok_or(SolveError::PivotNotCertified { index: i })?;
            if a[(j, i)].mig() == 0.0 {
                return Err(SolveError::PivotNotCertified { index: i });
            }
            if j != i {
                a.swap_rows(i, j);
            }
            let a_ii_recip = a[(i, i)].recip();
            for j in i + 1..n {
                a[(j, i)] *= a_ii_recip;
            }
            for k in i + 1..m {
                for j in i + 1..n {
                    let correction = a[(j, i)] * a[(i, k)];
                    a[(j, k)] -= correction;
                }
            }
            for j in i + 1..n {
                a[(j, i)] = T::ZERO;
            }
        }
        Ok(Self::from_inner(a))
    }

    /// Computes a verified forward elimination.
    #[must_use]
    pub fn forward_elimination(&self) -> Option<Self> {
        self.try_forward_elimination().ok()
    }
}

/// Verified interval Gaussian-elimination solver.
pub struct GaussianEliminationSolver;

impl<T, D> Solver<T, D> for GaussianEliminationSolver
where
    T: IntervalOps + Scalar,
    D: Dim + DimAdd<Const<1>>,
    DefaultAllocator: Allocator<D, D>
        + Allocator<D>
        + Allocator<D, DimSum<D, Const<1>>>
        + Reallocator<T, D, D, D, DimSum<D, Const<1>>>,
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
        let n = rhs.nrows();
        if lhs.nrows() != n || lhs.ncols() != n || n == 0 {
            return Err(SolveError::InvalidSystem);
        }
        if lhs.has_empty_entries()
            || lhs.has_nai_entries()
            || rhs.has_empty_entries()
            || rhs.has_nai_entries()
            || lhs.iter().any(|entry| !entry.is_bounded())
            || rhs.iter().any(|entry| !entry.is_bounded())
        {
            return Err(SolveError::InvalidInput);
        }
        let mut ab = lhs.as_inner().clone_owned().insert_column(n, T::ZERO);
        for i in 0..n {
            ab[(i, n)] = rhs[i];
        }
        let ab = IntervalMatrix::from_inner(ab).try_forward_elimination()?;
        let ab = ab.into_inner();
        let mut x = rhs.as_inner().clone_owned();
        for i in (0..n).rev() {
            let mut value = ab[(i, n)];
            for j in i + 1..n {
                value -= ab[(i, j)] * x[j];
            }
            x[i] = value * ab[(i, i)].recip();
        }
        Ok(IntervalMatrix::from_inner(x))
    }
}
