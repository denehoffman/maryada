#![allow(clippy::arithmetic_side_effects, clippy::indexing_slicing)]

use nalgebra::{
    Const, DefaultAllocator, Dim, DimAdd, DimSum, Scalar, Storage, StorageMut,
    allocator::{Allocator, Reallocator},
};

use crate::IntervalOps;

use super::{IntervalMatrix, OIntervalVector, Solver};

impl<T, R, C, S> IntervalMatrix<T, R, C, S>
where
    T: IntervalOps + Scalar,
    R: Dim,
    C: Dim,
    S: StorageMut<T, R, C> + Clone,
{
    /// Computes a verified forward elimination.
    ///
    /// Returns `None` if no pivot can be certified as nonzero.
    #[must_use]
    pub fn forward_elimination(&self) -> Option<Self> {
        // Algorithm 5.9
        let mut a = self.as_inner().clone();
        let (n, m) = a.shape();
        for i in 0..n.min(m) {
            let j = (i..n).max_by(|&k, &l| a[(k, i)].mig().total_cmp(&a[(l, i)].mig()))?;
            if a[(j, i)].mig() == 0.0 {
                return None;
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
        Some(Self::from_inner(a))
    }
}

/// Verified interval Gaussian-elimination solver.
pub struct GaussianElimination;

impl<T, D, SA, SB> Solver<T, D, SA, SB> for GaussianElimination
where
    T: IntervalOps + Scalar,
    D: Dim + DimAdd<Const<1>>,
    SA: Storage<T, D, D>,
    SB: Storage<T, D, Const<1>>,
    DefaultAllocator: Allocator<D, D>
        + Allocator<D>
        + Allocator<D, DimSum<D, Const<1>>>
        + Reallocator<T, D, D, D, DimSum<D, Const<1>>>,
{
    fn solve(
        &self,
        lhs: &IntervalMatrix<T, D, D, SA>,
        rhs: &IntervalMatrix<T, D, Const<1>, SB>,
    ) -> Option<OIntervalVector<T, D>> {
        let n = rhs.nrows();
        if lhs.nrows() != n || lhs.ncols() != n {
            return None;
        }
        let mut ab = lhs.as_inner().clone_owned().insert_column(n, T::ZERO);
        for i in 0..n {
            ab[(i, n)] = rhs[i];
        }
        let ab = IntervalMatrix::from_inner(ab).forward_elimination()?;
        let ab = ab.into_inner();
        let mut x = rhs.as_inner().clone_owned();
        for i in (0..n).rev() {
            let mut value = ab[(i, n)];
            for j in i + 1..n {
                value -= ab[(i, j)] * x[j];
            }
            x[i] = value * ab[(i, i)].recip();
        }
        Some(IntervalMatrix::from_inner(x))
    }
}
