#![allow(clippy::arithmetic_side_effects)]

use nalgebra::{
    Const, DefaultAllocator, Dim, DimMin, OMatrix, Scalar, Storage, allocator::Allocator,
};

use crate::{Interval, IntervalOps, OIntervalMatrix, Solver};

use super::IntervalMatrix;

/// Strategy used to left-precondition an interval linear system.
#[derive(Default)]
pub enum Preconditioner<D>
where
    D: Dim,
    DefaultAllocator: Allocator<D, D>,
{
    /// Uses the inverse of the midpoint matrix.
    #[default]
    InverseMidpoint,
    /// Uses the inverse of the midpoint diagonal.
    InverseDiagonalMidpoint,
    /// Uses a caller-provided real preconditioner matrix.
    Custom(OMatrix<f64, D, D>),
}

impl<D> Preconditioner<D>
where
    D: Dim,
    DefaultAllocator: Allocator<D, D>,
{
    /// Uses `matrix` as the real left-preconditioner.
    #[must_use]
    pub const fn custom(matrix: OMatrix<f64, D, D>) -> Self {
        Self::Custom(matrix)
    }

    /// Computes this preconditioner's point matrix as an interval matrix.
    #[must_use]
    pub fn matrix<T, S>(&self, lhs: &IntervalMatrix<T, D, D, S>) -> Option<OIntervalMatrix<T, D, D>>
    where
        T: IntervalOps + Scalar,
        S: Storage<T, D, D>,
        DefaultAllocator: Allocator<D, D> + Allocator<D>,
    {
        Some(match self {
            Self::InverseMidpoint => OIntervalMatrix::from(lhs.mid().try_inverse()?),
            Self::InverseDiagonalMidpoint => {
                OIntervalMatrix::from(OMatrix::from_diagonal(&lhs.mid().diagonal()).try_inverse()?)
            }
            Self::Custom(matrix) => {
                if matrix.shape() != lhs.shape() || matrix.iter().any(|entry| !entry.is_finite()) {
                    return None;
                }
                OIntervalMatrix::from_singletons(matrix)
            }
        })
    }

    /// Automatically selects a suitable preconditioner.
    #[must_use]
    pub fn auto<T, S>(lhs: &IntervalMatrix<T, D, D, S>) -> Option<Self>
    where
        T: IntervalOps + Scalar,
        S: Storage<T, D, D>,
        D: DimMin<D, Output = D>,
        DefaultAllocator: Allocator<D, D> + Allocator<D>,
    {
        if lhs.is_m_matrix() {
            return None;
        }
        let (n, _) = lhs.shape();
        for i in 0..n {
            let mut sum = Interval::ZERO;
            for j in 0..n {
                if i != j {
                    sum += Interval::singleton(lhs[(i, j)].mag());
                }
            }
            if lhs[(i, i)].mig() <= sum.inf() {
                return Some(Self::InverseMidpoint);
            }
        }
        None
    }
}

/// A solver adapter that left-preconditions a system before solving it.
pub struct Preconditioned<D, S>
where
    D: Dim,
    DefaultAllocator: Allocator<D, D>,
{
    solver: S,
    preconditioner: Option<Preconditioner<D>>,
}

impl<D, S> Preconditioned<D, S>
where
    D: Dim,
    DefaultAllocator: Allocator<D, D>,
{
    /// Wraps `solver` with the selected preconditioning strategy.
    #[must_use]
    pub const fn new(solver: S, preconditioner: Preconditioner<D>) -> Self {
        Self {
            solver,
            preconditioner: Some(preconditioner),
        }
    }

    /// Wraps `solver` with inverse-midpoint preconditioning.
    #[must_use]
    pub const fn inverse_midpoint(solver: S) -> Self {
        Self {
            solver,
            preconditioner: Some(Preconditioner::InverseMidpoint),
        }
    }

    /// Wraps `solver` with inverse-diagonal-midpoint preconditioning.
    #[must_use]
    pub const fn inverse_diagonal_midpoint(solver: S) -> Self {
        Self {
            solver,
            preconditioner: Some(Preconditioner::InverseDiagonalMidpoint),
        }
    }

    /// Wraps `solver` with a caller-provided real preconditioner matrix.
    #[must_use]
    pub const fn custom(solver: S, matrix: OMatrix<f64, D, D>) -> Self {
        Self {
            solver,
            preconditioner: Some(Preconditioner::Custom(matrix)),
        }
    }

    /// Automatically selects a suitable preconditioner.
    #[must_use]
    pub fn auto<T, SA>(solver: S, lhs: &IntervalMatrix<T, D, D, SA>) -> Self
    where
        T: IntervalOps + Scalar,
        SA: Storage<T, D, D>,
        D: DimMin<D, Output = D>,
        DefaultAllocator: Allocator<D, D> + Allocator<D>,
    {
        Self {
            solver,
            preconditioner: Preconditioner::auto(lhs),
        }
    }

    /// Replaces the selected preconditioning strategy.
    #[must_use]
    pub fn with_preconditioner(mut self, preconditioner: Preconditioner<D>) -> Self {
        self.preconditioner = Some(preconditioner);
        self
    }

    /// Borrows the wrapped solver.
    #[must_use]
    pub const fn solver(&self) -> &S {
        &self.solver
    }

    /// Returns the selected preconditioning strategy.
    #[must_use]
    pub const fn preconditioner(&self) -> Option<&Preconditioner<D>> {
        self.preconditioner.as_ref()
    }
}

impl<T, D, S> Solver<T, D> for Preconditioned<D, S>
where
    T: IntervalOps + Scalar,
    D: Dim,
    S: Solver<T, D>,
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    fn solve<SA, SB>(
        &self,
        lhs: &IntervalMatrix<T, D, D, SA>,
        rhs: &IntervalMatrix<T, D, Const<1>, SB>,
    ) -> Option<OIntervalMatrix<T, D, Const<1>>>
    where
        SA: Storage<T, D, D>,
        SB: Storage<T, D, Const<1>>,
    {
        if let Some(preconditioner) = &self.preconditioner {
            let c = preconditioner.matrix(lhs)?;
            self.solver.solve(&(&c * lhs), &(&c * rhs))
        } else {
            self.solver.solve(lhs, rhs)
        }
    }
}
