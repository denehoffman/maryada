#![allow(clippy::arithmetic_side_effects)]

use nalgebra::{
    ComplexField, Const, DefaultAllocator, Dim, OMatrix, Scalar, Storage, allocator::Allocator,
};

use crate::OIntervalMatrix;

use super::{EnclosureScalar, IntervalMatrix, SolveError, Solver};

/// Strategy used to left-precondition an interval linear system.
#[derive(Default)]
pub enum Preconditioner<D, P = f64>
where
    D: Dim,
    P: Scalar,
    DefaultAllocator: Allocator<D, D>,
{
    /// Uses the inverse of the midpoint matrix.
    #[default]
    InverseMidpoint,
    /// Uses the inverse of the midpoint diagonal.
    InverseDiagonalMidpoint,
    /// Uses a caller-provided point preconditioner matrix.
    ///
    /// The point scalar is selected by `P`; use the complex point scalar for
    /// complex interval systems.
    Custom(OMatrix<P, D, D>),
}

impl<D, P> Preconditioner<D, P>
where
    D: Dim,
    P: ComplexField + Copy,
    DefaultAllocator: Allocator<D, D>,
{
    /// Uses `matrix` as the point left-preconditioner.
    #[must_use]
    pub const fn custom(matrix: OMatrix<P, D, D>) -> Self {
        Self::Custom(matrix)
    }

    /// Computes this preconditioner's point matrix as an interval matrix with
    /// diagnostics.
    ///
    /// # Errors
    ///
    /// Returns [`SolveError::SingularMidpoint`] when a midpoint inverse cannot
    /// be formed, or [`SolveError::InvalidInput`] for invalid entries or a
    /// malformed custom preconditioner.
    #[allow(clippy::indexing_slicing)]
    pub fn try_matrix<T, S>(
        &self,
        lhs: &IntervalMatrix<T, D, D, S>,
    ) -> Result<OIntervalMatrix<T, D, D>, SolveError>
    where
        T: EnclosureScalar<Midpoint = P> + Scalar,
        S: Storage<T, D, D>,
        DefaultAllocator: Allocator<D, D> + Allocator<D>,
    {
        if lhs.is_empty() {
            return Err(SolveError::InvalidSystem);
        }
        if lhs.has_empty_entries()
            || lhs.has_nai_entries()
            || lhs.iter().any(|entry| !entry.is_bounded())
        {
            return Err(SolveError::InvalidInput);
        }
        Ok(match self {
            Self::InverseMidpoint => {
                let (dim, _) = lhs.shape_generic();
                let midpoint =
                    OMatrix::<P, D, D>::from_fn_generic(dim, dim, |i, j| lhs[(i, j)].mid());
                if midpoint.iter().any(|entry| !entry.is_finite()) {
                    return Err(SolveError::InvalidInput);
                }
                let inverse = midpoint.try_inverse().ok_or(SolveError::SingularMidpoint)?;
                OIntervalMatrix::from_fn_generic(dim, dim, |i, j| T::from_midpoint(inverse[(i, j)]))
            }
            Self::InverseDiagonalMidpoint => {
                let (dim, _) = lhs.shape_generic();
                let midpoint = OMatrix::<P, D, D>::from_fn_generic(dim, dim, |i, j| {
                    if i == j { lhs[(i, j)].mid() } else { P::zero() }
                });
                if midpoint.iter().any(|entry| !entry.is_finite()) {
                    return Err(SolveError::InvalidInput);
                }
                let inverse = midpoint.try_inverse().ok_or(SolveError::SingularMidpoint)?;
                OIntervalMatrix::from_fn_generic(dim, dim, |i, j| T::from_midpoint(inverse[(i, j)]))
            }
            Self::Custom(matrix) => {
                if matrix.shape() != lhs.shape() || matrix.iter().any(|entry| !entry.is_finite()) {
                    return Err(SolveError::InvalidInput);
                }
                let (dim, _) = lhs.shape_generic();
                OIntervalMatrix::from_fn_generic(dim, dim, |i, j| T::from_midpoint(matrix[(i, j)]))
            }
        })
    }

    /// Computes this preconditioner's point matrix as an interval matrix.
    #[must_use]
    pub fn matrix<T, S>(&self, lhs: &IntervalMatrix<T, D, D, S>) -> Option<OIntervalMatrix<T, D, D>>
    where
        T: EnclosureScalar<Midpoint = P> + Scalar,
        S: Storage<T, D, D>,
        DefaultAllocator: Allocator<D, D> + Allocator<D>,
    {
        self.try_matrix(lhs).ok()
    }

    /// Automatically selects a suitable preconditioner.
    #[must_use]
    pub fn auto<T, S>(lhs: &IntervalMatrix<T, D, D, S>) -> Option<Self>
    where
        T: EnclosureScalar<Midpoint = P> + Scalar,
        S: Storage<T, D, D>,
        DefaultAllocator: Allocator<D, D> + Allocator<D>,
    {
        let (n, _) = lhs.shape();
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                if i != j {
                    sum += lhs[(i, j)].mag();
                }
            }
            if lhs[(i, i)].mig() <= sum {
                return Some(Self::InverseMidpoint);
            }
        }
        None
    }
}

/// A solver adapter that left-preconditions a system before solving it.
pub struct Preconditioned<D, S, P = f64>
where
    D: Dim,
    P: Scalar + ComplexField + Copy,
    DefaultAllocator: Allocator<D, D>,
{
    solver: S,
    preconditioner: Option<Preconditioner<D, P>>,
}

impl<D, S, P> Preconditioned<D, S, P>
where
    D: Dim,
    P: Scalar + ComplexField + Copy,
    DefaultAllocator: Allocator<D, D>,
{
    /// Wraps `solver` with the selected preconditioning strategy.
    #[must_use]
    pub const fn new(solver: S, preconditioner: Preconditioner<D, P>) -> Self {
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

    /// Wraps `solver` with a caller-provided point preconditioner matrix.
    #[must_use]
    pub const fn custom(solver: S, matrix: OMatrix<P, D, D>) -> Self {
        Self {
            solver,
            preconditioner: Some(Preconditioner::Custom(matrix)),
        }
    }

    /// Automatically selects a suitable preconditioner.
    #[must_use]
    pub fn auto<T, SA>(solver: S, lhs: &IntervalMatrix<T, D, D, SA>) -> Self
    where
        T: EnclosureScalar<Midpoint = P> + Scalar,
        SA: Storage<T, D, D>,
        DefaultAllocator: Allocator<D, D> + Allocator<D>,
    {
        Self {
            solver,
            preconditioner: Preconditioner::auto(lhs),
        }
    }

    /// Replaces the selected preconditioning strategy.
    #[must_use]
    pub fn with_preconditioner(mut self, preconditioner: Preconditioner<D, P>) -> Self {
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
    pub const fn preconditioner(&self) -> Option<&Preconditioner<D, P>> {
        self.preconditioner.as_ref()
    }
}

impl<T, D, S, P> Solver<T, D> for Preconditioned<D, S, P>
where
    T: EnclosureScalar<Midpoint = P> + Scalar,
    P: ComplexField + Copy,
    D: Dim,
    S: Solver<T, D>,
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    fn try_solve<SA, SB>(
        &self,
        lhs: &IntervalMatrix<T, D, D, SA>,
        rhs: &IntervalMatrix<T, D, Const<1>, SB>,
    ) -> Result<OIntervalMatrix<T, D, Const<1>>, SolveError>
    where
        SA: Storage<T, D, D>,
        SB: Storage<T, D, Const<1>>,
    {
        if let Some(preconditioner) = &self.preconditioner {
            let c = preconditioner.try_matrix(lhs)?;
            self.solver.try_solve(&(&c * lhs), &(&c * rhs))
        } else {
            self.solver.try_solve(lhs, rhs)
        }
    }
}
