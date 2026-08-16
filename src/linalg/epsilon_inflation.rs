#![allow(clippy::arithmetic_side_effects)]

use nalgebra::{
    Const, DefaultAllocator, Dim, Scalar, Storage,
    allocator::Allocator,
    constraint::{AreMultipliable, ShapeConstraint},
};

use crate::IntervalOps;

use super::{IntervalMatrix, OIntervalMatrix, OIntervalVector, Solver};

/// Verified epsilon-inflation solver from Rump's dissertation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EpsilonInflationSolver {
    /// Relative inflation
    r: f64,
    /// Absolute inflation
    eps: f64,
    /// Maximum number of iterations
    max_iterations: usize,
}

impl Default for EpsilonInflationSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl EpsilonInflationSolver {
    /// Default relative inflation factor.
    pub const DEFAULT_RELATIVE_INFLATION: f64 = 0.1;
    /// Default absolute inflation radius.
    pub const DEFAULT_ABSOLUTE_INFLATION: f64 = 1e-20;
    /// Default maximum number of iterations.
    pub const DEFAULT_MAX_ITERATIONS: usize = 20;

    /// Creates a solver with the default inflation parameters.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            r: Self::DEFAULT_RELATIVE_INFLATION,
            eps: Self::DEFAULT_ABSOLUTE_INFLATION,
            max_iterations: Self::DEFAULT_MAX_ITERATIONS,
        }
    }

    /// Sets the relative inflation factor.
    ///
    /// A non-finite or negative value causes [`Solver::solve`] to return
    /// `None`.
    #[must_use]
    pub const fn with_relative_inflation(mut self, relative_inflation: f64) -> Self {
        self.r = relative_inflation;
        self
    }

    /// Sets the absolute inflation radius.
    ///
    /// A non-finite or negative value causes [`Solver::solve`] to return
    /// `None`.
    #[must_use]
    pub const fn with_absolute_inflation(mut self, absolute_inflation: f64) -> Self {
        self.eps = absolute_inflation;
        self
    }

    /// Sets the maximum number of iterations.
    #[must_use]
    pub const fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Returns the relative inflation factor.
    #[must_use]
    pub const fn relative_inflation(&self) -> f64 {
        self.r
    }

    /// Returns the absolute inflation radius.
    #[must_use]
    pub const fn absolute_inflation(&self) -> f64 {
        self.eps
    }

    /// Returns the maximum number of iterations.
    #[must_use]
    pub const fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    pub(super) fn solve_matrix<T, D, C, SA, SB>(
        &self,
        lhs: &IntervalMatrix<T, D, D, SA>,
        rhs: &IntervalMatrix<T, D, C, SB>,
    ) -> Option<OIntervalMatrix<T, D, C>>
    where
        T: IntervalOps + Scalar,
        D: Dim,
        C: Dim,
        SA: Storage<T, D, D>,
        SB: Storage<T, D, C>,
        DefaultAllocator: Allocator<D, D> + Allocator<D, C>,
        ShapeConstraint: AreMultipliable<D, D, D, D> + AreMultipliable<D, D, D, C>,
    {
        if !self.r.is_finite()
            || self.r < 0.0
            || !self.eps.is_finite()
            || self.eps < 0.0
            || !lhs.is_square()
            || lhs.is_empty()
            || lhs.nrows() != rhs.nrows()
            || lhs.has_empty_entries()
            || lhs.has_nai_entries()
            || rhs.has_empty_entries()
            || rhs.has_nai_entries()
        {
            return None;
        }

        let (dim, _) = lhs.shape_generic();
        let relative_inflation = T::new(1.0 - self.r, 1.0 + self.r);
        let absolute_inflation = T::new(-self.eps, self.eps);

        // C ~ inv(mid(A)).
        let midpoint = lhs.mid();
        if midpoint.iter().any(|entry| !entry.is_finite()) {
            return None;
        }
        let c = midpoint.try_inverse()?;
        if c.iter().any(|entry| !entry.is_finite()) {
            return None;
        }
        let c_interval = OIntervalMatrix::from_singletons(&c);
        let identity = OIntervalMatrix::<T, D, D>::identity_generic(dim);

        // Z = I - CA.
        let residual = &identity - &(&c_interval * lhs);

        // x^0 = Cb.
        let cb = &c_interval * rhs;
        let mut x = cb.clone();
        for _ in 0..self.max_iterations {
            // y = x^k * [1-r, 1+r] + [-eps, eps].
            let y = (&x * relative_inflation).add_scalar(absolute_inflation);

            // x^{k+1} = Cb + (I-CA)y.
            let next = &cb + &(&residual * &y);

            let enclosed = next.interior(&y);

            // Check if x^{k+1} is contained in the interior of y.
            if enclosed {
                return Some(next);
            }
            x = next;
        }
        None
    }
}

impl<T, D> Solver<T, D> for EpsilonInflationSolver
where
    T: IntervalOps + Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    fn solve<SA, SB>(
        &self,
        lhs: &IntervalMatrix<T, D, D, SA>,
        rhs: &IntervalMatrix<T, D, Const<1>, SB>,
    ) -> Option<OIntervalVector<T, D>>
    where
        SA: Storage<T, D, D>,
        SB: Storage<T, D, Const<1>>,
    {
        self.solve_matrix(lhs, rhs)
    }
}
