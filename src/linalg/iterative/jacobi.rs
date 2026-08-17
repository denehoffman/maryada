use nalgebra::{
    Const, DefaultAllocator, Dim, DimMin, Scalar, Storage,
    allocator::Allocator,
    constraint::{AreMultipliable, ShapeConstraint},
};

use crate::{
    InitialEnclosure, IntervalMatrix, IntervalOps, OIntervalMatrix, OIntervalVector, SolveError,
    Solver, StoppingTolerance,
    linalg::iterative::{enclosures_converged, validate_system},
};

/// The Jacobi iterative method for interval linear systems.
pub struct JacobiSolver<T, D>
where
    T: IntervalOps + Scalar,
    D: Dim + DimMin<D, Output = D>,
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    initializer: InitialEnclosure<T, D>,
    tolerance: StoppingTolerance,
    max_iterations: usize,
}

impl<T, D> JacobiSolver<T, D>
where
    T: IntervalOps + Scalar,
    D: Dim + DimMin<D, Output = D>,
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    /// Default maximum number of Jacobi iterations.
    pub const DEFAULT_MAX_ITERATIONS: usize = 20;

    /// Creates a solver with defaults derived from `lhs`.
    ///
    /// This uses the weighted-norm initializer, inverse-midpoint
    /// preconditioning, and [`Self::DEFAULT_MAX_ITERATIONS`].
    #[must_use]
    pub fn new<S>(lhs: &IntervalMatrix<T, D, D, S>) -> Self
    where
        S: Storage<T, D, D>,
    {
        Self::from_tolerance(StoppingTolerance::from_matrix(lhs))
    }

    /// Creates a solver with an explicit stopping tolerance and otherwise
    /// default configuration.
    #[must_use]
    pub const fn from_tolerance(tolerance: StoppingTolerance) -> Self {
        Self {
            initializer: InitialEnclosure::WeightedNorm,
            tolerance,
            max_iterations: Self::DEFAULT_MAX_ITERATIONS,
        }
    }

    /// Selects the initial-enclosure strategy.
    #[must_use]
    pub fn with_initial_enclosure(mut self, initializer: InitialEnclosure<T, D>) -> Self {
        self.initializer = initializer;
        self
    }

    /// Selects the stopping tolerance.
    #[must_use]
    pub const fn with_tolerance(mut self, tolerance: StoppingTolerance) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Selects the maximum number of iterations.
    #[must_use]
    pub const fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Borrows the configured initial-enclosure strategy.
    #[must_use]
    pub const fn initial_enclosure(&self) -> &InitialEnclosure<T, D> {
        &self.initializer
    }

    /// Returns the configured stopping tolerance.
    #[must_use]
    pub const fn tolerance(&self) -> StoppingTolerance {
        self.tolerance
    }

    /// Returns the configured maximum number of iterations.
    #[must_use]
    pub const fn max_iterations(&self) -> usize {
        self.max_iterations
    }
}

impl<T, D> Solver<T, D> for JacobiSolver<T, D>
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
        if !self.tolerance.is_valid() {
            return Err(SolveError::InvalidInput);
        }
        validate_system(lhs, rhs)?;
        let mut x = match &self.initializer {
            InitialEnclosure::Provided(enclosure) => {
                if enclosure.nrows() != lhs.nrows()
                    || enclosure.has_empty_entries()
                    || enclosure.has_nai_entries()
                {
                    return Err(SolveError::InvalidInput);
                }
                enclosure.clone()
            }
            InitialEnclosure::WeightedNorm => lhs.try_initial_enclosure_v_norm(rhs)?,
            InitialEnclosure::InfinityNorm => lhs.try_initial_enclosure_inf_norm(rhs)?,
        };
        let (dim, _) = lhs.shape_generic();
        let d_inv =
            OIntervalMatrix::<T, D, D>::from_diagonal(&lhs.diagonal().map(IntervalOps::recip));
        let j = OIntervalMatrix::<T, D, D>::from_fn_generic(dim, dim, |i, j| {
            if i == j { T::ZERO } else { lhs[(i, j)] }
        });
        for _ in 0..self.max_iterations {
            // NOTE: Horacek's thesis says this assumes that there are no intervals containing 0 on
            // the main diagonal of lhs. If this is not the case, "extended interval arithmetic can
            // be used". I need to read more on this.
            let y = &d_inv * (rhs - &j * &x);
            let next = y.intersection(&x);
            if next.has_empty_entries() || next.has_nai_entries() {
                return Err(SolveError::EmptyIntersection);
            }
            if enclosures_converged(&next, &x, self.tolerance) {
                return Ok(next);
            }
            x = next;
        }
        Err(SolveError::CertificationFailed {
            iterations: self.max_iterations,
        })
    }
}
