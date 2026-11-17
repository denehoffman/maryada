#![allow(clippy::arithmetic_side_effects)]

use nalgebra::{
    ComplexField, Const, DefaultAllocator, Dim, OMatrix, Scalar, Storage,
    allocator::Allocator,
    constraint::{AreMultipliable, ShapeConstraint},
};

use super::{
    EnclosureOps, IntervalMatrix, OIntervalMatrix, OIntervalVector, SolveError, Solver,
    solve_inverse_columns,
};

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

    /// Precomputes the midpoint inverse and interval iteration matrix for `lhs`.
    ///
    /// Reuse the result to solve with multiple right-hand sides. Each solve
    /// still verifies its own enclosure.
    ///
    /// # Errors
    ///
    /// Returns a [`SolveError`] for invalid parameters, an invalid matrix,
    /// or a midpoint matrix whose approximate inverse cannot be computed.
    pub fn prepare<'a, T, D, S>(
        &self,
        lhs: &'a IntervalMatrix<T, D, D, S>,
    ) -> Result<PreparedEpsilonInflation<'a, T, D, S>, SolveError>
    where
        T: EnclosureOps + Scalar,
        T::Midpoint: ComplexField,
        D: Dim,
        S: Storage<T, D, D>,
        DefaultAllocator: Allocator<D, D> + Allocator<D>,
        ShapeConstraint: AreMultipliable<D, D, D, D>,
    {
        self.validate_system(lhs, None)?;
        self.prepare_validated(lhs)
    }

    fn validate_system<T, D, S>(
        &self,
        lhs: &IntervalMatrix<T, D, D, S>,
        rhs_rows: Option<usize>,
    ) -> Result<(), SolveError>
    where
        T: EnclosureOps + Scalar,
        D: Dim,
        S: Storage<T, D, D>,
    {
        if !self.r.is_finite() || self.r < 0.0 || !self.eps.is_finite() || self.eps < 0.0 {
            return Err(SolveError::InvalidInput);
        }
        if !lhs.is_square() || lhs.is_empty() || rhs_rows.is_some_and(|rows| rows != lhs.nrows()) {
            return Err(SolveError::InvalidSystem);
        }
        if lhs.has_empty_entries() || lhs.has_nai_entries() {
            return Err(SolveError::InvalidInput);
        }
        Ok(())
    }

    #[allow(clippy::indexing_slicing)]
    fn prepare_validated<'a, T, D, S>(
        &self,
        lhs: &'a IntervalMatrix<T, D, D, S>,
    ) -> Result<PreparedEpsilonInflation<'a, T, D, S>, SolveError>
    where
        T: EnclosureOps + Scalar,
        T::Midpoint: ComplexField,
        D: Dim,
        S: Storage<T, D, D>,
        DefaultAllocator: Allocator<D, D> + Allocator<D>,
        ShapeConstraint: AreMultipliable<D, D, D, D>,
    {
        let (dim, _) = lhs.shape_generic();

        // R ~ inv(mid(A)).
        let midpoint = lhs.mid();
        if midpoint.iter().any(|entry| !entry.is_finite()) {
            return Err(SolveError::InvalidInput);
        }
        let preconditioner_point = midpoint.try_inverse().ok_or(SolveError::SingularMidpoint)?;
        if preconditioner_point.iter().any(|entry| !entry.is_finite()) {
            return Err(SolveError::SingularMidpoint);
        }
        let r_interval = OIntervalMatrix::from_fn_generic(dim, dim, |i, j| {
            <T as EnclosureOps>::from_midpoint(preconditioner_point[(i, j)])
        });
        let identity = OIntervalMatrix::<T, D, D>::identity_generic(dim);
        let iteration_matrix = &identity - &(&r_interval * lhs);
        Ok(PreparedEpsilonInflation {
            lhs,
            solver: *self,
            preconditioner_point,
            r_interval,
            iteration_matrix,
        })
    }
}

/// Reusable matrix setup for [`EpsilonInflationSolver`].
///
/// The coefficient matrix is borrowed for the lifetime of the setup. Each
/// call to [`Self::try_solve`] computes and certifies a fresh enclosure.
pub struct PreparedEpsilonInflation<'a, T, D, S>
where
    T: EnclosureOps + Scalar,
    T::Midpoint: ComplexField,
    D: Dim,
    S: Storage<T, D, D>,
    DefaultAllocator: Allocator<D, D>,
{
    lhs: &'a IntervalMatrix<T, D, D, S>,
    solver: EpsilonInflationSolver,
    preconditioner_point: OMatrix<T::Midpoint, D, D>,
    r_interval: OIntervalMatrix<T, D, D>,
    iteration_matrix: OIntervalMatrix<T, D, D>,
}

impl<T, D, S> PreparedEpsilonInflation<'_, T, D, S>
where
    T: EnclosureOps + Scalar,
    T::Midpoint: ComplexField,
    D: Dim,
    S: Storage<T, D, D>,
    DefaultAllocator: Allocator<D, D> + Allocator<D> + Allocator<D, Const<1>>,
    ShapeConstraint: AreMultipliable<D, D, D, Const<1>>,
{
    /// Computes and certifies a solution enclosure for `rhs`.
    ///
    /// # Errors
    ///
    /// Returns a [`SolveError`] for a mismatched or invalid right-hand side,
    /// or when the inflation iteration cannot certify an enclosure.
    pub fn try_solve<SB>(
        &self,
        rhs: &IntervalMatrix<T, D, Const<1>, SB>,
    ) -> Result<OIntervalVector<T, D>, SolveError>
    where
        SB: Storage<T, D, Const<1>>,
    {
        if self.lhs.nrows() != rhs.nrows() {
            return Err(SolveError::InvalidSystem);
        }
        if rhs.has_empty_entries() || rhs.has_nai_entries() {
            return Err(SolveError::InvalidInput);
        }
        self.solve_validated(rhs)
    }

    #[allow(clippy::indexing_slicing)]
    fn solve_validated<SB>(
        &self,
        rhs: &IntervalMatrix<T, D, Const<1>, SB>,
    ) -> Result<OIntervalVector<T, D>, SolveError>
    where
        SB: Storage<T, D, Const<1>>,
    {
        let (dim, _) = self.lhs.shape_generic();
        // Let x̃ = R mid(b). Writing x = x̃ + d gives
        //
        //     d = R (b - A x̃) + (I - R A) d = Z + C d.
        //
        // This residual-centered form is important for thick right-hand
        // sides (and is the form used by the verified Rump/Krawczyk method).
        let midpoint_rhs = rhs.mid();
        if midpoint_rhs.iter().any(|entry| !entry.is_finite()) {
            return Err(SolveError::InvalidInput);
        }
        let x_tilde_point = &self.preconditioner_point * &midpoint_rhs;
        if x_tilde_point.iter().any(|entry| !entry.is_finite()) {
            return Err(SolveError::InvalidInput);
        }
        let x_tilde: OIntervalVector<T, D> =
            OIntervalMatrix::from_fn_generic(dim, Const::<1>, |i, j| {
                <T as EnclosureOps>::from_midpoint(x_tilde_point[(i, j)])
            });
        let lhs_x_tilde: OIntervalVector<T, D> = self.lhs * &x_tilde;
        let residual_rhs: OIntervalVector<T, D> = rhs - &lhs_x_tilde;
        let correction = &self.r_interval * &residual_rhs;

        // Start from the residual enclosure. Every iterate below is an
        // enclosure for the correction d, not for the absolute solution x.
        let mut correction_enclosure = correction.clone();
        for _ in 0..self.solver.max_iterations {
            // Inflate the current correction enclosure before applying the
            // fixed-point map. The strict interior test is the existence
            // certificate: if F(Y) ⊂ int(Y), then the united solution set is
            // contained in F(Y), hence x̃ + F(Y) encloses every solution.
            let inflated = correction_enclosure
                .map(|entry| <T as EnclosureOps>::inflate(entry, self.solver.r, self.solver.eps));

            // x^{k+1} = Z + C y.
            let next = &correction + &(&self.iteration_matrix * &inflated);

            let enclosed = next.interior(&inflated);

            // Check if x^{k+1} is contained in the interior of y.
            if enclosed {
                return Ok(&x_tilde + &next);
            }
            correction_enclosure = next;
        }
        Err(SolveError::CertificationFailed {
            iterations: self.solver.max_iterations,
        })
    }
}

impl<T, D> Solver<T, D> for EpsilonInflationSolver
where
    T: EnclosureOps + Scalar,
    T::Midpoint: ComplexField,
    D: Dim,
    DefaultAllocator: Allocator<D, D> + Allocator<D> + Allocator<D, Const<1>>,
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
        self.validate_system(lhs, Some(rhs.nrows()))?;
        if rhs.has_empty_entries() || rhs.has_nai_entries() {
            return Err(SolveError::InvalidInput);
        }
        let prepared = self.prepare_validated(lhs)?;
        prepared.solve_validated(rhs)
    }

    fn try_inverse<S>(
        &self,
        mat: &IntervalMatrix<T, D, D, S>,
    ) -> Result<OIntervalMatrix<T, D, D>, SolveError>
    where
        S: Storage<T, D, D>,
    {
        let (dim, _) = mat.shape_generic();
        // The default inverse performs no solves when there are no columns.
        if mat.ncols() == 0 {
            return Ok(OIntervalMatrix::zeros_generic(dim, dim));
        }
        let prepared = self.prepare(mat)?;
        solve_inverse_columns(dim, mat.ncols(), |rhs| prepared.solve_validated(rhs))
    }
}
