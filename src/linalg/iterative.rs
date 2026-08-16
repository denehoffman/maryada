#![allow(clippy::arithmetic_side_effects)]
use super::{IntervalMatrix, OIntervalMatrix, OIntervalVector};
use crate::IntervalOps;
use nalgebra::{
    Const, DefaultAllocator, Dim, DimMin, OVector, Scalar, Storage, allocator::Allocator,
};

/// Absolute tolerance for comparing consecutive interval enclosures.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StoppingTolerance(f64);

impl StoppingTolerance {
    /// Scale applied to the narrowest coefficient of a system matrix.
    pub const DEFAULT_WIDTH_SCALE: f64 = 1e-5;

    /// Creates an explicit stopping tolerance.
    ///
    /// The convergence check rejects nonpositive and non-finite values.
    #[must_use]
    pub const fn new(epsilon: f64) -> Self {
        Self(epsilon)
    }

    /// Derives the default tolerance from the narrowest coefficient of `lhs`.
    ///
    /// This computes the smallest positive `wid(lhs_ij) * 10^-5`. Zero-width
    /// point coefficients are ignored so sparse matrices still receive a
    /// useful tolerance. If every coefficient is a point interval, this falls
    /// back to [`f64::EPSILON`].
    #[must_use]
    pub fn from_matrix<T, D, S>(lhs: &IntervalMatrix<T, D, D, S>) -> Self
    where
        T: IntervalOps + Scalar,
        D: Dim,
        S: Storage<T, D, D>,
    {
        let minimum_width = lhs
            .iter()
            .map(|entry| entry.wid())
            .filter(|width| width.is_finite() && *width > 0.0)
            .fold(f64::INFINITY, f64::min);
        if minimum_width.is_finite() {
            Self((minimum_width * Self::DEFAULT_WIDTH_SCALE).max(f64::MIN_POSITIVE))
        } else {
            Self(f64::EPSILON)
        }
    }

    /// Returns the stored epsilon.
    #[must_use]
    pub const fn epsilon(self) -> f64 {
        self.0
    }
}

/// Tests whether two consecutive enclosures satisfy the stopping criterion.
///
/// The lower and upper endpoints of every component are compared separately.
/// Returns `false` for a nonpositive or non-finite tolerance.
///
/// # Panics
///
/// Panics if `current` and `previous` have different lengths.
#[must_use]
#[allow(clippy::arithmetic_side_effects)]
pub fn enclosures_converged<T, D, SC, SP>(
    current: &IntervalMatrix<T, D, Const<1>, SC>,
    previous: &IntervalMatrix<T, D, Const<1>, SP>,
    tolerance: StoppingTolerance,
) -> bool
where
    T: IntervalOps + Scalar,
    D: Dim,
    SC: Storage<T, D, Const<1>>,
    SP: Storage<T, D, Const<1>>,
{
    assert_eq!(
        current.nrows(),
        previous.nrows(),
        "consecutive enclosure dimension mismatch",
    );
    let epsilon = tolerance.epsilon();
    epsilon.is_finite()
        && epsilon > 0.0
        && current
            .iter()
            .zip(previous.iter())
            .all(|(current, previous)| {
                endpoints_close::<T>(current.inf(), previous.inf(), epsilon)
                    && endpoints_close::<T>(current.sup(), previous.sup(), epsilon)
            })
}

fn endpoints_close<T>(current: f64, previous: f64, epsilon: f64) -> bool
where
    T: IntervalOps,
{
    current == previous || (T::singleton(current) - T::singleton(previous)).abs().sup() < epsilon
}

impl<T, D, SA> IntervalMatrix<T, D, D, SA>
where
    T: IntervalOps + Scalar,
    D: Dim + DimMin<D, Output = D>,
    SA: Storage<T, D, D>,
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    /// Finds an initial enclosure using a comparison-matrix weighted norm.
    ///
    /// `self` and `rhs` are expected to be the preconditioned system `CA` and
    /// `Cb`. The weight candidate `u` is obtained by approximately solving
    /// `<CA>u = e`; the positivity condition `<CA>u > 0` is then certified
    /// using interval arithmetic.
    #[must_use]
    #[allow(clippy::indexing_slicing)]
    pub fn initial_enclosure_v_norm<SB>(
        &self,
        rhs: &IntervalMatrix<T, D, Const<1>, SB>,
    ) -> Option<OIntervalVector<T, D>>
    where
        SB: Storage<T, D, Const<1>>,
    {
        if !valid_system(self, rhs) {
            return None;
        }

        let comparison = self.comparison_matrix()?;
        if comparison.iter().any(|entry| !entry.is_finite()) {
            return None;
        }
        let (dim, _) = self.shape_generic();
        let e = OVector::<f64, D>::repeat_generic(dim, Const::<1>, 1.0);
        let u = comparison.clone().lu().solve(&e)?;
        if u.iter().any(|entry| !entry.is_finite() || *entry < 0.0) {
            return None;
        }

        // Interval arithmetic certifies a componentwise lower bound for <CA>u.
        let v = OVector::<f64, D>::from_fn_generic(dim, Const::<1>, |row, _| {
            (0..self.ncols())
                .fold(T::ZERO, |sum, column| {
                    T::singleton(comparison[(row, column)]).mul_add(T::singleton(u[column]), sum)
                })
                .inf()
        });
        if v.iter().any(|entry| !entry.is_finite() || *entry <= 0.0) {
            return None;
        }

        let norm = rhs.v_norm(&v);
        if !norm.is_finite() {
            return None;
        }
        let enclosure = OIntervalVector::from_fn_generic(dim, Const::<1>, |row, _| {
            T::new(-u[row], u[row]) * norm
        });
        enclosure
            .iter()
            .all(|entry| entry.is_bounded())
            .then_some(enclosure)
    }

    /// Finds an initial enclosure using the induced infinity norm.
    ///
    /// `self` and `rhs` are expected to be the preconditioned system `CA` and
    /// `Cb`. This construction succeeds when `||I - CA||_inf < 1`.
    #[must_use]
    pub fn initial_enclosure_inf_norm<SB>(
        &self,
        rhs: &IntervalMatrix<T, D, Const<1>, SB>,
    ) -> Option<OIntervalVector<T, D>>
    where
        SB: Storage<T, D, Const<1>>,
    {
        if !valid_system(self, rhs) {
            return None;
        }

        let (dim, _) = self.shape_generic();
        let identity = OIntervalMatrix::<T, D, D>::identity_generic(dim);
        let residual = &identity - self;
        let residual_norm = residual.inf_norm();
        if !residual_norm.is_finite() || residual_norm >= 1.0 {
            return None;
        }

        let rhs_norm = rhs.inf_norm();
        if !rhs_norm.is_finite() {
            return None;
        }

        let denominator = T::ONE - T::singleton(residual_norm);
        if denominator.inf() <= 0.0 {
            return None;
        }
        let radius = (T::singleton(rhs_norm) / denominator).sup();
        if !radius.is_finite() {
            return None;
        }

        Some(OIntervalVector::from_fn_generic(dim, Const::<1>, |_, _| {
            T::new(-radius, radius)
        }))
    }
}

/// Strategy used to obtain the initial enclosure for an iterative solver.
pub enum InitialEnclosure<T, D>
where
    T: IntervalOps + Scalar,
    D: Dim + DimMin<D, Output = D>,
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    /// Uses a caller-provided enclosure known to contain the solution set.
    Provided(OIntervalVector<T, D>),
    /// Constructs an enclosure using the comparison-matrix weighted norm.
    WeightedNorm,
    /// Constructs an enclosure using the induced infinity norm.
    InfinityNorm,
}

mod krawczyk;
pub use krawczyk::KrawczykSolver;

mod jacobi;
pub use jacobi::JacobiSolver;

mod gauss_seidel;
pub use gauss_seidel::GaussSeidelSolver;

fn valid_system<T, D, SA, SB>(
    lhs: &IntervalMatrix<T, D, D, SA>,
    rhs: &IntervalMatrix<T, D, Const<1>, SB>,
) -> bool
where
    T: IntervalOps + Scalar,
    D: Dim,
    SA: Storage<T, D, D>,
    SB: Storage<T, D, Const<1>>,
{
    lhs.is_square()
        && !lhs.is_empty()
        && lhs.nrows() == rhs.nrows()
        && !lhs.has_empty_entries()
        && !lhs.has_nai_entries()
        && !rhs.has_empty_entries()
        && !rhs.has_nai_entries()
}

#[cfg(test)]
mod tests {
    use crate::{
        Interval, IntervalOps, Preconditioned, Preconditioner, SIntervalMatrix, SIntervalVector,
        Solver,
    };

    use super::{KrawczykSolver, StoppingTolerance, enclosures_converged};

    #[test]
    fn weighted_norm_enclosure_contains_the_solution() {
        let lhs = SIntervalMatrix::<Interval, 2, 2>::identity();
        let rhs = SIntervalVector::<Interval, 2>::from_column_slice(&[
            Interval::singleton(2.0),
            Interval::singleton(-3.0),
        ]);

        let enclosure = lhs.initial_enclosure_v_norm(&rhs);

        assert!(
            enclosure
                .as_ref()
                .is_some_and(|enclosure| enclosure[0].contains(2.0) && enclosure[1].contains(-3.0))
        );
    }

    #[test]
    fn infinity_norm_enclosure_contains_the_solution() {
        let lhs =
            SIntervalMatrix::<Interval, 2, 2>::from_diagonal(&SIntervalVector::from_column_slice(
                &[Interval::new(0.9, 1.1), Interval::new(0.9, 1.1)],
            ));
        let rhs = SIntervalVector::<Interval, 2>::from_column_slice(&[
            Interval::singleton(2.0),
            Interval::singleton(-3.0),
        ]);

        let enclosure = lhs.initial_enclosure_inf_norm(&rhs);

        assert!(
            enclosure
                .as_ref()
                .is_some_and(|enclosure| enclosure[0].contains(2.0) && enclosure[1].contains(-3.0))
        );
    }

    #[test]
    fn infinity_norm_enclosure_rejects_noncontraction() {
        let lhs = SIntervalMatrix::<Interval, 1, 1>::from_element(Interval::new(-1.0, 1.0));
        let rhs = SIntervalVector::<Interval, 1>::from_element(Interval::ONE);

        assert!(lhs.initial_enclosure_inf_norm(&rhs).is_none());
    }

    #[test]
    fn stopping_criterion_compares_both_endpoints() {
        let previous = SIntervalVector::<Interval, 2>::from_column_slice(&[
            Interval::new(-1.0, 1.0),
            Interval::new(2.0, 3.0),
        ]);
        let close = SIntervalVector::<Interval, 2>::from_column_slice(&[
            Interval::new(-1.000_001, 0.999_999),
            Interval::new(2.000_001, 3.000_001),
        ]);
        let far = SIntervalVector::<Interval, 2>::from_column_slice(&[
            Interval::new(-1.000_001, 0.9),
            Interval::new(2.000_001, 3.000_001),
        ]);
        let tolerance = StoppingTolerance::new(1e-5);

        assert!(enclosures_converged(&close, &previous, tolerance));
        assert!(!enclosures_converged(&far, &previous, tolerance));
    }

    #[test]
    fn stopping_tolerance_can_be_derived_from_the_system_matrix() {
        let lhs = SIntervalMatrix::<Interval, 2, 2>::from_row_slice(&[
            Interval::new(0.0, 2.0),
            Interval::new(-2.0, 2.0),
            Interval::new(-3.0, 3.0),
            Interval::new(1.0, 9.0),
        ]);

        let tolerance = StoppingTolerance::from_matrix(&lhs);

        assert_eq!(tolerance.epsilon(), 2e-5);
    }

    #[test]
    fn stopping_criterion_rejects_invalid_tolerances() {
        let enclosure = SIntervalVector::<Interval, 1>::from_element(Interval::ZERO);

        assert!(!enclosures_converged(
            &enclosure,
            &enclosure,
            StoppingTolerance::new(0.0),
        ));
        assert!(!enclosures_converged(
            &enclosure,
            &enclosure,
            StoppingTolerance::new(f64::NAN),
        ));
    }

    #[test]
    fn krawczyk_solver_uses_derived_defaults() {
        let lhs =
            SIntervalMatrix::<Interval, 2, 2>::from_diagonal(&SIntervalVector::from_column_slice(
                &[Interval::new(0.99, 1.01), Interval::new(0.99, 1.01)],
            ));
        let rhs = SIntervalVector::<Interval, 2>::from_column_slice(&[
            Interval::singleton(2.0),
            Interval::singleton(-3.0),
        ]);
        let solver = Preconditioned::new(KrawczykSolver::new(&lhs), Preconditioner::default());

        let solution = solver.solve(&lhs, &rhs);

        assert!(
            solution
                .as_ref()
                .is_some_and(|solution| solution[0].contains(2.0) && solution[1].contains(-3.0))
        );
    }
}
