use core::fmt;

/// Explains why a verified linear solve did not produce an enclosure.
///
/// A certification failure is deliberately distinct from singularity: verified
/// methods use sufficient tests, so failing to prove an enclosure is not proof
/// that the requested system has no solution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolveError {
    /// The matrix and right-hand side have incompatible dimensions or shapes.
    InvalidSystem,
    /// The input contains empty or invalid intervals, or the solver's settings
    /// are not finite and nonnegative where required.
    InvalidInput,
    /// The midpoint matrix could not be inverted in working precision.
    SingularMidpoint,
    /// The sufficient inclusion test did not succeed within the iteration
    /// limit.
    CertificationFailed {
        /// Number of verified iterations attempted.
        iterations: usize,
    },
}

impl fmt::Display for SolveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSystem => formatter.write_str("invalid interval linear system"),
            Self::InvalidInput => formatter.write_str("invalid interval linear-solver input"),
            Self::SingularMidpoint => formatter.write_str("midpoint matrix is singular"),
            Self::CertificationFailed { iterations } => write!(
                formatter,
                "could not certify a solution enclosure in {iterations} iterations"
            ),
        }
    }
}
