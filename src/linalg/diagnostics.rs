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
    /// Gaussian elimination could not certify a nonzero pivot.
    PivotNotCertified {
        /// Index of the pivot that could not be certified.
        index: usize,
    },
    /// The selected iterative initializer could not produce a bounded
    /// enclosure suitable for the method.
    InitialEnclosureFailed,
    /// The Hansen–Bliek–Rohn sufficient H-matrix condition was not certified.
    NotAnHMatrix,
    /// An iterative contraction no longer overlapped its previous enclosure.
    EmptyIntersection,
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
            Self::PivotNotCertified { index } => {
                write!(formatter, "could not certify pivot {index} as nonzero")
            }
            Self::InitialEnclosureFailed => {
                formatter.write_str("could not construct a valid initial enclosure")
            }
            Self::NotAnHMatrix => {
                formatter.write_str("could not certify the matrix as an H-matrix")
            }
            Self::EmptyIntersection => {
                formatter.write_str("iteration produced an empty solution enclosure")
            }
            Self::CertificationFailed { iterations } => write!(
                formatter,
                "could not certify a solution enclosure in {iterations} iterations"
            ),
        }
    }
}
