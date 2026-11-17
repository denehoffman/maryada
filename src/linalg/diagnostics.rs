use core::fmt;

/// A shape error from checked interval matrix arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixShapeError {
    /// The operands do not have compatible dimensions for the operation.
    IncompatibleOperands {
        /// Rows and columns of the left operand.
        lhs: (usize, usize),
        /// Rows and columns of the right operand.
        rhs: (usize, usize),
    },
    /// The caller-provided output has the wrong dimensions.
    OutputMismatch {
        /// Rows and columns required by the operation.
        expected: (usize, usize),
        /// Rows and columns of the supplied output.
        actual: (usize, usize),
    },
    /// The product dimensions exceed the representable allocation size.
    ProductTooLarge {
        /// Rows and columns of the left operand.
        lhs: (usize, usize),
        /// Rows and columns of the right operand.
        rhs: (usize, usize),
    },
}

impl fmt::Display for MatrixShapeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::IncompatibleOperands { lhs, rhs } => write!(
                formatter,
                "incompatible matrix operands of shape {}×{} and {}×{}",
                lhs.0, lhs.1, rhs.0, rhs.1
            ),
            Self::OutputMismatch { expected, actual } => write!(
                formatter,
                "matrix output requires shape {}×{} but has shape {}×{}",
                expected.0, expected.1, actual.0, actual.1
            ),
            Self::ProductTooLarge { lhs, rhs } => write!(
                formatter,
                "product of {}×{} and {}×{} matrices is too large",
                lhs.0, lhs.1, rhs.0, rhs.1
            ),
        }
    }
}

impl core::error::Error for MatrixShapeError {}

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
