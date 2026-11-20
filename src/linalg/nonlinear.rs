use core::fmt;

use nalgebra::{
    Const, DefaultAllocator, Dim,
    allocator::Allocator,
    constraint::{AreMultipliable, ShapeConstraint},
};

use crate::{EnclosureOps, Interval, OIntervalMatrix, OIntervalVector, Preconditioner};

/// Result of a Krawczyk contraction for a box in a nonlinear system.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KrawczykOutcome<B> {
    /// The input box contains no roots.
    Excluded,
    /// The returned box encloses the unique root in the input box.
    Unique(B),
    /// The returned box encloses every possible root in the input box.
    /// It may equal the input box if contraction made no progress.
    Unresolved(B),
}

/// An invalid input or callback failure during a Krawczyk contraction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KrawczykError<F, J> {
    /// The input box has no coordinates.
    ZeroDimensions,
    /// The function enclosure callback failed.
    Function(F),
    /// The Jacobian enclosure callback failed.
    Jacobian(J),
    /// The function callback returned a vector with the wrong number of rows.
    FunctionShape {
        /// Number of coordinates in the input box.
        expected: usize,
        /// Number of rows returned by the callback.
        actual: usize,
    },
    /// The Jacobian callback returned a matrix with the wrong shape.
    JacobianShape {
        /// Dimension of the expected square Jacobian.
        expected: usize,
        /// Rows and columns returned by the callback.
        actual: (usize, usize),
    },
    /// The function enclosure had an empty component on a nonempty box.
    EmptyFunctionEnclosure,
    /// The Jacobian enclosure had an empty entry on a nonempty box.
    EmptyJacobianEnclosure,
}

impl<F: fmt::Display, J: fmt::Display> fmt::Display for KrawczykError<F, J> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimensions => formatter.write_str("expected at least one coordinate"),
            Self::Function(error) => write!(formatter, "function evaluation failed: {error}"),
            Self::Jacobian(error) => write!(formatter, "Jacobian evaluation failed: {error}"),
            Self::FunctionShape { expected, actual } => {
                write!(
                    formatter,
                    "expected {expected} function components, got {actual}"
                )
            }
            Self::JacobianShape { expected, actual } => write!(
                formatter,
                "expected a {expected}×{expected} Jacobian, got {}×{}",
                actual.0, actual.1
            ),
            Self::EmptyFunctionEnclosure => {
                formatter.write_str("function enclosure has an empty component")
            }
            Self::EmptyJacobianEnclosure => {
                formatter.write_str("Jacobian enclosure has an empty entry")
            }
        }
    }
}

/// Applies one Krawczyk contraction to a box for a square nonlinear system.
///
/// The function must be continuously differentiable on a neighborhood of the
/// box. `function` must enclose each output component throughout its input box.
/// It may receive the whole box and a singleton midpoint box. `jacobian` must
/// enclose every entry of the Jacobian throughout the whole box.
///
/// The step uses a point inverse of the midpoint Jacobian as a preconditioner.
/// If that inverse cannot be formed, the box remains unresolved. The returned
/// enclosure can be submitted to another step or subdivided. An empty coordinate
/// excludes the box immediately.
/// A Krawczyk image contained in the input box with a certified contraction
/// factor below one proves that the box contains exactly one root.
///
/// # Errors
///
/// Returns [`KrawczykError`] for a zero-dimensional input box, a callback
/// failure, or a malformed enclosure returned by a callback.
///
/// ```
/// use core::convert::Infallible;
/// use maryada::{Interval, IntervalOps, KrawczykOutcome, SIntervalMatrix, SIntervalVector, krawczyk_contract};
///
/// type Vector = SIntervalVector<Interval, 2>;
/// type Jacobian = SIntervalMatrix<Interval, 2, 2>;
/// let domain = Vector::from_element(Interval::new(0.0, 3.0));
/// let result = krawczyk_contract(
///     &domain,
///     |x| Ok::<_, Infallible>(Vector::from_column_slice(&[
///         x[0] + x[1] - 3.0,
///         x[0] - x[1] - 1.0,
///     ])),
///     |_| Ok::<_, Infallible>(Jacobian::from_row_slice(&[
///         Interval::ONE, Interval::ONE,
///         Interval::ONE, Interval::from(-1.0),
///     ])),
/// )?;
/// assert_eq!(result, KrawczykOutcome::Unique(Vector::from_column_slice(&[
///     Interval::from(2.0), Interval::ONE,
/// ])));
/// # Ok::<(), maryada::KrawczykError<Infallible, Infallible>>(())
/// ```
#[allow(clippy::arithmetic_side_effects, clippy::type_complexity)]
pub fn krawczyk_contract<D, F, J, FE, JE>(
    domain: &OIntervalVector<Interval, D>,
    mut function: F,
    mut jacobian: J,
) -> Result<KrawczykOutcome<OIntervalVector<Interval, D>>, KrawczykError<FE, JE>>
where
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
    ShapeConstraint: AreMultipliable<D, D, D, D> + AreMultipliable<D, D, D, Const<1>>,
    F: FnMut(&OIntervalVector<Interval, D>) -> Result<OIntervalVector<Interval, D>, FE>,
    J: FnMut(&OIntervalVector<Interval, D>) -> Result<OIntervalMatrix<Interval, D, D>, JE>,
{
    let size = domain.nrows();
    if size == 0 {
        return Err(KrawczykError::ZeroDimensions);
    }
    if domain.has_empty_entries() {
        return Ok(KrawczykOutcome::Excluded);
    }
    let mut evaluate = |box_| {
        let values = function(box_).map_err(KrawczykError::Function)?;
        if values.nrows() != size {
            return Err(KrawczykError::FunctionShape {
                expected: size,
                actual: values.nrows(),
            });
        }
        if values.has_empty_entries() {
            return Err(KrawczykError::EmptyFunctionEnclosure);
        }
        Ok(values)
    };
    let range = evaluate(domain)?;
    if range.iter().any(|entry| !entry.contains(0.0)) {
        return Ok(KrawczykOutcome::Excluded);
    }
    if range.iter().all(|entry| *entry == Interval::ZERO) {
        return Ok(if domain.iter().all(|entry| entry.is_singleton()) {
            KrawczykOutcome::Unique(domain.clone())
        } else {
            KrawczykOutcome::Unresolved(domain.clone())
        });
    }

    let derivatives = jacobian(domain).map_err(KrawczykError::Jacobian)?;
    if derivatives.shape() != (size, size) {
        return Err(KrawczykError::JacobianShape {
            expected: size,
            actual: derivatives.shape(),
        });
    }
    if derivatives.has_empty_entries() {
        return Err(KrawczykError::EmptyJacobianEnclosure);
    }
    let Ok(inverse) = Preconditioner::InverseMidpoint.try_matrix(&derivatives) else {
        return Ok(KrawczykOutcome::Unresolved(domain.clone()));
    };
    if inverse.has_empty_entries() {
        return Ok(KrawczykOutcome::Unresolved(domain.clone()));
    }

    let midpoint: OIntervalVector<Interval, D> = domain.mid().into();
    let midpoint_value = evaluate(&midpoint)?;
    let (dimension, _) = domain.shape_generic();
    let identity = OIntervalMatrix::<Interval, D, D>::identity_generic(dimension);
    let residual = &identity - &(&inverse * &derivatives);
    // The mean-value formula shows the image contains all roots in the input box.
    let image = &(&midpoint - &(&inverse * &midpoint_value)) + &(&residual * &(domain - &midpoint));
    let contracted = domain.intersection(&image);
    if contracted.has_empty_entries() {
        return Ok(KrawczykOutcome::Excluded);
    }
    // When the image maps the box into itself, a norm below one gives a
    // unique fixed point. It also proves the point preconditioner nonsingular,
    // which makes that fixed point a root of the original system.
    Ok(if image.subset(domain) && residual.inf_norm() < 1.0 {
        KrawczykOutcome::Unique(contracted)
    } else {
        KrawczykOutcome::Unresolved(contracted)
    })
}
