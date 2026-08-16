//! Internal real-block fallback for rectangular complex interval systems.

#![allow(clippy::arithmetic_side_effects, clippy::indexing_slicing)]

use core::fmt;

use nalgebra::{Const, DefaultAllocator, Dim, Dyn, Scalar, Storage, allocator::Allocator};

use crate::{ComplexBox, IntervalOps};

use super::{
    DIntervalMatrix, DIntervalVector, EpsilonInflationSolver, IntervalMatrix, OIntervalVector,
    SolveError, Solver,
};

/// Solves the conservative real embedding
/// `[Re(A) -Im(A); Im(A) Re(A)] [Re(x); Im(x)] = [Re(b); Im(b)]`.
///
/// Repeated coefficient intervals are treated independently by the real
/// solver. This can widen the result but cannot invalidate its enclosure.
pub(super) fn try_solve_via_real_block<I, D, SA, SB>(
    lhs: &IntervalMatrix<ComplexBox<I>, D, D, SA>,
    rhs: &IntervalMatrix<ComplexBox<I>, D, Const<1>, SB>,
) -> Result<OIntervalVector<ComplexBox<I>, D>, SolveError>
where
    I: IntervalOps + Scalar + fmt::Debug + PartialEq + 'static,
    D: Dim,
    SA: Storage<ComplexBox<I>, D, D>,
    SB: Storage<ComplexBox<I>, D, Const<1>>,
    DefaultAllocator: Allocator<D>,
{
    let size = lhs.nrows();
    let doubled = size.checked_mul(2).ok_or(SolveError::InvalidSystem)?;
    let block = DIntervalMatrix::<I>::from_fn(doubled, doubled, |row, column| {
        let source_row = row % size;
        let source_column = column % size;
        let coefficient = lhs[(source_row, source_column)];
        match (row < size, column < size) {
            (true, true) | (false, false) => coefficient.re,
            (true, false) => -coefficient.im,
            (false, true) => coefficient.im,
        }
    });
    let stacked_rhs = DIntervalVector::<I>::from_fn(doubled, |row| {
        let coefficient = rhs[(row % size, 0)];
        if row < size {
            coefficient.re
        } else {
            coefficient.im
        }
    });
    let solution = <EpsilonInflationSolver as Solver<I, Dyn>>::try_solve(
        &EpsilonInflationSolver::default(),
        &block,
        &stacked_rhs,
    )?;
    let (dim, _) = lhs.shape_generic();
    Ok(OIntervalVector::from_fn_generic(
        dim,
        Const::<1>,
        |row, _| ComplexBox::new(solution[row], solution[row + size]),
    ))
}
