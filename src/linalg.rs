//! Matrix and vector storage types used by linear-algebra operations.
//!
//! Fixed-size types are available with the `linalg` feature. Dynamically sized
//! types additionally require the `alloc` feature.

// NOTE: This is mostly coming from Jaroslav Horacek's PhD thesis:
// <https://kam.mff.cuni.cz/~horacek/source/horacek_phdthesis.pdf>
// I'll add a nice citation for this and associated papers later.

use core::ops::{Add, Div, Mul, Sub};

use nalgebra::{
    ArrayStorage, Const, DefaultAllocator, Dim, DimMin, Matrix, OMatrix, OVector, Scalar, Storage,
    allocator::{Allocator, SameShapeAllocator, SameShapeC, SameShapeR},
    constraint::{AreMultipliable, SameNumberOfColumns, SameNumberOfRows, ShapeConstraint},
    storage::Owned,
};

#[cfg(feature = "alloc")]
use nalgebra::{Dyn, VecStorage};

use crate::{
    IntervalOps,
    rounding::{self, Direction},
};

/// A nalgebra matrix whose entries are real intervals.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntervalMatrix<T, R, C, S>(Matrix<T, R, C, S>)
where
    T: IntervalOps + Scalar,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>;

/// An owned interval matrix with allocator-selected storage.
pub type OIntervalMatrix<T, R, C> = IntervalMatrix<T, R, C, Owned<T, R, C>>;
/// An owned interval column vector with allocator-selected storage.
pub type OIntervalVector<T, D> = OIntervalMatrix<T, D, Const<1>>;
/// A statically sized interval matrix.
pub type SIntervalMatrix<T, const R: usize, const C: usize> =
    IntervalMatrix<T, Const<R>, Const<C>, ArrayStorage<T, R, C>>;
/// A statically sized interval column vector.
pub type SIntervalVector<T, const D: usize> = SIntervalMatrix<T, D, 1>;

/// A dynamically sized interval matrix.
#[cfg(feature = "alloc")]
pub type DIntervalMatrix<T> = IntervalMatrix<T, Dyn, Dyn, VecStorage<T, Dyn, Dyn>>;
/// A dynamically sized interval column vector.
#[cfg(feature = "alloc")]
pub type DIntervalVector<T> = IntervalMatrix<T, Dyn, Const<1>, VecStorage<T, Dyn, Const<1>>>;

impl<T, R, C, S> IntervalMatrix<T, R, C, S>
where
    T: IntervalOps + Scalar,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
{
    /// Wraps a nalgebra matrix whose scalar type is an interval type.
    pub const fn from_inner(inner: Matrix<T, R, C, S>) -> Self {
        Self(inner)
    }

    /// Borrows the underlying nalgebra matrix.
    pub const fn as_inner(&self) -> &Matrix<T, R, C, S> {
        &self.0
    }

    /// Unwraps this value into its underlying nalgebra matrix.
    pub fn into_inner(self) -> Matrix<T, R, C, S> {
        self.0
    }
}

impl<T, R, C, S> From<Matrix<T, R, C, S>> for IntervalMatrix<T, R, C, S>
where
    T: IntervalOps + Scalar,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
{
    fn from(value: Matrix<T, R, C, S>) -> Self {
        Self(value)
    }
}

impl<T, R, C, S> From<Matrix<f64, R, C, S>> for OIntervalMatrix<T, R, C>
where
    T: IntervalOps + Scalar,
    R: Dim,
    C: Dim,
    S: Storage<f64, R, C>,
    DefaultAllocator: Allocator<R, C>,
{
    fn from(value: Matrix<f64, R, C, S>) -> Self {
        Self(value.map(T::from))
    }
}

impl<T, R, C, S> IntervalMatrix<T, R, C, S>
where
    T: IntervalOps + Scalar,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
{
    fn has_invalid_entries(&self) -> bool {
        self.0
            .iter()
            .copied()
            .any(|entry| entry.is_empty() || entry.is_nai())
    }
}

impl<T, R, C, S> IntervalMatrix<T, R, C, S>
where
    T: IntervalOps + Scalar,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
{
    /// Maps interval entries into an ordinary owned nalgebra matrix.
    pub fn map_inner<T2, F>(&self, f: F) -> OMatrix<T2, R, C>
    where
        T2: Scalar,
        F: FnMut(T) -> T2,
        DefaultAllocator: Allocator<R, C>,
    {
        self.0.map(f)
    }

    /// Maps interval entries into another owned interval matrix.
    pub fn map<T2, F>(&self, f: F) -> OIntervalMatrix<T2, R, C>
    where
        T2: IntervalOps + Scalar,
        F: FnMut(T) -> T2,
        DefaultAllocator: Allocator<R, C>,
    {
        IntervalMatrix::from_inner(self.map_inner(f))
    }
}

impl<T, R1, C1, SA> IntervalMatrix<T, R1, C1, SA>
where
    T: IntervalOps + Scalar,
    R1: Dim,
    C1: Dim,
    SA: Storage<T, R1, C1>,
{
    pub fn component_mul<SB>(
        &self,
        rhs: &IntervalMatrix<T, R1, C1, SB>,
    ) -> OIntervalMatrix<T, R1, C1>
    where
        SB: Storage<T, R1, C1>,
        DefaultAllocator: Allocator<R1, C1>,
    {
        assert_eq!(
            self.0.shape(),
            rhs.0.shape(),
            "interval matrix component_mul dimension mismatch",
        );
        let (nrows, ncols) = self.0.shape_generic();
        let elements = self
            .0
            .iter()
            .copied()
            .zip(rhs.0.iter().copied())
            .map(|(lhs, rhs)| Mul::mul(lhs, rhs));

        IntervalMatrix(Matrix::from_iterator_generic(nrows, ncols, elements))
    }

    pub fn component_div<SB>(
        &self,
        rhs: &IntervalMatrix<T, R1, C1, SB>,
    ) -> OIntervalMatrix<T, R1, C1>
    where
        SB: Storage<T, R1, C1>,
        DefaultAllocator: Allocator<R1, C1>,
    {
        assert_eq!(
            self.0.shape(),
            rhs.0.shape(),
            "interval matrix component_div dimension mismatch",
        );
        let (nrows, ncols) = self.0.shape_generic();
        let elements = self
            .0
            .iter()
            .copied()
            .zip(rhs.0.iter().copied())
            .map(|(lhs, rhs)| Div::div(lhs, rhs));

        IntervalMatrix(Matrix::from_iterator_generic(nrows, ncols, elements))
    }

    pub fn component_sub<SB>(
        &self,
        rhs: &IntervalMatrix<T, R1, C1, SB>,
    ) -> OIntervalMatrix<T, R1, C1>
    where
        SB: Storage<T, R1, C1>,
        DefaultAllocator: Allocator<R1, C1>,
    {
        assert_eq!(
            self.0.shape(),
            rhs.0.shape(),
            "interval matrix component_sub dimension mismatch",
        );
        let (nrows, ncols) = self.0.shape_generic();
        let elements = self
            .0
            .iter()
            .copied()
            .zip(rhs.0.iter().copied())
            .map(|(lhs, rhs)| Sub::sub(lhs, rhs));

        IntervalMatrix(Matrix::from_iterator_generic(nrows, ncols, elements))
    }

    pub fn scale(&self, value: T) -> OIntervalMatrix<T, R1, C1>
    where
        DefaultAllocator: Allocator<R1, C1>,
    {
        self.map(|a| Mul::mul(a, value))
    }

    pub fn unscale(&self, value: T) -> OIntervalMatrix<T, R1, C1>
    where
        DefaultAllocator: Allocator<R1, C1>,
    {
        self.map(|a| Div::div(a, value))
    }
}

impl<T, D, S> IntervalMatrix<T, D, D, S>
where
    T: IntervalOps + Scalar,
    D: Dim,
    S: Storage<T, D, D>,
{
    pub fn diagonal(&self) -> OIntervalVector<T, D>
    where
        DefaultAllocator: Allocator<D>,
    {
        IntervalMatrix::from_inner(self.0.diagonal())
    }

    pub fn is_z_matrix(&self) -> bool {
        // Def 4.4
        if !self.0.is_square() || self.has_invalid_entries() {
            return false;
        }
        let n = self.0.nrows();
        (0..n).all(|i| (0..n).all(|j| i == j || self.0[(i, j)].sup() <= 0.0))
    }
}

impl<T, D, S> IntervalMatrix<T, D, D, S>
where
    T: IntervalOps + Scalar,
    D: Dim,
    S: Storage<T, D, D>,
    DefaultAllocator: Allocator<D, D>,
{
    pub fn comparison_matrix(&self) -> Option<OMatrix<f64, D, D>> {
        if !self.0.is_square() || self.has_invalid_entries() {
            return None;
        }
        let (dim, _) = self.0.shape_generic();
        Some(OMatrix::from_fn_generic(dim, dim, |i, j| {
            if i == j {
                self.0[(i, j)].mig()
            } else {
                -self.0[(i, j)].mag()
            }
        }))
    }
}

impl<T, D, S> IntervalMatrix<T, D, D, S>
where
    T: IntervalOps + Scalar,
    D: Dim + DimMin<D, Output = D>,
    S: Storage<T, D, D>,
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    pub fn is_m_matrix(&self) -> bool {
        // Def 4.5
        if !self.is_z_matrix() {
            return false;
        }
        let n = self.0.nrows();
        if n == 0 {
            return false;
        }
        let (dim, _) = self.0.shape_generic();

        // Theorem 4.9(3)
        let a_inf = self.inf();
        let e = OVector::<f64, D>::repeat_generic(dim, Const::<1>, 1.0);
        let Some(u) = a_inf.lu().solve(&e) else {
            return false; // Theorem 4.9(4)
        };

        // Check inv(inf(A)) e = u > 0
        if u.iter().any(|&entry| !entry.is_finite() || entry <= 0.0) {
            return false;
        }

        // Check Au > 0
        (0..n).all(|i| {
            let mut row_sum = T::ZERO;
            for j in 0..n {
                row_sum = self.0[(i, j)].mul_add(T::singleton(u[j]), row_sum);
            }
            row_sum.inf() > 0.0
        })
    }

    pub fn is_h_matrix(&self) -> bool {
        let Some(comparison) = self.comparison_matrix() else {
            return false; // not square or invalid entries
        };
        let n = comparison.nrows();
        if n == 0 || comparison.iter().any(|entry| !entry.is_finite()) {
            return false;
        }
        let (dim, _) = self.0.shape_generic();
        // Theorem 4.18(3)
        let e = OVector::<f64, D>::repeat_generic(dim, Const::<1>, 1.0);
        let Some(u) = comparison.clone().lu().solve(&e) else {
            return false;
        };
        if u.iter().any(|&entry| !entry.is_finite() || entry <= 0.0) {
            return false;
        }
        // NOTE: Technically the following rechecks this, but it uses IA. The previous check is just
        // a fast path to check for failure quickly.
        (0..n).all(|i| {
            let mut row_sum = T::ZERO;
            for j in 0..n {
                let c_ij = T::singleton(comparison[(i, j)]);
                let u_j = T::singleton(u[j]);
                row_sum = c_ij.mul_add(u_j, row_sum);
            }
            row_sum.inf() > 0.0
        })
    }

    pub fn is_strongly_regular(&self) -> bool {
        if !self.0.is_square() || self.0.nrows() == 0 || self.has_invalid_entries() {
            return false;
        }

        let a_c = self.mid();
        if a_c.iter().any(|entry| !entry.is_finite()) {
            return false;
        }
        let Some(c) = a_c.try_inverse() else {
            return false;
        };

        if c.iter().any(|entry| !entry.is_finite()) {
            return false;
        }
        let c_interval: OIntervalMatrix<T, D, D> = c.into();
        // Theorem 4.33(5)
        let preconditioned = Mul::mul(&c_interval, self);
        preconditioned.is_h_matrix()
    }
}

impl<T, R, C, S> IntervalMatrix<T, R, C, S>
where
    T: IntervalOps + Scalar,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
    DefaultAllocator: Allocator<R, C>,
{
    pub fn inf(&self) -> OMatrix<f64, R, C> {
        self.map_inner(IntervalOps::inf)
    }
    pub fn sup(&self) -> OMatrix<f64, R, C> {
        self.map_inner(IntervalOps::sup)
    }
}

impl<T, R, C, S> IntervalMatrix<T, R, C, S>
where
    T: IntervalOps + Scalar,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
    DefaultAllocator: Allocator<R, C>,
{
    pub fn rad(&self) -> OMatrix<f64, R, C> {
        self.map_inner(IntervalOps::rad)
    }
    pub fn inner_rad(&self) -> OMatrix<f64, R, C> {
        self.map_inner(IntervalOps::inner_rad)
    }
}

impl<T, R, C, S> IntervalMatrix<T, R, C, S>
where
    T: IntervalOps + Scalar,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
    DefaultAllocator: Allocator<R, C>,
{
    pub fn mid(&self) -> OMatrix<f64, R, C> {
        self.map_inner(IntervalOps::mid)
    }
}

impl<T, R, C, S> IntervalMatrix<T, R, C, S>
where
    T: IntervalOps + Scalar,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
    DefaultAllocator: Allocator<R, C>,
{
    pub fn mag(&self) -> OMatrix<f64, R, C> {
        self.map_inner(IntervalOps::mag)
    }

    pub fn mig(&self) -> OMatrix<f64, R, C> {
        self.map_inner(IntervalOps::mig)
    }

    pub fn norm1(&self) -> f64 {
        let mut norm: f64 = 0.0;
        for column in self.mag().column_iter() {
            let mut sum = 0.0;
            for row_entry in column.iter() {
                sum = rounding::add(sum, *row_entry, Direction::Up);
                if sum.is_nan() {
                    return f64::NAN;
                }
                norm = norm.max(sum);
            }
        }
        norm
    }

    pub fn inf_norm(&self) -> f64 {
        let mut norm: f64 = 0.0;
        for row in self.mag().row_iter() {
            let mut sum = 0.0;
            for column_entry in row.iter() {
                sum = rounding::add(sum, *column_entry, Direction::Up);
                if sum.is_nan() {
                    return f64::NAN;
                }
                norm = norm.max(sum);
            }
        }
        norm
    }
}

impl<'a, 'b, T, R1, C1, R2, C2, SA, SB> Add<&'b IntervalMatrix<T, R2, C2, SB>>
    for &'a IntervalMatrix<T, R1, C1, SA>
where
    T: IntervalOps + Scalar,
    R1: Dim,
    C1: Dim,
    R2: Dim,
    C2: Dim,
    SA: Storage<T, R1, C1>,
    SB: Storage<T, R2, C2>,
    DefaultAllocator: SameShapeAllocator<R1, C1, R2, C2>,
    ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>,
{
    type Output = OIntervalMatrix<T, SameShapeR<R1, R2>, SameShapeC<C1, C2>>;

    fn add(self, rhs: &'b IntervalMatrix<T, R2, C2, SB>) -> Self::Output {
        assert_eq!(
            self.0.shape(),
            rhs.0.shape(),
            "interval matrix addition dimension mismatch",
        );
        let nrows: SameShapeR<R1, R2> = Dim::from_usize(self.0.nrows());
        let ncols: SameShapeC<C1, C2> = Dim::from_usize(self.0.ncols());
        let elements = self
            .0
            .iter()
            .copied()
            .zip(rhs.0.iter().copied())
            .map(|(lhs, rhs)| Add::add(lhs, rhs));

        IntervalMatrix(Matrix::from_iterator_generic(nrows, ncols, elements))
    }
}

impl<'a, 'b, T, R1, C1, R2, C2, SA, SB> Sub<&'b IntervalMatrix<T, R2, C2, SB>>
    for &'a IntervalMatrix<T, R1, C1, SA>
where
    T: IntervalOps + Scalar,
    R1: Dim,
    C1: Dim,
    R2: Dim,
    C2: Dim,
    SA: Storage<T, R1, C1>,
    SB: Storage<T, R2, C2>,
    DefaultAllocator: SameShapeAllocator<R1, C1, R2, C2>,
    ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>,
{
    type Output = OIntervalMatrix<T, SameShapeR<R1, R2>, SameShapeC<C1, C2>>;

    fn sub(self, rhs: &'b IntervalMatrix<T, R2, C2, SB>) -> Self::Output {
        assert_eq!(
            self.0.shape(),
            rhs.0.shape(),
            "interval matrix subtraction dimension mismatch",
        );
        let nrows: SameShapeR<R1, R2> = Dim::from_usize(self.0.nrows());
        let ncols: SameShapeC<C1, C2> = Dim::from_usize(self.0.ncols());
        let elements = self
            .0
            .iter()
            .copied()
            .zip(rhs.0.iter().copied())
            .map(|(lhs, rhs)| Sub::sub(lhs, rhs));

        IntervalMatrix(Matrix::from_iterator_generic(nrows, ncols, elements))
    }
}

impl<'a, 'b, T, R1, C1, R2, C2, SA, SB> Mul<&'b IntervalMatrix<T, R2, C2, SB>>
    for &'a IntervalMatrix<T, R1, C1, SA>
where
    T: IntervalOps + Scalar,
    R1: Dim,
    C1: Dim,
    R2: Dim,
    C2: Dim,
    SA: Storage<T, R1, C1>,
    SB: Storage<T, R2, C2>,
    DefaultAllocator: Allocator<R1, C2>,
    ShapeConstraint: AreMultipliable<R1, C1, R2, C2>,
{
    type Output = OIntervalMatrix<T, R1, C2>;

    fn mul(self, rhs: &'b IntervalMatrix<T, R2, C2, SB>) -> Self::Output {
        assert_eq!(
            self.0.ncols(),
            rhs.0.nrows(),
            "interval matrix multiplication dimension mismatch",
        );
        let (nrows, _) = self.0.shape_generic();
        let (_, ncols) = rhs.0.shape_generic();
        let mut output = OMatrix::<T, R1, C2>::from_element_generic(nrows, ncols, T::ZERO);
        for (mut output_column, rhs_column) in output.column_iter_mut().zip(rhs.0.column_iter()) {
            for (lhs_column, rhs_entry) in self.0.column_iter().zip(rhs_column.iter().copied()) {
                for (output_entry, lhs_entry) in
                    output_column.iter_mut().zip(lhs_column.iter().copied())
                {
                    *output_entry = lhs_entry.mul_add(rhs_entry, *output_entry);
                }
            }
        }
        IntervalMatrix(output)
    }
}

pub trait Solver<T, D, SA, SB>
where
    T: IntervalOps + Scalar,
    D: Dim,
    SA: Storage<T, D, D>,
    SB: Storage<T, D, Const<1>>,
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    fn solve(
        &self,
        lhs: &IntervalMatrix<T, D, D, SA>,
        rhs: &IntervalMatrix<T, D, Const<1>, SB>,
    ) -> Option<OIntervalVector<T, D>>;
}

/// Epsilon-inflation method
mod ei;
pub use ei::EpsilonInflation;

/// Gaussian elimination
mod ge;
pub use ge::GaussianElimination;

// Hansen-Bliek-Rohn-Ning-Kearfott-Neumaier method
mod hbr;
pub use hbr::HBR;
