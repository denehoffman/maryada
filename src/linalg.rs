//! Matrix and vector storage types used by linear-algebra operations.
//!
//! Fixed-size types are available with the `linalg` feature. Dynamically sized
//! types additionally require the `alloc` feature.

// NOTE: This is mostly coming from Jaroslav Horacek's PhD thesis:
// <https://kam.mff.cuni.cz/~horacek/source/horacek_phdthesis.pdf>
// I'll add a nice citation for this and associated papers later.

use core::ops::{Add, Mul, Sub};

use nalgebra::{
    DefaultAllocator, Dim, DimMin, DimName, Matrix, MatrixSum, OMatrix, Scalar, Storage,
    allocator::{Allocator, SameShapeAllocator, SameShapeC, SameShapeR},
    constraint::{AreMultipliable, SameNumberOfColumns, SameNumberOfRows, ShapeConstraint},
};

use crate::{
    DecoratedInterval, Enclosure, EnclosureArithmetic, Interval, Magnitude, Midpoint, Radius,
    rounding::{self, Direction},
};

#[repr(transparent)]
pub struct IntervalMatrix<M>(M);

impl<M> IntervalMatrix<M> {
    pub const fn from_inner(inner: M) -> Self {
        Self(inner)
    }

    pub fn as_inner(&self) -> &M {
        &self.0
    }

    pub fn into_inner(self) -> M {
        self.0
    }
}

impl<T, R, C, S> From<Matrix<f64, R, C, S>> for IntervalMatrix<OMatrix<T, R, C>>
where
    T: Enclosure + Scalar,
    R: Dim,
    C: Dim,
    S: Storage<f64, R, C>,
    DefaultAllocator: Allocator<R, C>,
{
    fn from(value: Matrix<f64, R, C, S>) -> Self {
        Self(value.map(T::from_f64))
    }
}

fn enclose<T, R, C, S>(mat: Matrix<f64, R, C, S>) -> IntervalMatrix<OMatrix<T, R, C>>
where
    T: Enclosure + Scalar,
    R: Dim,
    C: Dim,
    S: Storage<f64, R, C>,
    DefaultAllocator: Allocator<R, C>,
{
    mat.into()
}

impl<T, R, C, S> IntervalMatrix<Matrix<T, R, C, S>>
where
    T: Enclosure + Scalar,
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
// impl<R, C, S> From<Matrix<f64, R, C, S>> for IntervalMatrix<OMatrix<Interval, R, C>>
// where
//     R: Dim,
//     C: Dim,
//     S: Storage<f64, R, C>,
//     DefaultAllocator: Allocator<R, C>,
// {
//     fn from(value: Matrix<f64, R, C, S>) -> Self {
//         Self(value.map(Interval::from))
//     }
// }
//
// impl<R, C, S> From<Matrix<f64, R, C, S>> for IntervalMatrix<OMatrix<DecoratedInterval, R, C>>
// where
//     R: Dim,
//     C: Dim,
//     S: Storage<f64, R, C>,
//     DefaultAllocator: Allocator<R, C>,
// {
//     fn from(value: Matrix<f64, R, C, S>) -> Self {
//         Self(value.map(DecoratedInterval::from))
//     }
// }

impl<T, R1, C1, SA> IntervalMatrix<Matrix<T, R1, C1, SA>>
where
    T: EnclosureArithmetic + Scalar,
    R1: Dim,
    C1: Dim,
    SA: Storage<T, R1, C1>,
{
    pub fn component_mul<R2, C2, SB>(
        &self,
        rhs: &IntervalMatrix<Matrix<T, R2, C2, SB>>,
    ) -> IntervalMatrix<MatrixSum<T, R1, C1, R2, C2>>
    where
        R2: Dim,
        C2: Dim,
        SB: Storage<T, R2, C2>,
        DefaultAllocator: SameShapeAllocator<R1, C1, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>,
    {
        assert_eq!(
            self.0.shape(),
            rhs.0.shape(),
            "interval matrix component_mul dimension mismatch",
        );
        let nrows: SameShapeR<R1, R2> = Dim::from_usize(self.0.nrows());
        let ncols: SameShapeC<C1, C2> = Dim::from_usize(self.0.ncols());
        let elements = self
            .0
            .iter()
            .copied()
            .zip(rhs.0.iter().copied())
            .map(|(lhs, rhs)| EnclosureArithmetic::mul(lhs, rhs));

        IntervalMatrix(Matrix::from_iterator_generic(nrows, ncols, elements))
    }

    pub fn component_div<R2, C2, SB>(
        &self,
        rhs: &IntervalMatrix<Matrix<T, R2, C2, SB>>,
    ) -> IntervalMatrix<MatrixSum<T, R1, C1, R2, C2>>
    where
        R2: Dim,
        C2: Dim,
        SB: Storage<T, R2, C2>,
        DefaultAllocator: SameShapeAllocator<R1, C1, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>,
    {
        assert_eq!(
            self.0.shape(),
            rhs.0.shape(),
            "interval matrix component_mul dimension mismatch",
        );
        let nrows: SameShapeR<R1, R2> = Dim::from_usize(self.0.nrows());
        let ncols: SameShapeC<C1, C2> = Dim::from_usize(self.0.ncols());
        let elements = self
            .0
            .iter()
            .copied()
            .zip(rhs.0.iter().copied())
            .map(|(lhs, rhs)| EnclosureArithmetic::div(lhs, rhs));

        IntervalMatrix(Matrix::from_iterator_generic(nrows, ncols, elements))
    }

    pub fn scale(&self, value: T) -> IntervalMatrix<OMatrix<T, R1, C1>>
    where
        DefaultAllocator: Allocator<R1, C1>,
    {
        let nrows = Dim::from_usize(self.0.nrows());
        let ncols = Dim::from_usize(self.0.ncols());
        IntervalMatrix(Matrix::from_iterator_generic(
            nrows,
            ncols,
            self.0
                .iter()
                .copied()
                .map(|a| EnclosureArithmetic::mul(a, value)),
        ))
    }

    pub fn unscale(&self, value: T) -> IntervalMatrix<OMatrix<T, R1, C1>>
    where
        DefaultAllocator: Allocator<R1, C1>,
    {
        let nrows = Dim::from_usize(self.0.nrows());
        let ncols = Dim::from_usize(self.0.ncols());
        IntervalMatrix(Matrix::from_iterator_generic(
            nrows,
            ncols,
            self.0
                .iter()
                .copied()
                .map(|a| EnclosureArithmetic::div(a, value)),
        ))
    }
}

pub enum RegularityResult {
    ProvenRegular,
    ProvenSingular,
    Inconclusive,
}

impl<T, D, S> IntervalMatrix<Matrix<T, D, D, S>>
where
    T: EnclosureArithmetic + Magnitude + Midpoint<Point = f64> + Radius + Scalar,
    D: DimMin<D, Output = D> + DimName,
    S: Storage<T, D, D>,
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    pub fn is_regular(&self) -> RegularityResult {
        if self.has_invalid_entries() {
            return RegularityResult::Inconclusive;
        }
        let midpoint = self.mid();
        if midpoint.iter().any(|value| !value.is_finite()) {
            return RegularityResult::Inconclusive;
        }
        let Some(inv) = midpoint.try_inverse() else {
            return RegularityResult::Inconclusive;
        };
        if inv.iter().any(|value| !value.is_finite()) {
            return RegularityResult::Inconclusive;
        }
        let inv_enc = enclose::<T, D, D, _>(inv);
        let (nrows, ncols) = self.0.shape_generic();
        // Beeck
        let id: IntervalMatrix<OMatrix<T, D, D>> =
            IntervalMatrix::from_inner(Matrix::from_fn_generic(nrows, ncols, |i, j| {
                if i == j { T::one() } else { T::zero() }
            }));
        let preconditioned = &inv_enc * self;
        let residual = &id - &preconditioned;
        let bound = residual.inf_norm();
        if bound.is_finite() && bound < 1.0 {
            return RegularityResult::ProvenRegular;
        }
        // Rump

        todo!()
    }
}

impl<T, R, C, S> IntervalMatrix<Matrix<T, R, C, S>>
where
    T: Radius + Scalar,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
    DefaultAllocator: Allocator<R, C>,
{
    pub fn rad(&self) -> OMatrix<f64, R, C> {
        self.0.map(Radius::rad)
    }
    pub fn inner_rad(&self) -> OMatrix<f64, R, C> {
        self.0.map(Radius::inner_rad)
    }
}

impl<T, R, C, S> IntervalMatrix<Matrix<T, R, C, S>>
where
    T: Midpoint<Point = f64> + Scalar,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
    DefaultAllocator: Allocator<R, C>,
{
    pub fn mid(&self) -> OMatrix<f64, R, C> {
        self.0.map(Midpoint::mid)
    }
}

impl<T, R, C, S> IntervalMatrix<Matrix<T, R, C, S>>
where
    T: Magnitude + Scalar,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
    DefaultAllocator: Allocator<R, C>,
{
    pub fn mag(&self) -> OMatrix<f64, R, C> {
        self.0.map(Magnitude::mag)
    }

    pub fn mig(&self) -> OMatrix<f64, R, C> {
        self.0.map(Magnitude::mig)
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

impl<'a, 'b, T, R1, C1, R2, C2, SA, SB> Add<&'b IntervalMatrix<Matrix<T, R2, C2, SB>>>
    for &'a IntervalMatrix<Matrix<T, R1, C1, SA>>
where
    T: EnclosureArithmetic + Scalar,
    R1: Dim,
    C1: Dim,
    R2: Dim,
    C2: Dim,
    SA: Storage<T, R1, C1>,
    SB: Storage<T, R2, C2>,
    DefaultAllocator: SameShapeAllocator<R1, C1, R2, C2>,
    ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>,
{
    type Output = IntervalMatrix<MatrixSum<T, R1, C1, R2, C2>>;

    fn add(self, rhs: &'b IntervalMatrix<Matrix<T, R2, C2, SB>>) -> Self::Output {
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
            .map(|(lhs, rhs)| EnclosureArithmetic::add(lhs, rhs));

        IntervalMatrix(Matrix::from_iterator_generic(nrows, ncols, elements))
    }
}

impl<'a, 'b, T, R1, C1, R2, C2, SA, SB> Sub<&'b IntervalMatrix<Matrix<T, R2, C2, SB>>>
    for &'a IntervalMatrix<Matrix<T, R1, C1, SA>>
where
    T: EnclosureArithmetic + Scalar,
    R1: Dim,
    C1: Dim,
    R2: Dim,
    C2: Dim,
    SA: Storage<T, R1, C1>,
    SB: Storage<T, R2, C2>,
    DefaultAllocator: SameShapeAllocator<R1, C1, R2, C2>,
    ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>,
{
    type Output = IntervalMatrix<MatrixSum<T, R1, C1, R2, C2>>;

    fn sub(self, rhs: &'b IntervalMatrix<Matrix<T, R2, C2, SB>>) -> Self::Output {
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
            .map(|(lhs, rhs)| EnclosureArithmetic::sub(lhs, rhs));

        IntervalMatrix(Matrix::from_iterator_generic(nrows, ncols, elements))
    }
}

impl<'a, 'b, T, R1, C1, R2, C2, SA, SB> Mul<&'b IntervalMatrix<Matrix<T, R2, C2, SB>>>
    for &'a IntervalMatrix<Matrix<T, R1, C1, SA>>
where
    T: EnclosureArithmetic + Scalar,
    R1: Dim,
    C1: Dim,
    R2: Dim,
    C2: Dim,
    SA: Storage<T, R1, C1>,
    SB: Storage<T, R2, C2>,
    DefaultAllocator: Allocator<R1, C2>,
    ShapeConstraint:
        AreMultipliable<R1, C1, R2, C2> + SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>,
{
    type Output = IntervalMatrix<OMatrix<T, R1, C2>>;

    fn mul(self, rhs: &'b IntervalMatrix<Matrix<T, R2, C2, SB>>) -> Self::Output {
        assert_eq!(
            self.0.ncols(),
            rhs.0.nrows(),
            "interval matrix multiplication dimension mismatch",
        );
        let (nrows, _) = self.0.shape_generic();
        let (_, ncols) = rhs.0.shape_generic();
        let mut output =
            OMatrix::<T, R1, C2>::from_element_generic(nrows, ncols, <T as Enclosure>::zero());
        for (mut output_column, rhs_column) in output.column_iter_mut().zip(rhs.0.column_iter()) {
            for (lhs_column, rhs_entry) in self.0.column_iter().zip(rhs_column.iter().copied()) {
                for (output_entry, lhs_entry) in
                    output_column.iter_mut().zip(lhs_column.iter().copied())
                {
                    *output_entry =
                        EnclosureArithmetic::mul_add(lhs_entry, rhs_entry, *output_entry);
                }
            }
        }
        IntervalMatrix(output)
    }
}
