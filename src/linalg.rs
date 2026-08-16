//! Matrix and vector storage types used by linear-algebra operations.
//!
//! Fixed-size types are available with the `linalg` feature. Dynamically sized
//! types additionally require the `alloc` feature.
//!
//! The majority of the algorithmic work here was written by following the thesis of Jaroslav Horáček (see reference).
//!
//! # References
//!
//! J. Horáček, *Interval Linear and Nonlinear Systems*, Ph.D. thesis, Charles University, 2019. [https://dspace.cuni.cz/handle/20.500.11956/111301](https://dspace.cuni.cz/handle/20.500.11956/111301)

use core::{
    fmt,
    fmt::Write as _,
    ops::{Index, IndexMut, Mul},
};

use nalgebra::{
    ArrayStorage, Const, DefaultAllocator, Dim, DimMin, Matrix, OMatrix, OVector, Scalar, Storage,
    StorageMut,
    allocator::{Allocator, SameShapeAllocator, SameShapeC, SameShapeR},
    constraint::{SameNumberOfColumns, SameNumberOfRows, ShapeConstraint},
    iter::{ColumnIter, ColumnIterMut, MatrixIter, MatrixIterMut, RowIter, RowIterMut},
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
/// An owned interval row vector with allocator-selected storage.
pub type OIntervalRowVector<T, D> = OIntervalMatrix<T, Const<1>, D>;
/// A statically sized interval matrix.
pub type SIntervalMatrix<T, const R: usize, const C: usize> =
    IntervalMatrix<T, Const<R>, Const<C>, ArrayStorage<T, R, C>>;
/// A statically sized interval column vector.
pub type SIntervalVector<T, const D: usize> = SIntervalMatrix<T, D, 1>;
/// A statically sized interval row vector.
pub type SIntervalRowVector<T, const D: usize> = SIntervalMatrix<T, 1, D>;

/// A dynamically sized interval matrix.
#[cfg(feature = "alloc")]
pub type DIntervalMatrix<T> = IntervalMatrix<T, Dyn, Dyn, VecStorage<T, Dyn, Dyn>>;
/// A dynamically sized interval column vector.
#[cfg(feature = "alloc")]
pub type DIntervalVector<T> = IntervalMatrix<T, Dyn, Const<1>, VecStorage<T, Dyn, Const<1>>>;
/// A dynamically sized interval row vector.
#[cfg(feature = "alloc")]
pub type DIntervalRowVector<T> = IntervalMatrix<T, Const<1>, Dyn, VecStorage<T, Const<1>, Dyn>>;

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

    /// Returns the number of rows and columns.
    pub fn shape(&self) -> (usize, usize) {
        self.0.shape()
    }

    /// Returns the row and column dimensions at the type level.
    pub fn shape_generic(&self) -> (R, C) {
        self.0.shape_generic()
    }

    /// Returns the number of rows.
    pub fn nrows(&self) -> usize {
        self.0.nrows()
    }

    /// Returns the number of columns.
    pub fn ncols(&self) -> usize {
        self.0.ncols()
    }

    /// Returns the number of entries.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns whether this matrix has no entries.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns whether this matrix has the same number of rows and columns.
    pub fn is_square(&self) -> bool {
        self.0.is_square()
    }

    /// Returns an entry, or `None` if the row or column is out of bounds.
    pub fn get(&self, row: usize, column: usize) -> Option<&T> {
        self.0.get((row, column))
    }

    /// Iterates over entries in column-major order.
    pub fn iter(&self) -> MatrixIter<'_, T, R, C, S> {
        self.0.iter()
    }

    /// Iterates over rows.
    pub const fn row_iter(&self) -> RowIter<'_, T, R, C, S> {
        self.0.row_iter()
    }

    /// Iterates over columns.
    pub fn column_iter(&self) -> ColumnIter<'_, T, R, C, S> {
        self.0.column_iter()
    }
}

struct CharCounter(usize);

impl fmt::Write for CharCounter {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        self.0 = self.0.saturating_add(text.chars().count());
        Ok(())
    }
}

macro_rules! impl_matrix_format {
    ($trait:path, $without_precision:literal, $with_precision:literal) => {
        impl<T, R, C, S> $trait for IntervalMatrix<T, R, C, S>
        where
            T: IntervalOps + Scalar + $trait,
            R: Dim,
            C: Dim,
            S: Storage<T, R, C>,
        {
            #[allow(clippy::arithmetic_side_effects, clippy::indexing_slicing)]
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                fn value_width<T: $trait>(
                    value: &T,
                    precision: Option<usize>,
                ) -> Result<usize, fmt::Error> {
                    let mut counter = CharCounter(0);
                    match precision {
                        Some(precision) => {
                            write!(&mut counter, $with_precision, value, precision)?;
                        }
                        None => {
                            write!(&mut counter, $without_precision, value)?;
                        }
                    }
                    Ok(counter.0)
                }

                let (rows, columns) = self.shape();
                if rows == 0 || columns == 0 {
                    return formatter.write_str("[ ]");
                }

                let precision = formatter.precision();
                let mut element_width = 0;
                for row in 0..rows {
                    for column in 0..columns {
                        element_width =
                            element_width.max(value_width(&self[(row, column)], precision)?);
                    }
                }
                let cell_width = element_width + 1;
                let interior_width = cell_width * columns - 1;

                writeln!(formatter)?;
                writeln!(formatter, "  ┌ {:>interior_width$} ┐", "")?;
                for row in 0..rows {
                    formatter.write_str("  │")?;
                    for column in 0..columns {
                        let value = &self[(row, column)];
                        let padding = element_width - value_width(value, precision)?;
                        write!(formatter, " {:padding$}", "")?;
                        match precision {
                            Some(precision) => {
                                write!(formatter, $with_precision, value, precision)?;
                            }
                            None => write!(formatter, $without_precision, value)?,
                        }
                    }
                    writeln!(formatter, " │")?;
                }
                writeln!(formatter, "  └ {:>interior_width$} ┘", "")?;
                writeln!(formatter)
            }
        }
    };
}

impl_matrix_format!(fmt::Display, "{}", "{:.1$}");
impl_matrix_format!(fmt::LowerExp, "{:e}", "{:.1$e}");
impl_matrix_format!(fmt::UpperExp, "{:E}", "{:.1$E}");
impl_matrix_format!(fmt::LowerHex, "{:x}", "{:.1$x}");
impl_matrix_format!(fmt::UpperHex, "{:X}", "{:.1$X}");

impl<T, R, C, S> IntervalMatrix<T, R, C, S>
where
    T: IntervalOps + Scalar,
    R: Dim,
    C: Dim,
    S: StorageMut<T, R, C>,
{
    /// Mutably borrows the underlying nalgebra matrix.
    pub const fn as_inner_mut(&mut self) -> &mut Matrix<T, R, C, S> {
        &mut self.0
    }

    /// Returns a mutable entry, or `None` if the row or column is out of bounds.
    pub fn get_mut(&mut self, row: usize, column: usize) -> Option<&mut T> {
        self.0.get_mut((row, column))
    }

    /// Mutably iterates over entries in column-major order.
    pub fn iter_mut(&mut self) -> MatrixIterMut<'_, T, R, C, S> {
        self.0.iter_mut()
    }

    /// Mutably iterates over rows.
    pub const fn row_iter_mut(&mut self) -> RowIterMut<'_, T, R, C, S> {
        self.0.row_iter_mut()
    }

    /// Mutably iterates over columns.
    pub fn column_iter_mut(&mut self) -> ColumnIterMut<'_, T, R, C, S> {
        self.0.column_iter_mut()
    }
}

impl<T, R, C, S> AsRef<Matrix<T, R, C, S>> for IntervalMatrix<T, R, C, S>
where
    T: IntervalOps + Scalar,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
{
    fn as_ref(&self) -> &Matrix<T, R, C, S> {
        self.as_inner()
    }
}

impl<T, R, C, S> AsMut<Matrix<T, R, C, S>> for IntervalMatrix<T, R, C, S>
where
    T: IntervalOps + Scalar,
    R: Dim,
    C: Dim,
    S: StorageMut<T, R, C>,
{
    fn as_mut(&mut self) -> &mut Matrix<T, R, C, S> {
        self.as_inner_mut()
    }
}

impl<'a, T, R, C, S> IntoIterator for &'a IntervalMatrix<T, R, C, S>
where
    T: IntervalOps + Scalar,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
{
    type Item = &'a T;
    type IntoIter = MatrixIter<'a, T, R, C, S>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, R, C, S> IntoIterator for &'a mut IntervalMatrix<T, R, C, S>
where
    T: IntervalOps + Scalar,
    R: Dim,
    C: Dim,
    S: StorageMut<T, R, C>,
{
    type Item = &'a mut T;
    type IntoIter = MatrixIterMut<'a, T, R, C, S>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T, R, C, S> Index<(usize, usize)> for IntervalMatrix<T, R, C, S>
where
    T: IntervalOps + Scalar,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
{
    type Output = T;

    #[allow(clippy::indexing_slicing)]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.0[index]
    }
}

impl<T, R, C, S> IndexMut<(usize, usize)> for IntervalMatrix<T, R, C, S>
where
    T: IntervalOps + Scalar,
    R: Dim,
    C: Dim,
    S: StorageMut<T, R, C>,
{
    #[allow(clippy::indexing_slicing)]
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<T, R, S> Index<usize> for IntervalMatrix<T, R, Const<1>, S>
where
    T: IntervalOps + Scalar,
    R: Dim,
    S: Storage<T, R, Const<1>>,
{
    type Output = T;

    #[allow(clippy::indexing_slicing)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T, R, S> IndexMut<usize> for IntervalMatrix<T, R, Const<1>, S>
where
    T: IntervalOps + Scalar,
    R: Dim,
    S: StorageMut<T, R, Const<1>>,
{
    #[allow(clippy::indexing_slicing)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
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

impl<T, R, C> OIntervalMatrix<T, R, C>
where
    T: IntervalOps + Scalar,
    R: Dim,
    C: Dim,
    DefaultAllocator: Allocator<R, C>,
{
    /// Creates an owned interval matrix filled with one value.
    pub fn from_element_generic(nrows: R, ncols: C, element: T) -> Self {
        OMatrix::from_element_generic(nrows, ncols, element).into()
    }

    /// Creates an owned interval matrix by calling `f` for each `(row, column)`.
    pub fn from_fn_generic<F>(nrows: R, ncols: C, f: F) -> Self
    where
        F: FnMut(usize, usize) -> T,
    {
        OMatrix::from_fn_generic(nrows, ncols, f).into()
    }

    /// Creates an owned interval matrix from a column-major iterator.
    pub fn from_iterator_generic<I>(nrows: R, ncols: C, iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        OMatrix::from_iterator_generic(nrows, ncols, iter).into()
    }

    /// Creates an owned interval matrix from a row-major iterator.
    pub fn from_row_iterator_generic<I>(nrows: R, ncols: C, iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        OMatrix::from_row_iterator_generic(nrows, ncols, iter).into()
    }

    /// Creates an owned interval matrix from a row-major slice.
    pub fn from_row_slice_generic(nrows: R, ncols: C, entries: &[T]) -> Self {
        OMatrix::from_row_slice_generic(nrows, ncols, entries).into()
    }

    /// Creates an owned interval matrix from a column-major slice.
    pub fn from_column_slice_generic(nrows: R, ncols: C, entries: &[T]) -> Self {
        OMatrix::from_column_slice_generic(nrows, ncols, entries).into()
    }

    /// Creates an owned interval matrix from interval row vectors.
    ///
    /// # Panics
    ///
    /// Panics if the supplied dimensions do not match the number or width of the rows.
    #[allow(clippy::indexing_slicing)]
    pub fn from_rows_generic<SR>(
        nrows: R,
        ncols: C,
        rows: &[IntervalMatrix<T, Const<1>, C, SR>],
    ) -> Self
    where
        SR: Storage<T, Const<1>, C>,
    {
        assert_eq!(
            rows.len(),
            nrows.value(),
            "interval matrix row count mismatch"
        );
        assert!(
            rows.iter().all(|row| row.ncols() == ncols.value()),
            "interval matrix row width mismatch",
        );
        Self::from_fn_generic(nrows, ncols, |i, j| rows[i][(0, j)])
    }

    /// Creates an owned interval matrix from interval column vectors.
    ///
    /// # Panics
    ///
    /// Panics if the supplied dimensions do not match the number or height of the columns.
    #[allow(clippy::indexing_slicing)]
    pub fn from_columns_generic<SC>(
        nrows: R,
        ncols: C,
        columns: &[IntervalMatrix<T, R, Const<1>, SC>],
    ) -> Self
    where
        SC: Storage<T, R, Const<1>>,
    {
        assert_eq!(
            columns.len(),
            ncols.value(),
            "interval matrix column count mismatch",
        );
        assert!(
            columns.iter().all(|column| column.nrows() == nrows.value()),
            "interval matrix column height mismatch",
        );
        Self::from_fn_generic(nrows, ncols, |i, j| columns[j][i])
    }

    /// Creates singleton intervals from an ordinary real matrix.
    pub fn from_singletons<S2>(values: &Matrix<f64, R, C, S2>) -> Self
    where
        S2: Storage<f64, R, C>,
    {
        let (nrows, ncols) = values.shape_generic();
        Self::from_iterator_generic(nrows, ncols, values.iter().copied().map(T::singleton))
    }

    /// Creates intervals from corresponding lower- and upper-bound matrices.
    ///
    /// # Panics
    ///
    /// Panics if `lower` and `upper` have different dimensions.
    pub fn from_bounds<SL, SU>(lower: &Matrix<f64, R, C, SL>, upper: &Matrix<f64, R, C, SU>) -> Self
    where
        SL: Storage<f64, R, C>,
        SU: Storage<f64, R, C>,
    {
        assert_eq!(
            lower.shape(),
            upper.shape(),
            "interval matrix bounds dimension mismatch"
        );
        let (nrows, ncols) = lower.shape_generic();
        let entries = lower
            .iter()
            .copied()
            .zip(upper.iter().copied())
            .map(|(lower, upper)| T::new(lower, upper));
        Self::from_iterator_generic(nrows, ncols, entries)
    }

    /// Creates intervals from corresponding midpoint and radius matrices.
    ///
    /// # Panics
    ///
    /// Panics if `midpoint` and `radius` have different dimensions.
    pub fn from_mid_rad<SM, SR>(
        midpoint: &Matrix<f64, R, C, SM>,
        radius: &Matrix<f64, R, C, SR>,
    ) -> Self
    where
        SM: Storage<f64, R, C>,
        SR: Storage<f64, R, C>,
    {
        assert_eq!(
            midpoint.shape(),
            radius.shape(),
            "interval matrix midpoint-radius dimension mismatch",
        );
        let (nrows, ncols) = midpoint.shape_generic();
        let entries =
            midpoint
                .iter()
                .copied()
                .zip(radius.iter().copied())
                .map(|(midpoint, radius)| {
                    T::new(
                        rounding::sub(midpoint, radius, Direction::Down),
                        rounding::add(midpoint, radius, Direction::Up),
                    )
                });
        Self::from_iterator_generic(nrows, ncols, entries)
    }

    /// Creates an owned interval matrix filled with singleton zero intervals.
    pub fn zeros_generic(nrows: R, ncols: C) -> Self {
        Self::from_element_generic(nrows, ncols, T::ZERO)
    }
}

impl<T, D> OIntervalMatrix<T, D, D>
where
    T: IntervalOps + Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D, D>,
{
    /// Creates an identity interval matrix.
    pub fn identity_generic(dim: D) -> Self {
        OMatrix::from_fn_generic(dim, dim, |i, j| if i == j { T::ONE } else { T::ZERO }).into()
    }
}

impl<T, D> OIntervalMatrix<T, D, D>
where
    T: IntervalOps + Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    /// Creates a square interval matrix from its diagonal.
    pub fn from_diagonal<S>(diagonal: &IntervalMatrix<T, D, Const<1>, S>) -> Self
    where
        S: Storage<T, D, Const<1>>,
    {
        let (dim, _) = diagonal.shape_generic();
        Self::from_fn_generic(dim, dim, |i, j| if i == j { diagonal[i] } else { T::ZERO })
    }
}

impl<T, const R: usize, const C: usize> SIntervalMatrix<T, R, C>
where
    T: IntervalOps + Scalar,
{
    /// Creates a statically sized interval matrix filled with one value.
    pub fn from_element(element: T) -> Self {
        Self::from_element_generic(Const::<R>, Const::<C>, element)
    }

    /// Creates a statically sized interval matrix by calling `f` for each coordinate.
    pub fn from_fn<F>(f: F) -> Self
    where
        F: FnMut(usize, usize) -> T,
    {
        Self::from_fn_generic(Const::<R>, Const::<C>, f)
    }

    /// Creates a statically sized interval matrix from a column-major iterator.
    pub fn from_iterator<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        Self::from_iterator_generic(Const::<R>, Const::<C>, iter)
    }

    /// Creates a statically sized interval matrix from a row-major iterator.
    pub fn from_row_iterator<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        Self::from_row_iterator_generic(Const::<R>, Const::<C>, iter)
    }

    /// Creates a statically sized interval matrix from a row-major slice.
    pub fn from_row_slice(entries: &[T]) -> Self {
        Self::from_row_slice_generic(Const::<R>, Const::<C>, entries)
    }

    /// Creates a statically sized interval matrix from a column-major slice.
    pub fn from_column_slice(entries: &[T]) -> Self {
        Self::from_column_slice_generic(Const::<R>, Const::<C>, entries)
    }

    /// Creates a statically sized interval matrix from row vectors.
    pub fn from_rows<SR>(rows: &[IntervalMatrix<T, Const<1>, Const<C>, SR>]) -> Self
    where
        SR: Storage<T, Const<1>, Const<C>>,
    {
        Self::from_rows_generic(Const::<R>, Const::<C>, rows)
    }

    /// Creates a statically sized interval matrix from column vectors.
    pub fn from_columns<SC>(columns: &[IntervalMatrix<T, Const<R>, Const<1>, SC>]) -> Self
    where
        SC: Storage<T, Const<R>, Const<1>>,
    {
        Self::from_columns_generic(Const::<R>, Const::<C>, columns)
    }

    /// Creates a statically sized zero interval matrix.
    #[must_use]
    pub fn zeros() -> Self {
        Self::zeros_generic(Const::<R>, Const::<C>)
    }
}

impl<T, const D: usize> SIntervalMatrix<T, D, D>
where
    T: IntervalOps + Scalar,
{
    /// Creates a statically sized identity interval matrix.
    #[must_use]
    pub fn identity() -> Self {
        Self::identity_generic(Const::<D>)
    }
}

#[cfg(feature = "alloc")]
impl<T> DIntervalMatrix<T>
where
    T: IntervalOps + Scalar,
{
    /// Creates a dynamically sized interval matrix filled with one value.
    pub fn from_element(nrows: usize, ncols: usize, element: T) -> Self {
        Self::from_element_generic(Dyn(nrows), Dyn(ncols), element)
    }

    /// Creates a dynamically sized interval matrix by calling `f` for each coordinate.
    pub fn from_fn<F>(nrows: usize, ncols: usize, f: F) -> Self
    where
        F: FnMut(usize, usize) -> T,
    {
        Self::from_fn_generic(Dyn(nrows), Dyn(ncols), f)
    }

    /// Creates a dynamically sized interval matrix from a column-major iterator.
    pub fn from_iterator<I>(nrows: usize, ncols: usize, iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        Self::from_iterator_generic(Dyn(nrows), Dyn(ncols), iter)
    }

    /// Creates a dynamically sized interval matrix from a row-major iterator.
    pub fn from_row_iterator<I>(nrows: usize, ncols: usize, iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        Self::from_row_iterator_generic(Dyn(nrows), Dyn(ncols), iter)
    }

    /// Creates a dynamically sized interval matrix from a row-major slice.
    pub fn from_row_slice(nrows: usize, ncols: usize, entries: &[T]) -> Self {
        Self::from_row_slice_generic(Dyn(nrows), Dyn(ncols), entries)
    }

    /// Creates a dynamically sized interval matrix from a column-major slice.
    pub fn from_column_slice(nrows: usize, ncols: usize, entries: &[T]) -> Self {
        Self::from_column_slice_generic(Dyn(nrows), Dyn(ncols), entries)
    }

    /// Creates a dynamically sized interval matrix from row vectors.
    pub fn from_rows(rows: &[DIntervalRowVector<T>]) -> Self {
        let ncols = rows.first().map_or(0, IntervalMatrix::ncols);
        Self::from_rows_generic(Dyn(rows.len()), Dyn(ncols), rows)
    }

    /// Creates a dynamically sized interval matrix from column vectors.
    pub fn from_columns(columns: &[DIntervalVector<T>]) -> Self {
        let nrows = columns.first().map_or(0, IntervalMatrix::nrows);
        Self::from_columns_generic(Dyn(nrows), Dyn(columns.len()), columns)
    }

    /// Creates a dynamically sized zero interval matrix.
    #[must_use]
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self::zeros_generic(Dyn(nrows), Dyn(ncols))
    }

    /// Creates a dynamically sized identity interval matrix.
    #[must_use]
    pub fn identity(dim: usize) -> Self {
        Self::identity_generic(Dyn(dim))
    }
}

#[cfg(feature = "alloc")]
impl<T> DIntervalVector<T>
where
    T: IntervalOps + Scalar,
{
    /// Creates a dynamically sized interval column vector filled with one value.
    pub fn from_element(nrows: usize, element: T) -> Self {
        Self::from_element_generic(Dyn(nrows), Const::<1>, element)
    }

    /// Creates a dynamically sized interval column vector by calling `f` for each row.
    pub fn from_fn<F>(nrows: usize, mut f: F) -> Self
    where
        F: FnMut(usize) -> T,
    {
        Self::from_fn_generic(Dyn(nrows), Const::<1>, |i, _| f(i))
    }

    /// Creates a dynamically sized interval column vector from an iterator.
    pub fn from_iterator<I>(nrows: usize, iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        Self::from_iterator_generic(Dyn(nrows), Const::<1>, iter)
    }

    /// Creates a dynamically sized interval column vector from a slice.
    pub fn from_column_slice(entries: &[T]) -> Self {
        Self::from_column_slice_generic(Dyn(entries.len()), Const::<1>, entries)
    }

    /// Creates a dynamically sized zero interval column vector.
    #[must_use]
    pub fn zeros(nrows: usize) -> Self {
        Self::zeros_generic(Dyn(nrows), Const::<1>)
    }
}

#[cfg(feature = "alloc")]
impl<T> DIntervalRowVector<T>
where
    T: IntervalOps + Scalar,
{
    /// Creates a dynamically sized interval row vector filled with one value.
    pub fn from_element(ncols: usize, element: T) -> Self {
        Self::from_element_generic(Const::<1>, Dyn(ncols), element)
    }

    /// Creates a dynamically sized interval row vector by calling `f` for each column.
    pub fn from_fn<F>(ncols: usize, mut f: F) -> Self
    where
        F: FnMut(usize) -> T,
    {
        Self::from_fn_generic(Const::<1>, Dyn(ncols), |_, j| f(j))
    }

    /// Creates a dynamically sized interval row vector from an iterator.
    pub fn from_iterator<I>(ncols: usize, iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        Self::from_iterator_generic(Const::<1>, Dyn(ncols), iter)
    }

    /// Creates a dynamically sized interval row vector from a slice.
    pub fn from_row_slice(entries: &[T]) -> Self {
        Self::from_row_slice_generic(Const::<1>, Dyn(entries.len()), entries)
    }

    /// Creates a dynamically sized zero interval row vector.
    #[must_use]
    pub fn zeros(ncols: usize) -> Self {
        Self::zeros_generic(Const::<1>, Dyn(ncols))
    }
}

impl<T, R, C, S> IntervalMatrix<T, R, C, S>
where
    T: IntervalOps + Scalar,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
{
    /// Returns whether any entry is empty.
    #[must_use]
    pub fn has_empty_entries(&self) -> bool {
        self.any(IntervalOps::is_empty)
    }

    /// Returns whether any entry is `NaI`.
    #[must_use]
    pub fn has_nai_entries(&self) -> bool {
        self.any(IntervalOps::is_nai)
    }

    /// Returns whether any entry is entire.
    #[must_use]
    pub fn has_entire_entries(&self) -> bool {
        self.any(IntervalOps::is_entire)
    }

    /// Intersects corresponding entries of two matrices.
    ///
    /// # Panics
    ///
    /// Panics if the matrices have different dimensions.
    #[must_use]
    pub fn intersection<R2, C2, S2>(
        &self,
        rhs: &IntervalMatrix<T, R2, C2, S2>,
    ) -> OIntervalMatrix<T, SameShapeR<R, R2>, SameShapeC<C, C2>>
    where
        R2: Dim,
        C2: Dim,
        S2: Storage<T, R2, C2>,
        DefaultAllocator: SameShapeAllocator<R, C, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        self.zip_map(rhs, IntervalOps::intersection)
    }

    /// Computes the convex hull of corresponding entries of two matrices.
    ///
    /// # Panics
    ///
    /// Panics if the matrices have different dimensions.
    #[must_use]
    pub fn convex_hull<R2, C2, S2>(
        &self,
        rhs: &IntervalMatrix<T, R2, C2, S2>,
    ) -> OIntervalMatrix<T, SameShapeR<R, R2>, SameShapeC<C, C2>>
    where
        R2: Dim,
        C2: Dim,
        S2: Storage<T, R2, C2>,
        DefaultAllocator: SameShapeAllocator<R, C, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        self.zip_map(rhs, IntervalOps::convex_hull)
    }

    /// Returns whether corresponding entries are interval-equal.
    ///
    /// # Panics
    ///
    /// Panics if the matrices have different dimensions.
    #[must_use]
    pub fn equal<R2, C2, S2>(&self, rhs: &IntervalMatrix<T, R2, C2, S2>) -> bool
    where
        R2: Dim,
        C2: Dim,
        S2: Storage<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "interval matrix equality dimension mismatch",
        );
        self.iter()
            .copied()
            .zip(rhs.iter().copied())
            .all(|(lhs, rhs)| lhs.equal(rhs))
    }

    /// Returns whether every entry is a subset of the corresponding `rhs` entry.
    ///
    /// # Panics
    ///
    /// Panics if the matrices have different dimensions.
    #[must_use]
    pub fn subset<R2, C2, S2>(&self, rhs: &IntervalMatrix<T, R2, C2, S2>) -> bool
    where
        R2: Dim,
        C2: Dim,
        S2: Storage<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "interval matrix subset dimension mismatch",
        );
        self.iter()
            .copied()
            .zip(rhs.iter().copied())
            .all(|(lhs, rhs)| lhs.subset(rhs))
    }

    /// Returns whether every entry lies in the interior of the corresponding `rhs` entry.
    ///
    /// # Panics
    ///
    /// Panics if the matrices have different dimensions.
    #[must_use]
    pub fn interior<R2, C2, S2>(&self, rhs: &IntervalMatrix<T, R2, C2, S2>) -> bool
    where
        R2: Dim,
        C2: Dim,
        S2: Storage<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "interval matrix interior dimension mismatch",
        );
        self.iter()
            .copied()
            .zip(rhs.iter().copied())
            .all(|(lhs, rhs)| lhs.interior(rhs))
    }
}

impl<T, D, S> IntervalMatrix<T, D, D, S>
where
    T: IntervalOps + Scalar,
    D: Dim,
    S: Storage<T, D, D>,
{
    /// Returns whether every off-diagonal interval is nonpositive.
    #[must_use]
    pub fn is_z_matrix(&self) -> bool {
        // Def 4.4
        if !self.is_square() || self.has_empty_entries() || self.has_nai_entries() {
            return false;
        }
        let n = self.nrows();
        (0..n).all(|i| (0..n).all(|j| i == j || self[(i, j)].sup() <= 0.0))
    }
}

impl<T, D, S> IntervalMatrix<T, D, D, S>
where
    T: IntervalOps + Scalar,
    D: Dim,
    S: Storage<T, D, D>,
    DefaultAllocator: Allocator<D, D>,
{
    /// Builds the real comparison matrix of this interval matrix.
    ///
    /// Diagonal entries are the mignitudes of the corresponding intervals;
    /// off-diagonal entries are the negated magnitudes.
    ///
    /// Returns `None` when the matrix is not square or contains an empty or
    /// `NaI` entry.
    #[must_use]
    pub fn comparison_matrix(&self) -> Option<OMatrix<f64, D, D>> {
        if !self.is_square() || self.has_empty_entries() || self.has_nai_entries() {
            return None;
        }
        let (dim, _) = self.shape_generic();
        Some(OMatrix::from_fn_generic(dim, dim, |i, j| {
            if i == j {
                self[(i, j)].mig()
            } else {
                -self[(i, j)].mag()
            }
        }))
    }
}

#[allow(clippy::indexing_slicing)]
impl<T, D, S> IntervalMatrix<T, D, D, S>
where
    T: IntervalOps + Scalar,
    D: Dim + DimMin<D, Output = D>,
    S: Storage<T, D, D>,
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    /// Tests whether this interval matrix is an M-matrix.
    ///
    /// Returns `false` when the property cannot be certified, including for
    /// empty, invalid, or singular systems.
    #[must_use]
    pub fn is_m_matrix(&self) -> bool {
        // Def 4.5
        if !self.is_z_matrix() {
            return false;
        }
        let n = self.nrows();
        if n == 0 {
            return false;
        }
        let (dim, _) = self.shape_generic();

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
                row_sum = self[(i, j)].mul_add(T::singleton(u[j]), row_sum);
            }
            row_sum.inf() > 0.0
        })
    }

    /// Tests whether this interval matrix is an H-matrix.
    ///
    /// Returns `false` when its comparison matrix is invalid, empty,
    /// non-finite, singular, or does not satisfy the positive-vector test.
    #[must_use]
    pub fn is_h_matrix(&self) -> bool {
        let Some(comparison) = self.comparison_matrix() else {
            return false; // not square or invalid entries
        };
        let n = comparison.nrows();
        if n == 0 || comparison.iter().any(|entry| !entry.is_finite()) {
            return false;
        }
        let (dim, _) = self.shape_generic();
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

    /// Tests strong regularity using midpoint-inverse preconditioning.
    ///
    /// Returns `false` when the midpoint inverse or the resulting H-matrix
    /// condition cannot be certified.
    #[must_use]
    pub fn is_strongly_regular(&self) -> bool {
        if !self.is_square()
            || self.is_empty()
            || self.has_empty_entries()
            || self.has_nai_entries()
        {
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
    /// Returns the matrix of lower endpoints.
    #[must_use]
    pub fn inf(&self) -> OMatrix<f64, R, C> {
        self.map_inner(IntervalOps::inf)
    }
    /// Returns the matrix of upper endpoints.
    #[must_use]
    pub fn sup(&self) -> OMatrix<f64, R, C> {
        self.map_inner(IntervalOps::sup)
    }

    /// Returns the matrix of outward-rounded radii.
    #[must_use]
    pub fn rad(&self) -> OMatrix<f64, R, C> {
        self.map_inner(IntervalOps::rad)
    }
    /// Returns the matrix of inward-rounded radii.
    #[must_use]
    pub fn inner_rad(&self) -> OMatrix<f64, R, C> {
        self.map_inner(IntervalOps::inner_rad)
    }

    /// Returns the matrix of interval midpoints.
    #[must_use]
    pub fn mid(&self) -> OMatrix<f64, R, C> {
        self.map_inner(IntervalOps::mid)
    }

    /// Returns the matrix of interval magnitudes.
    #[must_use]
    pub fn mag(&self) -> OMatrix<f64, R, C> {
        self.map_inner(IntervalOps::mag)
    }

    /// Returns the matrix of interval mignitudes.
    #[must_use]
    pub fn mig(&self) -> OMatrix<f64, R, C> {
        self.map_inner(IntervalOps::mig)
    }

    /// Returns an upward-rounded upper bound for the induced matrix 1-norm.
    #[must_use]
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

    /// Returns an upward-rounded upper bound for the induced infinity norm.
    #[must_use]
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

impl<T, D, S> IntervalMatrix<T, D, Const<1>, S>
where
    T: IntervalOps + Scalar,
    D: Dim,
    S: Storage<T, D, Const<1>>,
{
    /// Returns an upward-rounded upper bound for the weighted maximum norm.
    ///
    /// The norm is `max_i mag(self[i]) / v[i]`. Returns `NaN` if a weight is
    /// nonpositive or non-finite, or if an interval magnitude is `NaN`.
    ///
    /// # Panics
    ///
    /// Panics if `self` and `v` have different lengths.
    #[must_use]
    #[allow(clippy::arithmetic_side_effects)]
    pub fn v_norm<SV>(&self, v: &Matrix<f64, D, Const<1>, SV>) -> f64
    where
        SV: Storage<f64, D, Const<1>>,
    {
        assert_eq!(
            self.nrows(),
            v.nrows(),
            "interval vector weighted norm dimension mismatch",
        );
        self.iter()
            .zip(v.iter())
            .try_fold(0.0_f64, |norm, (entry, &weight)| {
                if !weight.is_finite() || weight <= 0.0 {
                    None
                } else {
                    let weighted_magnitude = (*entry / weight).mag();
                    (!weighted_magnitude.is_nan()).then(|| norm.max(weighted_magnitude))
                }
            })
            .unwrap_or(f64::NAN)
    }
}

mod ops;

/// A verified solver for square interval linear systems.
pub trait Solver<T, D>
where
    T: IntervalOps + Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    /// Computes an interval enclosure of the solution.
    ///
    /// Returns `None` when the system is dimensionally invalid or the method
    /// cannot certify an enclosure.
    #[must_use]
    fn solve<SA, SB>(
        &self,
        lhs: &IntervalMatrix<T, D, D, SA>,
        rhs: &IntervalMatrix<T, D, Const<1>, SB>,
    ) -> Option<OIntervalVector<T, D>>
    where
        SA: Storage<T, D, D>,
        SB: Storage<T, D, Const<1>>;

    /// Computes an interval enclosure of the inverse.
    ///
    /// Returns `None` if any column of the inverse cannot be certified.
    #[must_use]
    #[allow(clippy::indexing_slicing)]
    fn inverse<S>(&self, mat: &IntervalMatrix<T, D, D, S>) -> Option<OIntervalMatrix<T, D, D>>
    where
        S: Storage<T, D, D>,
    {
        let (dim, _) = mat.shape_generic();
        let mut inverse = OIntervalMatrix::zeros_generic(dim, dim);
        for column in 0..mat.ncols() {
            let rhs = OIntervalVector::from_fn_generic(dim, Const::<1>, |row, _| {
                if row == column { T::ONE } else { T::ZERO }
            });
            let solution = self.solve(mat, &rhs)?;
            for row in 0..mat.nrows() {
                inverse[(row, column)] = solution[row];
            }
        }
        Some(inverse)
    }
}

/// Epsilon-inflation method.
mod epsilon_inflation;
pub use epsilon_inflation::EpsilonInflationSolver;

/// Gaussian elimination method.
mod gaussian_elimination;
pub use gaussian_elimination::GaussianEliminationSolver;

/// Hansen–Bliek–Rohn method.
mod hansen_bliek_rohn;
pub use hansen_bliek_rohn::HansenBliekRohnSolver;

/// Iterative methods for matrix solves.
mod iterative;
pub use iterative::{
    GaussSeidelSolver, InitialEnclosure, JacobiSolver, KrawczykSolver, StoppingTolerance,
};

mod preconditioned;
pub use preconditioned::{Preconditioned, Preconditioner};

impl<T, D, S> IntervalMatrix<T, D, D, S>
where
    T: IntervalOps + Scalar,
    D: Dim,
    S: Storage<T, D, D>,
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    /// Computes an interval enclosure of the solution using Gaussian elimination.
    #[must_use]
    pub fn solve<SR>(
        &self,
        rhs: &IntervalMatrix<T, D, Const<1>, SR>,
    ) -> Option<OIntervalVector<T, D>>
    where
        SR: Storage<T, D, Const<1>>,
        D: DimMin<D, Output = D>,
        GaussianEliminationSolver: Solver<T, D>,
    {
        let solver = Preconditioned::auto(GaussianEliminationSolver, self);
        solver.solve(self, rhs)
    }

    /// Computes an interval enclosure of the solution using `solver`.
    #[must_use]
    pub fn solve_with<SR, V>(
        &self,
        rhs: &IntervalMatrix<T, D, Const<1>, SR>,
        solver: &V,
    ) -> Option<OIntervalVector<T, D>>
    where
        SR: Storage<T, D, Const<1>>,
        V: Solver<T, D>,
    {
        solver.solve(self, rhs)
    }

    /// Computes an interval enclosure of the inverse using Gaussian elimination.
    #[must_use]
    pub fn inverse(&self) -> Option<OIntervalMatrix<T, D, D>>
    where
        D: DimMin<D, Output = D>,
        GaussianEliminationSolver: Solver<T, D>,
    {
        let solver = Preconditioned::auto(GaussianEliminationSolver, self);
        solver.inverse(self)
    }

    /// Computes an interval enclosure of the inverse using `solver`.
    #[must_use]
    pub fn inverse_with<V>(&self, solver: &V) -> Option<OIntervalMatrix<T, D, D>>
    where
        V: Solver<T, D>,
    {
        solver.inverse(self)
    }
}

#[cfg(test)]
#[allow(clippy::indexing_slicing)]
mod tests {
    extern crate std;

    use nalgebra::SMatrix;

    use crate::{DecoratedInterval, Interval, IntervalOps};

    use super::*;

    #[test]
    fn basic_access_matches_underlying_matrix() {
        let mut matrix: SIntervalMatrix<Interval, 2, 2> =
            SMatrix::<Interval, 2, 2>::from_row_slice(&[
                Interval::singleton(1.0),
                Interval::singleton(2.0),
                Interval::singleton(3.0),
                Interval::singleton(4.0),
            ])
            .into();

        assert_eq!(matrix.shape(), (2, 2));
        assert_eq!(matrix.nrows(), 2);
        assert_eq!(matrix.ncols(), 2);
        assert_eq!(matrix.len(), 4);
        assert!(!matrix.is_empty());
        assert!(matrix.is_square());
        assert_eq!(matrix[(0, 1)], Interval::singleton(2.0));
        assert_eq!(matrix.get(1, 0), Some(&Interval::singleton(3.0)));
        assert_eq!(matrix.get(2, 0), None);
        assert_eq!(matrix.iter().next(), Some(&Interval::singleton(1.0)));
        assert_eq!((&matrix).into_iter().count(), 4);
        assert_eq!(matrix.row_iter().count(), 2);
        assert_eq!(matrix.column_iter().count(), 2);

        let changed = matrix.get_mut(1, 1).is_some_and(|entry| {
            *entry = Interval::singleton(5.0);
            true
        });
        assert!(changed);
        matrix.iter_mut().for_each(|entry| *entry += 1.0);
        assert_eq!((&mut matrix).into_iter().count(), 4);
        assert_eq!(matrix.row_iter_mut().count(), 2);
        assert_eq!(matrix.column_iter_mut().count(), 2);

        let inner: &SMatrix<Interval, 2, 2> = matrix.as_ref();
        assert_eq!(inner[(1, 1)], Interval::singleton(6.0));
        let inner: &mut SMatrix<Interval, 2, 2> = matrix.as_mut();
        inner[(0, 0)] = Interval::ZERO;
        assert_eq!(matrix[(0, 0)], Interval::ZERO);
    }

    #[test]
    fn static_constructors_preserve_layout_and_interval_semantics() {
        let one = Interval::singleton(1.0);
        let two = Interval::singleton(2.0);
        let three = Interval::singleton(3.0);
        let four = Interval::singleton(4.0);

        let filled = SIntervalMatrix::<Interval, 2, 2>::from_element(one);
        assert!(filled.iter().all(|entry| *entry == one));

        let from_fn = SIntervalMatrix::<Interval, 2, 2>::from_fn(|i, j| {
            Interval::singleton(match (i, j) {
                (0, 0) => 0.0,
                (1, 1) => 2.0,
                _ => 1.0,
            })
        });
        assert_eq!(from_fn[(1, 1)], Interval::singleton(2.0));

        let column_major =
            SIntervalMatrix::<Interval, 2, 2>::from_iterator([one, two, three, four]);
        let row_major =
            SIntervalMatrix::<Interval, 2, 2>::from_row_iterator([one, two, three, four]);
        assert_eq!(column_major[(0, 1)], three);
        assert_eq!(row_major[(0, 1)], two);

        assert_eq!(
            SIntervalMatrix::<Interval, 2, 2>::from_row_slice(&[one, two, three, four]),
            row_major,
        );
        assert_eq!(
            SIntervalMatrix::<Interval, 2, 2>::from_column_slice(&[one, two, three, four]),
            column_major,
        );

        let rows = [
            SIntervalRowVector::<Interval, 2>::from_row_slice(&[one, two]),
            SIntervalRowVector::<Interval, 2>::from_row_slice(&[three, four]),
        ];
        let columns = [
            SIntervalVector::<Interval, 2>::from_column_slice(&[one, three]),
            SIntervalVector::<Interval, 2>::from_column_slice(&[two, four]),
        ];
        assert_eq!(SIntervalMatrix::from_rows(&rows), row_major);
        assert_eq!(SIntervalMatrix::from_columns(&columns), row_major);

        let diagonal = SIntervalVector::<Interval, 2>::from_column_slice(&[two, three]);
        let diagonal_matrix = SIntervalMatrix::<Interval, 2, 2>::from_diagonal(&diagonal);
        assert_eq!(diagonal_matrix[(0, 0)], two);
        assert_eq!(diagonal_matrix[(0, 1)], Interval::ZERO);
        assert_eq!(
            SIntervalMatrix::<Interval, 2, 2>::zeros()[(1, 1)],
            Interval::ZERO
        );
        assert_eq!(
            SIntervalMatrix::<Interval, 2, 2>::identity()[(1, 1)],
            Interval::ONE
        );

        let points = SMatrix::<f64, 2, 2>::from_row_slice(&[1.0, 2.0, 3.0, 4.0]);
        let singletons = SIntervalMatrix::<Interval, 2, 2>::from_singletons(&points);
        assert_eq!(singletons, row_major);

        let lower = SMatrix::<f64, 1, 2>::from_row_slice(&[0.0, 1.0]);
        let upper = SMatrix::<f64, 1, 2>::from_row_slice(&[2.0, 3.0]);
        let bounds = SIntervalMatrix::<Interval, 1, 2>::from_bounds(&lower, &upper);
        assert_eq!(bounds[(0, 0)], Interval::new(0.0, 2.0));

        let midpoint = SMatrix::<f64, 1, 1>::from_element(1.0);
        let radius = SMatrix::<f64, 1, 1>::from_element(0.1);
        let mid_rad = SIntervalMatrix::<Interval, 1, 1>::from_mid_rad(&midpoint, &radius);
        assert!(mid_rad[(0, 0)].contains(0.9));
        assert!(mid_rad[(0, 0)].contains(1.1));
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn dynamic_constructors_infer_dimensions_from_rows_and_columns() {
        let one = Interval::singleton(1.0);
        let two = Interval::singleton(2.0);
        let three = Interval::singleton(3.0);
        let four = Interval::singleton(4.0);
        let rows = [
            DIntervalRowVector::from_row_slice(&[one, two]),
            DIntervalRowVector::from_row_slice(&[three, four]),
        ];
        let columns = [
            DIntervalVector::from_column_slice(&[one, three]),
            DIntervalVector::from_column_slice(&[two, four]),
        ];

        let from_rows = DIntervalMatrix::from_rows(&rows);
        let from_columns = DIntervalMatrix::from_columns(&columns);

        assert_eq!(from_rows, from_columns);
        assert_eq!(from_rows.shape(), (2, 2));
        assert_eq!(DIntervalMatrix::<Interval>::zeros(2, 3).shape(), (2, 3));
        assert_eq!(
            DIntervalMatrix::<Interval>::identity(2)[(0, 0)],
            Interval::ONE
        );
    }

    #[test]
    fn interval_predicates_and_set_operations_are_componentwise() {
        let narrow = SIntervalMatrix::<Interval, 1, 2>::from_row_slice(&[
            Interval::new(1.0, 2.0),
            Interval::new(3.0, 4.0),
        ]);
        let wide = SIntervalMatrix::<Interval, 1, 2>::from_row_slice(&[
            Interval::new(0.0, 3.0),
            Interval::new(2.0, 5.0),
        ]);

        assert!(narrow.subset(&wide));
        assert!(narrow.interior(&wide));
        assert!(narrow.equal(&narrow));
        assert_eq!(narrow.intersection(&wide), narrow);
        assert_eq!(narrow.convex_hull(&wide), wide);
        assert!(narrow.all(IntervalOps::is_bounded));
        assert!(narrow.any(|entry| entry.contains(1.5)));

        let exceptional =
            SIntervalMatrix::<Interval, 1, 2>::from_row_slice(&[Interval::EMPTY, Interval::ENTIRE]);
        assert!(exceptional.has_empty_entries());
        assert!(exceptional.has_entire_entries());
        assert!(!exceptional.has_nai_entries());

        let decorated =
            SIntervalMatrix::<DecoratedInterval, 1, 1>::from_element(DecoratedInterval::NAI);
        assert!(decorated.has_nai_entries());
    }

    #[test]
    fn solvers_accept_borrowed_matrix_and_vector_views() {
        let lhs_storage = SIntervalMatrix::<Interval, 3, 3>::from_fn(|row, column| {
            if row == column {
                Interval::singleton(2.0)
            } else {
                Interval::ZERO
            }
        });
        let rhs_storage = SIntervalVector::<Interval, 3>::from_column_slice(&[
            Interval::singleton(4.0),
            Interval::singleton(6.0),
            Interval::singleton(8.0),
        ]);
        let lhs = IntervalMatrix::from_inner(lhs_storage.as_inner().fixed_view::<2, 2>(0, 0));
        let rhs = IntervalMatrix::from_inner(rhs_storage.as_inner().fixed_rows::<2>(0));

        let solution = lhs.solve_with(&rhs, &GaussianEliminationSolver);
        assert!(
            solution
                .as_ref()
                .is_some_and(|solution| solution[0].contains(2.0) && solution[1].contains(3.0))
        );

        let inverse = lhs.inverse_with(&GaussianEliminationSolver);
        assert!(
            inverse.as_ref().is_some_and(
                |inverse| inverse[(0, 0)].contains(0.5) && inverse[(1, 1)].contains(0.5)
            )
        );
    }

    #[test]
    fn weighted_vector_norm_uses_positive_component_weights() {
        let vector = SIntervalVector::<Interval, 2>::from_column_slice(&[
            Interval::new(-2.0, 1.0),
            Interval::new(3.0, 4.0),
        ]);
        let weights = nalgebra::SVector::<f64, 2>::from_column_slice(&[4.0, 2.0]);

        assert_eq!(vector.v_norm(&weights), 2.0);
    }

    #[test]
    fn matrix_display_forwards_precision_and_measures_interval_widths() {
        let matrix = SIntervalMatrix::<Interval, 2, 1>::from_column_slice(&[
            Interval::new(-1.25, 2.5),
            Interval::new(-100.0, 200.0),
        ]);

        let rendered = std::format!("{matrix:.2}");
        assert!(rendered.contains("[-1.25,2.50]"));
        assert!(rendered.contains("[-100.00,200.00]"));
        let row_widths: std::vec::Vec<_> = rendered
            .lines()
            .filter(|line| line.contains('│'))
            .map(str::len)
            .collect();
        assert!(row_widths.windows(2).all(|widths| widths[0] == widths[1]));
        assert!(std::format!("{matrix:.1e}").contains('e'));
        assert!(std::format!("{matrix:x}").contains("0x"));
    }
}
