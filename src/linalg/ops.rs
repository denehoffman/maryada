//! Standard matrix arithmetic operations.

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use nalgebra::{
    DefaultAllocator, Dim, Scalar, Storage, StorageMut,
    allocator::{Allocator, SameShapeAllocator, SameShapeC, SameShapeR},
    constraint::{AreMultipliable, SameNumberOfColumns, SameNumberOfRows, ShapeConstraint},
};

use crate::IntervalOps;

use super::{IntervalMatrix, OIntervalMatrix, OIntervalVector};

impl<T, D, S> IntervalMatrix<T, D, D, S>
where
    T: IntervalOps + Scalar,
    D: Dim,
    S: Storage<T, D, D>,
{
    /// Returns an owned vector containing the main diagonal.
    #[must_use]
    pub fn diagonal(&self) -> OIntervalVector<T, D>
    where
        DefaultAllocator: Allocator<D>,
    {
        IntervalMatrix::from_inner(self.as_inner().diagonal())
    }
}

impl<T, R1, C1, SA> IntervalMatrix<T, R1, C1, SA>
where
    T: IntervalOps + Scalar,
    R1: Dim,
    C1: Dim,
    SA: Storage<T, R1, C1>,
{
    /// Multiplies two matrices componentwise.
    ///
    /// # Panics
    ///
    /// Panics if the matrices have different dimensions.
    #[must_use]
    pub fn component_mul<R2, C2, SB>(
        &self,
        rhs: &IntervalMatrix<T, R2, C2, SB>,
    ) -> OIntervalMatrix<T, SameShapeR<R1, R2>, SameShapeC<C1, C2>>
    where
        R2: Dim,
        C2: Dim,
        SB: Storage<T, R2, C2>,
        DefaultAllocator: SameShapeAllocator<R1, C1, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>,
    {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "interval matrix component_mul dimension mismatch",
        );
        let nrows: SameShapeR<R1, R2> = Dim::from_usize(self.nrows());
        let ncols: SameShapeC<C1, C2> = Dim::from_usize(self.ncols());
        let elements = self
            .iter()
            .copied()
            .zip(rhs.iter().copied())
            .map(|(lhs, rhs)| Mul::mul(lhs, rhs));

        OIntervalMatrix::from_iterator_generic(nrows, ncols, elements)
    }

    /// Divides two matrices componentwise.
    ///
    /// # Panics
    ///
    /// Panics if the matrices have different dimensions.
    #[must_use]
    pub fn component_div<R2, C2, SB>(
        &self,
        rhs: &IntervalMatrix<T, R2, C2, SB>,
    ) -> OIntervalMatrix<T, SameShapeR<R1, R2>, SameShapeC<C1, C2>>
    where
        R2: Dim,
        C2: Dim,
        SB: Storage<T, R2, C2>,
        DefaultAllocator: SameShapeAllocator<R1, C1, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>,
    {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "interval matrix component_div dimension mismatch",
        );
        let nrows: SameShapeR<R1, R2> = Dim::from_usize(self.nrows());
        let ncols: SameShapeC<C1, C2> = Dim::from_usize(self.ncols());
        let elements = self
            .iter()
            .copied()
            .zip(rhs.iter().copied())
            .map(|(lhs, rhs)| Div::div(lhs, rhs));

        OIntervalMatrix::from_iterator_generic(nrows, ncols, elements)
    }

    /// Multiplies every entry by a real scalar.
    #[must_use = "Did you mean to use scale_mut()?"]
    pub fn scale(&self, value: f64) -> OIntervalMatrix<T, R1, C1>
    where
        DefaultAllocator: Allocator<R1, C1>,
    {
        self.map(|entry| Mul::mul(entry, value))
    }

    /// Divides every entry by a real scalar.
    #[must_use = "Did you mean to use unscale_mut()?"]
    pub fn unscale(&self, value: f64) -> OIntervalMatrix<T, R1, C1>
    where
        DefaultAllocator: Allocator<R1, C1>,
    {
        self.map(|entry| Div::div(entry, value))
    }

    /// Adds an interval scalar to every entry.
    #[must_use = "Did you mean to use add_scalar_mut()?"]
    pub fn add_scalar(&self, value: T) -> OIntervalMatrix<T, R1, C1>
    where
        DefaultAllocator: Allocator<R1, C1>,
    {
        self.map(|entry| Add::add(entry, value))
    }
}

impl<T, R, C, S> IntervalMatrix<T, R, C, S>
where
    T: IntervalOps + Scalar,
    R: Dim,
    C: Dim,
    S: StorageMut<T, R, C>,
{
    /// Multiplies this matrix by `rhs` componentwise.
    ///
    /// # Panics
    ///
    /// Panics if the matrices have different dimensions.
    pub fn component_mul_assign<R2, C2, SB>(&mut self, rhs: &IntervalMatrix<T, R2, C2, SB>)
    where
        R2: Dim,
        C2: Dim,
        SB: Storage<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "interval matrix component_mul_assign dimension mismatch",
        );
        self.iter_mut()
            .zip(rhs.iter().copied())
            .for_each(|(lhs, rhs)| MulAssign::mul_assign(lhs, rhs));
    }

    /// Divides this matrix by `rhs` componentwise.
    ///
    /// # Panics
    ///
    /// Panics if the matrices have different dimensions.
    pub fn component_div_assign<R2, C2, SB>(&mut self, rhs: &IntervalMatrix<T, R2, C2, SB>)
    where
        R2: Dim,
        C2: Dim,
        SB: Storage<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2> + SameNumberOfColumns<C, C2>,
    {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "interval matrix component_div_assign dimension mismatch",
        );
        self.iter_mut()
            .zip(rhs.iter().copied())
            .for_each(|(lhs, rhs)| DivAssign::div_assign(lhs, rhs));
    }

    /// Adds an interval scalar to every entry in place.
    pub fn add_scalar_mut(&mut self, value: T) {
        self.iter_mut()
            .for_each(|entry| AddAssign::add_assign(entry, value));
    }

    /// Multiplies every entry by a real scalar in place.
    pub fn scale_mut(&mut self, value: f64) {
        self.iter_mut()
            .for_each(|entry| MulAssign::mul_assign(entry, value));
    }

    /// Divides every entry by a real scalar in place.
    pub fn unscale_mut(&mut self, value: f64) {
        self.iter_mut()
            .for_each(|entry| DivAssign::div_assign(entry, value));
    }
}

impl<T, R1, C1, R2, C2, SA, SB> Add<&IntervalMatrix<T, R2, C2, SB>>
    for &IntervalMatrix<T, R1, C1, SA>
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

    fn add(self, rhs: &IntervalMatrix<T, R2, C2, SB>) -> Self::Output {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "interval matrix addition dimension mismatch",
        );
        let nrows: SameShapeR<R1, R2> = Dim::from_usize(self.nrows());
        let ncols: SameShapeC<C1, C2> = Dim::from_usize(self.ncols());
        let elements = self
            .iter()
            .copied()
            .zip(rhs.iter().copied())
            .map(|(lhs, rhs)| Add::add(lhs, rhs));

        OIntervalMatrix::from_iterator_generic(nrows, ncols, elements)
    }
}

impl<T, R1, C1, R2, C2, SA, SB> Add<&IntervalMatrix<T, R2, C2, SB>>
    for IntervalMatrix<T, R1, C1, SA>
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

    fn add(self, rhs: &IntervalMatrix<T, R2, C2, SB>) -> Self::Output {
        Add::add(&self, rhs)
    }
}

impl<T, R1, C1, R2, C2, SA, SB> Add<IntervalMatrix<T, R2, C2, SB>>
    for &IntervalMatrix<T, R1, C1, SA>
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

    fn add(self, rhs: IntervalMatrix<T, R2, C2, SB>) -> Self::Output {
        Add::add(self, &rhs)
    }
}

impl<T, R1, C1, R2, C2, SA, SB> Add<IntervalMatrix<T, R2, C2, SB>> for IntervalMatrix<T, R1, C1, SA>
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

    fn add(self, rhs: IntervalMatrix<T, R2, C2, SB>) -> Self::Output {
        Add::add(&self, &rhs)
    }
}

impl<T, R1, C1, R2, C2, SA, SB> Sub<&IntervalMatrix<T, R2, C2, SB>>
    for &IntervalMatrix<T, R1, C1, SA>
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

    fn sub(self, rhs: &IntervalMatrix<T, R2, C2, SB>) -> Self::Output {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "interval matrix subtraction dimension mismatch",
        );
        let nrows: SameShapeR<R1, R2> = Dim::from_usize(self.nrows());
        let ncols: SameShapeC<C1, C2> = Dim::from_usize(self.ncols());
        let elements = self
            .iter()
            .copied()
            .zip(rhs.iter().copied())
            .map(|(lhs, rhs)| Sub::sub(lhs, rhs));

        OIntervalMatrix::from_iterator_generic(nrows, ncols, elements)
    }
}

impl<T, R1, C1, R2, C2, SA, SB> Sub<&IntervalMatrix<T, R2, C2, SB>>
    for IntervalMatrix<T, R1, C1, SA>
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

    fn sub(self, rhs: &IntervalMatrix<T, R2, C2, SB>) -> Self::Output {
        Sub::sub(&self, rhs)
    }
}

impl<T, R1, C1, R2, C2, SA, SB> Sub<IntervalMatrix<T, R2, C2, SB>>
    for &IntervalMatrix<T, R1, C1, SA>
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

    fn sub(self, rhs: IntervalMatrix<T, R2, C2, SB>) -> Self::Output {
        Sub::sub(self, &rhs)
    }
}

impl<T, R1, C1, R2, C2, SA, SB> Mul<&IntervalMatrix<T, R2, C2, SB>>
    for &IntervalMatrix<T, R1, C1, SA>
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

    fn mul(self, rhs: &IntervalMatrix<T, R2, C2, SB>) -> Self::Output {
        assert_eq!(
            self.ncols(),
            rhs.nrows(),
            "interval matrix multiplication dimension mismatch",
        );
        let (nrows, _) = self.shape_generic();
        let (_, ncols) = rhs.shape_generic();
        let mut output = OIntervalMatrix::<T, R1, C2>::zeros_generic(nrows, ncols);
        for (mut output_column, rhs_column) in output.column_iter_mut().zip(rhs.column_iter()) {
            for (lhs_column, rhs_entry) in self.column_iter().zip(rhs_column.iter().copied()) {
                for (output_entry, lhs_entry) in
                    output_column.iter_mut().zip(lhs_column.iter().copied())
                {
                    *output_entry = lhs_entry.mul_add(rhs_entry, *output_entry);
                }
            }
        }
        output
    }
}

impl<T, R1, C1, R2, C2, SA, SB> Mul<&IntervalMatrix<T, R2, C2, SB>>
    for IntervalMatrix<T, R1, C1, SA>
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

    fn mul(self, rhs: &IntervalMatrix<T, R2, C2, SB>) -> Self::Output {
        Mul::mul(&self, rhs)
    }
}

impl<T, R1, C1, R2, C2, SA, SB> Mul<IntervalMatrix<T, R2, C2, SB>>
    for &IntervalMatrix<T, R1, C1, SA>
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

    fn mul(self, rhs: IntervalMatrix<T, R2, C2, SB>) -> Self::Output {
        Mul::mul(self, &rhs)
    }
}

impl<T, R1, C1, R2, C2, SA, SB> Mul<IntervalMatrix<T, R2, C2, SB>> for IntervalMatrix<T, R1, C1, SA>
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

    fn mul(self, rhs: IntervalMatrix<T, R2, C2, SB>) -> Self::Output {
        Mul::mul(&self, &rhs)
    }
}

impl<T, R, C, S> Neg for &IntervalMatrix<T, R, C, S>
where
    T: IntervalOps + Scalar,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
    DefaultAllocator: Allocator<R, C>,
{
    type Output = OIntervalMatrix<T, R, C>;

    fn neg(self) -> Self::Output {
        self.map(Neg::neg)
    }
}

impl<T, R, C, S> Neg for IntervalMatrix<T, R, C, S>
where
    T: IntervalOps + Scalar,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
    DefaultAllocator: Allocator<R, C>,
{
    type Output = OIntervalMatrix<T, R, C>;

    fn neg(self) -> Self::Output {
        Neg::neg(&self)
    }
}

impl<T, R, C, S> Mul<T> for &IntervalMatrix<T, R, C, S>
where
    T: IntervalOps + Scalar,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
    DefaultAllocator: Allocator<R, C>,
{
    type Output = OIntervalMatrix<T, R, C>;

    fn mul(self, rhs: T) -> Self::Output {
        self.map(|entry| Mul::mul(entry, rhs))
    }
}

impl<T, R, C, S> Mul<T> for IntervalMatrix<T, R, C, S>
where
    T: IntervalOps + Scalar,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
    DefaultAllocator: Allocator<R, C>,
{
    type Output = OIntervalMatrix<T, R, C>;

    fn mul(self, rhs: T) -> Self::Output {
        Mul::mul(&self, rhs)
    }
}

impl<T, R, C, S> Div<T> for &IntervalMatrix<T, R, C, S>
where
    T: IntervalOps + Scalar,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
    DefaultAllocator: Allocator<R, C>,
{
    type Output = OIntervalMatrix<T, R, C>;

    fn div(self, rhs: T) -> Self::Output {
        self.map(|entry| Div::div(entry, rhs))
    }
}

impl<T, R, C, S> Div<T> for IntervalMatrix<T, R, C, S>
where
    T: IntervalOps + Scalar,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
    DefaultAllocator: Allocator<R, C>,
{
    type Output = OIntervalMatrix<T, R, C>;

    fn div(self, rhs: T) -> Self::Output {
        Div::div(&self, rhs)
    }
}

impl<T, R1, C1, R2, C2, SA, SB> AddAssign<&IntervalMatrix<T, R2, C2, SB>>
    for IntervalMatrix<T, R1, C1, SA>
where
    T: IntervalOps + Scalar,
    R1: Dim,
    C1: Dim,
    R2: Dim,
    C2: Dim,
    SA: StorageMut<T, R1, C1>,
    SB: Storage<T, R2, C2>,
    ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>,
{
    fn add_assign(&mut self, rhs: &IntervalMatrix<T, R2, C2, SB>) {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "interval matrix addition assignment dimension mismatch",
        );
        self.iter_mut()
            .zip(rhs.iter().copied())
            .for_each(|(lhs, rhs)| AddAssign::add_assign(lhs, rhs));
    }
}

impl<T, R1, C1, R2, C2, SA, SB> AddAssign<IntervalMatrix<T, R2, C2, SB>>
    for IntervalMatrix<T, R1, C1, SA>
where
    T: IntervalOps + Scalar,
    R1: Dim,
    C1: Dim,
    R2: Dim,
    C2: Dim,
    SA: StorageMut<T, R1, C1>,
    SB: Storage<T, R2, C2>,
    ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>,
{
    fn add_assign(&mut self, rhs: IntervalMatrix<T, R2, C2, SB>) {
        AddAssign::add_assign(self, &rhs);
    }
}

impl<T, R1, C1, R2, C2, SA, SB> SubAssign<&IntervalMatrix<T, R2, C2, SB>>
    for IntervalMatrix<T, R1, C1, SA>
where
    T: IntervalOps + Scalar,
    R1: Dim,
    C1: Dim,
    R2: Dim,
    C2: Dim,
    SA: StorageMut<T, R1, C1>,
    SB: Storage<T, R2, C2>,
    ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>,
{
    fn sub_assign(&mut self, rhs: &IntervalMatrix<T, R2, C2, SB>) {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "interval matrix subtraction assignment dimension mismatch",
        );
        self.iter_mut()
            .zip(rhs.iter().copied())
            .for_each(|(lhs, rhs)| SubAssign::sub_assign(lhs, rhs));
    }
}

impl<T, R1, C1, R2, C2, SA, SB> SubAssign<IntervalMatrix<T, R2, C2, SB>>
    for IntervalMatrix<T, R1, C1, SA>
where
    T: IntervalOps + Scalar,
    R1: Dim,
    C1: Dim,
    R2: Dim,
    C2: Dim,
    SA: StorageMut<T, R1, C1>,
    SB: Storage<T, R2, C2>,
    ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>,
{
    fn sub_assign(&mut self, rhs: IntervalMatrix<T, R2, C2, SB>) {
        SubAssign::sub_assign(self, &rhs);
    }
}

impl<T, R, C, S> MulAssign<T> for IntervalMatrix<T, R, C, S>
where
    T: IntervalOps + Scalar,
    R: Dim,
    C: Dim,
    S: StorageMut<T, R, C>,
{
    fn mul_assign(&mut self, rhs: T) {
        self.iter_mut()
            .for_each(|entry| MulAssign::mul_assign(entry, rhs));
    }
}

impl<T, R, C, S> DivAssign<T> for IntervalMatrix<T, R, C, S>
where
    T: IntervalOps + Scalar,
    R: Dim,
    C: Dim,
    S: StorageMut<T, R, C>,
{
    fn div_assign(&mut self, rhs: T) {
        self.iter_mut()
            .for_each(|entry| DivAssign::div_assign(entry, rhs));
    }
}

impl<T, D, SA, SB> MulAssign<&IntervalMatrix<T, D, D, SB>> for IntervalMatrix<T, D, D, SA>
where
    T: IntervalOps + Scalar,
    D: Dim,
    SA: StorageMut<T, D, D>,
    SB: Storage<T, D, D>,
    DefaultAllocator: Allocator<D, D>,
    ShapeConstraint: AreMultipliable<D, D, D, D>,
{
    fn mul_assign(&mut self, rhs: &IntervalMatrix<T, D, D, SB>) {
        let product = Mul::mul(&*self, rhs);
        self.iter_mut()
            .zip(product.iter().copied())
            .for_each(|(entry, product_entry)| *entry = product_entry);
    }
}

impl<T, D, SA, SB> MulAssign<IntervalMatrix<T, D, D, SB>> for IntervalMatrix<T, D, D, SA>
where
    T: IntervalOps + Scalar,
    D: Dim,
    SA: StorageMut<T, D, D>,
    SB: Storage<T, D, D>,
    DefaultAllocator: Allocator<D, D>,
    ShapeConstraint: AreMultipliable<D, D, D, D>,
{
    fn mul_assign(&mut self, rhs: IntervalMatrix<T, D, D, SB>) {
        MulAssign::mul_assign(self, &rhs);
    }
}

#[cfg(test)]
mod tests {
    use crate::{Interval, IntervalOps};

    #[cfg(feature = "alloc")]
    use super::super::DIntervalMatrix;
    use super::super::SIntervalMatrix;

    #[test]
    fn component_and_scalar_arithmetic_are_complete() {
        let one = Interval::singleton(1.0);
        let two = Interval::singleton(2.0);
        let three = Interval::singleton(3.0);
        let four = Interval::singleton(4.0);
        let lhs = SIntervalMatrix::<Interval, 2, 2>::from_row_slice(&[one, two, three, four]);
        let rhs = SIntervalMatrix::<Interval, 2, 2>::from_row_slice(&[four, three, two, one]);

        let sum = SIntervalMatrix::<Interval, 2, 2>::from_element(Interval::singleton(5.0));
        assert_eq!(&lhs + &rhs, sum);
        assert_eq!(lhs + &rhs, sum);
        assert_eq!(&lhs + rhs, sum);
        assert_eq!(lhs + rhs, sum);

        let difference = SIntervalMatrix::<Interval, 2, 2>::from_row_slice(&[
            Interval::singleton(-3.0),
            Interval::singleton(-1.0),
            Interval::singleton(1.0),
            Interval::singleton(3.0),
        ]);
        assert_eq!(&lhs - &rhs, difference);
        assert_eq!(lhs - &rhs, difference);
        assert_eq!(&lhs - rhs, difference);
        assert_eq!(lhs - rhs, difference);

        let product = SIntervalMatrix::<Interval, 2, 2>::from_row_slice(&[
            four,
            Interval::singleton(6.0),
            Interval::singleton(6.0),
            four,
        ]);
        assert_eq!(lhs.component_mul(&rhs), product);
        assert_eq!(product.component_div(&rhs), lhs);

        let scalar = Interval::singleton(2.0);
        assert_eq!(lhs.add_scalar(scalar)[(0, 0)], Interval::singleton(3.0));
        assert_eq!(lhs / scalar, lhs.unscale(2.0));
        assert_eq!(&lhs * scalar, lhs.scale(2.0));
        assert_eq!((-&lhs)[(0, 1)], Interval::singleton(-2.0));
        assert_eq!((-lhs)[(1, 0)], Interval::singleton(-3.0));

        let mut real_scalar = lhs;
        real_scalar.add_scalar_mut(Interval::singleton(1.0));
        real_scalar.add_scalar_mut(Interval::singleton(-1.0));
        real_scalar.scale_mut(2.0);
        real_scalar.unscale_mut(2.0);
        assert_eq!(real_scalar, lhs);
    }

    #[test]
    fn assignment_and_owned_matrix_multiplication_match_borrowed_operations() {
        let one = Interval::singleton(1.0);
        let two = Interval::singleton(2.0);
        let three = Interval::singleton(3.0);
        let four = Interval::singleton(4.0);
        let lhs = SIntervalMatrix::<Interval, 2, 2>::from_row_slice(&[one, two, three, four]);
        let rhs = SIntervalMatrix::<Interval, 2, 2>::from_row_slice(&[four, three, two, one]);
        let identity = SIntervalMatrix::<Interval, 2, 2>::identity();

        let borrowed_product = &lhs * &rhs;
        assert_eq!(lhs * &rhs, borrowed_product);
        assert_eq!(&lhs * rhs, borrowed_product);
        assert_eq!(lhs * rhs, borrowed_product);

        let mut assigned = lhs;
        assigned += &rhs;
        assigned -= rhs;
        assert_eq!(assigned, lhs);
        assigned *= Interval::singleton(2.0);
        assigned /= Interval::singleton(2.0);
        assigned *= &identity;
        assert_eq!(assigned, lhs);

        assigned.component_mul_assign(&rhs);
        assigned.component_div_assign(&rhs);
        assert_eq!(assigned, lhs);
    }

    #[cfg(feature = "alloc")]
    #[test]
    #[should_panic(expected = "interval matrix addition dimension mismatch")]
    fn component_arithmetic_rejects_dynamic_dimension_mismatches() {
        let lhs = DIntervalMatrix::<Interval>::zeros(1, 2);
        let rhs = DIntervalMatrix::<Interval>::zeros(2, 1);
        let _ = &lhs + &rhs;
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn component_arithmetic_accepts_compatible_static_and_dynamic_dimensions() {
        let one = Interval::singleton(1.0);
        let static_matrix = SIntervalMatrix::<Interval, 1, 2>::from_element(one);
        let dynamic_matrix = DIntervalMatrix::<Interval>::from_element(1, 2, one);

        let sum = &static_matrix + &dynamic_matrix;
        assert!(sum.iter().all(|entry| *entry == Interval::singleton(2.0)));

        let mut assigned = static_matrix;
        assigned += &dynamic_matrix;
        assigned -= dynamic_matrix;
        assert_eq!(assigned, static_matrix);
    }
}

impl<T, R1, C1, R2, C2, SA, SB> Sub<IntervalMatrix<T, R2, C2, SB>> for IntervalMatrix<T, R1, C1, SA>
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

    fn sub(self, rhs: IntervalMatrix<T, R2, C2, SB>) -> Self::Output {
        Sub::sub(&self, &rhs)
    }
}
