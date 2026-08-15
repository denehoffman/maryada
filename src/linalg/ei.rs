use super::*;

/// From Rump's dissertation
pub struct EpsilonInflation {
    /// Relative inflation
    r: f64,
    /// Absolute inflation
    eps: f64,
    /// Maximum number of iterations
    max_iterations: usize,
}

impl Default for EpsilonInflation {
    fn default() -> Self {
        Self {
            r: 0.1,
            eps: 1e-20,
            max_iterations: 20,
        }
    }
}

impl EpsilonInflation {
    pub(super) fn solve_matrix<T, D, C, SA, SB>(
        &self,
        lhs: &IntervalMatrix<T, D, D, SA>,
        rhs: &IntervalMatrix<T, D, C, SB>,
    ) -> Option<OIntervalMatrix<T, D, C>>
    where
        T: IntervalOps + Scalar,
        D: Dim,
        C: Dim,
        SA: Storage<T, D, D>,
        SB: Storage<T, D, C>,
        DefaultAllocator: Allocator<D, D> + Allocator<D, C>,
        ShapeConstraint: AreMultipliable<D, D, D, D> + AreMultipliable<D, D, D, C>,
    {
        if !lhs.is_square()
            || lhs.is_empty()
            || lhs.nrows() != rhs.nrows()
            || lhs.has_invalid_entries()
            || rhs.has_invalid_entries()
        {
            return None;
        }

        let (dim, _) = lhs.shape_generic();
        let relative_inflation = T::new(1.0 - self.r, 1.0 + self.r);
        let absolute_inflation = T::new(-self.eps, self.eps);

        // C ~ inv(mid(A)).
        let midpoint = lhs.mid();
        if midpoint.iter().any(|entry| !entry.is_finite()) {
            return None;
        }
        let Some(c) = midpoint.try_inverse() else {
            return None;
        };
        if c.iter().any(|entry| !entry.is_finite()) {
            return None;
        }
        let c_interval = OIntervalMatrix::from_singletons(&c);
        let identity = OIntervalMatrix::<T, D, D>::identity_generic(dim);

        // Z = I - CA.
        let residual = &identity - &(&c_interval * lhs);

        // x^0 = Cb.
        let cb = &c_interval * rhs;
        let mut x = cb.clone();
        for _ in 0..self.max_iterations {
            // y = x^k * [1-r, 1+r] + [-eps, eps].
            let y = x
                .scale(relative_inflation)
                .map(|entry| entry + absolute_inflation);

            // x^{k+1} = Cb + (I-CA)y.
            let next = &cb + &(&residual * &y);

            let enclosed = next
                .iter()
                .copied()
                .zip(y.iter().copied())
                .all(|(next_entry, inflated_entry)| next_entry.interior(inflated_entry));

            // Check if x^{k+1} is contained in the interior of y.
            if enclosed {
                return Some(next);
            }
            x = next;
        }
        None
    }
}

impl<T, D, SA, SB> Solver<T, D, SA, SB> for EpsilonInflation
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
    ) -> Option<OIntervalVector<T, D>> {
        self.solve_matrix(lhs, rhs)
    }
}
