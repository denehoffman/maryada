use core::any::Any;

use nalgebra::{Dim, Matrix, Scalar, Storage, StorageMut};

use crate::{
    Interval,
    rounding::{Direction, FloatClass, pack_integer},
};

pub(super) const ROW_BLOCK: usize = 16;

struct Grid {
    exponent: i32,
    bits: u32,
}

impl Grid {
    #[allow(clippy::arithmetic_side_effects)]
    fn for_entries<'a, T: Scalar>(
        entries: impl Iterator<Item = &'a T>,
        max_bits: u32,
    ) -> Option<Self> {
        let mut lowest = i32::MAX;
        let mut highest = i32::MIN;
        for entry in entries {
            let entry: &dyn Any = entry;
            let interval = entry.downcast_ref::<Interval>()?;
            for endpoint in [interval.inf_raw(), interval.sup_raw()] {
                match FloatClass::new(endpoint) {
                    FloatClass::Finite(value) => {
                        let used = u64::BITS - value.significand.leading_zeros();
                        let trailing = value.significand.trailing_zeros();
                        lowest = lowest.min(value.exponent + trailing.cast_signed());
                        highest = highest.max(value.exponent + used.cast_signed());
                        if highest.abs_diff(lowest) > max_bits {
                            return None;
                        }
                    }
                    FloatClass::Zero { .. } => {}
                    _ => return None,
                }
            }
        }
        if lowest == i32::MAX {
            return Some(Self {
                exponent: 0,
                bits: 0,
            });
        }
        Some(Self {
            exponent: lowest,
            bits: highest.abs_diff(lowest),
        })
    }

    // The scan established this type, exact divisibility on the chosen
    // grid, and a magnitude strictly below 2^63.
    #[allow(clippy::arithmetic_side_effects, clippy::expect_used)]
    fn coordinates<T: Scalar>(&self, entry: &T) -> (i64, i64) {
        let entry: &dyn Any = entry;
        let interval = entry
            .downcast_ref::<Interval>()
            .expect("validated bare interval");
        let scale = |endpoint| {
            let FloatClass::Finite(value) = FloatClass::new(endpoint) else {
                debug_assert_eq!(endpoint, 0.0);
                return 0;
            };
            let shift = value.exponent - self.exponent;
            let magnitude = if shift >= 0 {
                value.significand << shift.cast_unsigned()
            } else {
                value.significand >> shift.unsigned_abs()
            }
            .cast_signed();
            if value.negative {
                -magnitude
            } else {
                magnitude
            }
        };
        (scale(interval.inf_raw()), scale(interval.sup_raw()))
    }
}

// The caller checked the type and shapes. No output changes on fallback.
#[allow(clippy::arithmetic_side_effects, clippy::expect_used)]
pub(super) fn accumulate_grid_product<T, R, K, K2, C, RO, CO, SA, SB, SO>(
    left: &Matrix<T, R, K, SA>,
    right: &Matrix<T, K2, C, SB>,
    output: &mut Matrix<T, RO, CO, SO>,
) -> Option<()>
where
    T: Scalar,
    R: Dim,
    K: Dim,
    K2: Dim,
    C: Dim,
    RO: Dim,
    CO: Dim,
    SA: Storage<T, R, K>,
    SB: Storage<T, K2, C>,
    SO: StorageMut<T, RO, CO>,
{
    let inner = left.ncols();
    let left_grid = Grid::for_entries(left.iter(), i64::BITS - 1)?;
    let sum_bits = usize::BITS - (inner - 1).leading_zeros();
    // Even the sum of the magnitudes of all products must fit in i128.
    let remaining = (i128::BITS - 1).checked_sub(left_grid.bits + sum_bits)?;
    let right_grid = Grid::for_entries(right.iter(), remaining.min(i64::BITS - 1))?;
    let exponent = left_grid.exponent + right_grid.exponent;

    for (mut output_column, right_column) in output.column_iter_mut().zip(right.column_iter()) {
        for row in (0..left.nrows()).step_by(ROW_BLOCK) {
            let count = ROW_BLOCK.min(left.nrows() - row);
            let mut totals = [(0_i128, 0_i128); ROW_BLOCK];
            for (left_column, right_entry) in left.column_iter().zip(right_column.iter()) {
                let (b0, b1) = right_grid.coordinates(right_entry);
                if b0 == 0 && b1 == 0 {
                    continue;
                }
                let (b0, b1) = (i128::from(b0), i128::from(b1));
                let left_rows = left_column.rows(row, count);
                for ((lower, upper), left_entry) in totals.iter_mut().zip(left_rows.iter()) {
                    let (a0, a1) = left_grid.coordinates(left_entry);
                    let (a0, a1) = (i128::from(a0), i128::from(a1));
                    let (p00, p01, p10, p11) = (a0 * b0, a0 * b1, a1 * b0, a1 * b1);
                    *lower += p00.min(p01).min(p10).min(p11);
                    *upper += p00.max(p01).max(p10).max(p11);
                }
            }
            let mut output_rows = output_column.rows_mut(row, count);
            for (slot, (lower, upper)) in output_rows.iter_mut().zip(totals) {
                let slot: &mut dyn Any = slot;
                *slot
                    .downcast_mut::<Interval>()
                    .expect("validated bare interval") = Interval::from_valid_bounds(
                    pack_integer(lower, exponent, Direction::Down),
                    pack_integer(upper, exponent, Direction::Up),
                );
            }
        }
    }
    Some(())
}
