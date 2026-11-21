#![allow(missing_docs)]

use maryada::{
    DecoratedInterval, Decoration, DotLengthMismatch, EnclosureOps, Interval, IntervalOps, dot,
    set_dec, sum,
};

#[test]
fn sum_rounds_once_after_cancellation() {
    let max = Interval::from(f64::MAX);
    let tiny = Interval::from(f64::from_bits(1));
    assert_eq!(sum(&[max, max, tiny, -max, -max]), tiny);
    assert_eq!(sum(&[-max, -max, -tiny, max, max]), -tiny);
    assert_eq!(
        sum(&[Interval::ONE, tiny]),
        Interval::new(1.0, 1.0_f64.next_up())
    );
    assert_eq!(sum(&[max, max]), Interval::new(f64::MAX, f64::INFINITY));
}

#[test]
fn iterators_sum_owned_and_borrowed_intervals() {
    let max = Interval::from(f64::MAX);
    let tiny = Interval::from(f64::from_bits(1));
    let terms = [max, max, tiny, -max, -max];
    assert_eq!(terms.iter().sum::<Interval>(), tiny);
    assert_eq!(
        terms.into_iter().map(core::ops::Neg::neg).sum::<Interval>(),
        -tiny
    );

    let defined = set_dec(Interval::ONE, Decoration::Def);
    let terms = [DecoratedInterval::ONE, defined];
    let expected = set_dec(Interval::from(2.0), Decoration::Def);
    assert_eq!(terms.iter().sum::<DecoratedInterval>(), expected);
    assert_eq!(terms.into_iter().sum::<DecoratedInterval>(), expected);
    assert_eq!(
        core::iter::empty::<Interval>().sum::<Interval>(),
        Interval::ZERO
    );

    let terms = [DecoratedInterval::EMPTY, DecoratedInterval::NAI];
    let mut consumed = 0_usize;
    let result: DecoratedInterval = terms
        .into_iter()
        .inspect(|_| consumed = consumed.saturating_add(1))
        .sum();
    assert!(result.is_nai());
    assert_eq!(consumed, terms.len());
}

#[test]
fn dot_accumulates_products_before_rounding_or_overflow() {
    let max = Interval::from(f64::MAX);
    let tiny = Interval::from(f64::from_bits(1));
    assert_eq!(
        dot(&[max, max, tiny], &[max, -max, Interval::ONE]),
        Ok(tiny)
    );

    let half = Interval::from(0.5);
    assert_eq!(dot(&[tiny, tiny], &[half, half]), Ok(tiny));
    assert_eq!(dot(&[-tiny, -tiny], &[half, half]), Ok(-tiny));
    assert_eq!(
        dot(&[tiny], &[half]),
        Ok(Interval::new(0.0, f64::from_bits(1)))
    );
}

#[test]
fn reductions_handle_interval_widths_and_unbounded_terms() {
    let left = [Interval::new(-2.0, 3.0), Interval::new(2.0, 3.0)];
    let right = [Interval::new(-4.0, 5.0), Interval::new(4.0, 5.0)];
    assert_eq!(sum(&left), Interval::new(0.0, 6.0));
    assert_eq!(dot(&left, &right), Ok(Interval::new(-4.0, 30.0)));

    let positive = Interval::new(2.0, f64::INFINITY);
    assert_eq!(dot(&[positive], &[Interval::ZERO]), Ok(Interval::ZERO));
    assert_eq!(
        dot(&[positive], &[Interval::new(-2.0, -1.5)]),
        Ok(Interval::new(f64::NEG_INFINITY, -3.0))
    );
    assert_eq!(sum(&[positive, -positive]), Interval::ENTIRE);
}

#[test]
fn reductions_validate_lengths_and_propagate_empty_and_nai() {
    assert_eq!(sum::<Interval>(&[]), Interval::ZERO);
    assert_eq!(
        dot::<DecoratedInterval>(&[], &[]),
        Ok(DecoratedInterval::ZERO)
    );
    assert_eq!(
        dot(&[Interval::ONE], &[]),
        Err(DotLengthMismatch { left: 1, right: 0 })
    );
    assert_eq!(sum(&[Interval::EMPTY, Interval::ONE]), Interval::EMPTY);
    assert_eq!(
        dot(&[Interval::EMPTY], &[Interval::ZERO]),
        Ok(Interval::EMPTY)
    );

    let empty_then_nai = [DecoratedInterval::EMPTY, DecoratedInterval::NAI];
    assert!(sum(&empty_then_nai).is_nai());
    assert!(dot(&empty_then_nai, &[DecoratedInterval::ONE; 2]).is_ok_and(EnclosureOps::is_nai));
    let defined = set_dec(Interval::ONE, Decoration::Def);
    assert_eq!(
        sum(&[DecoratedInterval::ONE, defined]).decoration(),
        Decoration::Def
    );
    assert_eq!(dot(&[DecoratedInterval::ONE], &[defined]), Ok(defined));
}
