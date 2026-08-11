#![allow(missing_docs)]
use maryada::{
    DecoratedInterval, Interval, IntervalOps, Signal, SignalFlags, interval_from_be_bytes,
    interval_part,
};

#[test]
fn required_exception_conditions_are_reported_and_accumulated() {
    let mut signals = SignalFlags::NONE;

    assert!(Interval::nums_to_interval(2.0, 1.0, &mut signals).is_empty());
    assert!(signals.contains(Signal::UndefinedOperation));

    assert!(interval_part(DecoratedInterval::NAI, &mut signals).is_empty());
    assert!(signals.contains(Signal::IntvlPartOfNaI));

    // (+0, +0) is not the standard inf-sup representation: the lower zero
    // must be negative zero.
    assert!(interval_from_be_bytes(&[0; 16], &mut signals).is_empty());
    assert!(signals.contains(Signal::InvalidOperand));

    let raised = signals.take();
    assert!(signals.is_empty());
    assert!(raised.contains(Signal::UndefinedOperation));
    assert!(raised.contains(Signal::IntvlPartOfNaI));
    assert!(raised.contains(Signal::InvalidOperand));
}
