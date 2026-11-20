#![cfg(feature = "branch")]
#![allow(missing_docs, clippy::arithmetic_side_effects, clippy::unwrap_used)]

use core::convert::Infallible;

use maryada::{EnclosureOps, Interval, IntervalOps, RootIsolationError, RootIsolator, RunStatus};

#[test]
fn extended_newton_division_keeps_two_disjoint_candidates() {
    let mut search = RootIsolator::new(
        Interval::new(-2.0, 2.0),
        |x: &Interval| Ok::<_, Infallible>(x.sqr() - 2.0),
        |x: &Interval| Ok::<_, Infallible>(2.0 * *x),
    );

    let progress = search.run(1).unwrap();
    assert_eq!(progress.status, RunStatus::StepLimit);
    assert_eq!(progress.evaluated, 1);
    assert_eq!(
        progress.unresolved,
        [Interval::new(-2.0, -0.5), Interval::new(0.5, 2.0)]
    );
    assert_eq!(progress.unique, []);

    let report = search.run(2).unwrap();
    assert_eq!(report.status, RunStatus::Exhausted);
    assert_eq!(report.unresolved, []);
    assert_eq!(report.evaluated, 3);
    assert_eq!(report.unique.len(), 2);
    assert!(report.unique.first().unwrap().contains(-2.0_f64.sqrt()));
    assert!(report.unique.last().unwrap().contains(2.0_f64.sqrt()));
}

#[test]
fn root_exclusion_can_use_the_function_range_or_newton_image() {
    let domain = Interval::new(0.0, 1.0);
    let mut positive = RootIsolator::new(
        domain,
        |x: &Interval| Ok::<_, Infallible>(*x + 10.0),
        |_: &Interval| Err::<Interval, _>("derivative is unnecessary"),
    );
    let mut overestimated = RootIsolator::new(
        domain,
        |x: &Interval| {
            Ok::<_, Infallible>(if x.is_singleton() {
                *x + 10.0
            } else {
                Interval::new(0.0, 11.0)
            })
        },
        |_: &Interval| Ok::<_, Infallible>(Interval::ONE),
    );
    for report in [positive.run(1).unwrap(), overestimated.run(1).unwrap()] {
        assert_eq!(report.status, RunStatus::Exhausted);
        assert_eq!(report.excluded, [domain]);
        assert_eq!(report.unique, []);
        assert_eq!(report.unresolved, []);
    }
}

#[test]
fn zero_in_both_enclosures_keeps_uncertain_roots() {
    let domain = Interval::new(-1.0, 1.0);
    let mut repeated = RootIsolator::new(
        domain,
        |x: &Interval| Ok::<_, Infallible>(x.sqr()),
        |x: &Interval| Ok::<_, Infallible>(2.0 * *x),
    );
    let report = repeated.run(12).unwrap();
    assert_eq!(report.unique, []);
    assert!(report.unresolved.iter().any(|region| region.contains(0.0)));

    let mut constant = RootIsolator::new(
        domain,
        |_: &Interval| Ok::<_, Infallible>(Interval::ZERO),
        |_: &Interval| Ok::<_, Infallible>(Interval::ZERO),
    );
    let report = constant.run(1).unwrap();
    assert_eq!(report.status, RunStatus::Exhausted);
    assert_eq!(report.unresolved, [domain]);
    assert_eq!(report.unique, []);
}

#[test]
fn roots_on_shared_boundaries_are_reported_once() {
    let mut search = RootIsolator::new(
        Interval::new(-1.0, 1.0),
        |x: &Interval| Ok::<_, Infallible>(*x),
        |x: &Interval| {
            Ok::<_, Infallible>(if x.inf() < 0.0 && x.sup() > 0.0 {
                Interval::ENTIRE
            } else {
                Interval::ONE
            })
        },
    );
    assert_eq!(search.run(1).unwrap().status, RunStatus::StepLimit);
    let report = search.run(2).unwrap();
    assert_eq!(report.status, RunStatus::Exhausted);
    assert_eq!(report.unique, [Interval::ZERO]);
    assert_eq!(report.unresolved, []);
}

#[test]
fn endpoint_signs_certify_a_root_between_adjacent_floats() {
    const SCALE: f64 = 9_007_199_254_740_992.0;
    let domain = Interval::new(1.0, 1.0_f64.next_up());
    let mut search = RootIsolator::new(
        domain,
        |x: &Interval| Ok::<_, Infallible>(((*x - 1.0) * SCALE) - 1.0),
        |_: &Interval| Ok::<_, Infallible>(Interval::from(SCALE)),
    );
    let report = search.run(1).unwrap();
    assert_eq!(report.unique, [domain]);
    assert_eq!(report.unresolved, []);
}

#[test]
fn callback_failure_preserves_the_original_region_for_retry() {
    let domain = Interval::new(1.0, 2.0);
    let mut failed = false;
    let mut search = RootIsolator::new(
        domain,
        |x: &Interval| {
            if x.is_singleton() && !failed {
                failed = true;
                return Err("retry the point evaluation");
            }
            Ok(x.sqr() - 2.0)
        },
        |x: &Interval| Ok::<_, Infallible>(2.0 * *x),
    );
    assert_eq!(
        search.run(1),
        Err(RootIsolationError::Function("retry the point evaluation"))
    );
    let pending = search.run(0).unwrap();
    assert_eq!(pending.evaluated, 0);
    assert_eq!(pending.unresolved, [domain]);
    let report = search.run(1).unwrap();
    assert_eq!(report.status, RunStatus::Exhausted);
    assert_eq!(report.evaluated, 1);
    assert_eq!(report.unique.len(), 1);
}

#[test]
fn empty_domains_and_empty_enclosures_have_distinct_results() {
    let domain = Interval::new(-1.0, 1.0);
    let mut search = RootIsolator::new(
        domain,
        |_: &Interval| Ok::<_, Infallible>(Interval::EMPTY),
        |_: &Interval| Ok::<_, Infallible>(Interval::ONE),
    );
    assert_eq!(
        search.run(1),
        Err(RootIsolationError::EmptyFunctionEnclosure(domain))
    );

    let mut search = RootIsolator::new(
        domain,
        |x: &Interval| Ok::<_, Infallible>(*x),
        |_: &Interval| Ok::<_, Infallible>(Interval::EMPTY),
    );
    assert_eq!(
        search.run(1),
        Err(RootIsolationError::EmptyDerivativeEnclosure(domain))
    );

    let mut search = RootIsolator::new(
        Interval::EMPTY,
        |_: &Interval| Err::<Interval, _>("empty domain should skip evaluation"),
        |_: &Interval| Err::<Interval, _>("empty domain should skip evaluation"),
    );
    let report = search.run(1).unwrap();
    assert_eq!(report.status, RunStatus::Exhausted);
    assert_eq!(report.excluded, []);
    assert_eq!(report.unresolved, []);
}
