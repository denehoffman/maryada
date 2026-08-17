#![cfg(feature = "branch")]
#![allow(missing_docs)]
#![allow(clippy::float_cmp, clippy::unwrap_used)]
#![allow(clippy::indexing_slicing)]

use maryada::{
    BestFirstQueue, Bisect, BranchAndBound, EnclosureOps, GlobalMinimizationResult,
    GlobalMinimizer, GlobalMinimizerError, GlobalMinimizerOptions, GlobalMinimizerOptionsError,
    Interval, IntervalOps, PruneDecision, Pruner, RunStatus, StepOutcome, Unresolved, WorkQueue,
};

#[test]
fn widest_bisection_selects_the_largest_component() {
    let interval = Interval::new(-2.0, 6.0);
    assert_eq!(interval.branch_width(), 8.0);
    assert_eq!(
        Bisect::bisect(&interval),
        Some((Interval::new(-2.0, 2.0), Interval::new(2.0, 6.0)))
    );
    assert!(Bisect::bisect(&Interval::from(2.0)).is_none());

    let vector = vec![Interval::new(0.0, 1.0), Interval::new(-4.0, 4.0)];
    let (left, right) = vector.bisect().unwrap();
    assert_eq!(left[0], vector[0]);
    assert_eq!(right[0], vector[0]);
    assert_eq!(left[1], Interval::new(-4.0, 0.0));
    assert_eq!(right[1], Interval::new(0.0, 4.0));
}

#[test]
fn best_first_queue_is_stable_for_equal_priorities() {
    let mut queue = BestFirstQueue::new();
    queue.push("first", 2.0);
    queue.push("best", -1.0);
    queue.push("second", 2.0);

    assert_eq!(queue.pop().unwrap().domain, "best");
    assert_eq!(queue.pop().unwrap().domain, "first");
    assert_eq!(queue.pop().unwrap().domain, "second");
    assert!(queue.pop().is_none());
}

struct WidthPruner;

impl Pruner<Interval, Interval> for WidthPruner {
    type Error = core::convert::Infallible;

    fn prune(
        &mut self,
        domain: &mut Interval,
        _value: &Interval,
    ) -> Result<PruneDecision, Self::Error> {
        if domain.wid() <= 1.0 {
            Ok(PruneDecision::Certify)
        } else {
            Ok(PruneDecision::Branch {
                priority: domain.inf(),
            })
        }
    }
}

#[test]
fn branch_and_bound_steps_and_resumes() {
    let mut search = BranchAndBound::new(
        Interval::new(0.0, 4.0),
        |domain: &Interval| Ok::<Interval, core::convert::Infallible>(*domain),
        WidthPruner,
    );

    assert_eq!(search.step().unwrap(), StepOutcome::Branched);
    let limited = search.run(1).unwrap();
    assert_eq!(limited.status, RunStatus::StepLimit);
    assert!(limited.unresolved.len() >= 2);
    assert_eq!(limited.branched, 2);

    let complete = search.run(16).unwrap();
    assert_eq!(complete.status, RunStatus::Exhausted);
    assert_eq!(complete.certified.len(), 4);
    assert!(complete.unresolved.is_empty());
}

#[test]
fn retained_and_contracted_domains_are_reported() {
    struct Retain;

    impl Pruner<Interval, Interval> for Retain {
        type Error = core::convert::Infallible;

        fn prune(
            &mut self,
            domain: &mut Interval,
            _value: &Interval,
        ) -> Result<PruneDecision, Self::Error> {
            *domain = Interval::new(domain.inf(), 1.0);
            Ok(PruneDecision::Retain { priority: -2.0 })
        }
    }

    let mut search = BranchAndBound::new(
        Interval::new(-1.0, 2.0),
        |domain: &Interval| Ok::<Interval, core::convert::Infallible>(*domain),
        Retain,
    );
    let report = search.run(1).unwrap();
    assert_eq!(report.status, RunStatus::Exhausted);
    assert_eq!(report.retained, 1);
    assert_eq!(
        report.unresolved,
        vec![Unresolved {
            domain: Interval::new(-1.0, 1.0),
            priority: -2.0
        }]
    );
}

#[test]
fn global_minimizer_encloses_a_quadratic_minimum() {
    let options = GlobalMinimizerOptions {
        value_tolerance: Some(0.05),
        domain_tolerance: Some(0.05),
        gap_tolerance: None,
        max_steps: None,
    };
    let mut minimizer = GlobalMinimizer::with_options(
        Interval::new(-2.0, 3.0),
        |domain: &Interval| {
            let centered = *domain - 1.0;
            Ok::<Interval, core::convert::Infallible>(centered.sqr())
        },
        options,
    );
    let result: GlobalMinimizationResult<Interval> = minimizer.solve().unwrap();

    assert_eq!(result.status, RunStatus::Exhausted);
    assert!(result.minimum.contains(0.0));
    assert!(result.minimum.sup() >= 0.0);
    assert!(!result.candidates.is_empty());
}

#[test]
fn global_minimizer_default_is_unbounded_and_convergent() {
    let mut minimizer = GlobalMinimizer::new(Interval::new(0.0, 1.0), |domain: &Interval| {
        Ok::<Interval, core::convert::Infallible>(*domain)
    });

    assert_eq!(minimizer.options().max_steps, None);
    let result = minimizer.solve().unwrap();

    assert_eq!(result.status, RunStatus::Converged);
    assert!(result.minimum.contains(0.0));
    assert!(result.minimum.wid() <= 1.0e-6);
}

#[test]
fn global_minimizer_rejects_an_unbounded_configuration_without_tolerances() {
    let mut minimizer = GlobalMinimizer::with_options(
        Interval::new(0.0, 1.0),
        |domain: &Interval| Ok::<Interval, core::convert::Infallible>(*domain),
        GlobalMinimizerOptions {
            value_tolerance: None,
            domain_tolerance: None,
            gap_tolerance: None,
            max_steps: None,
        },
    );

    assert_eq!(
        minimizer.solve(),
        Err(GlobalMinimizerError::InvalidOptions(
            GlobalMinimizerOptionsError::NoTerminationCriterion
        ))
    );
}

#[test]
fn global_minimizer_rejects_nonpositive_tolerances() {
    let mut minimizer = GlobalMinimizer::new(Interval::new(0.0, 1.0), |domain: &Interval| {
        Ok::<Interval, core::convert::Infallible>(*domain)
    })
    .with_value_tolerance(0.0);

    assert_eq!(
        minimizer.solve(),
        Err(GlobalMinimizerError::InvalidOptions(
            GlobalMinimizerOptionsError::InvalidTolerance {
                name: "value_tolerance",
                value: 0.0,
            }
        ))
    );
}

#[test]
fn global_minimizer_uses_a_configured_step_limit_for_solve() {
    let mut minimizer = GlobalMinimizer::new(Interval::new(0.0, 1.0), |domain: &Interval| {
        Ok::<Interval, core::convert::Infallible>(*domain)
    })
    .with_max_steps(1);

    let result = minimizer.solve().unwrap();
    assert_eq!(result.status, RunStatus::StepLimit);
}

#[test]
#[allow(clippy::arithmetic_side_effects, clippy::unnecessary_wraps)]
fn global_minimizer_reports_the_domain_for_its_best_value_bound() {
    fn objective(domain: &Interval) -> Result<Interval, core::convert::Infallible> {
        Ok((domain.sqr() - 1.0).sqr() + 0.6 * *domain + 2.0)
    }

    let mut minimizer = GlobalMinimizer::new(Interval::new(-1.5, 1.5), objective);
    let result = minimizer.advance(20).unwrap();
    let domain = result.best_domain.unwrap();
    let value = result.best_value.unwrap();
    let point = result.best_point.unwrap();

    assert_eq!(result.status, RunStatus::StepLimit);
    assert!(point.is_singleton());
    assert!(domain.contains(point.mid()));
    assert!(domain.contains(-1.0679));
    assert!(value.contains(1.3790));
    assert!(result.minimum.contains(1.3790));
}

#[test]
#[allow(clippy::arithmetic_side_effects, clippy::unnecessary_wraps)]
fn global_minimizer_keeps_live_parameter_boxes_in_the_result() {
    fn objective(domain: &Interval) -> Result<Interval, core::convert::Infallible> {
        Ok((domain.sqr() - 1.0).sqr() + 0.6 * *domain + 2.0)
    }

    let mut minimizer = GlobalMinimizer::new(Interval::new(-1.5, 1.5), objective);
    let result = minimizer.advance(10).unwrap();
    let domain = Interval::new(-1.21875, -1.125);
    let value = objective(&domain).unwrap();

    assert!(
        result
            .unresolved
            .iter()
            .any(|pending| pending.domain == domain)
    );
    assert!(value.inf() > 1.339);
    assert!(value.sup() < 1.561);
    assert!(result.minimum.inf() > 1.1);
}

#[cfg(feature = "complex")]
#[test]
fn complex_boxes_split_the_widest_real_or_imaginary_component() {
    use maryada::ComplexBox;

    let box_domain = ComplexBox::new(Interval::new(-1.0, 1.0), Interval::new(-4.0, 4.0));
    let (left, right) = box_domain.bisect().unwrap();
    assert_eq!(left.re, box_domain.re);
    assert_eq!(right.re, box_domain.re);
    assert_eq!(left.im, Interval::new(-4.0, 0.0));
    assert_eq!(right.im, Interval::new(0.0, 4.0));
}
