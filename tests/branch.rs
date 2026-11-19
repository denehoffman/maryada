#![allow(missing_docs)]
#![cfg(feature = "branch")]
#![allow(clippy::float_cmp, clippy::unwrap_used)]
#![allow(clippy::indexing_slicing)]

use maryada::{
    BestFirstQueue, Bisect, BranchAndBound, BranchRule, EnclosureOps, GlobalMinimizationResult,
    GlobalMinimizer, GlobalMinimizerError, GlobalMinimizerOptions, GlobalMinimizerOptionsError,
    Interval, IntervalOps, PruneDecision, Pruner, RunStatus, SearchError, StepOutcome, Unresolved,
    WorkQueue,
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

    let vector = vec![
        Interval::new(0.0, 1.0),
        Interval::new(-4.0, 4.0),
        Interval::new(2.0, 10.0),
    ];
    let (left, right) = vector.bisect().unwrap();
    assert_eq!(left[0], vector[0]);
    assert_eq!(right[0], vector[0]);
    assert_eq!(left[1], Interval::new(-4.0, 0.0));
    assert_eq!(right[1], Interval::new(0.0, 4.0));
    assert_eq!(left[2], vector[2]);
    assert_eq!(right[2], vector[2]);
}

#[test]
fn vector_bisection_skips_unsplittable_coordinates() {
    let adjacent = Interval::new(1e16, 1e16_f64.next_up());
    let narrow = Interval::new(0.0, 0.25);
    let splittable = Interval::new(0.0, 1.0);
    let vector = vec![adjacent, narrow, splittable];

    assert_eq!(
        vector.bisect(),
        Some((
            vec![adjacent, narrow, Interval::new(0.0, 0.5)],
            vec![adjacent, narrow, Interval::new(0.5, 1.0)],
        ))
    );
    assert!(vec![adjacent, adjacent].bisect().is_none());
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
fn branch_and_bound_retries_the_original_domain_after_callback_errors() {
    struct RetryPruner(bool);

    impl Pruner<Interval, Interval> for RetryPruner {
        type Error = &'static str;

        fn prune(
            &mut self,
            domain: &mut Interval,
            _: &Interval,
        ) -> Result<PruneDecision, Self::Error> {
            if domain.wid() > 1.0 {
                return Ok(PruneDecision::Branch { priority: -2.0 });
            }
            if core::mem::take(&mut self.0) {
                *domain = Interval::ZERO;
                return Err("transient pruning error");
            }
            Ok(PruneDecision::Certify)
        }
    }

    let mut fail_evaluation = true;
    let mut search = BranchAndBound::new(
        Interval::new(-1.0, 1.0),
        |domain: &Interval| {
            if domain.wid() <= 1.0 && core::mem::take(&mut fail_evaluation) {
                return Err("transient evaluation error");
            }
            Ok(*domain)
        },
        RetryPruner(true),
    );
    assert_eq!(search.step().unwrap(), StepOutcome::Branched);
    let before = search.report(RunStatus::StepLimit);

    assert_eq!(
        search.run(1),
        Err(SearchError::Evaluation("transient evaluation error"))
    );
    assert_eq!(search.report(RunStatus::StepLimit), before);
    assert_eq!(
        search.step(),
        Err(SearchError::Pruning("transient pruning error"))
    );
    assert_eq!(search.pending_count(), 2);
    assert_eq!(search.pending_domains(), before.unresolved);

    let retried = search.run(1).unwrap();
    assert_eq!(retried.status, RunStatus::StepLimit);
    assert_eq!(retried.evaluated, 3);
    assert_eq!(retried.certified, vec![Interval::new(-1.0, 0.0)]);
    assert_eq!(
        retried.unresolved,
        vec![Unresolved {
            domain: Interval::new(0.0, 1.0),
            priority: -2.0
        }]
    );
}

#[test]
fn branch_and_bound_resumes_when_only_the_failed_domain_remains() {
    let mut fail_once = true;
    let root = Interval::new(0.0, 1.0);
    let mut search = BranchAndBound::new(
        root,
        |domain: &Interval| {
            if core::mem::take(&mut fail_once) {
                return Err("transient evaluation error");
            }
            Ok(*domain)
        },
        WidthPruner,
    );

    assert_eq!(
        search.run(1),
        Err(SearchError::Evaluation("transient evaluation error"))
    );
    assert_eq!(search.run(0).unwrap().status, RunStatus::StepLimit);

    let retried = search.run(1).unwrap();
    assert_eq!(retried.status, RunStatus::Exhausted);
    assert_eq!(retried.evaluated, 1);
    assert_eq!(retried.certified, vec![root]);
    assert!(retried.unresolved.is_empty());
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
        absolute_gap_tolerance: None,
        relative_gap_tolerance: None,
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
fn global_minimizer_retries_after_box_or_midpoint_errors() {
    for fail_on in [1_usize, 2] {
        let mut calls = 0_usize;
        let mut minimizer = GlobalMinimizer::new(Interval::new(-1.0, 1.0), |domain: &Interval| {
            calls = calls.saturating_add(1);
            if calls == fail_on {
                return Err("transient objective error");
            }
            Ok(domain.sqr())
        });
        let mut expected = minimizer.result(RunStatus::StepLimit);

        assert_eq!(
            minimizer.solve(),
            Err(GlobalMinimizerError::Evaluation(
                "transient objective error"
            ))
        );
        assert_eq!(minimizer.pending_count(), 1);
        assert!(minimizer.best_sample().is_none());
        expected.evaluations = fail_on;
        assert_eq!(minimizer.result(RunStatus::StepLimit), expected);

        let retried = minimizer.solve().unwrap();
        assert_eq!(retried.status, RunStatus::Converged);
        assert_eq!(retried.minimum, Interval::ZERO);
        assert_eq!(retried.evaluations, fail_on.saturating_add(2));
        assert_eq!(
            minimizer.best_sample(),
            Some((&Interval::ZERO, Interval::ZERO))
        );
    }
}

#[test]
fn global_minimizer_tracks_a_sample_independently_of_the_best_box() {
    let root = Interval::new(-2.0, 2.0);
    let mut minimizer = GlobalMinimizer::new(root, |domain: &Interval| {
        // These are enclosures of the same constant function. The root box
        // gives a stronger upper bound than its first sampled midpoint.
        let value = if *domain == Interval::ZERO {
            Interval::new(1.0, 2.0)
        } else if domain.is_singleton() {
            Interval::ONE
        } else {
            Interval::new(0.0, 1.0)
        };
        Ok::<_, core::convert::Infallible>(value)
    })
    .without_absolute_gap_tolerance()
    .without_relative_gap_tolerance()
    .without_value_tolerance()
    .without_domain_tolerance();

    assert!(minimizer.best_sample().is_none());
    let progress = minimizer.advance_progress(1).unwrap();
    assert_eq!(progress.evaluations, 2);
    let result = minimizer.result(progress.status);
    assert_eq!(result.best_point, Some(root));
    assert_eq!(result.best_domain, Some(root));
    assert_eq!(result.minimum.sup(), 1.0);
    assert_eq!(
        minimizer.best_sample(),
        Some((&Interval::ZERO, Interval::new(1.0, 2.0)))
    );

    let progress = minimizer.advance_progress(2).unwrap();
    assert_eq!(progress.evaluations, 6);
    assert_eq!(
        minimizer.best_sample(),
        Some((&Interval::from(-1.0), Interval::ONE))
    );
    assert_eq!(minimizer.result(progress.status).best_domain, Some(root));
}

#[test]
fn global_minimizer_accepts_singleton_domains_and_skips_empty_samples() {
    let mut singleton = GlobalMinimizer::new(vec![Interval::ONE], |_: &Vec<Interval>| {
        Ok::<_, core::convert::Infallible>(Interval::new(1.0, 2.0))
    });
    assert_eq!(singleton.advance_progress(1).unwrap().evaluations, 1);
    let (point, value) = singleton.best_sample().unwrap();
    assert_eq!(point.as_slice(), &[Interval::ONE]);
    assert_eq!(value, Interval::new(1.0, 2.0));

    let mut empty =
        GlobalMinimizer::new(vec![Interval::ONE, Interval::EMPTY], |_: &Vec<Interval>| {
            Ok::<_, core::convert::Infallible>(Interval::ONE)
        });
    empty.advance_progress(1).unwrap();
    assert!(empty.best_sample().is_none());

    let mut undefined = GlobalMinimizer::new(Interval::new(-1.0, 1.0), |domain: &Interval| {
        if domain.is_singleton() {
            Ok::<_, core::convert::Infallible>(Interval::EMPTY)
        } else {
            Ok(Interval::new(0.0, 1.0))
        }
    });
    undefined.advance_progress(1).unwrap();
    assert!(undefined.best_sample().is_none());
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
fn global_minimizer_rounds_the_absolute_gap_upward() {
    let value = Interval::new(-f64::from_bits(1), 1.0);
    let mut minimizer = GlobalMinimizer::new(Interval::new(-1.0, 1.0), |_: &Interval| {
        Ok::<Interval, core::convert::Infallible>(value)
    })
    .without_value_tolerance()
    .without_domain_tolerance()
    .without_relative_gap_tolerance()
    .with_absolute_gap_tolerance(1.0);

    let result = minimizer.advance(1).unwrap();
    assert_eq!(result.minimum, value);
    assert_eq!(result.unresolved.len(), 2);
    assert_eq!(result.status, RunStatus::StepLimit);

    let result = minimizer
        .with_absolute_gap_tolerance(1.0_f64.next_up())
        .advance(0)
        .unwrap();
    assert_eq!(result.status, RunStatus::Converged);
}

#[test]
fn global_minimizer_certifies_the_relative_gap_at_rounding_boundaries() {
    for (value, below, enough) in [
        // Rounding the width or threshold to nearest would accept `below`.
        (
            Interval::new(-f64::from_bits(1), 1.0),
            1.0,
            1.0_f64.next_up(),
        ),
        (
            Interval::new(2.0, 3.0),
            1.0 / 3.0,
            (1.0_f64 / 3.0).next_up(),
        ),
        // The width overflows before scaling, but the relative gap is exactly 2.
        (Interval::new(-f64::MAX, f64::MAX), 2.0_f64.next_down(), 2.0),
    ] {
        let mut minimizer = GlobalMinimizer::new(Interval::new(-1.0, 1.0), |_: &Interval| {
            Ok::<_, core::convert::Infallible>(value)
        })
        .without_absolute_gap_tolerance()
        .without_value_tolerance()
        .without_domain_tolerance()
        .with_relative_gap_tolerance(below);

        let progress = minimizer.advance_progress(1).unwrap();
        assert_eq!(progress.minimum, value);
        assert_eq!(progress.status, RunStatus::StepLimit);

        let progress_after = minimizer
            .with_relative_gap_tolerance(enough)
            .advance_progress(0)
            .unwrap();
        assert_eq!(progress_after.status, RunStatus::Converged);
        assert_eq!(progress_after.evaluations, progress.evaluations);
    }
}

#[test]
fn global_minimizer_enables_both_gap_checks_by_default() {
    for (value, absolute_only, relative_only) in [
        (
            Interval::new(0.0, 5.0e-7),
            RunStatus::Converged,
            RunStatus::StepLimit,
        ),
        (
            Interval::new(9_999_999.0, 10_000_000.0),
            RunStatus::StepLimit,
            RunStatus::Converged,
        ),
    ] {
        let mut minimizer = GlobalMinimizer::new(Interval::new(-1.0, 1.0), |_: &Interval| {
            Ok::<_, core::convert::Infallible>(value)
        })
        .without_value_tolerance()
        .without_domain_tolerance();
        assert_eq!(minimizer.options().absolute_gap_tolerance, Some(1.0e-6));
        assert_eq!(minimizer.options().relative_gap_tolerance, Some(1.0e-6));
        assert_eq!(
            minimizer.advance_progress(1).unwrap().status,
            RunStatus::Converged
        );

        minimizer = minimizer.without_absolute_gap_tolerance();
        assert_eq!(minimizer.advance_progress(0).unwrap().status, relative_only);

        minimizer = minimizer
            .with_absolute_gap_tolerance(1.0e-6)
            .without_relative_gap_tolerance();
        assert_eq!(minimizer.advance_progress(0).unwrap().status, absolute_only);

        minimizer = minimizer.without_absolute_gap_tolerance();
        assert_eq!(
            minimizer.advance_progress(0).unwrap().status,
            RunStatus::StepLimit
        );
    }
}

#[test]
fn global_minimizer_can_stop_with_either_gap_tolerance() {
    for (value, absolute) in [
        (Interval::new(9.0, 10.0), None),
        (Interval::new(-1000.0, -900.0), Some(1.0e-6)),
        (Interval::new(2.0, 3.0), Some(1.0)),
        (Interval::ZERO, None),
    ] {
        let mut minimizer = GlobalMinimizer::with_options(
            Interval::new(-1.0, 1.0),
            |_: &Interval| Ok::<_, core::convert::Infallible>(value),
            GlobalMinimizerOptions {
                value_tolerance: None,
                domain_tolerance: None,
                absolute_gap_tolerance: absolute,
                relative_gap_tolerance: Some(0.125),
                max_steps: None,
            },
        );
        assert_eq!(minimizer.solve().unwrap().status, RunStatus::Converged);
    }
}

#[test]
fn global_minimizer_validates_and_can_disable_relative_gap_tolerance() {
    let mut minimizer = GlobalMinimizer::new(Interval::new(-1.0, 1.0), |_: &Interval| {
        Ok::<_, core::convert::Infallible>(Interval::ENTIRE)
    })
    .without_absolute_gap_tolerance()
    .without_value_tolerance()
    .without_domain_tolerance();

    for value in [0.0, f64::INFINITY] {
        minimizer = minimizer.with_relative_gap_tolerance(value);
        assert_eq!(minimizer.options().relative_gap_tolerance, Some(value));
        assert_eq!(
            minimizer.advance_progress(1),
            Err(GlobalMinimizerError::InvalidOptions(
                GlobalMinimizerOptionsError::InvalidTolerance {
                    name: "relative_gap_tolerance",
                    value,
                }
            ))
        );
    }

    minimizer = minimizer.with_relative_gap_tolerance(2.0);
    assert_eq!(minimizer.advance_progress(0).unwrap().evaluations, 0);
    assert_eq!(
        minimizer.advance_progress(1).unwrap().status,
        RunStatus::StepLimit
    );

    minimizer = minimizer.without_relative_gap_tolerance();
    assert_eq!(minimizer.options().relative_gap_tolerance, None);
    assert_eq!(
        minimizer.solve(),
        Err(GlobalMinimizerError::InvalidOptions(
            GlobalMinimizerOptionsError::NoTerminationCriterion
        ))
    );
}

#[test]
fn global_minimizer_progress_keeps_lower_bounds_from_terminal_domains() {
    struct KeepLeft;

    impl BranchRule<Interval> for KeepLeft {
        fn split(&self, domain: &Interval) -> Option<(Interval, Interval)> {
            if *domain == Interval::new(0.0, 0.5) {
                None
            } else {
                Bisect::bisect(domain)
            }
        }
    }

    for value_tolerance in [None, Some(0.5)] {
        // All enclosures contain the constant objective 0.25. The left child
        // becomes terminal and has a smaller lower bound than the right descendants.
        let objective = |domain: &Interval| {
            let value = if Bisect::is_singleton(domain) {
                Interval::from(0.25)
            } else if *domain == Interval::new(0.0, 0.5) {
                Interval::new(0.0, 0.5)
            } else if domain.inf() >= 0.5 {
                Interval::new(0.25, 2.0)
            } else {
                Interval::new(0.0, 2.0)
            };
            Ok::<_, core::convert::Infallible>(value)
        };
        let options = GlobalMinimizerOptions {
            value_tolerance,
            domain_tolerance: None,
            absolute_gap_tolerance: Some(0.125),
            relative_gap_tolerance: None,
            max_steps: None,
        };
        let mut minimizer = GlobalMinimizer::with_branch_and_options(
            Interval::new(0.0, 1.0),
            objective,
            KeepLeft,
            options,
        );
        assert_eq!(minimizer.advance_progress(1).unwrap().pending, 2);
        let progress = minimizer.advance_progress(2).unwrap();
        assert_eq!(progress.status, RunStatus::StepLimit);
        assert_eq!(progress.minimum, Interval::new(0.0, 0.25));
        assert_eq!(progress.pending, 2);
        assert_eq!(progress.candidates, usize::from(value_tolerance.is_some()));
        assert_eq!(progress.unresolved, usize::from(value_tolerance.is_none()));
        assert_eq!(progress.evaluations, 6);
        assert_eq!(progress.branched, 2);
        assert_eq!(minimizer.advance_progress(0).unwrap(), progress);

        let result = minimizer.result(progress.status);
        assert_eq!(result.minimum, progress.minimum);
        assert_eq!(result.candidates.len(), progress.candidates);
        assert_eq!(
            result.unresolved.len(),
            if value_tolerance.is_some() { 2 } else { 3 }
        );
    }
}

#[test]
fn global_minimizer_rejects_an_unbounded_configuration_without_tolerances() {
    let mut minimizer = GlobalMinimizer::with_options(
        Interval::new(0.0, 1.0),
        |domain: &Interval| Ok::<Interval, core::convert::Infallible>(*domain),
        GlobalMinimizerOptions {
            value_tolerance: None,
            domain_tolerance: None,
            absolute_gap_tolerance: None,
            relative_gap_tolerance: None,
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

    minimizer = minimizer
        .without_value_tolerance()
        .with_absolute_gap_tolerance(0.0);
    assert_eq!(
        minimizer.solve(),
        Err(GlobalMinimizerError::InvalidOptions(
            GlobalMinimizerOptionsError::InvalidTolerance {
                name: "absolute_gap_tolerance",
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

#[cfg(feature = "complex")]
#[test]
fn complex_bisection_skips_unsplittable_coordinates() {
    use maryada::ComplexBox;

    let adjacent = Interval::new(1e16, 1e16_f64.next_up());
    let splittable = Interval::new(0.0, 1.0);
    let lower_half = Interval::new(0.0, 0.5);
    let upper_half = Interval::new(0.5, 1.0);

    assert_eq!(
        ComplexBox::new(adjacent, splittable).bisect(),
        Some((
            ComplexBox::new(adjacent, lower_half),
            ComplexBox::new(adjacent, upper_half),
        ))
    );
    for imaginary in [adjacent, splittable] {
        assert_eq!(
            ComplexBox::new(splittable, imaginary).bisect(),
            Some((
                ComplexBox::new(lower_half, imaginary),
                ComplexBox::new(upper_half, imaginary),
            ))
        );
    }
    assert!(ComplexBox::new(adjacent, adjacent).bisect().is_none());
}
