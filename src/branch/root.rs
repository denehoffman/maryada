use alloc::vec::Vec;
use core::{convert::Infallible, fmt};

use crate::{EnclosureOps, Interval, IntervalOps, div_split};

use super::{
    BestFirstQueue, Bisect, BranchAndBound, BranchRule, Evaluator, PruneDecision, Pruner,
    RunStatus, SearchError,
};

/// An error encountered while isolating scalar roots.
#[derive(Debug, Eq, PartialEq)]
pub enum RootIsolationError<F, D> {
    /// The function enclosure callback failed.
    Function(F),
    /// The derivative enclosure callback failed.
    Derivative(D),
    /// The function callback returned Empty for this nonempty input.
    EmptyFunctionEnclosure(Interval),
    /// The derivative callback returned Empty for this nonempty input.
    EmptyDerivativeEnclosure(Interval),
}

impl<F: fmt::Display, D: fmt::Display> fmt::Display for RootIsolationError<F, D> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Function(error) => write!(formatter, "function evaluation failed: {error}"),
            Self::Derivative(error) => write!(formatter, "derivative evaluation failed: {error}"),
            Self::EmptyFunctionEnclosure(domain) => {
                write!(formatter, "empty function enclosure for {domain}")
            }
            Self::EmptyDerivativeEnclosure(domain) => {
                write!(formatter, "empty derivative enclosure for {domain}")
            }
        }
    }
}

/// Cumulative results from a scalar interval Newton search.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RootIsolationReport {
    /// Whether the queue is exhausted or the step budget was reached.
    pub status: RunStatus,
    /// Enclosures that each contain exactly one root, ordered from left to right.
    pub unique: Vec<Interval>,
    /// Evaluated regions proved to contain no root. Contraction may also
    /// eliminate parts of a region without recording them here.
    pub excluded: Vec<Interval>,
    /// Remaining regions without a uniqueness or exclusion proof, including
    /// pending work and regions the search could not subdivide.
    pub unresolved: Vec<Interval>,
    /// Number of regions successfully evaluated and classified so far.
    pub evaluated: usize,
}

/// A resumable interval Newton search for roots of a scalar function.
///
/// The function must be continuous on the initial domain and differentiable
/// in its interior. The callbacks must enclose the function and its derivative
/// throughout each input interval. The function callback also receives
/// singleton intervals for point evaluations.
///
/// The search reports a root as unique as soon as it has a proof. The width
/// of the reported enclosure depends on the contraction that proved it.
///
/// ```
/// use core::convert::Infallible;
/// use maryada::{Interval, IntervalDual, IntervalOps, RootIsolator, RunStatus};
///
/// let polynomial = |x: &Interval| IntervalDual::variable(*x).sqr() - 2.0;
/// let mut search = RootIsolator::new(
///     Interval::new(-2.0, 2.0),
///     |x: &Interval| Ok::<_, Infallible>(polynomial(x).value),
///     |x: &Interval| polynomial(x).derivative.ok_or("derivative unavailable"),
/// );
/// let report = search.run(32).unwrap();
/// assert_eq!(report.status, RunStatus::Exhausted);
/// assert_eq!(report.unique.len(), 2);
/// assert!(report.unresolved.is_empty());
/// ```
pub struct RootIsolator<F, D>
where
    F: Evaluator<Interval, Value = Interval>,
    D: Evaluator<Interval, Value = Interval>,
{
    search: BranchAndBound<RootCandidate, NewtonEvaluator<F, D>, RootPruner, RootBranch>,
}

impl<F, D> RootIsolator<F, D>
where
    F: Evaluator<Interval, Value = Interval>,
    D: Evaluator<Interval, Value = Interval>,
{
    /// Creates a search over `domain` with function and derivative enclosures.
    #[must_use]
    pub fn new(domain: Interval, function: F, derivative: D) -> Self {
        Self {
            search: BranchAndBound::with_branch_and_queue(
                RootCandidate::single(domain),
                NewtonEvaluator {
                    function,
                    derivative,
                },
                RootPruner {
                    excluded: Vec::new(),
                },
                RootBranch,
                BestFirstQueue::new(),
            ),
        }
    }

    /// Processes at most `max_steps` regions and returns the current report.
    /// A later call resumes any pending work.
    ///
    /// # Errors
    ///
    /// Returns the first callback failure or empty enclosure on a nonempty
    /// region. The original region is kept for retry.
    pub fn run(
        &mut self,
        max_steps: usize,
    ) -> Result<RootIsolationReport, RootIsolationError<F::Error, D::Error>> {
        let report = match self.search.run(max_steps) {
            Ok(report) => report,
            Err(SearchError::Evaluation(error)) => return Err(error),
            Err(SearchError::Pruning(never)) => match never {},
        };
        let mut unique: Vec<_> = report
            .certified
            .into_iter()
            .map(|entry| entry.interval)
            .collect();
        unique.sort_unstable_by(|left, right| {
            left.inf()
                .total_cmp(&right.inf())
                .then_with(|| left.sup().total_cmp(&right.sup()))
        });
        unique.dedup();
        Ok(RootIsolationReport {
            status: report.status,
            unique,
            excluded: self.search.pruner.excluded.clone(),
            unresolved: report
                .unresolved
                .into_iter()
                .map(|entry| entry.domain.interval)
                .filter(|domain| !domain.is_empty())
                .collect(),
            evaluated: report.evaluated,
        })
    }
}

#[derive(Clone, Copy)]
struct RootCandidate {
    interval: Interval,
    // Set only between pruning a region and queuing two disjoint Newton images.
    other: Interval,
}

impl RootCandidate {
    const fn single(interval: Interval) -> Self {
        Self {
            interval,
            other: Interval::EMPTY,
        }
    }
}

struct RootBranch;

impl BranchRule<RootCandidate> for RootBranch {
    fn split(&self, candidate: &RootCandidate) -> Option<(RootCandidate, RootCandidate)> {
        let (left, right) = if candidate.other.is_empty() {
            Bisect::bisect(&candidate.interval)?
        } else {
            (candidate.interval, candidate.other)
        };
        Some((RootCandidate::single(left), RootCandidate::single(right)))
    }
}

#[derive(Clone, Copy)]
enum RootDecision {
    Exclude,
    Unique(Interval),
    Retain(Interval),
    Subdivide(RootCandidate),
}

struct RootPruner {
    excluded: Vec<Interval>,
}

impl Pruner<RootCandidate, RootDecision> for RootPruner {
    type Error = Infallible;

    fn prune(
        &mut self,
        candidate: &mut RootCandidate,
        decision: &RootDecision,
    ) -> Result<PruneDecision, Self::Error> {
        Ok(match *decision {
            RootDecision::Exclude => {
                if !candidate.interval.is_empty() {
                    self.excluded.push(candidate.interval);
                }
                PruneDecision::Discard
            }
            RootDecision::Unique(interval) => {
                *candidate = RootCandidate::single(interval);
                PruneDecision::Certify
            }
            RootDecision::Retain(interval) => {
                *candidate = RootCandidate::single(interval);
                PruneDecision::Retain {
                    priority: -interval.wid(),
                }
            }
            RootDecision::Subdivide(region) => {
                *candidate = region;
                PruneDecision::Branch {
                    priority: -region.interval.wid().max(region.other.wid()),
                }
            }
        })
    }
}

struct NewtonEvaluator<F, D> {
    function: F,
    derivative: D,
}

impl<F, D> NewtonEvaluator<F, D>
where
    F: Evaluator<Interval, Value = Interval>,
    D: Evaluator<Interval, Value = Interval>,
{
    fn function_range(
        &mut self,
        domain: Interval,
    ) -> Result<Interval, RootIsolationError<F::Error, D::Error>> {
        let value = self
            .function
            .evaluate(&domain)
            .map_err(RootIsolationError::Function)?;
        if value.is_empty() {
            Err(RootIsolationError::EmptyFunctionEnclosure(domain))
        } else {
            Ok(value)
        }
    }

    fn classify_endpoints(
        &mut self,
        domain: Interval,
        midpoint: Interval,
        midpoint_value: Interval,
    ) -> Result<RootDecision, RootIsolationError<F::Error, D::Error>> {
        if !domain.is_bounded() {
            return Ok(RootDecision::Retain(domain));
        }
        let lower = Interval::from(domain.inf());
        let upper = Interval::from(domain.sup());
        let lower_value = if lower == midpoint {
            midpoint_value
        } else {
            self.function_range(lower)?
        };
        if lower_value == Interval::ZERO {
            return Ok(RootDecision::Unique(lower));
        }
        if lower == upper {
            return Ok(if lower_value.contains(0.0) {
                RootDecision::Retain(domain)
            } else {
                RootDecision::Exclude
            });
        }
        let upper_value = if upper == midpoint {
            midpoint_value
        } else {
            self.function_range(upper)?
        };
        if upper_value == Interval::ZERO {
            return Ok(RootDecision::Unique(upper));
        }
        // The caller has proved strict monotonicity. Endpoint signs then
        // distinguish an interior root from a region with none.
        Ok(if lower_value.contains(0.0) || upper_value.contains(0.0) {
            RootDecision::Retain(domain)
        } else if (lower_value.sup() < 0.0) == (upper_value.sup() < 0.0) {
            RootDecision::Exclude
        } else {
            RootDecision::Unique(domain)
        })
    }
}

impl<F, D> Evaluator<RootCandidate> for NewtonEvaluator<F, D>
where
    F: Evaluator<Interval, Value = Interval>,
    D: Evaluator<Interval, Value = Interval>,
{
    type Value = RootDecision;
    type Error = RootIsolationError<F::Error, D::Error>;

    #[allow(clippy::arithmetic_side_effects)]
    fn evaluate(&mut self, candidate: &RootCandidate) -> Result<Self::Value, Self::Error> {
        let domain = candidate.interval;
        if domain.is_empty() {
            return Ok(RootDecision::Exclude);
        }
        let value = self.function_range(domain)?;
        if !value.contains(0.0) {
            return Ok(RootDecision::Exclude);
        }
        if domain.is_singleton() || value == Interval::ZERO {
            return Ok(if domain.is_singleton() && value == Interval::ZERO {
                RootDecision::Unique(domain)
            } else {
                RootDecision::Retain(domain)
            });
        }

        let derivative = self
            .derivative
            .evaluate(&domain)
            .map_err(RootIsolationError::Derivative)?;
        if derivative.is_empty() {
            return Err(RootIsolationError::EmptyDerivativeEnclosure(domain));
        }
        let midpoint = Interval::from(domain.mid());
        let midpoint_value = self.function_range(midpoint)?;
        if derivative.contains(0.0) {
            if midpoint_value.contains(0.0) {
                // The mean-value equation may be 0 = 0 * (m - root).
                return Ok(RootDecision::Subdivide(RootCandidate::single(domain)));
            }
            let quotients = div_split(midpoint_value, derivative);
            let mut images = quotients
                .components()
                .iter()
                .rev()
                .map(|part| domain.intersection(midpoint - *part))
                .filter(|interval| !interval.is_empty());
            return Ok(match (images.next(), images.next()) {
                (None, _) => RootDecision::Exclude,
                (Some(interval), Some(other)) if interval.disjoint(other) => {
                    RootDecision::Subdivide(RootCandidate { interval, other })
                }
                (Some(interval), other) => {
                    let interval = other.map_or(interval, |other| interval.convex_hull(other));
                    RootDecision::Subdivide(RootCandidate::single(interval))
                }
            });
        }

        // The mean-value theorem puts every root in the Newton image.
        let image = midpoint - midpoint_value / derivative;
        let contracted = domain.intersection(image);
        if contracted.is_empty() {
            return Ok(RootDecision::Exclude);
        }
        // Interior inclusion proves existence, and the nonzero derivative
        // proves uniqueness. A singleton image can also certify an endpoint.
        if image.is_bounded()
            && (image.interior(domain) || (image.is_singleton() && image.subset(domain)))
        {
            return Ok(RootDecision::Unique(image));
        }
        if Bisect::bisect(&contracted).is_none() {
            return self.classify_endpoints(contracted, midpoint, midpoint_value);
        }
        Ok(RootDecision::Subdivide(RootCandidate::single(contracted)))
    }
}
