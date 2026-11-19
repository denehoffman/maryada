//! Allocation-backed branch-and-bound search primitives.
//!
//! The search layer deliberately knows nothing about linear algebra or about
//! the function being enclosed.  An [`Evaluator`] maps a domain to an
//! enclosure, a [`Pruner`] can contract that domain and classify it, and a
//! [`BranchRule`] supplies the subdivision policy.  This separation makes it
//! possible to use the same scheduler for scalar, complex, and matrix-valued
//! interval calculations.
//!
//! The module is intended for `no_std` programs which enable the crate's
//! `alloc` feature.  The built-in queue uses [`alloc::collections::BinaryHeap`]
//! and is deterministic for equal priorities.

extern crate alloc;

use alloc::{
    collections::{BinaryHeap, binary_heap::PeekMut},
    vec::Vec,
};
use core::{cmp::Ordering, fmt};

use crate::{
    EnclosureOps, Interval, IntervalOps,
    rounding::{self, Direction},
};

#[cfg(feature = "complex")]
use crate::ComplexBox;

/// A domain that can be split into two covering subdomains.
///
/// `branch_width` is the component selected by the default widest-first
/// policy.  A `NaN` width is treated as non-branchable by the built-in
/// implementations.  Implementations should return `None` for empty,
/// singleton, or otherwise unsplittable domains.
pub trait Bisect: Sized {
    /// Returns a nonnegative measure used for widest-first subdivision.
    fn branch_width(&self) -> f64;

    /// Splits this domain into two children which cover the parent.
    fn bisect(&self) -> Option<(Self, Self)>;

    /// Returns whether this domain represents exactly one point.
    ///
    /// The default uses zero branch width. Override it when zero width can
    /// also describe a domain with no points or more than one point.
    fn is_singleton(&self) -> bool {
        self.branch_width() == 0.0
    }
}

/// Computes a midpoint domain when a concrete point evaluation is useful.
///
/// The global minimizer uses this capability to obtain an incumbent upper
/// bound from a midpoint before subdividing a box.  Custom domains may omit
/// this trait; the generic branch-and-bound engine does not require it.
pub trait Midpoint: Sized {
    /// Returns a singleton midpoint, or `None` when no finite midpoint exists.
    fn midpoint(&self) -> Option<Self>;
}

/// The default widest-component subdivision rule.
#[derive(Clone, Copy, Debug, Default)]
pub struct WidestBranch;

/// A user-supplied subdivision policy.
pub trait BranchRule<D> {
    /// Splits `domain`, returning two covering children when possible.
    fn split(&self, domain: &D) -> Option<(D, D)>;
}

impl<D: Bisect> BranchRule<D> for WidestBranch {
    fn split(&self, domain: &D) -> Option<(D, D)> {
        domain.bisect()
    }
}

/// An enclosure evaluator used by [`BranchAndBound`].
pub trait Evaluator<D> {
    /// The enclosure produced by this evaluator.
    type Value;
    /// An evaluation failure.
    type Error;

    /// Evaluates one domain.
    ///
    /// # Errors
    ///
    /// Returns the evaluator's domain-specific failure.
    fn evaluate(&mut self, domain: &D) -> Result<Self::Value, Self::Error>;
}

impl<D, V, E, F> Evaluator<D> for F
where
    F: FnMut(&D) -> Result<V, E>,
{
    type Value = V;
    type Error = E;

    fn evaluate(&mut self, domain: &D) -> Result<Self::Value, Self::Error> {
        self(domain)
    }
}

/// A pruning classification returned after evaluating a domain.
///
/// The pruner receives a mutable domain, so it may contract it before
/// returning a decision.  `Retain` records a terminal unresolved candidate;
/// `Branch` asks the search engine to apply its [`BranchRule`] to the
/// contracted domain.  The priority is lower-is-better and is used by the
/// best-first queue.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PruneDecision {
    /// Discard this domain because it cannot contribute to the result.
    Discard,
    /// Keep this domain as unresolved without subdividing it further.
    Retain {
        /// Queue-like priority to expose in reports.
        priority: f64,
    },
    /// Subdivide this domain and enqueue both children.
    Branch {
        /// Priority assigned to both children.
        priority: f64,
    },
    /// Certify this domain according to the enclosing algorithm.
    Certify,
}

/// A pruning policy for evaluated domains.
pub trait Pruner<D, V> {
    /// A pruning failure.
    type Error;

    /// Contracts and classifies an evaluated domain.
    ///
    /// # Errors
    ///
    /// Returns the pruner's domain-specific failure.
    fn prune(&mut self, domain: &mut D, value: &V) -> Result<PruneDecision, Self::Error>;
}

/// An item returned by a [`WorkQueue`].
#[derive(Clone, Debug, PartialEq)]
pub struct QueueEntry<D> {
    /// The queued domain.
    pub domain: D,
    /// Lower-is-better scheduling priority.
    pub priority: f64,
}

/// A deterministic priority queue for search domains.
pub trait WorkQueue<D> {
    /// Adds a domain with a lower-is-better priority.
    fn push(&mut self, domain: D, priority: f64);

    /// Removes the highest-priority domain, if any.
    fn pop(&mut self) -> Option<QueueEntry<D>>;

    /// Returns the number of pending domains.
    fn len(&self) -> usize;

    /// Returns whether no domains are pending.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a clone of the pending entries for reporting or inspection.
    fn snapshot(&self) -> Vec<QueueEntry<D>>
    where
        D: Clone;
}

struct HeapEntry<D> {
    domain: D,
    priority: f64,
    sequence: u64,
}

impl<D> PartialEq for HeapEntry<D> {
    fn eq(&self, other: &Self) -> bool {
        self.priority.to_bits() == other.priority.to_bits() && self.sequence == other.sequence
    }
}

impl<D> Eq for HeapEntry<D> {}

impl<D> PartialOrd for HeapEntry<D> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<D> Ord for HeapEntry<D> {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap.  Reversing priority gives lower values
        // first; reversing sequence makes earlier insertions win ties.
        other
            .priority
            .total_cmp(&self.priority)
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

/// The built-in deterministic allocation-backed best-first queue.
pub struct BestFirstQueue<D> {
    entries: BinaryHeap<HeapEntry<D>>,
    next_sequence: u64,
}

impl<D> Default for BestFirstQueue<D> {
    fn default() -> Self {
        Self {
            entries: BinaryHeap::new(),
            next_sequence: 0,
        }
    }
}

impl<D> BestFirstQueue<D> {
    /// Creates an empty queue.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Removes all entries from the queue.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

impl<D> WorkQueue<D> for BestFirstQueue<D> {
    fn push(&mut self, domain: D, priority: f64) {
        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.wrapping_add(1);
        self.entries.push(HeapEntry {
            domain,
            priority,
            sequence,
        });
    }

    fn pop(&mut self) -> Option<QueueEntry<D>> {
        self.entries.pop().map(|entry| QueueEntry {
            domain: entry.domain,
            priority: entry.priority,
        })
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn snapshot(&self) -> Vec<QueueEntry<D>>
    where
        D: Clone,
    {
        self.entries
            .iter()
            .map(|entry| QueueEntry {
                domain: entry.domain.clone(),
                priority: entry.priority,
            })
            .collect()
    }
}

/// A terminal unresolved domain and its scheduling priority.
#[derive(Clone, Debug, PartialEq)]
pub struct Unresolved<D> {
    /// The surviving domain.
    pub domain: D,
    /// Lower-is-better priority supplied by the pruner.
    pub priority: f64,
}

/// A one-node processing result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StepOutcome {
    /// No node was available.
    Idle,
    /// The node was discarded.
    Discarded,
    /// The node was retained as unresolved.
    Retained,
    /// The node was certified.
    Certified,
    /// The node was subdivided and its children were queued.
    Branched,
    /// The node could not be subdivided and was retained as unresolved.
    Unsplittable,
}

/// The stopping state of a search run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RunStatus {
    /// The queue is empty; no further work is pending.
    Exhausted,
    /// An enabled absolute or relative global optimality gap test passed.
    Converged,
    /// The step budget expired while work remained.
    StepLimit,
}

/// A typed evaluator or pruning failure.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, PartialEq)]
pub enum SearchError<E, P> {
    /// The evaluator failed.
    Evaluation(E),
    /// The pruner failed.
    Pruning(P),
}

impl<E: fmt::Display, P: fmt::Display> fmt::Display for SearchError<E, P> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Evaluation(error) => write!(formatter, "interval evaluation failed: {error}"),
            Self::Pruning(error) => write!(formatter, "interval pruning failed: {error}"),
        }
    }
}

/// A cumulative search report.  The engine remains usable after a report
/// with [`RunStatus::StepLimit`]; call [`BranchAndBound::run`] again to resume.
#[derive(Clone, Debug, PartialEq)]
pub struct RunReport<D> {
    /// Whether the search converged, exhausted its queue, or reached its step limit.
    pub status: RunStatus,
    /// Domains certified by the pruner.
    pub certified: Vec<D>,
    /// Terminal unresolved domains plus domains still in the queue.
    pub unresolved: Vec<Unresolved<D>>,
    /// Number of successful evaluator calls, including calls followed by a pruning error.
    pub evaluated: usize,
    /// Number of discarded domains.
    pub discarded: usize,
    /// Number of successful subdivisions.
    pub branched: usize,
    /// Number of terminal unresolved domains.
    pub retained: usize,
}

/// A decomposed branch-and-bound search engine.
pub struct BranchAndBound<D, E, P, B = WidestBranch, Q = BestFirstQueue<D>>
where
    E: Evaluator<D>,
    P: Pruner<D, E::Value>,
    B: BranchRule<D>,
    Q: WorkQueue<D>,
{
    evaluator: E,
    pruner: P,
    brancher: B,
    queue: Q,
    retry: Option<QueueEntry<D>>,
    certified: Vec<D>,
    unresolved: Vec<Unresolved<D>>,
    evaluated: usize,
    discarded: usize,
    branched: usize,
    retained: usize,
}

impl<D, E, P> BranchAndBound<D, E, P, WidestBranch, BestFirstQueue<D>>
where
    E: Evaluator<D>,
    P: Pruner<D, E::Value>,
    D: Bisect,
{
    /// Creates a widest-first search with a built-in queue.
    #[must_use]
    pub fn new(root: D, evaluator: E, pruner: P) -> Self {
        Self::with_branch(root, evaluator, pruner, WidestBranch)
    }
}

impl<D, E, P, B, Q> BranchAndBound<D, E, P, B, Q>
where
    E: Evaluator<D>,
    P: Pruner<D, E::Value>,
    B: BranchRule<D>,
    Q: WorkQueue<D>,
{
    /// Creates a search with custom branching and queue policies.
    #[must_use]
    pub fn with_branch_and_queue(
        root: D,
        evaluator: E,
        pruner: P,
        brancher: B,
        mut queue: Q,
    ) -> Self {
        queue.push(root, 0.0);
        Self {
            evaluator,
            pruner,
            brancher,
            queue,
            retry: None,
            certified: Vec::new(),
            unresolved: Vec::new(),
            evaluated: 0,
            discarded: 0,
            branched: 0,
            retained: 0,
        }
    }

    /// Processes exactly one queued domain.
    ///
    /// The domain is cloned before pruning so a failed contraction can be retried.
    ///
    /// # Errors
    ///
    /// Returns an evaluator or pruner error and keeps the original domain and
    /// priority for the next step.
    pub fn step(&mut self) -> Result<StepOutcome, SearchError<E::Error, P::Error>>
    where
        D: Clone,
    {
        let Some(mut entry) = self.retry.take().or_else(|| self.queue.pop()) else {
            return Ok(StepOutcome::Idle);
        };
        let value = match self.evaluator.evaluate(&entry.domain) {
            Ok(value) => value,
            Err(error) => {
                self.retry = Some(entry);
                return Err(SearchError::Evaluation(error));
            }
        };
        self.evaluated = self.evaluated.saturating_add(1);
        let original = entry.clone();
        let decision = match self.pruner.prune(&mut entry.domain, &value) {
            Ok(decision) => decision,
            Err(error) => {
                self.retry = Some(original);
                return Err(SearchError::Pruning(error));
            }
        };
        let domain = entry.domain;
        match decision {
            PruneDecision::Discard => {
                self.discarded = self.discarded.saturating_add(1);
                Ok(StepOutcome::Discarded)
            }
            PruneDecision::Retain { priority } => {
                self.unresolved.push(Unresolved { domain, priority });
                self.retained = self.retained.saturating_add(1);
                Ok(StepOutcome::Retained)
            }
            PruneDecision::Certify => {
                self.certified.push(domain);
                Ok(StepOutcome::Certified)
            }
            PruneDecision::Branch { priority } => {
                if let Some((left, right)) = self.brancher.split(&domain) {
                    self.queue.push(left, priority);
                    self.queue.push(right, priority);
                    self.branched = self.branched.saturating_add(1);
                    Ok(StepOutcome::Branched)
                } else {
                    self.unresolved.push(Unresolved { domain, priority });
                    self.retained = self.retained.saturating_add(1);
                    Ok(StepOutcome::Unsplittable)
                }
            }
        }
    }

    /// Runs at most `max_steps`, preserving queued work for a later call.
    ///
    /// # Errors
    ///
    /// Returns an evaluator or pruner error from the first failed step.
    pub fn run(&mut self, max_steps: usize) -> Result<RunReport<D>, SearchError<E::Error, P::Error>>
    where
        D: Clone,
    {
        let mut steps = 0usize;
        while steps < max_steps {
            if self.pending_count() == 0 {
                break;
            }
            let outcome = self.step()?;
            if outcome == StepOutcome::Idle {
                break;
            }
            steps = steps.saturating_add(1);
        }
        let status = if self.pending_count() == 0 {
            RunStatus::Exhausted
        } else {
            RunStatus::StepLimit
        };
        Ok(self.report(status))
    }

    /// Returns the current report without changing search state.
    #[must_use]
    pub fn report(&self, status: RunStatus) -> RunReport<D>
    where
        D: Clone,
    {
        let mut unresolved = self.unresolved.clone();
        if let Some(entry) = &self.retry {
            unresolved.push(Unresolved {
                domain: entry.domain.clone(),
                priority: entry.priority,
            });
        }
        unresolved.extend(self.queue.snapshot().into_iter().map(|entry| Unresolved {
            domain: entry.domain,
            priority: entry.priority,
        }));
        RunReport {
            status,
            certified: self.certified.clone(),
            unresolved,
            evaluated: self.evaluated,
            discarded: self.discarded,
            branched: self.branched,
            retained: self.retained,
        }
    }

    /// Returns the number of domains still scheduled for evaluation.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.queue
            .len()
            .saturating_add(usize::from(self.retry.is_some()))
    }

    /// Returns the number of terminal unresolved domains.
    #[must_use]
    pub const fn retained_count(&self) -> usize {
        self.unresolved.len()
    }

    /// Returns a clone of all domains that can still be resumed or inspected.
    #[must_use]
    pub fn pending_domains(&self) -> Vec<Unresolved<D>>
    where
        D: Clone,
    {
        self.report(RunStatus::StepLimit).unresolved
    }
}

impl<D, E, P, B, Q> BranchAndBound<D, E, P, B, Q>
where
    E: Evaluator<D>,
    P: Pruner<D, E::Value>,
    B: BranchRule<D>,
    Q: WorkQueue<D> + Default,
{
    /// Creates a search with a custom branching policy and a default queue.
    #[must_use]
    pub fn with_branch(root: D, evaluator: E, pruner: P, brancher: B) -> Self {
        Self::with_branch_and_queue(root, evaluator, pruner, brancher, Q::default())
    }
}

impl<D, E, P, B, Q> fmt::Debug for BranchAndBound<D, E, P, B, Q>
where
    E: Evaluator<D>,
    P: Pruner<D, E::Value>,
    B: BranchRule<D>,
    Q: WorkQueue<D>,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BranchAndBound")
            .field("pending", &self.pending_count())
            .field("certified", &self.certified.len())
            .field("unresolved", &self.unresolved.len())
            .field("evaluated", &self.evaluated)
            .finish_non_exhaustive()
    }
}

/// An invalid [`GlobalMinimizer`] stopping configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GlobalMinimizerOptionsError {
    /// A configured tolerance was not finite and strictly positive.
    InvalidTolerance {
        /// The name of the invalid option.
        name: &'static str,
        /// The invalid tolerance value.
        value: f64,
    },
    /// No tolerance was enabled and no step limit was configured.
    NoTerminationCriterion,
}

impl fmt::Display for GlobalMinimizerOptionsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTolerance { name, value } => write!(
                formatter,
                "{name} must be finite and strictly positive, got {value}"
            ),
            Self::NoTerminationCriterion => write!(
                formatter,
                "at least one stopping tolerance or max_steps must be configured"
            ),
        }
    }
}

/// A global-minimizer evaluation or configuration failure.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, PartialEq)]
pub enum GlobalMinimizerError<E> {
    /// The objective evaluator failed.
    Evaluation(E),
    /// The minimizer options were invalid.
    InvalidOptions(GlobalMinimizerOptionsError),
}

impl<E: fmt::Display> fmt::Display for GlobalMinimizerError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Evaluation(error) => write!(formatter, "interval evaluation failed: {error}"),
            Self::InvalidOptions(error) => write!(formatter, "invalid minimizer options: {error}"),
        }
    }
}

/// Configuration for [`GlobalMinimizer`].
///
/// The absolute and relative gap tolerances both default to `1e-6`. Either
/// enabled test can stop the search. Both tests require a finite lower bound
/// `L` and upper bound `U` on the global minimum.
///
/// # Examples
///
/// Set the absolute gap tolerance and disable the relative gap test using
/// either the options or the builder methods.
///
/// ```
/// use maryada::{GlobalMinimizer, GlobalMinimizerOptions, Interval, IntervalOps};
///
/// let domain = Interval::new(0.0, 1.0);
/// let objective = |x: &Interval| Ok::<_, core::convert::Infallible>(*x);
/// let options = GlobalMinimizerOptions {
///     absolute_gap_tolerance: Some(1e-8),
///     relative_gap_tolerance: None,
///     ..GlobalMinimizerOptions::default()
/// };
/// let configured = GlobalMinimizer::with_options(domain, objective, options);
/// let built = GlobalMinimizer::new(domain, objective)
///     .with_absolute_gap_tolerance(1e-8)
///     .without_relative_gap_tolerance();
/// assert_eq!(configured.options(), built.options());
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlobalMinimizerOptions {
    /// Stop subdividing a domain when its objective enclosure is no wider
    /// than this value.  `None` disables this box-resolution criterion.
    pub value_tolerance: Option<f64>,
    /// Stop subdividing a domain when its branch width is no wider than this
    /// value.  `None` disables this parameter-resolution criterion.
    pub domain_tolerance: Option<f64>,
    /// Stop when `U - L <= tolerance` is certified. The absolute gap tolerance
    /// must be finite and strictly positive. `None` disables the absolute gap test.
    pub absolute_gap_tolerance: Option<f64>,
    /// Stop when `U - L <= tolerance * max(|L|, |U|)` is certified. The relative
    /// gap tolerance must be finite and strictly positive. An exact zero
    /// enclosure satisfies the test. `None` disables the relative gap test.
    pub relative_gap_tolerance: Option<f64>,
    /// Maximum number of domain steps used by [`GlobalMinimizer::solve`].
    /// `None` allows `solve` to continue until another stopping criterion is
    /// reached.
    pub max_steps: Option<usize>,
}

impl Default for GlobalMinimizerOptions {
    fn default() -> Self {
        Self {
            value_tolerance: Some(1.0e-6),
            domain_tolerance: Some(1.0e-6),
            absolute_gap_tolerance: Some(1.0e-6),
            relative_gap_tolerance: Some(1.0e-6),
            max_steps: None,
        }
    }
}

impl GlobalMinimizerOptions {
    fn validate(&self) -> Result<(), GlobalMinimizerOptionsError> {
        for (name, tolerance) in [
            ("value_tolerance", self.value_tolerance),
            ("domain_tolerance", self.domain_tolerance),
            ("absolute_gap_tolerance", self.absolute_gap_tolerance),
            ("relative_gap_tolerance", self.relative_gap_tolerance),
        ] {
            if let Some(value) = tolerance
                && (!value.is_finite() || value <= 0.0)
            {
                return Err(GlobalMinimizerOptionsError::InvalidTolerance { name, value });
            }
        }
        if self.max_steps.is_none()
            && self.value_tolerance.is_none()
            && self.domain_tolerance.is_none()
            && self.absolute_gap_tolerance.is_none()
            && self.relative_gap_tolerance.is_none()
        {
            return Err(GlobalMinimizerOptionsError::NoTerminationCriterion);
        }
        Ok(())
    }
}

/// One step of a global-minimization search.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GlobalStep {
    /// No domain was pending.
    Idle,
    /// The objective enclosure was dominated by the incumbent.
    Discarded,
    /// The domain was subdivided.
    Branched,
    /// The domain met a stopping tolerance.
    Certified,
    /// The domain was not splittable at the current precision.
    Unsplittable,
}

/// A Moore--Skelboe-style global minimization result.
#[derive(Clone, Debug, PartialEq)]
pub struct GlobalMinimizationResult<D> {
    /// A rigorous interval containing the global minimum value.
    ///
    /// The lower endpoint is the smallest lower bound among the queued,
    /// unresolved, and certified domains, while the upper endpoint is the best
    /// upper bound obtained from an evaluated domain or midpoint.  This
    /// interval is therefore a bound on the optimum value, not necessarily the
    /// image of [`Self::best_domain`].
    ///
    /// In particular, `minimum` and `best_value` answer different questions:
    /// the former bounds the global optimum, while the latter encloses the
    /// objective over the parameter box that supplied the current upper bound.
    pub minimum: Interval,
    /// The argument whose objective evaluation gave the best upper bound.
    ///
    /// This can be a singleton midpoint or the full [`Self::best_domain`].
    /// Use [`GlobalMinimizer::best_sample`] to retrieve an evaluated singleton
    /// and its own objective enclosure.
    pub best_point: Option<D>,
    /// The parameter domain associated with the best known upper bound.
    ///
    /// When a midpoint produced the upper bound, this field is the box that
    /// contained that midpoint. It identifies where the bound was established.
    pub best_domain: Option<D>,
    /// The objective enclosure evaluated over [`Self::best_domain`].
    pub best_value: Option<Interval>,
    /// Domains certified at the configured tolerances.
    pub candidates: Vec<D>,
    /// Terminal unresolved domains and their priorities.
    pub unresolved: Vec<Unresolved<D>>,
    /// Whether the search converged, exhausted its queue, or reached its step limit.
    pub status: RunStatus,
    /// Number of objective calls, including midpoint calls and failed attempts.
    pub evaluations: usize,
    /// Number of pruned domains.
    pub discarded: usize,
    /// Number of successful subdivisions.
    pub branched: usize,
}

/// A summary of a minimization run without copying its domains.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlobalMinimizationProgress {
    /// A rigorous enclosure of the global minimum, as in [`GlobalMinimizationResult::minimum`].
    pub minimum: Interval,
    /// Whether the search converged, exhausted its queue, or reached its step limit.
    pub status: RunStatus,
    /// Number of domains queued for evaluation.
    pub pending: usize,
    /// Number of domains certified at the configured tolerances.
    pub candidates: usize,
    /// Number of terminal unresolved domains.
    pub unresolved: usize,
    /// Number of objective calls, including midpoint calls and failed attempts.
    pub evaluations: usize,
    /// Number of pruned domains.
    pub discarded: usize,
    /// Number of successful subdivisions.
    pub branched: usize,
}

/// An objective-driven interval global minimizer.
///
/// The default configuration enables both the absolute and relative gap tests.
/// Use [`Self::without_absolute_gap_tolerance`] or
/// [`Self::without_relative_gap_tolerance`] to disable either test.
pub struct GlobalMinimizer<D, O, B = WidestBranch>
where
    D: Bisect + Midpoint + Clone,
    O: Evaluator<D, Value = Interval>,
    B: BranchRule<D>,
{
    objective: O,
    brancher: B,
    queue: BestFirstQueue<D>,
    candidates: Vec<D>,
    unresolved: Vec<Unresolved<D>>,
    options: GlobalMinimizerOptions,
    lower: f64,
    upper: f64,
    terminal_lower_bound: f64,
    best_point: Option<D>,
    best_domain: Option<D>,
    best_value: Option<Interval>,
    best_sample: Option<(D, Interval)>,
    evaluations: usize,
    discarded: usize,
    branched: usize,
}

impl<D, O> GlobalMinimizer<D, O, WidestBranch>
where
    D: Bisect + Midpoint + Clone,
    O: Evaluator<D, Value = Interval>,
{
    /// Creates a widest-first minimizer with the default stopping policy.
    ///
    /// The default uses finite value, parameter-width, absolute gap, and
    /// relative gap tolerances and has no step limit. Call [`Self::solve`]
    /// to run until convergence or queue exhaustion.
    #[must_use]
    pub fn new(root: D, objective: O) -> Self {
        Self::with_options(root, objective, GlobalMinimizerOptions::default())
    }

    /// Creates a widest-first minimizer with an explicit stopping policy.
    #[must_use]
    pub fn with_options(root: D, objective: O, options: GlobalMinimizerOptions) -> Self {
        Self::with_branch_and_options(root, objective, WidestBranch, options)
    }
}

impl<D, O, B> GlobalMinimizer<D, O, B>
where
    D: Bisect + Midpoint + Clone,
    O: Evaluator<D, Value = Interval>,
    B: BranchRule<D>,
{
    /// Creates a minimizer with a custom subdivision policy.
    #[must_use]
    pub fn with_branch(root: D, objective: O, brancher: B) -> Self {
        Self::with_branch_and_options(root, objective, brancher, GlobalMinimizerOptions::default())
    }

    /// Creates a minimizer with custom subdivision and stopping policies.
    #[must_use]
    pub fn with_branch_and_options(
        root: D,
        objective: O,
        brancher: B,
        options: GlobalMinimizerOptions,
    ) -> Self {
        let mut queue = BestFirstQueue::new();
        queue.push(root, f64::NEG_INFINITY);
        Self {
            objective,
            brancher,
            queue,
            candidates: Vec::new(),
            unresolved: Vec::new(),
            options,
            lower: f64::INFINITY,
            upper: f64::INFINITY,
            terminal_lower_bound: f64::INFINITY,
            best_point: None,
            best_domain: None,
            best_value: None,
            best_sample: None,
            evaluations: 0,
            discarded: 0,
            branched: 0,
        }
    }

    /// Returns the current stopping configuration.
    #[must_use]
    pub const fn options(&self) -> GlobalMinimizerOptions {
        self.options
    }

    /// Sets the maximum width of a box objective enclosure used for local
    /// resolution and returns the minimizer for builder-style configuration.
    #[must_use]
    pub const fn with_value_tolerance(mut self, tolerance: f64) -> Self {
        self.options.value_tolerance = Some(tolerance);
        self
    }

    /// Disables the box objective-width stopping criterion.
    #[must_use]
    pub const fn without_value_tolerance(mut self) -> Self {
        self.options.value_tolerance = None;
        self
    }

    /// Sets the maximum parameter-box width used for local resolution and
    /// returns the minimizer for builder-style configuration.
    #[must_use]
    pub const fn with_domain_tolerance(mut self, tolerance: f64) -> Self {
        self.options.domain_tolerance = Some(tolerance);
        self
    }

    /// Disables the parameter-width stopping criterion.
    #[must_use]
    pub const fn without_domain_tolerance(mut self) -> Self {
        self.options.domain_tolerance = None;
        self
    }

    /// Sets [`GlobalMinimizerOptions::absolute_gap_tolerance`].
    #[must_use]
    pub const fn with_absolute_gap_tolerance(mut self, tolerance: f64) -> Self {
        self.options.absolute_gap_tolerance = Some(tolerance);
        self
    }

    /// Disables the absolute gap test.
    #[must_use]
    pub const fn without_absolute_gap_tolerance(mut self) -> Self {
        self.options.absolute_gap_tolerance = None;
        self
    }

    /// Sets [`GlobalMinimizerOptions::relative_gap_tolerance`].
    #[must_use]
    pub const fn with_relative_gap_tolerance(mut self, tolerance: f64) -> Self {
        self.options.relative_gap_tolerance = Some(tolerance);
        self
    }

    /// Disables the relative gap test.
    #[must_use]
    pub const fn without_relative_gap_tolerance(mut self) -> Self {
        self.options.relative_gap_tolerance = None;
        self
    }

    /// Sets the step limit used by [`Self::solve`].
    #[must_use]
    pub const fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.options.max_steps = Some(max_steps);
        self
    }

    /// Removes the step limit used by [`Self::solve`].
    #[must_use]
    pub const fn without_max_steps(mut self) -> Self {
        self.options.max_steps = None;
        self
    }

    /// Processes one domain, including a midpoint objective call when
    /// available.
    ///
    /// # Errors
    ///
    /// Returns the objective error from either call. The domain stays queued
    /// and the incumbent, bounds, and sample are updated only after all calls succeed.
    pub fn step(&mut self) -> Result<GlobalStep, O::Error> {
        let Some(entry) = self.queue.entries.peek_mut() else {
            return Ok(GlobalStep::Idle);
        };
        let mut evaluate = |domain: &D| {
            self.evaluations = self.evaluations.saturating_add(1);
            self.objective.evaluate(domain)
        };
        let value = evaluate(&entry.domain)?;
        let valid_value = !value.is_nai() && !value.is_empty();
        let singleton = valid_value && entry.domain.is_singleton();
        let midpoint = if valid_value
            && !singleton
            && let Some(point) = entry.domain.midpoint()
        {
            let point_value = evaluate(&point)?;
            Some((point, point_value))
        } else {
            None
        };

        let domain = PeekMut::pop(entry).domain;
        if value.is_nai() {
            self.terminal_lower_bound = f64::NEG_INFINITY;
            self.unresolved.push(Unresolved {
                domain,
                priority: f64::NEG_INFINITY,
            });
            return Ok(GlobalStep::Unsplittable);
        }
        if value.is_empty() {
            self.discarded = self.discarded.saturating_add(1);
            return Ok(GlobalStep::Discarded);
        }
        self.update_bounds(value);
        self.update_upper_bound(&domain, value, &domain, value);
        if singleton {
            self.update_sample(value, || domain.clone());
        }

        if let Some((point, point_value)) = midpoint
            && !point_value.is_empty()
            && !point_value.is_nai()
        {
            self.update_bounds(point_value);
            self.update_upper_bound(&domain, value, &point, point_value);
            self.update_sample(point_value, || point);
        }

        let priority = value.inf();
        if priority > self.upper {
            self.discarded = self.discarded.saturating_add(1);
            return Ok(GlobalStep::Discarded);
        }

        let value_small = self
            .options
            .value_tolerance
            .is_some_and(|tolerance| value.wid() <= tolerance);
        let domain_small = self
            .options
            .domain_tolerance
            .is_some_and(|tolerance| domain.branch_width() <= tolerance);
        if value_small || domain_small {
            self.terminal_lower_bound = self.terminal_lower_bound.min(priority);
            self.candidates.push(domain);
            return Ok(GlobalStep::Certified);
        }

        if let Some((left, right)) = self.brancher.split(&domain) {
            self.queue.push(left, priority);
            self.queue.push(right, priority);
            self.branched = self.branched.saturating_add(1);
            Ok(GlobalStep::Branched)
        } else {
            self.terminal_lower_bound = self.terminal_lower_bound.min(priority);
            self.unresolved.push(Unresolved { domain, priority });
            Ok(GlobalStep::Unsplittable)
        }
    }

    fn update_bounds(&mut self, value: Interval) {
        if value.inf() < self.lower {
            self.lower = value.inf();
        }
    }

    fn update_upper_bound(
        &mut self,
        domain: &D,
        domain_value: Interval,
        point: &D,
        bound: Interval,
    ) {
        if bound.sup() < self.upper {
            self.upper = bound.sup();
            self.best_point = Some(point.clone());
            self.best_domain = Some(domain.clone());
            self.best_value = Some(domain_value);
        }
    }

    fn update_sample(&mut self, value: Interval, point: impl FnOnce() -> D) {
        if self
            .best_sample
            .as_ref()
            .is_none_or(|(_, best)| value.sup() < best.sup())
        {
            self.best_sample = Some((point(), value));
        }
    }

    fn frontier_lower_bound(&self) -> f64 {
        // A queued priority comes from a valid enclosure or is -infinity for the root.
        let queued_lower = self
            .queue
            .entries
            .peek()
            .map_or(f64::INFINITY, |entry| entry.priority);
        self.terminal_lower_bound.min(queued_lower)
    }

    fn gap_reached(&self) -> bool {
        let absolute = self.options.absolute_gap_tolerance;
        let relative = self.options.relative_gap_tolerance;
        if (absolute.is_none() && relative.is_none()) || !self.upper.is_finite() {
            return false;
        }
        let lower = self.frontier_lower_bound();
        if !lower.is_finite() {
            return false;
        }
        let gap = rounding::sub(self.upper, lower, Direction::Up);
        if absolute.is_some_and(|tolerance| gap <= tolerance) {
            return true;
        }
        let Some(tolerance) = relative else {
            return false;
        };

        let scale = self.upper.abs().max(lower.abs());
        if gap.is_finite() {
            // Compare an upper bound on the gap with a lower bound on the threshold.
            gap <= rounding::mul(tolerance, scale, Direction::Down)
        } else {
            // The unscaled subtraction overflowed. Scale the two bounds first
            // and round them outward before subtracting them.
            let upper = rounding::div(self.upper, scale, Direction::Up);
            let lower = rounding::div(lower, scale, Direction::Down);
            rounding::sub(upper, lower, Direction::Up) <= tolerance
        }
    }

    fn advance_with_limit(
        &mut self,
        max_steps: Option<usize>,
    ) -> Result<RunStatus, GlobalMinimizerError<O::Error>> {
        GlobalMinimizerOptions {
            max_steps,
            ..self.options
        }
        .validate()
        .map_err(GlobalMinimizerError::InvalidOptions)?;

        let mut steps = 0usize;
        let status = loop {
            if self.queue.is_empty() {
                break RunStatus::Exhausted;
            }
            if self.gap_reached() {
                break RunStatus::Converged;
            }
            if max_steps.is_some_and(|limit| steps >= limit) {
                break RunStatus::StepLimit;
            }
            let outcome = self.step().map_err(GlobalMinimizerError::Evaluation)?;
            if outcome == GlobalStep::Idle {
                break RunStatus::Exhausted;
            }
            steps = steps.saturating_add(1);
        };
        Ok(status)
    }

    /// Solves using the stopping policy configured on this minimizer.
    ///
    /// With the default configuration, this runs without a step limit until
    /// the absolute or relative gap test passes or the queue is exhausted.
    /// Configure a finite limit with
    /// [`Self::with_max_steps`] when an anytime result is preferred.
    ///
    /// # Errors
    ///
    /// Returns an invalid-options error before doing work, or the objective
    /// error from the first failed objective call.
    pub fn solve(&mut self) -> Result<GlobalMinimizationResult<D>, GlobalMinimizerError<O::Error>> {
        let status = self.advance_with_limit(self.options.max_steps)?;
        Ok(self.result(status))
    }

    /// Performs at most `max_steps` domain iterations and returns a rigorous
    /// result.  The minimizer retains its queue, so a later call resumes the
    /// same search.  Midpoint calls are included in the evaluation count but
    /// not in the step budget.
    ///
    /// This explicit budget is independent of the optional limit configured
    /// for [`Self::solve`].
    ///
    /// # Errors
    ///
    /// Returns an invalid-options error before doing work, or the objective
    /// error from the first failed objective call.
    pub fn advance(
        &mut self,
        max_steps: usize,
    ) -> Result<GlobalMinimizationResult<D>, GlobalMinimizerError<O::Error>> {
        let status = self.advance_with_limit(Some(max_steps))?;
        Ok(self.result(status))
    }

    /// Performs at most `max_steps` domain iterations and summarizes the run
    /// without cloning the domains for a result snapshot. A zero budget reports
    /// the current enclosure, counts, and stopping status without evaluating a domain.
    /// Use [`Self::result`] when the full domain snapshot is needed.
    /// # Errors
    /// Returns an invalid-options error before doing work, or the objective
    /// error from the first failed objective call.
    pub fn advance_progress(
        &mut self,
        max_steps: usize,
    ) -> Result<GlobalMinimizationProgress, GlobalMinimizerError<O::Error>> {
        let status = self.advance_with_limit(Some(max_steps))?;
        Ok(GlobalMinimizationProgress {
            minimum: self.minimum(),
            status,
            pending: self.pending_count(),
            candidates: self.candidates.len(),
            unresolved: self.unresolved.len(),
            evaluations: self.evaluations,
            discarded: self.discarded,
            branched: self.branched,
        })
    }

    /// Returns the best sampled point and its objective enclosure.
    ///
    /// A sample comes from evaluating a singleton domain or a midpoint with a
    /// nonempty objective enclosure. The sample with the smallest upper endpoint
    /// wins, with ties keeping the earliest sample. Its enclosure bounds the
    /// objective at that point. The global minimum enclosure can use a tighter
    /// upper bound from a whole box. The returned point borrows from the
    /// minimizer, so this can be used after [`Self::advance_progress`].
    #[must_use]
    pub fn best_sample(&self) -> Option<(&D, Interval)> {
        self.best_sample
            .as_ref()
            .map(|(point, value)| (point, *value))
    }

    fn minimum(&self) -> Interval {
        if self.lower == f64::INFINITY {
            return Interval::EMPTY;
        }
        let frontier_lower = self.frontier_lower_bound();
        let lower = if frontier_lower == f64::INFINITY {
            self.lower
        } else {
            frontier_lower
        };
        Interval::new(lower, self.upper)
    }

    /// Returns a result snapshot without changing search state.
    #[must_use]
    pub fn result(&self, status: RunStatus) -> GlobalMinimizationResult<D> {
        let mut unresolved = self.unresolved.clone();
        unresolved.extend(self.queue.snapshot().into_iter().map(|entry| Unresolved {
            domain: entry.domain,
            priority: entry.priority,
        }));
        GlobalMinimizationResult {
            minimum: self.minimum(),
            best_point: self.best_point.clone(),
            best_domain: self.best_domain.clone(),
            best_value: self.best_value,
            candidates: self.candidates.clone(),
            unresolved,
            status,
            evaluations: self.evaluations,
            discarded: self.discarded,
            branched: self.branched,
        }
    }

    /// Returns the number of domains still scheduled for evaluation.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.queue.len()
    }
}

// A bare or decorated interval has exactly one scalar branch component.
impl<I: IntervalOps> Bisect for I {
    fn branch_width(&self) -> f64 {
        (*self).wid()
    }

    fn bisect(&self) -> Option<(Self, Self)> {
        if (*self).is_nai() || (*self).is_empty() || (*self).is_singleton() {
            None
        } else {
            let midpoint = (*self).mid();
            if midpoint == (*self).inf() || midpoint == (*self).sup() {
                None
            } else {
                Some((*self).bisect())
            }
        }
    }
}

impl<I: IntervalOps> Midpoint for I {
    fn midpoint(&self) -> Option<Self> {
        if (*self).is_nai() || (*self).is_empty() || (*self).mid().is_nan() {
            None
        } else {
            Some(I::singleton((*self).mid()))
        }
    }
}

fn widest_bisection<'a, D: Bisect + 'a>(
    coordinates: impl IntoIterator<Item = &'a D>,
) -> Option<(usize, (D, D))> {
    let mut selected = None;
    let mut widest = f64::NAN;
    for (index, coordinate) in coordinates.into_iter().enumerate() {
        let width = coordinate.branch_width();
        if !width.is_nan()
            && (widest.is_nan() || width > widest)
            && let Some(split) = coordinate.bisect()
        {
            selected = Some((index, split));
            widest = width;
        }
    }
    selected
}

#[cfg(feature = "complex")]
#[allow(clippy::use_self)]
impl<I: IntervalOps> Bisect for ComplexBox<I> {
    fn branch_width(&self) -> f64 {
        let re = self.re.wid();
        let im = self.im.wid();
        if re.is_nan() {
            im
        } else if im.is_nan() || re >= im {
            re
        } else {
            im
        }
    }

    fn is_singleton(&self) -> bool {
        EnclosureOps::is_singleton(self.re) && EnclosureOps::is_singleton(self.im)
    }

    fn bisect(&self) -> Option<(Self, Self)> {
        if self.is_nai() || self.is_empty() || self.is_singleton() {
            return None;
        }
        let (index, (left, right)) = widest_bisection([&self.re, &self.im])?;
        if index == 0 {
            Some((
                ComplexBox::new(left, self.im),
                ComplexBox::new(right, self.im),
            ))
        } else {
            Some((
                ComplexBox::new(self.re, left),
                ComplexBox::new(self.re, right),
            ))
        }
    }
}

#[cfg(feature = "complex")]
#[allow(clippy::use_self)]
impl<I: IntervalOps> Midpoint for ComplexBox<I> {
    fn midpoint(&self) -> Option<Self> {
        if self.is_nai() || self.is_empty() {
            None
        } else {
            Some(ComplexBox::new(
                I::singleton(self.re.mid()),
                I::singleton(self.im.mid()),
            ))
        }
    }
}

impl<T> Bisect for Vec<T>
where
    T: Bisect + Clone,
{
    fn branch_width(&self) -> f64 {
        self.iter()
            .map(Bisect::branch_width)
            .filter(|width| !width.is_nan())
            .fold(f64::NAN, |best, width| {
                if best.is_nan() || width > best {
                    width
                } else {
                    best
                }
            })
    }

    fn is_singleton(&self) -> bool {
        !self.is_empty() && self.iter().all(Bisect::is_singleton)
    }

    fn bisect(&self) -> Option<(Self, Self)> {
        let (index, (left, right)) = widest_bisection(self)?;
        let mut first = self.clone();
        let mut second = self.clone();
        *first.get_mut(index)? = left;
        *second.get_mut(index)? = right;
        Some((first, second))
    }
}

impl<T> Midpoint for Vec<T>
where
    T: Midpoint + Clone,
{
    fn midpoint(&self) -> Option<Self> {
        let mut midpoint = Self::with_capacity(self.len());
        for value in self {
            midpoint.push(value.midpoint()?);
        }
        Some(midpoint)
    }
}
