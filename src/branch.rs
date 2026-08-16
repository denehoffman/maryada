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

use alloc::{collections::BinaryHeap, vec::Vec};
use core::{cmp::Ordering, fmt};

use crate::{Interval, IntervalOps};

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

    /// Returns whether this domain has zero branch width.
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

/// The result of a bounded run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RunStatus {
    /// The queue is empty; no further work is pending.
    Exhausted,
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
    /// Whether the search exhausted its queue or reached its step limit.
    pub status: RunStatus,
    /// Domains certified by the pruner.
    pub certified: Vec<D>,
    /// Terminal unresolved domains plus domains still in the queue.
    pub unresolved: Vec<Unresolved<D>>,
    /// Number of successfully evaluated domains.
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
    /// # Errors
    ///
    /// Returns an evaluator or pruner error without silently classifying the
    /// failed domain.
    pub fn step(&mut self) -> Result<StepOutcome, SearchError<E::Error, P::Error>> {
        let Some(entry) = self.queue.pop() else {
            return Ok(StepOutcome::Idle);
        };
        let mut domain = entry.domain;
        let value = self
            .evaluator
            .evaluate(&domain)
            .map_err(SearchError::Evaluation)?;
        self.evaluated = self.evaluated.saturating_add(1);
        let decision = self
            .pruner
            .prune(&mut domain, &value)
            .map_err(SearchError::Pruning)?;
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
            if self.queue.is_empty() {
                break;
            }
            let outcome = self.step()?;
            if outcome == StepOutcome::Idle {
                break;
            }
            steps = steps.saturating_add(1);
        }
        let status = if self.queue.is_empty() {
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
        self.queue.len()
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
            .field("pending", &self.queue.len())
            .field("certified", &self.certified.len())
            .field("unresolved", &self.unresolved.len())
            .field("evaluated", &self.evaluated)
            .finish_non_exhaustive()
    }
}

/// Configuration for [`GlobalMinimizer`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlobalMinimizerOptions {
    /// Stop subdividing a domain when its objective enclosure is no wider
    /// than this value.  Zero means only thin objective intervals stop.
    pub value_tolerance: f64,
    /// Stop subdividing a domain when its branch width is no wider than this
    /// value.  Zero means only singleton domains stop.
    pub domain_tolerance: f64,
}

impl Default for GlobalMinimizerOptions {
    fn default() -> Self {
        Self {
            value_tolerance: 0.0,
            domain_tolerance: 0.0,
        }
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
    /// A rigorous interval containing the best objective value found.
    pub minimum: Interval,
    /// Domains certified at the configured tolerances.
    pub candidates: Vec<D>,
    /// Terminal unresolved domains and their priorities.
    pub unresolved: Vec<Unresolved<D>>,
    /// Whether the search exhausted its queue or reached its step limit.
    pub status: RunStatus,
    /// Number of objective calls, including midpoint calls.
    pub evaluations: usize,
    /// Number of pruned domains.
    pub discarded: usize,
    /// Number of successful subdivisions.
    pub branched: usize,
}

/// An objective-driven interval global minimizer.
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
    evaluations: usize,
    discarded: usize,
    branched: usize,
}

impl<D, O> GlobalMinimizer<D, O, WidestBranch>
where
    D: Bisect + Midpoint + Clone,
    O: Evaluator<D, Value = Interval>,
{
    /// Creates a widest-first minimizer with default zero tolerances.
    #[must_use]
    pub fn new(root: D, objective: O) -> Self {
        Self::with_options(root, objective, GlobalMinimizerOptions::default())
    }

    /// Creates a widest-first minimizer with explicit tolerances.
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

    /// Creates a minimizer with custom subdivision and tolerance policies.
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
            evaluations: 0,
            discarded: 0,
            branched: 0,
        }
    }

    /// Processes one domain, including a midpoint objective call when
    /// available.
    ///
    /// # Errors
    ///
    /// Returns the objective error from either the box or midpoint call.
    pub fn step(&mut self) -> Result<GlobalStep, O::Error> {
        let Some(entry) = self.queue.pop() else {
            return Ok(GlobalStep::Idle);
        };
        let domain = entry.domain;
        let value = self.evaluate(&domain)?;
        if value.is_nai() {
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

        if !domain.is_singleton()
            && let Some(midpoint) = domain.midpoint()
        {
            let midpoint_value = self.evaluate(&midpoint)?;
            if !midpoint_value.is_empty() && !midpoint_value.is_nai() {
                self.update_bounds(midpoint_value);
            }
        }

        if value.inf() > self.upper {
            self.discarded = self.discarded.saturating_add(1);
            return Ok(GlobalStep::Discarded);
        }

        let value_small = value.wid() <= self.options.value_tolerance;
        let domain_small = domain.branch_width() <= self.options.domain_tolerance;
        if value_small || domain_small {
            self.candidates.push(domain);
            return Ok(GlobalStep::Certified);
        }

        if let Some((left, right)) = self.brancher.split(&domain) {
            let priority = value.inf();
            self.queue.push(left, priority);
            self.queue.push(right, priority);
            self.branched = self.branched.saturating_add(1);
            Ok(GlobalStep::Branched)
        } else {
            self.unresolved.push(Unresolved {
                domain,
                priority: value.inf(),
            });
            Ok(GlobalStep::Unsplittable)
        }
    }

    fn evaluate(&mut self, domain: &D) -> Result<Interval, O::Error> {
        self.evaluations = self.evaluations.saturating_add(1);
        self.objective.evaluate(domain)
    }

    fn update_bounds(&mut self, value: Interval) {
        if value.inf() < self.lower {
            self.lower = value.inf();
        }
        if value.sup() < self.upper {
            self.upper = value.sup();
        }
    }

    /// Runs at most `max_steps` domain iterations and returns a rigorous
    /// result.  Midpoint calls are included in the evaluation count but not
    /// in the step budget.
    ///
    /// # Errors
    ///
    /// Returns the objective error from the first failed objective call.
    pub fn run(&mut self, max_steps: usize) -> Result<GlobalMinimizationResult<D>, O::Error> {
        let mut steps = 0usize;
        while steps < max_steps {
            if self.queue.is_empty() {
                break;
            }
            let outcome = self.step()?;
            if outcome == GlobalStep::Idle {
                break;
            }
            steps = steps.saturating_add(1);
        }
        let status = if self.queue.is_empty() {
            RunStatus::Exhausted
        } else {
            RunStatus::StepLimit
        };
        Ok(self.result(status))
    }

    /// Returns a result snapshot without changing search state.
    #[must_use]
    pub fn result(&self, status: RunStatus) -> GlobalMinimizationResult<D> {
        let minimum = if self.lower == f64::INFINITY {
            Interval::EMPTY
        } else {
            Interval::new(self.lower, self.upper)
        };
        let mut unresolved = self.unresolved.clone();
        unresolved.extend(self.queue.snapshot().into_iter().map(|entry| Unresolved {
            domain: entry.domain,
            priority: entry.priority,
        }));
        GlobalMinimizationResult {
            minimum,
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
            Some((*self).bisect())
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

    fn bisect(&self) -> Option<(Self, Self)> {
        if self.is_nai() || self.is_empty() || self.is_singleton() {
            return None;
        }
        if self.re.wid() >= self.im.wid() {
            let (left, right) = Bisect::bisect(&self.re)?;
            Some((
                ComplexBox::new(left, self.im),
                ComplexBox::new(right, self.im),
            ))
        } else {
            let (left, right) = Bisect::bisect(&self.im)?;
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

    fn bisect(&self) -> Option<(Self, Self)> {
        let mut index = None;
        let mut widest = f64::NAN;
        for (candidate, value) in self.iter().enumerate() {
            let width = value.branch_width();
            if !width.is_nan() && (widest.is_nan() || width > widest) {
                index = Some(candidate);
                widest = width;
            }
        }
        let index = index?;
        let value = self.get(index)?;
        let (left, right) = value.bisect()?;
        let mut first = self.clone();
        let mut second = self.clone();
        if let Some(slot) = first.get_mut(index) {
            *slot = left;
        }
        if let Some(slot) = second.get_mut(index) {
            *slot = right;
        }
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
