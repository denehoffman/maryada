#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
/// An IEEE 1788 exception condition reported by an interval operation.
pub enum Signal {
    /// The operation is undefined for the supplied input.
    UndefinedOperation = 1 << 0,
    /// An accuracy-relaxed text constructor may have produced a wider result.
    PossiblyUndefinedOperation = 1 << 1,
    /// The interval part of `NaI` was requested.
    IntvlPartOfNaI = 1 << 2,
    /// A binary interchange operand has an invalid representation.
    InvalidOperand = 1 << 3,
}

const fn signal_bit(signal: Signal) -> u8 {
    match signal {
        Signal::UndefinedOperation => 1,
        Signal::PossiblyUndefinedOperation => 2,
        Signal::IntvlPartOfNaI => 4,
        Signal::InvalidOperand => 8,
    }
}

impl From<Signal> for u8 {
    fn from(value: Signal) -> Self {
        signal_bit(value)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(transparent)]
/// An accumulating bit set of raised [`Signal`] values.
pub struct SignalFlags(u8);

impl SignalFlags {
    /// An empty signal set.
    pub const NONE: Self = Self(0);

    /// Returns the raw signal bits.
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Returns `true` when no signals have been raised.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns whether `signal` is present in this set.
    #[must_use]
    pub const fn contains(self, signal: Signal) -> bool {
        self.0 & signal_bit(signal) != 0
    }

    /// Adds a signal to this set.
    pub const fn insert(&mut self, signal: Signal) {
        self.0 |= signal_bit(signal);
    }

    /// Removes all accumulated signals.
    pub const fn clear(&mut self) {
        self.0 = 0;
    }

    /// Returns the accumulated signals and clears this set.
    #[must_use]
    pub const fn take(&mut self) -> Self {
        let previous = *self;
        self.clear();
        previous
    }
}

/// Receives exception conditions raised by interval operations.
///
/// Use [`SignalFlags`] to accumulate conditions, or `()` to ignore them.
pub trait SignalSink {
    /// Reports one exception condition.
    fn raise(&mut self, signal: Signal);
}

impl SignalSink for SignalFlags {
    fn raise(&mut self, signal: Signal) {
        self.insert(signal);
    }
}

/// Pass `&mut ()` when exception flags are not needed.
impl SignalSink for () {
    fn raise(&mut self, _: Signal) {}
}
