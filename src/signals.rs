#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
/// An IEEE 1788 exception condition reported by an interval operation.
pub enum Signal {
    /// The operation is undefined for the supplied input.
    UndefinedOperation = 1 << 0,
    /// An accuracy-relaxed text constructor may have produced a wider result.
    PossiblyUndefinedOperation = 1 << 1,
    /// The interval part of NaI was requested.
    IntvlPartOfNaI = 1 << 2,
    /// A binary interchange operand has an invalid representation.
    InvalidOperand = 1 << 3,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(transparent)]
/// An accumulating bit set of raised [`Signal`] values.
pub struct SignalFlags(u8);

impl SignalFlags {
    /// An empty signal set.
    pub const NONE: Self = Self(0);

    /// Returns the raw signal bits.
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Returns `true` when no signals have been raised.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns whether `signal` is present in this set.
    pub const fn contains(self, signal: Signal) -> bool {
        self.0 & signal as u8 != 0
    }

    /// Adds a signal to this set.
    pub fn insert(&mut self, signal: Signal) {
        self.0 |= signal as u8;
    }

    /// Removes all accumulated signals.
    pub fn clear(&mut self) {
        self.0 = 0;
    }

    /// Returns the accumulated signals and clears this set.
    pub fn take(&mut self) -> Self {
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
