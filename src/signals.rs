#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Signal {
    UndefinedOperation = 1 << 0,
    PossiblyUndefinedOperation = 1 << 1,
    IntvlPartOfNaI = 1 << 2,
    InvalidOperand = 1 << 3,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(transparent)]
pub struct SignalFlags(u8);

impl SignalFlags {
    pub const NONE: Self = Self(0);

    pub const fn bits(self) -> u8 {
        self.0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub const fn contains(self, signal: Signal) -> bool {
        self.0 & signal as u8 != 0
    }

    pub fn insert(&mut self, signal: Signal) {
        self.0 |= signal as u8;
    }

    pub fn clear(&mut self) {
        self.0 = 0;
    }

    pub fn take(&mut self) -> Self {
        let previous = *self;
        self.clear();
        previous
    }
}

pub trait SignalSink {
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
