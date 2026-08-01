#![no_std]

mod interchange;
mod ops;
mod rounding;
mod signals;
mod text;
mod types;
mod ux;

pub use interchange::*;
pub use ops::*;
pub use signals::{Signal, SignalFlags, SignalSink};
pub use text::{TextError, interval_to_text};
pub use types::{DecoratedInterval, Decoration, Interval, IntervalDatum, InvalidDecoration};

pub mod prelude {
    pub use crate::types::{DecoratedInterval, Decoration, Interval};
}
