use crate::{
    DecoratedInterval, Decoration, Interval,
    signals::{Signal, SignalSink},
};

pub const INTERVAL_ENCODED_LEN: usize = 16;
pub const DECORATED_INTERVAL_ENCODED_LEN: usize = 17;

const POSITIVE_INFINITY_BITS: u64 = 0x7ff0_0000_0000_0000;
const NEGATIVE_INFINITY_BITS: u64 = 0xfff0_0000_0000_0000;
const NEGATIVE_ZERO_BITS: u64 = 0x8000_0000_0000_0000;
const POSITIVE_ZERO_BITS: u64 = 0x0000_0000_0000_0000;
const CANONICAL_QUIET_NAN_BITS: u64 = 0x7ff8_0000_0000_0000;

// Level 4 export.

pub fn interval_to_be_bytes(x: Interval) -> [u8; INTERVAL_ENCODED_LEN] {
    let mut output = [0_u8; INTERVAL_ENCODED_LEN];
    output[..8].copy_from_slice(&x.inf_raw().to_be_bytes());
    output[8..].copy_from_slice(&x.sup_raw().to_be_bytes());
    output
}

pub fn interval_to_le_bytes(x: Interval) -> [u8; INTERVAL_ENCODED_LEN] {
    let mut output = [0_u8; INTERVAL_ENCODED_LEN];
    output[..8].copy_from_slice(&x.inf_raw().to_le_bytes());
    output[8..].copy_from_slice(&x.sup_raw().to_le_bytes());
    output
}

pub fn decorated_interval_to_be_bytes(
    x: DecoratedInterval,
) -> [u8; DECORATED_INTERVAL_ENCODED_LEN] {
    let mut output = [0_u8; DECORATED_INTERVAL_ENCODED_LEN];
    if x.is_nai_raw() {
        let nan = CANONICAL_QUIET_NAN_BITS.to_be_bytes();
        output[..8].copy_from_slice(&nan);
        output[8..16].copy_from_slice(&nan);
        output[16] = Decoration::Ill as u8;
        return output;
    }
    output[..16].copy_from_slice(&interval_to_be_bytes(x.interval_raw()));
    output[16] = x.decoration_raw() as u8;
    output
}

pub fn decorated_interval_to_le_bytes(
    x: DecoratedInterval,
) -> [u8; DECORATED_INTERVAL_ENCODED_LEN] {
    let mut output = [0_u8; DECORATED_INTERVAL_ENCODED_LEN];
    if x.is_nai_raw() {
        let nan = CANONICAL_QUIET_NAN_BITS.to_le_bytes();

        output[..8].copy_from_slice(&nan);
        output[8..16].copy_from_slice(&nan);
        output[16] = Decoration::Ill as u8;

        return output;
    }
    output[..16].copy_from_slice(&interval_to_le_bytes(x.interval_raw()));
    output[16] = x.decoration_raw() as u8;
    output
}

// Level 4 import with InvalidOperand signaling.

pub fn interval_from_be_bytes<S: SignalSink>(bytes: &[u8], signals: &mut S) -> Interval {
    match decode_interval(bytes, Endian::Big) {
        Some(value) => value,
        None => {
            signals.raise(Signal::InvalidOperand);
            Interval::EMPTY
        }
    }
}

pub fn interval_from_le_bytes<S: SignalSink>(bytes: &[u8], signals: &mut S) -> Interval {
    match decode_interval(bytes, Endian::Little) {
        Some(value) => value,
        None => {
            signals.raise(Signal::InvalidOperand);
            Interval::EMPTY
        }
    }
}

pub fn decorated_interval_from_be_bytes<S: SignalSink>(
    bytes: &[u8],
    signals: &mut S,
) -> DecoratedInterval {
    match decode_decorated(bytes, Endian::Big) {
        Some(value) => value,
        None => {
            signals.raise(Signal::InvalidOperand);
            DecoratedInterval::NAI
        }
    }
}

pub fn decorated_interval_from_le_bytes<S: SignalSink>(
    bytes: &[u8],
    signals: &mut S,
) -> DecoratedInterval {
    match decode_decorated(bytes, Endian::Little) {
        Some(value) => value,
        None => {
            signals.raise(Signal::InvalidOperand);
            DecoratedInterval::NAI
        }
    }
}

#[derive(Clone, Copy)]
enum Endian {
    Big,
    Little,
}

fn decode_interval(bytes: &[u8], endian: Endian) -> Option<Interval> {
    let bytes: &[u8; INTERVAL_ENCODED_LEN] = bytes.try_into().ok()?;
    let inf_bits = read_u64(&bytes[..8], endian)?;
    let sup_bits = read_u64(&bytes[8..], endian)?;
    decode_interval_bits(inf_bits, sup_bits)
}

fn decode_interval_bits(inf_bits: u64, sup_bits: u64) -> Option<Interval> {
    if inf_bits == POSITIVE_INFINITY_BITS && sup_bits == NEGATIVE_INFINITY_BITS {
        return Some(Interval::EMPTY);
    }
    let inf = f64::from_bits(inf_bits);
    let sup = f64::from_bits(sup_bits);
    if inf.is_nan() || sup.is_nan() || inf > sup || inf == f64::INFINITY || sup == f64::NEG_INFINITY
    {
        return None;
    }
    if inf == 0.0 && inf_bits != NEGATIVE_ZERO_BITS {
        return None;
    }
    if sup == 0.0 && sup_bits != POSITIVE_ZERO_BITS {
        return None;
    }
    Some(Interval::from_valid_bounds(inf, sup))
}

fn decode_decorated(bytes: &[u8], endian: Endian) -> Option<DecoratedInterval> {
    let bytes: &[u8; DECORATED_INTERVAL_ENCODED_LEN] = bytes.try_into().ok()?;
    let inf_bits = read_u64(&bytes[..8], endian)?;
    let sup_bits = read_u64(&bytes[8..16], endian)?;
    let decoration = Decoration::try_from(bytes[16]).ok()?;
    let inf = f64::from_bits(inf_bits);
    let sup = f64::from_bits(sup_bits);
    if decoration == Decoration::Ill {
        return if inf.is_nan() && sup.is_nan() {
            Some(DecoratedInterval::NAI)
        } else {
            None
        };
    }
    if inf.is_nan() || sup.is_nan() {
        return None;
    }
    let interval = decode_interval_bits(inf_bits, sup_bits)?;
    if interval.is_empty_raw() && decoration != Decoration::Trv {
        return None;
    }
    if !interval.is_bounded_raw() && decoration == Decoration::Com {
        return None;
    }
    Some(DecoratedInterval::set_dec_raw(interval, decoration))
}

fn read_u64(bytes: &[u8], endian: Endian) -> Option<u64> {
    let bytes: [u8; 8] = bytes.try_into().ok()?;
    Some(match endian {
        Endian::Big => u64::from_be_bytes(bytes),
        Endian::Little => u64::from_le_bytes(bytes),
    })
}
