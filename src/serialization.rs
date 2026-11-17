use core::{fmt, str};

use serde::{Deserialize, Deserializer, Serialize, Serializer, de, ser};

use crate::{
    DecoratedInterval, Interval, IntervalDatum, ParseIntervalDiagnostic, SignalFlags,
    interval_to_text,
};

fn serialize_text<T: IntervalDatum, S: Serializer>(
    value: T,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let mut buffer = [0; 64];
    let len = interval_to_text(value, None, &mut buffer).map_err(ser::Error::custom)?;
    let text = buffer
        .get(..len)
        .and_then(|bytes| str::from_utf8(bytes).ok())
        .ok_or_else(|| ser::Error::custom("invalid interval text encoding"))?;
    serializer.serialize_str(text)
}

struct TextVisitor<T> {
    parse: fn(&str) -> Result<(T, SignalFlags), ParseIntervalDiagnostic>,
}

impl<T> de::Visitor<'_> for TextVisitor<T> {
    type Value = T;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an interval literal")
    }

    fn visit_str<E: de::Error>(self, text: &str) -> Result<T, E> {
        let (value, signals) = (self.parse)(text).map_err(E::custom)?;
        if !signals.is_empty() {
            return Err(E::custom("possibly undefined interval literal"));
        }
        Ok(value)
    }
}

struct BytesVisitor<T> {
    decode: fn(&[u8], &mut SignalFlags) -> T,
}

impl<T> de::Visitor<'_> for BytesVisitor<T> {
    type Value = T;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("big-endian interval interchange bytes")
    }

    fn visit_bytes<E: de::Error>(self, bytes: &[u8]) -> Result<T, E> {
        let mut signals = SignalFlags::NONE;
        let value = (self.decode)(bytes, &mut signals);
        if !signals.is_empty() {
            return Err(E::custom("invalid interval interchange bytes"));
        }
        Ok(value)
    }
}

impl Serialize for Interval {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if serializer.is_human_readable() {
            serialize_text(*self, serializer)
        } else {
            serializer.serialize_bytes(&self.to_be_bytes())
        }
    }
}

impl<'de> Deserialize<'de> for Interval {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        if deserializer.is_human_readable() {
            deserializer.deserialize_str(TextVisitor {
                parse: Self::parse_checked,
            })
        } else {
            deserializer.deserialize_bytes(BytesVisitor {
                decode: Self::from_be_bytes,
            })
        }
    }
}

impl Serialize for DecoratedInterval {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if serializer.is_human_readable() {
            serialize_text(*self, serializer)
        } else {
            serializer.serialize_bytes(&self.to_be_bytes())
        }
    }
}

impl<'de> Deserialize<'de> for DecoratedInterval {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        if deserializer.is_human_readable() {
            deserializer.deserialize_str(TextVisitor {
                parse: Self::parse_checked,
            })
        } else {
            deserializer.deserialize_bytes(BytesVisitor {
                decode: Self::from_be_bytes,
            })
        }
    }
}
