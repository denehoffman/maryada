#![cfg(feature = "serde")]
#![allow(missing_docs, clippy::unwrap_used)]

use maryada::{DecoratedInterval, Decoration, Interval, IntervalOps, set_dec};
use serde::{Serialize, de::DeserializeOwned};

fn round_trip<T, const N: usize>(value: T, to_be_bytes: impl Fn(T) -> [u8; N])
where
    T: Serialize + DeserializeOwned + Copy,
{
    let expected_bytes = to_be_bytes(value);
    let json = serde_json::to_string(&value).unwrap();
    let from_json = serde_json::from_str(&json).unwrap();
    assert_eq!(to_be_bytes(from_json), expected_bytes);

    let mut cbor = Vec::new();
    ciborium::into_writer(&value, &mut cbor).unwrap();
    // Both payload sizes fit in a single CBOR byte-string header.
    assert_eq!(cbor.first(), Some(&(0x40 | u8::try_from(N).unwrap())));
    assert_eq!(cbor.get(1..), Some(expected_bytes.as_slice()));
    let from_cbor = ciborium::from_reader(cbor.as_slice()).unwrap();
    assert_eq!(to_be_bytes(from_cbor), expected_bytes);
}

#[test]
fn serde_round_trips_exact_endpoints_special_values_and_decorations() {
    let narrow = Interval::new(f64::from_bits(1), f64::MIN_POSITIVE);
    for value in [Interval::EMPTY, Interval::ENTIRE, Interval::ZERO, narrow] {
        round_trip(value, Interval::to_be_bytes);
    }
    for value in [
        DecoratedInterval::EMPTY,
        DecoratedInterval::ENTIRE,
        DecoratedInterval::ZERO,
        DecoratedInterval::NAI,
        set_dec(Interval::new(-f64::MAX, f64::MAX), Decoration::Com),
        set_dec(narrow, Decoration::Trv),
        set_dec(narrow, Decoration::Def),
        set_dec(narrow, Decoration::Dac),
        set_dec(narrow, Decoration::Com),
    ] {
        round_trip(value, DecoratedInterval::to_be_bytes);
    }

    assert_eq!(
        serde_json::to_string(&Interval::new(1.0, 2.0)).unwrap(),
        r#""[0x1p+0,0x1p+1]""#
    );
    assert_eq!(
        serde_json::to_string(&DecoratedInterval::NAI).unwrap(),
        r#""[nai]""#
    );
    assert_eq!(
        serde_json::from_str::<Interval>(r#""[1,2]""#).unwrap(),
        Interval::new(1.0, 2.0)
    );
}

#[test]
fn serde_rejects_malformed_and_diagnostic_input() {
    let reversed = serde_json::from_str::<Interval>(r#""[2,1]""#).unwrap_err();
    assert!(
        reversed
            .to_string()
            .contains("lower endpoint exceeds upper endpoint at byte 3")
    );
    assert!(serde_json::from_str::<Interval>(r#""[0.100000000000000001,1/10]""#).is_err());
    assert!(serde_json::from_str::<Interval>(r#""[nai]""#).is_err());
    assert!(serde_json::from_str::<Interval>("[1,2]").is_err());
    assert!(serde_json::from_str::<DecoratedInterval>(r#""[empty]_com""#).is_err());

    // A zero lower endpoint must carry a negative sign bit in interchange.
    let mut noncanonical_zero = [0_u8; 17];
    *noncanonical_zero.first_mut().unwrap() = 0x50;
    assert!(ciborium::from_reader::<Interval, _>(noncanonical_zero.as_slice()).is_err());
    assert!(ciborium::from_reader::<Interval, _>(&[0x40][..]).is_err());

    let mut invalid_decoration = Vec::from([0x51]);
    invalid_decoration.extend_from_slice(&DecoratedInterval::ONE.to_be_bytes());
    *invalid_decoration.last_mut().unwrap() = 0xff;
    assert!(ciborium::from_reader::<DecoratedInterval, _>(invalid_decoration.as_slice()).is_err());
}
