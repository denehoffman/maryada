#![allow(missing_docs)]
#![allow(
    clippy::float_cmp,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::unwrap_used
)]
use maryada::{
    DecoratedInterval, Decoration, Interval, IntervalOps, Signal, SignalFlags, TextError,
    decoration_part, interval_to_text, set_dec, subset,
};

#[test]
fn required_text_literal_forms_are_accepted() {
    let mut signals = SignalFlags::NONE;

    assert_eq!(
        Interval::text_to_interval("[1, 2]", &mut signals).bounds(),
        (1.0, 2.0)
    );
    assert_eq!(
        Interval::text_to_interval("[0x1p-1]", &mut signals).bounds(),
        (0.5, 0.5)
    );
    assert_eq!(
        Interval::text_to_interval("[+1.25E+2]", &mut signals).bounds(),
        (125.0, 125.0)
    );
    assert_eq!(
        Interval::text_to_interval("[0X1.Ap+2]", &mut signals).bounds(),
        (6.5, 6.5)
    );
    assert!(Interval::text_to_interval("[-INFINITY,+Inf]", &mut signals).is_entire());
    assert!(Interval::text_to_interval("[,]", &mut signals).is_entire());
    assert_eq!(
        Interval::text_to_interval("[,2]", &mut signals).bounds(),
        (f64::NEG_INFINITY, 2.0)
    );
    assert_eq!(
        Interval::text_to_interval("[1,]", &mut signals).bounds(),
        (1.0, f64::INFINITY)
    );
    assert!(Interval::text_to_interval("[ ]", &mut signals).is_empty());
    assert!(Interval::text_to_interval("[empty]", &mut signals).is_empty());
    assert!(Interval::text_to_interval("[EnTiRe]", &mut signals).is_entire());

    let rational = Interval::text_to_interval("[1/3, 2/3]", &mut signals);
    assert_eq!(
        rational.bounds(),
        (
            f64::from_bits(0x3fd5_5555_5555_5555),
            f64::from_bits(0x3fe5_5555_5555_5556),
        )
    );

    assert_eq!(
        Interval::text_to_interval("[-1/3]", &mut signals).bounds(),
        (
            f64::from_bits(0xbfd5_5555_5555_5556),
            f64::from_bits(0xbfd5_5555_5555_5555),
        )
    );

    let uncertain = Interval::text_to_interval("1.0?", &mut signals);
    assert!(uncertain.contains(1.0));
    assert!(uncertain.inf() < uncertain.sup());

    let decorated = DecoratedInterval::text_to_interval("[1,2]_COM", &mut signals);
    assert_eq!(decoration_part(decorated), Decoration::Com);
    assert!(DecoratedInterval::text_to_interval("[NAI]", &mut signals).is_nai());
    assert!(signals.is_empty());
}

#[test]
fn uncertain_literals_cover_symmetric_directed_unbounded_and_scaled_forms() {
    let symmetric = Interval::text_to_interval("1.0?", &mut ());
    let symmetric_reference = Interval::text_to_interval("[19/20,21/20]", &mut ());
    assert!(subset(symmetric_reference, symmetric));

    let radius = Interval::text_to_interval("1.0?2", &mut ());
    let radius_reference = Interval::text_to_interval("[4/5,6/5]", &mut ());
    assert!(subset(radius_reference, radius));

    let down = Interval::text_to_interval("1.0?d", &mut ());
    assert_eq!(down.sup(), 1.0);
    assert!(down.inf() <= 0.95);

    let up = Interval::text_to_interval("1.0?2U", &mut ());
    assert_eq!(up.inf(), 1.0);
    assert!(up.sup() >= 1.2);

    assert_eq!(
        Interval::text_to_interval("1.0??d", &mut ()).bounds(),
        (f64::NEG_INFINITY, 1.0)
    );
    assert_eq!(
        Interval::text_to_interval("1.0??U", &mut ()).bounds(),
        (1.0, f64::INFINITY)
    );
    assert_eq!(
        Interval::text_to_interval("1.0?2uE3", &mut ()).bounds(),
        (1000.0, 1200.0)
    );
}

#[test]
fn decorated_literals_enforce_permitted_combinations_and_default_decorations() {
    for (literal, expected) in [
        ("[1]_trv", Decoration::Trv),
        ("[1]_def", Decoration::Def),
        ("[1]_dac", Decoration::Dac),
        ("[1]_com", Decoration::Com),
    ] {
        let value = DecoratedInterval::text_to_interval(literal, &mut ());
        assert_eq!(decoration_part(value), expected);
    }

    assert_eq!(
        decoration_part(DecoratedInterval::text_to_interval("[1,2]", &mut ())),
        Decoration::Com
    );
    assert_eq!(
        decoration_part(DecoratedInterval::text_to_interval("[,]", &mut ())),
        Decoration::Dac
    );
    assert_eq!(
        decoration_part(DecoratedInterval::text_to_interval("[empty]", &mut ())),
        Decoration::Trv
    );

    let positive_overflow = DecoratedInterval::text_to_interval("[1e999]_com", &mut ());
    assert_eq!(positive_overflow.bounds(), (f64::MAX, f64::INFINITY));
    assert_eq!(decoration_part(positive_overflow), Decoration::Dac);

    let negative_overflow = DecoratedInterval::text_to_interval("[-1e999]_com", &mut ());
    assert_eq!(negative_overflow.bounds(), (f64::NEG_INFINITY, -f64::MAX));
    assert_eq!(decoration_part(negative_overflow), Decoration::Dac);

    let mut signals = SignalFlags::NONE;
    assert!(DecoratedInterval::text_to_interval("[empty]_com", &mut signals).is_nai());
    assert!(signals.contains(Signal::UndefinedOperation));
    signals.clear();
    assert!(DecoratedInterval::text_to_interval("[,]_com", &mut signals).is_nai());
    assert!(signals.contains(Signal::UndefinedOperation));
}

#[test]
fn invalid_text_signals_and_output_round_trips() {
    let mut signals = SignalFlags::NONE;
    for literal in [
        "not an interval",
        "[.]",
        "[1e]",
        "[0x1]",
        "[0x.p1]",
        "[1/0]",
        "[1/-2]",
        "[+inf,0]",
        "[0,-inf]",
        "[nai]",
        "[2,1]",
        "[1 2]",
        "[1,2,3]",
        "[1;2]",
        "[1,2] trailing",
        "1e2?",
        "1.0 ?",
        "1.0?-1",
        "1.0?x",
        "1.0?2du",
        "1.0?2e",
    ] {
        assert!(
            Interval::text_to_interval(literal, &mut signals).is_empty(),
            "bare constructor accepted {literal:?}"
        );
        assert!(signals.contains(Signal::UndefinedOperation));
        signals.clear();
    }

    for literal in [
        "[1]_ill",
        "[nai]_trv",
        "[1]_com_extra",
        "[empty]_def",
        "[empty]_dac",
        "[empty]_com",
        "[entire]_com",
    ] {
        assert!(
            DecoratedInterval::text_to_interval(literal, &mut signals).is_nai(),
            "decorated constructor accepted {literal:?}"
        );
        assert!(signals.contains(Signal::UndefinedOperation));
        signals.clear();
    }

    let relaxed = Interval::text_to_interval("[0.1,0x1p0]", &mut signals);
    assert!(relaxed.inf() <= 0.1 && relaxed.sup() >= 1.0);
    assert!(signals.is_empty());

    for literal in ["[1.0,0x0p0]", "[2/3,1/3]"] {
        assert!(Interval::text_to_interval(literal, &mut signals).is_empty());
        assert!(signals.contains(Signal::UndefinedOperation));
        signals.clear();
    }

    let value = Interval::new(-1.25, 3.5);
    let mut output = [0; 64];
    let length = interval_to_text(value, None, &mut output).unwrap();
    let text = core::str::from_utf8(&output[..length]).unwrap();
    assert_eq!(Interval::text_to_interval(text, &mut ()), value);

    for value in [Interval::EMPTY, Interval::ENTIRE, Interval::ZERO] {
        let length = interval_to_text(value, Some("invalid"), &mut output).unwrap();
        let text = core::str::from_utf8(&output[..length]).unwrap();
        assert_eq!(Interval::text_to_interval(text, &mut ()), value);
    }

    let length = interval_to_text(DecoratedInterval::NAI, Some("hex"), &mut output).unwrap();
    assert_eq!(&output[..length], b"[nai]");
}

#[test]
fn interchange_encoding_and_validation_match_the_standard_representation() {
    let value = set_dec(Interval::new(-1.0, 3.0), Decoration::Com);
    let expected_be = [
        0xbf, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // -1
        0x40, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 3
        0x10, // com
    ];
    let expected_little_endian = [
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0xbf, // -1
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x40, // 3
        0x10, // com
    ];

    assert_eq!(value.to_be_bytes(), expected_be);
    assert_eq!(value.to_le_bytes(), expected_little_endian);
    assert_eq!(
        DecoratedInterval::from_be_bytes(&expected_be, &mut ()),
        value
    );
    assert_eq!(
        DecoratedInterval::from_le_bytes(&expected_little_endian, &mut ()),
        value
    );

    for (interval, expected) in [
        (
            Interval::EMPTY,
            [
                0x7f, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // +inf
                0xff, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // -inf
            ],
        ),
        (
            Interval::ENTIRE,
            [
                0xff, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // -inf
                0x7f, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // +inf
            ],
        ),
        (
            Interval::ZERO,
            [
                0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // -0
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // +0
            ],
        ),
    ] {
        assert_eq!(interval.to_be_bytes(), expected);
        assert_eq!(Interval::from_be_bytes(&expected, &mut ()), interval);

        let mut little_endian = expected;
        little_endian[..8].reverse();
        little_endian[8..].reverse();
        assert_eq!(interval.to_le_bytes(), little_endian);
        assert_eq!(Interval::from_le_bytes(&little_endian, &mut ()), interval);
    }

    for (decoration, octet) in [
        (Decoration::Trv, 0x04),
        (Decoration::Def, 0x08),
        (Decoration::Dac, 0x0c),
        (Decoration::Com, 0x10),
    ] {
        let decorated = set_dec(Interval::new(-1.0, 3.0), decoration);
        let encoded = decorated.to_be_bytes();
        assert_eq!(encoded[16], octet);
        assert_eq!(
            DecoratedInterval::from_be_bytes(&encoded, &mut ()),
            decorated
        );
    }

    let nai = DecoratedInterval::NAI.to_be_bytes();
    assert_eq!(nai[..8], 0x7ff8_0000_0000_0000_u64.to_be_bytes());
    assert_eq!(nai[8..16], 0x7ff8_0000_0000_0000_u64.to_be_bytes());
    assert_eq!(nai[16], 0x00);
    assert!(DecoratedInterval::from_be_bytes(&nai, &mut ()).is_nai());

    let invalid_bare = [
        [
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // +0 lower
            0x3f, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 1
        ],
        [
            0xbf, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // -1
            0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // -0 upper
        ],
        [
            0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 2
            0x3f, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 1
        ],
        [
            0x7f, 0xf8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // NaN
            0x3f, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 1
        ],
    ];
    let mut signals = SignalFlags::NONE;
    for encoded in invalid_bare {
        assert!(Interval::from_be_bytes(&encoded, &mut signals).is_empty());
        assert!(signals.contains(Signal::InvalidOperand));
        signals.clear();
    }
    assert!(Interval::from_be_bytes(&[0; 15], &mut signals).is_empty());
    assert!(signals.contains(Signal::InvalidOperand));

    let invalid_decorated = [
        [
            0x7f, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // +inf
            0xff, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // -inf
            0x10, // Empty_com
        ],
        [
            0xff, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // -inf
            0x7f, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // +inf
            0x10, // Entire_com
        ],
        [
            0xbf, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // -1
            0x40, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 3
            0x00, // ill with numeric bounds
        ],
        [
            0x7f, 0xf8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // NaN
            0x7f, 0xf8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // NaN
            0x04, // trv with NaN bounds
        ],
        [
            0xbf, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // -1
            0x40, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 3
            0x01, // invalid decoration octet
        ],
    ];
    signals.clear();
    for encoded in invalid_decorated {
        assert!(DecoratedInterval::from_be_bytes(&encoded, &mut signals).is_nai());
        assert!(signals.contains(Signal::InvalidOperand));
        signals.clear();
    }
    assert!(DecoratedInterval::from_be_bytes(&[0; 16], &mut signals).is_nai());
    assert!(signals.contains(Signal::InvalidOperand));
}

#[test]
fn uncertain_literals_cover_long_input_and_checked_arithmetic_boundaries() {
    let long_center =
        Interval::text_to_interval("1234567890123456789012345678901234567890?1", &mut ());
    assert!(long_center.is_bounded());
    assert!(long_center.contains(1.234_567_890_123_456_8e39));

    let doubled_center_fallback =
        Interval::text_to_interval("170141183460469231731687303715884105727?1", &mut ());
    assert!(doubled_center_fallback.is_bounded());

    for literal in [
        "170141183460469231731687303715884105727??",
        "170141183460469231731687303715884105727??d",
        "170141183460469231731687303715884105727??U",
    ] {
        let value = Interval::text_to_interval(literal, &mut ());
        assert!(!value.is_empty(), "fallback rejected {literal:?}");
    }

    for literal in [
        "1?170141183460469231731687303715884105727",
        "85070591730234615865843651857942052863?85070591730234615865843651857942052863",
    ] {
        assert!(
            Interval::text_to_interval(literal, &mut ()).is_entire(),
            "boundary did not conservatively widen {literal:?}",
        );
    }

    let huge_positive =
        Interval::text_to_interval("1?1e999999999999999999999999999999999999999", &mut ());
    assert!(!huge_positive.is_empty());
    assert_eq!(huge_positive.sup(), f64::INFINITY);
    let huge_negative =
        Interval::text_to_interval("1?1e-999999999999999999999999999999999999999", &mut ());
    assert!(!huge_negative.is_empty());
    assert!(huge_negative.contains(0.0));

    let mut signals = SignalFlags::NONE;
    for literal in [
        "1?9999999999999999999999999999999999999999",
        "1234567890123456789012345678901234567890.0.0?1",
        "1?1e+",
        "1?1e-",
        "1?1e-not-a-number",
    ] {
        assert!(
            Interval::text_to_interval(literal, &mut signals).is_empty(),
            "accepted malformed uncertain literal {literal:?}",
        );
        assert!(
            signals.contains(Signal::UndefinedOperation),
            "missing signal for {literal:?}"
        );
        signals.clear();
    }

    for literal in ["[inf]", "[-inf]", "[+infinity]", "[-INFINITY]"] {
        assert!(Interval::text_to_interval(literal, &mut signals).is_empty());
        assert!(signals.contains(Signal::UndefinedOperation));
        signals.clear();
    }
}

#[test]
fn hexadecimal_output_handles_binary64_edges_decorations_and_small_buffers() {
    let values = [
        Interval::ZERO,
        Interval::new(f64::from_bits(1), f64::from_bits(1)),
        Interval::from(1.625),
        Interval::from(-1.625),
        Interval::from(f64::MAX),
        Interval::new(f64::NEG_INFINITY, -1.0),
        Interval::new(1.0, f64::INFINITY),
        Interval::EMPTY,
        Interval::ENTIRE,
    ];

    let mut output = [0; 128];
    for value in values {
        let length = interval_to_text(value, Some("hex"), &mut output).unwrap();
        let text = core::str::from_utf8(&output[..length]).unwrap();
        assert_eq!(Interval::text_to_interval(text, &mut ()), value, "{text}");
    }

    let length = interval_to_text(Interval::ZERO, None, &mut output).unwrap();
    assert_eq!(&output[..length], b"[-0x0p+0,0x0p+0]");
    let length = interval_to_text(Interval::from(f64::from_bits(1)), None, &mut output).unwrap();
    assert_eq!(&output[..length], b"[0x1p-1074,0x1p-1074]");
    let length = interval_to_text(Interval::from(1.625), None, &mut output).unwrap();
    assert_eq!(&output[..length], b"[0x1.ap+0,0x1.ap+0]");

    for decoration in [
        Decoration::Trv,
        Decoration::Def,
        Decoration::Dac,
        Decoration::Com,
    ] {
        let value = set_dec(Interval::new(-1.625, 2.5), decoration);
        let length = interval_to_text(value, Some("hex"), &mut output).unwrap();
        let text = core::str::from_utf8(&output[..length]).unwrap();
        let parsed = DecoratedInterval::text_to_interval(text, &mut ());
        assert_eq!(parsed, value, "{text}");
        assert_eq!(decoration_part(parsed), decoration);
    }

    let ill_length = interval_to_text(DecoratedInterval::NAI, None, &mut output).unwrap();
    assert_eq!(&output[..ill_length], b"[nai]");

    for (value, capacity) in [
        (Interval::EMPTY, 0usize),
        (Interval::ENTIRE, 4usize),
        (Interval::from(1.625), 8usize),
    ] {
        let error = interval_to_text(value, None, &mut output[..capacity]).unwrap_err();
        assert!(matches!(error, TextError::BufferTooSmall { required } if required > capacity));
    }
}
