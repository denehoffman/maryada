#![allow(missing_docs)]
#![allow(clippy::float_cmp, clippy::unwrap_used, clippy::too_many_lines)]

use maryada::{
    DecoratedInterval, Decoration, EnclosureOps, Interval, IntervalOps, decoration_part, div_split,
    recip_split, set_dec,
};

#[test]
fn split_division_preserves_gaps_and_handles_zero_and_empty() {
    let positive = Interval::new(1.0, 2.0);
    let negative = Interval::new(-2.0, -1.0);
    let crossing = Interval::new(-2.0, 1.0);
    for (numerator, denominator, expected) in [
        (
            positive,
            crossing,
            vec![
                Interval::new(f64::NEG_INFINITY, -0.5),
                Interval::new(1.0, f64::INFINITY),
            ],
        ),
        (
            negative,
            crossing,
            vec![
                Interval::new(f64::NEG_INFINITY, -1.0),
                Interval::new(0.5, f64::INFINITY),
            ],
        ),
        (Interval::new(-1.0, 1.0), crossing, vec![Interval::ENTIRE]),
        (Interval::new(0.0, 2.0), crossing, vec![Interval::ENTIRE]),
        (Interval::ZERO, crossing, vec![Interval::ZERO]),
        (
            positive,
            Interval::new(0.0, 2.0),
            vec![Interval::new(0.5, f64::INFINITY)],
        ),
        (
            negative,
            Interval::new(0.0, 2.0),
            vec![Interval::new(f64::NEG_INFINITY, -0.5)],
        ),
        (
            positive,
            Interval::new(-2.0, 0.0),
            vec![Interval::new(f64::NEG_INFINITY, -0.5)],
        ),
        (positive, positive, vec![Interval::new(0.5, 2.0)]),
        (Interval::ONE, Interval::ENTIRE, vec![Interval::ENTIRE]),
        (positive, Interval::ZERO, vec![]),
        (Interval::ZERO, Interval::ZERO, vec![]),
        (positive, Interval::EMPTY, vec![]),
        (Interval::EMPTY, crossing, vec![]),
    ] {
        let result = div_split(numerator, denominator);
        assert_eq!(
            result.components(),
            expected,
            "{numerator:?} / {denominator:?}"
        );
        assert_eq!(result.hull(), numerator / denominator);
    }
}

#[test]
fn split_reciprocals_round_outward_and_merge_touching_bounds() {
    let one_third_down = f64::from_bits(0x3fd5_5555_5555_5555);
    for (input, expected) in [
        (
            Interval::new(-3.0, 3.0),
            vec![
                Interval::new(f64::NEG_INFINITY, -one_third_down),
                Interval::new(one_third_down, f64::INFINITY),
            ],
        ),
        (
            Interval::new(-f64::from_bits(1), f64::from_bits(1)),
            vec![
                Interval::new(f64::NEG_INFINITY, -f64::MAX),
                Interval::new(f64::MAX, f64::INFINITY),
            ],
        ),
        (
            Interval::new(0.0, 2.0),
            vec![Interval::new(0.5, f64::INFINITY)],
        ),
        (
            Interval::new(-2.0, 0.0),
            vec![Interval::new(f64::NEG_INFINITY, -0.5)],
        ),
        (Interval::new(2.0, 4.0), vec![Interval::new(0.25, 0.5)]),
        (Interval::ENTIRE, vec![Interval::ENTIRE]),
        (Interval::ZERO, vec![]),
        (Interval::EMPTY, vec![]),
    ] {
        let result = recip_split(input);
        assert_eq!(result.components(), expected, "1 / {input:?}");
        assert_eq!(result.hull(), input.recip());
    }
    let underflow = div_split(
        Interval::from(f64::from_bits(1)),
        Interval::new(-f64::MAX, f64::MAX),
    );
    assert_eq!(underflow.components(), &[Interval::ENTIRE]);
}

#[test]
#[allow(clippy::arithmetic_side_effects)]
fn split_quotients_enclose_point_values_and_match_ordinary_hulls_at_binary64_edges() {
    let tiny = f64::from_bits(1);
    let endpoints = [
        f64::NEG_INFINITY,
        -f64::MAX,
        -2.0,
        -tiny,
        0.0,
        tiny,
        2.0,
        f64::MAX,
        f64::INFINITY,
    ];
    let mut intervals = vec![Interval::EMPTY];
    for (index, &lower) in endpoints.iter().enumerate() {
        for &upper in endpoints.iter().skip(index) {
            if let Ok(interval) = Interval::try_new(lower, upper) {
                intervals.push(interval);
            }
        }
    }
    for &numerator in &intervals {
        assert_eq!(recip_split(numerator).hull(), numerator.recip());
        for &denominator in &intervals {
            let result = div_split(numerator, denominator);
            assert_eq!(
                result.hull(),
                numerator / denominator,
                "{numerator:?} / {denominator:?}"
            );
            if let [left, right] = result.components() {
                assert!(left.sup() < right.inf());
            }
            assert!(
                result
                    .components()
                    .iter()
                    .all(|component| !component.is_empty())
            );
            for x in [-2.0, -1.0, -tiny, 0.0, tiny, 1.0, 2.0] {
                for y in [-2.0, -1.0, -tiny, tiny, 1.0, 2.0] {
                    if numerator.contains(x) && denominator.contains(y) {
                        let point = x / y;
                        if point.is_finite() {
                            assert!(
                                result
                                    .components()
                                    .iter()
                                    .any(|component| component.contains(point)),
                                "missed {x} / {y} in {result:?}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn split_decorations_use_the_original_denominator_and_propagate_nai() {
    let positive = Interval::new(1.0, 2.0);
    for left_decoration in [
        Decoration::Trv,
        Decoration::Def,
        Decoration::Dac,
        Decoration::Com,
    ] {
        for right_decoration in [
            Decoration::Trv,
            Decoration::Def,
            Decoration::Dac,
            Decoration::Com,
        ] {
            let numerator = set_dec(positive, left_decoration);
            for denominator in [
                Interval::new(2.0, 4.0),
                Interval::new(0.0, 4.0),
                Interval::new(-4.0, 4.0),
            ] {
                let divisor = set_dec(denominator, right_decoration);
                let expected = if denominator.contains(0.0) {
                    Decoration::Trv
                } else {
                    left_decoration.min(right_decoration)
                };
                let result = div_split(numerator, divisor);
                assert_eq!(result.hull(), numerator / divisor);
                for &component in result.components() {
                    assert_eq!(decoration_part(component), expected);
                }
                assert_eq!(recip_split(divisor).hull(), divisor.recip());
            }
        }
    }
    for (left, right) in [
        (DecoratedInterval::NAI, DecoratedInterval::ONE),
        (DecoratedInterval::ONE, DecoratedInterval::NAI),
        (DecoratedInterval::NAI, DecoratedInterval::EMPTY),
        (DecoratedInterval::EMPTY, DecoratedInterval::NAI),
    ] {
        let result = div_split(left, right);
        assert_eq!(result.components().len(), 1);
        assert!(result.components().first().unwrap().is_nai());
        assert!(result.hull().is_nai());
    }
    let result = recip_split(DecoratedInterval::NAI);
    assert_eq!(result.components().len(), 1);
    assert!(result.hull().is_nai());
    for denominator in [DecoratedInterval::ZERO, DecoratedInterval::EMPTY] {
        let result = div_split(DecoratedInterval::ONE, denominator);
        assert_eq!(result.components(), []);
        assert_eq!(result.hull(), DecoratedInterval::EMPTY);
    }
}
