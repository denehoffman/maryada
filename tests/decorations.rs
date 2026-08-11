#![allow(missing_docs)]
use maryada::{
    DecoratedInterval, Decoration, Interval, IntervalOps, add, ceil, decoration_part, fma,
    intersection, neg, new_dec, set_dec, sign, sqrt,
};

fn assert_decoration(value: DecoratedInterval, expected: Decoration) {
    assert_eq!(decoration_part(value), expected);
}

#[test]
fn initialization_assembly_and_comparison_follow_the_propagation_order() {
    assert_decoration(new_dec(Interval::EMPTY), Decoration::Trv);
    assert_decoration(new_dec(Interval::ENTIRE), Decoration::Dac);
    assert_decoration(new_dec(Interval::new(1.0, 2.0)), Decoration::Com);

    assert_decoration(set_dec(Interval::EMPTY, Decoration::Def), Decoration::Trv);
    assert_decoration(set_dec(Interval::ENTIRE, Decoration::Com), Decoration::Dac);
    assert!(set_dec(Interval::ZERO, Decoration::Ill).is_nai());

    assert!(Decoration::Ill < Decoration::Trv);
    assert!(Decoration::Trv < Decoration::Def);
    assert!(Decoration::Def < Decoration::Dac);
    assert!(Decoration::Dac < Decoration::Com);
}

#[test]
fn arithmetic_uses_the_min_rule_and_propagates_nai_for_every_arity() {
    let com = set_dec(Interval::new(1.0, 2.0), Decoration::Com);
    let dac = set_dec(Interval::new(3.0, 4.0), Decoration::Dac);
    let def = set_dec(Interval::new(5.0, 6.0), Decoration::Def);

    assert_decoration(neg(def), Decoration::Def);
    assert_decoration(add(dac, def), Decoration::Def);
    assert_decoration(fma(com, dac, def), Decoration::Def);

    let nai = DecoratedInterval::NAI;
    assert!(neg(nai).is_nai());
    assert!(add(nai, com).is_nai());
    assert!(add(com, nai).is_nai());
    assert!(fma(nai, com, com).is_nai());
    assert!(fma(com, nai, com).is_nai());
    assert!(fma(com, com, nai).is_nai());
}

#[test]
fn local_decorations_distinguish_domain_and_continuity_classes() {
    let bounded = |inf, sup| new_dec(Interval::new(inf, sup));

    assert_decoration(neg(bounded(1.0, 2.0)), Decoration::Com);
    assert_decoration(neg(new_dec(Interval::ENTIRE)), Decoration::Dac);
    assert_decoration(sqrt(bounded(-1.0, 4.0)), Decoration::Trv);

    // Integer functions are defined everywhere but discontinuous. A
    // one-sided restriction can be continuous (dac), while crossing the
    // discontinuity leaves only def.
    assert_decoration(sign(bounded(0.0, 0.0)), Decoration::Dac);
    assert_decoration(sign(bounded(-1.0, 1.0)), Decoration::Def);
    assert_decoration(ceil(bounded(-0.5, 0.0)), Decoration::Dac);
    assert_decoration(ceil(bounded(0.0, 0.5)), Decoration::Def);

    // Set operations have the trivial decorated versions prescribed by 5.7.
    assert_decoration(
        intersection(bounded(0.0, 2.0), bounded(1.0, 3.0)),
        Decoration::Trv,
    );
}
