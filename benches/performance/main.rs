//! Host benchmarks for interval arithmetic, linear algebra, and search.

#![allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_precision_loss
)]

use std::{convert::Infallible, hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput};
use maryada::{
    DIntervalMatrix, DIntervalVector, EnclosureOps, EpsilonInflationSolver, GlobalMinimizer,
    Interval, IntervalOps, RunStatus, Solver,
};
use nalgebra::{DMatrix, DVector};

fn benchmark<I, O>(
    criterion: &mut Criterion,
    name: &str,
    inputs: &[I],
    operation: impl Fn(&I) -> O,
) {
    criterion
        .benchmark_group(name)
        .throughput(Throughput::Elements(inputs.len() as u64))
        .bench_function(BenchmarkId::from_parameter(inputs.len()), |bencher| {
            bencher.iter(|| {
                for input in inputs {
                    let _ = black_box(operation(black_box(input)));
                }
            });
        });
}

#[derive(Clone, Copy)]
enum ScalarWorkload {
    Singleton,
    SameSign,
    ZeroCrossing,
    Unbounded,
    Subnormal,
}

struct ScalarInput {
    left: Interval,
    right: Interval,
    addend: Interval,
}

fn scalar_input(workload: ScalarWorkload, index: u32) -> ScalarInput {
    let first = 0.25 + f64::from(index % 61) / 32.0;
    let second = 0.5 + f64::from(index * 17 % 59) / 32.0;
    let signed = |interval: Interval, sign_bit| {
        if index & sign_bit == 0 {
            interval
        } else {
            -interval
        }
    };
    let (left, right) = match workload {
        ScalarWorkload::Singleton => (
            signed(Interval::from(first), 1),
            signed(Interval::from(second), 2),
        ),
        ScalarWorkload::SameSign => (
            signed(Interval::new(first, first + 0.125), 1),
            signed(Interval::new(second, second + 0.25), 2),
        ),
        ScalarWorkload::ZeroCrossing => (
            Interval::new(-first, second),
            if index & 1 == 0 {
                Interval::new(-second, first)
            } else {
                signed(Interval::new(second, second + 0.25), 2)
            },
        ),
        ScalarWorkload::Unbounded => (
            signed(Interval::new(first, f64::INFINITY), 1),
            if index & 2 == 0 {
                signed(Interval::new(second, f64::INFINITY), 4)
            } else {
                Interval::new(-second, second)
            },
        ),
        ScalarWorkload::Subnormal => {
            let bits = u64::from(index) + 1;
            (
                signed(
                    Interval::new(f64::from_bits(bits), f64::from_bits(bits + 8)),
                    1,
                ),
                signed(
                    Interval::new(f64::from_bits(bits + 3), f64::from_bits(bits + 20)),
                    2,
                ),
            )
        }
    };
    let addend = if matches!(workload, ScalarWorkload::Subnormal) {
        f64::from_bits(u64::from(index) + 27)
    } else {
        second / 8.0
    };
    ScalarInput {
        left,
        right,
        addend: signed(Interval::from(addend), 4),
    }
}

fn benchmark_scalar(criterion: &mut Criterion) {
    for (group, workload) in [
        ("singleton", ScalarWorkload::Singleton),
        ("same-sign", ScalarWorkload::SameSign),
        ("zero-crossing", ScalarWorkload::ZeroCrossing),
        ("unbounded", ScalarWorkload::Unbounded),
        ("subnormal", ScalarWorkload::Subnormal),
    ] {
        let inputs: Vec<_> = (0..512)
            .map(|index| scalar_input(workload, index))
            .collect();
        macro_rules! measure {
            ($operation:literal, $calculate:expr) => {
                benchmark(
                    criterion,
                    &format!("scalar/{group}/{}", $operation),
                    &inputs,
                    $calculate,
                )
            };
        }
        measure!("add", |x: &ScalarInput| x.left + x.right);
        measure!("mul", |x: &ScalarInput| x.left * x.right);
        measure!("div", |x: &ScalarInput| x.left / x.right);
        measure!("fma", |x: &ScalarInput| x.left.mul_add(x.right, x.addend));
        measure!("exp", |x: &ScalarInput| x.left.exp());
        measure!("log", |x: &ScalarInput| x.left.log());
        measure!("sin", |x: &ScalarInput| x.left.sin());
        measure!("tan", |x: &ScalarInput| x.left.tan());
    }
}

#[allow(clippy::expect_used)]
fn benchmark_matrices(criterion: &mut Criterion) {
    let solver = EpsilonInflationSolver::default();
    for size in [4, 32] {
        // Dyadic midpoint entries make A*x exact in binary64 for the
        // known solutions below. The interval coefficients enclose A.
        let midpoint = DMatrix::from_fn(size, size, |row, column| {
            if row == column {
                size as f64 / 2.0 + 1.0
            } else {
                (((row * 13 + column * 7 + row * column) % 9) as f64 - 4.0) / 16.0
            }
        });
        let matrix: DIntervalMatrix<Interval> = midpoint
            .map(|value| Interval::new(value - 1.0 / 1024.0, value + 1.0 / 1024.0))
            .into();
        benchmark(
            criterion,
            &format!("matrix/mul/{size}x{size}"),
            std::slice::from_ref(&matrix),
            |matrix| matrix * matrix,
        );

        let right_hand_sides: Vec<_> = (0..8)
            .map(|index| {
                let witness = DVector::from_fn(size, |row, _| {
                    (((row * 5 + index * 7) % 17) as f64 - 8.0) / 4.0
                });
                let rhs: DIntervalVector<Interval> = (&midpoint * &witness)
                    .map(|value| Interval::new(value - 1.0 / 128.0, value + 1.0 / 128.0))
                    .into();
                let enclosure = solver
                    .try_solve(&matrix, &rhs)
                    .expect("benchmark system should certify");
                assert_eq!(enclosure.len(), witness.len());
                for (entry, &point) in enclosure.iter().zip(witness.iter()) {
                    assert!(entry.contains(point), "solve missed the known solution");
                }
                rhs
            })
            .collect();
        benchmark(
            criterion,
            &format!("matrix/solve/epsilon-inflation/{size}x{size}"),
            &right_hand_sides,
            |rhs| solver.try_solve(&matrix, rhs),
        );
    }
}

#[allow(clippy::expect_used)]
fn benchmark_search(criterion: &mut Criterion) {
    let root = vec![Interval::new(-1.0, 1.0); 8];
    for (mode, gap_tolerance) in [("gap-on", Some(0.25)), ("gap-off", None)] {
        // A constant enclosure keeps both children in play and exercises
        // the gap check throughout the step budget.
        let advance = |root: &Vec<Interval>| {
            let mut search = GlobalMinimizer::new(root.clone(), |_: &Vec<Interval>| {
                Ok::<_, Infallible>(Interval::new(0.0, 1.0))
            })
            .without_value_tolerance()
            .without_domain_tolerance()
            .without_gap_tolerance();
            if let Some(tolerance) = gap_tolerance {
                search = search.with_gap_tolerance(tolerance);
            }
            search.advance(512).expect("valid search benchmark options")
        };
        let result = advance(&root);
        assert_eq!(result.status, RunStatus::StepLimit);
        assert_eq!(result.unresolved.len(), 513);
        assert_eq!(result.minimum, Interval::new(0.0, 1.0));

        benchmark(
            criterion,
            &format!("search/advance/8d-512-steps/{mode}"),
            std::slice::from_ref(&root),
            advance,
        );
    }
}

fn main() {
    let mut criterion = Criterion::default()
        .warm_up_time(Duration::from_millis(250))
        .measurement_time(Duration::from_secs(1))
        .sample_size(30)
        .configure_from_args();
    benchmark_scalar(&mut criterion);
    benchmark_matrices(&mut criterion);
    benchmark_search(&mut criterion);
    criterion.final_summary();
}
