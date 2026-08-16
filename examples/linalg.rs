//! Several methods for
//!
use maryada::{GaussianElimination, Interval, IntervalMatrix, SIntervalVector};
use nalgebra::{Matrix4, Vector4};

fn main() {
    let low = Matrix4::new(
        4.0, -1.0, -1.0, -1.0, -1.0, -6.0, -1.0, -1.0, -1.0, -1.0, 9.0, -1.0, -1.0, -1.0, -1.0,
        -11.0,
    );
    let high = Matrix4::new(
        6.0, 1.0, 1.0, 1.0, 1.0, -4.0, 1.0, 1.0, 1.0, 1.0, 11.0, 1.0, 1.0, 1.0, 1.0, -9.0,
    );

    let a = IntervalMatrix::<Interval, _, _, _>::from_bounds(&low, &high);
    let b = SIntervalVector::<Interval, 4>::from_bounds(
        &Vector4::new(-2.0, 1.0, -4.0, 2.0),
        &Vector4::new(4.0, 8.0, 10.0, 12.0),
    );
    let solver = GaussianElimination;
    let Some(x) = a.solve_with(&b, &solver) else {
        eprintln!("the interval system could not be solved");
        return;
    };
    println!("{x}");
}
