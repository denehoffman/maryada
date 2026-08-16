//! Several methods for
//!
use maryada::{HansenBliekRohnSolver, Interval, IntervalMatrix, JacobiSolver, SIntervalVector};
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

    println!("Solving Ax = b for x\nA = {a:.2}\nb = {b:.2}");

    let Some(x) = a.solve(&b) else {
        eprintln!("the interval system could not be solved");
        return;
    };
    println!("Gaussian Elimination (with automatic preconditioning):\n\t{x:.2}");

    let solver = HansenBliekRohnSolver;
    let Some(x) = a.solve_with(&b, &solver) else {
        eprintln!("the interval system could not be solved");
        return;
    };
    println!("Hansen-Bliek-Rohn:\n\t{x:.2}");

    let solver = JacobiSolver::new(&a);
    let Some(x) = a.solve_with(&b, &solver) else {
        eprintln!("the interval system could not be solved");
        return;
    };
    println!("Jacobi:\n\t{x:.2}");
}
