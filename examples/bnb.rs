#![allow(
    clippy::arithmetic_side_effects,
    clippy::unnecessary_wraps,
    clippy::unwrap_used,
    missing_docs
)]

use std::convert::Infallible;

use maryada::{GlobalMinimizer, Interval, IntervalOps};

fn main() {
    fn objective(x: &Interval) -> Result<Interval, Infallible> {
        Ok((x.sqr() - 1.0).sqr() + 0.6 * *x + 2.0)
    }
    println!("Global Minimization of f(x) = (x^2 − 1)^2 + 0.6x + 2 for x ∈ [-1.5, 1.5]");
    let mut minimizer = GlobalMinimizer::new(Interval::new(-1.5, 1.5), objective);
    let res = minimizer.solve().unwrap();
    println!("status: {:?}", res.status);
    println!(
        "global minimum value enclosure:\n\tf(x) ∈ {} (truth: ≈ 1.3790)",
        res.minimum
    );
    if let (Some(domain), Some(value)) = (res.best_domain.as_ref(), res.best_value) {
        println!("converged interval:\n\tx ∈ {domain} (truth: ≈ -1.0679)");
        println!("f(x) on that interval:\n\tf(x) ∈ {value}");
    }

    fn schwefel_function(x: &Vec<Interval>) -> Result<Interval, Infallible> {
        let mut res = Interval::ZERO;
        let d = x.len() as f64;
        for x_i in x {
            res -= *x_i * x_i.abs().sqrt().sin();
        }
        res += Interval::from(418.9828872724337 * d);
        Ok(res)
    }
    println!(
        "Global Minimization of f(x) = -∑ᵢ xᵢ sin(√|xᵢ|) + 418.9828872724337 * D for x ∈ [-500, 500]^D with D = 3"
    );
    let mut minimizer =
        GlobalMinimizer::new(vec![Interval::new(-500.0, 500.0); 3], schwefel_function)
            .with_domain_tolerance(0.4)
            .with_value_tolerance(0.4);
    let res = minimizer.solve().unwrap();
    println!("status: {:?}", res.status);
    println!(
        "global minimum value enclosure:\n\tf(x) ∈ {} (truth: 0.0)",
        res.minimum
    );
    if let (Some(domain), Some(value)) = (res.best_domain.as_ref(), res.best_value) {
        println!(
            "converged interval:\n\tx ∈ ({:.2}, {:.2}, {:.2}) (truth: ≈(420.9687, 420.9687, 420.9687))",
            domain[0], domain[1], domain[2]
        );
        println!("f(x) on that interval:\n\tf(x) ∈ {value}");
    }
}
