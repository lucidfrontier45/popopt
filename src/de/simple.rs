use anyhow::Result as AnyResult;
use ordered_float::NotNan;
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::interface::{Problem, Score, Variable};

use super::base::{CrossoverOperator, Initializer, MutationOperator, Selector};

pub struct SimpleInitializer {
    bounds: Vec<(NotNan<f64>, NotNan<f64>)>,
    seed: u64,
}

impl SimpleInitializer {
    pub fn new(bounds: Vec<(NotNan<f64>, NotNan<f64>)>) -> Self {
        let mut rng = StdRng::from_os_rng();
        let seed: u64 = rng.random();
        Self::with_seed(bounds, seed)
    }

    pub fn from_single_bound(lower: NotNan<f64>, upper: NotNan<f64>, dim: usize) -> Self {
        Self::new(vec![(lower, upper); dim])
    }

    pub fn with_seed(bounds: Vec<(NotNan<f64>, NotNan<f64>)>, seed: u64) -> Self {
        Self { bounds, seed }
    }
}

impl Initializer for SimpleInitializer {
    fn initialize(
        &self,
        problem: &dyn Problem,
        population_size: usize,
    ) -> AnyResult<(Vec<Score>, Vec<Variable>)> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut scores = Vec::new();
        let mut variables = Vec::new();
        for _ in 0..population_size {
            let mut x = Vec::new();
            for (lower, upper) in &self.bounds {
                x.push(rng.random_range(lower.into_inner()..upper.into_inner()));
            }
            let x = Variable::from_vec(x);
            let score = problem.evaluate(&x)?;
            variables.push(x);
            scores.push(score);
        }
        Ok((scores, variables))
    }
}

pub struct SimpleMutationOperator {
    scale: NotNan<f64>,
}

impl SimpleMutationOperator {
    pub fn new(scale: NotNan<f64>) -> Self {
        Self { scale }
    }
}

impl MutationOperator for SimpleMutationOperator {
    fn mutate_one(&self, current_population: &[Variable]) -> AnyResult<Variable> {
        // randomly select three distinct indices
        let n = current_population.len();
        let mut rng = rand::rng();
        let mut indices = vec![];
        while indices.len() < 3 {
            let index = rng.random_range(0..n);
            if !indices.contains(&index) {
                indices.push(index);
            }
        }
        let (i0, i1, i2) = (indices[0], indices[1], indices[2]);

        // perform mutation
        let v = &current_population[i0]
            + self.scale.into_inner() * (&current_population[i1] - &current_population[i2]);
        Ok(v)
    }
}

pub struct SimpleCrossoverOperator {
    crossover_rate: NotNan<f64>,
}

impl SimpleCrossoverOperator {
    pub fn new(crossover_rate: NotNan<f64>) -> Self {
        Self { crossover_rate }
    }
}

impl CrossoverOperator for SimpleCrossoverOperator {
    fn crossover_one(&self, v_current: &Variable, v_mutant: &Variable) -> AnyResult<Variable> {
        let mut v_trial: Vec<f64> = Vec::with_capacity(v_current.len());
        let mut rng = rand::rng();
        for (x_current, x_mutant) in v_current.iter().zip(v_mutant.iter()) {
            let r: f64 = rng.random_range(0.0..1.0);
            let x_trial = if r < self.crossover_rate.into_inner() {
                x_mutant
            } else {
                x_current
            };
            v_trial.push(*x_trial);
        }
        Ok(Variable::from_vec(v_trial))
    }
}

#[derive(Default)]
pub struct SimpleSelector {}

impl SimpleSelector {
    pub fn new() -> Self {
        Self {}
    }
}

impl Selector for SimpleSelector {
    fn select_one(
        &self,
        problem: &dyn Problem,
        s_current: Score,
        v_current: Variable,
        v_trial: Variable,
    ) -> AnyResult<(Score, Variable)> {
        let s_trial = problem.evaluate(&v_trial)?;
        if s_trial < s_current {
            Ok((s_trial, v_trial))
        } else {
            Ok((s_current, v_current))
        }
    }
}
