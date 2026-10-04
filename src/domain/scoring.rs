use thiserror::Error;

#[derive(Debug, Error)]
pub enum ScoreError {
    #[error(
        "accuracy must be between zero and one; cost and latency must be finite and non-negative"
    )]
    InvalidMetrics,
}

#[derive(Debug, Clone, Copy)]
pub struct ScoreParts {
    pub accuracy: f64,
    pub cost_factor: f64,
    pub latency_factor: f64,
    pub score: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct ScorePolicy {
    pub id: &'static str,
    pub reference_cost_usd: f64,
    pub free_latency_ms: f64,
}

impl ScorePolicy {
    pub fn decision_v1() -> Self {
        Self {
            id: "decision-v1",
            reference_cost_usd: 0.001,
            free_latency_ms: 500.0,
        }
    }

    pub fn score(
        &self,
        accuracy: f64,
        mean_cost_usd: f64,
        latency_ms: f64,
    ) -> Result<ScoreParts, ScoreError> {
        if !accuracy.is_finite()
            || !(0.0..=1.0).contains(&accuracy)
            || !mean_cost_usd.is_finite()
            || mean_cost_usd < 0.0
            || !latency_ms.is_finite()
            || latency_ms < 0.0
        {
            return Err(ScoreError::InvalidMetrics);
        }
        let cost_factor = 1.0 / (1.0 + (mean_cost_usd / self.reference_cost_usd).sqrt());
        let latency_penalty = ((latency_ms - self.free_latency_ms).max(0.0) / 1000.0).powi(2);
        let latency_factor = 1.0 / (1.0 + latency_penalty);
        Ok(ScoreParts {
            accuracy,
            cost_factor,
            latency_factor,
            score: accuracy * cost_factor * latency_factor,
        })
    }
}
