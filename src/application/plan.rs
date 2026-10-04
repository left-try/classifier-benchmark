use crate::{
    application::budget::{estimate_plan, EstimateError, Money},
    config::AppConfig,
    domain::{
        model::ModelCandidate,
        suite::{TestSuite, WorkUnit},
    },
};
use std::collections::{BTreeMap, HashMap};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PlanError {
    #[error("budget must be positive and finite")]
    InvalidBudget,
    #[error("no compatible candidate models were selected")]
    NoModels,
    #[error("case {case_id} request bound {bytes} bytes exceeds configured limit {limit}")]
    InputTooLarge {
        case_id: String,
        bytes: usize,
        limit: usize,
    },
    #[error("estimated run cost ${estimated:.6} exceeds budget ${budget:.6}")]
    OverBudget { estimated: f64, budget: f64 },
    #[error(transparent)]
    Estimate(#[from] EstimateError),
}

#[derive(Debug, Clone)]
pub struct RunPlan {
    pub models: Vec<ModelCandidate>,
    pub work_units: Vec<WorkUnit>,
    pub estimated_total: Money,
    pub budget: Money,
    pub max_input_bytes: usize,
    pub max_output_tokens: u32,
    pub concurrency: usize,
    pub max_retries: u32,
    pub model_spacing_ms: u64,
    pub seed: u64,
    pub protocol_overrides: BTreeMap<String, String>,
    pub input_token_bounds: HashMap<String, u64>,
}

impl RunPlan {
    pub fn build(
        config: &AppConfig,
        suite: &TestSuite,
        models: Vec<ModelCandidate>,
        budget_usd: f64,
    ) -> Result<Self, PlanError> {
        if !budget_usd.is_finite() || budget_usd <= 0.0 {
            return Err(PlanError::InvalidBudget);
        }
        if models.is_empty() {
            return Err(PlanError::NoModels);
        }
        let budget = Money::from_usd_ceil(budget_usd).map_err(|_| PlanError::InvalidBudget)?;
        let ids = models.iter().map(|m| m.id.clone()).collect::<Vec<_>>();
        let work_units = suite.work_units(&ids, config.repeats);
        let mut input_token_bounds = HashMap::new();
        for case in &suite.cases {
            let payload = serde_json::json!({
                "scenario": case.scenario,
                "input": case.input,
                "output_schema": suite.output_schema(&case.scenario)
            });
            let bytes = serde_json::to_vec(&payload)
                .map_err(|_| PlanError::InvalidBudget)?
                .len()
                .saturating_add(384);
            if bytes > config.max_input_bytes {
                return Err(PlanError::InputTooLarge {
                    case_id: case.case_id.clone(),
                    bytes,
                    limit: config.max_input_bytes,
                });
            }
            input_token_bounds.insert(case.case_id.clone(), bytes as u64);
        }
        let mut plan = Self {
            models,
            work_units,
            estimated_total: Money(0),
            budget,
            max_input_bytes: config.max_input_bytes,
            max_output_tokens: config.max_output_tokens,
            concurrency: config.concurrency,
            max_retries: config.max_retries,
            model_spacing_ms: config.model_spacing_ms,
            seed: config.seed,
            protocol_overrides: config.protocol_overrides.clone(),
            input_token_bounds,
        };
        plan.estimated_total = estimate_plan(&plan)?;
        if plan.estimated_total > budget {
            return Err(PlanError::OverBudget {
                estimated: plan.estimated_total.as_usd(),
                budget: budget.as_usd(),
            });
        }
        Ok(plan)
    }
}
