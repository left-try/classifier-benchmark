use crate::domain::suite::WorkUnit;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunManifest {
    pub run_id: String,
    pub selected_models: Vec<String>,
    pub suite: String,
    pub dataset: String,
    pub scoring_policy: String,
    pub catalog_snapshot: String,
    pub config: Value,
    pub budget_usd: f64,
    pub seed: u64,
    pub concurrency: usize,
    pub started_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
    pub status: RunStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Planned,
    Running,
    Complete,
    Incomplete,
    StoppedBudget,
    StoppedError,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallResult {
    pub run_id: String,
    pub case_id: String,
    pub scenario: String,
    pub model_id: String,
    pub requested_model_id: String,
    pub protocol: String,
    pub prediction: Option<Value>,
    pub valid_output: bool,
    pub format_compliant: Option<bool>,
    pub latency_ms: f64,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cost_usd: Option<f64>,
    pub estimated_retry_cost_usd: f64,
    pub repeat: u32,
    pub timestamp: DateTime<Utc>,
    pub retry_count: u32,
    pub error: Option<String>,
}

impl CallResult {
    pub fn work_unit(&self) -> WorkUnit {
        WorkUnit {
            model_id: self.requested_model_id.clone(),
            case_id: self.case_id.clone(),
            repeat: self.repeat,
        }
    }

    pub fn successful(&self) -> bool {
        self.error.is_none() && self.prediction.is_some()
    }
}

#[derive(Debug, Clone)]
pub struct StoredRun {
    pub manifest: RunManifest,
    pub results: Vec<CallResult>,
}
