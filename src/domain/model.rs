use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCandidate {
    pub id: String,
    pub name: String,
    pub creator: String,
    pub created: Option<i64>,
    pub input_modalities: Vec<String>,
    pub output_modalities: Vec<String>,
    pub context_length: u64,
    pub supported_parameters: Vec<String>,
    pub prompt_usd_per_million: Option<f64>,
    pub completion_usd_per_million: Option<f64>,
    #[serde(default)]
    pub exclusion_reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogSnapshot {
    pub source: String,
    pub fetched_at: DateTime<Utc>,
    pub models: Vec<ModelCandidate>,
    pub raw_json: serde_json::Value,
}
