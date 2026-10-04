use crate::{
    application::catalog::CatalogQuery,
    domain::{
        model::{CatalogSnapshot, ModelCandidate},
        run::{CallResult, RunManifest, RunStatus, StoredRun},
        suite::WorkUnit,
    },
};
use async_trait::async_trait;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AdapterError {
    #[error("OpenRouter request failed: {0}")]
    Http(String),
    #[error("provider returned invalid data")]
    InvalidData,
}

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("run data could not be read or written")]
    Io,
    #[error("run data is invalid")]
    Invalid,
    #[error("run id was not found")]
    NotFound,
}

#[async_trait]
pub trait ModelCatalog: Send + Sync {
    async fn list_models(&self, query: &CatalogQuery) -> Result<CatalogSnapshot, AdapterError>;
}

#[async_trait]
pub trait ModelRunner: Send + Sync {
    async fn execute(&self, request: &ModelRequest) -> Result<CallResult, CallError>;
}

#[async_trait]
pub trait RunStore: Send + Sync {
    async fn create(&self, manifest: &RunManifest) -> Result<(), StoreError>;
    async fn append(&self, run_id: &str, result: &CallResult) -> Result<(), StoreError>;
    async fn load(&self, run_id: &str) -> Result<StoredRun, StoreError>;
    async fn set_status(&self, run_id: &str, status: RunStatus) -> Result<(), StoreError>;
    async fn save_catalog(
        &self,
        run_id: &str,
        snapshot: &CatalogSnapshot,
    ) -> Result<(), StoreError>;
    async fn catalog(&self, run_id: &str) -> Result<CatalogSnapshot, StoreError>;
}

#[derive(Debug, Clone)]
pub struct ModelRequest {
    pub run_id: String,
    pub unit: WorkUnit,
    pub model: ModelCandidate,
    pub protocol: String,
    pub scenario: String,
    pub input: serde_json::Value,
    pub output_schema: serde_json::Value,
    pub max_output_tokens: u32,
}

#[derive(Debug, Error)]
pub enum CallError {
    #[error("transient provider error: {0}")]
    Transient(String),
    #[error("provider rejected request: {0}")]
    Permanent(String),
}

pub fn load_api_key() -> Option<String> {
    if let Ok(value) = std::env::var("OPENROUTER_API_KEY") {
        if !value.trim().is_empty() {
            return Some(value);
        }
    }
    let content = std::fs::read_to_string(Path::new(".env")).ok()?;
    content.lines().find_map(|line| {
        let (name, value) = line.split_once('=')?;
        if name.trim() != "OPENROUTER_API_KEY" {
            return None;
        }
        let value = value
            .trim()
            .trim_matches(|character| character == '\'' || character == '"');
        (!value.is_empty()).then(|| value.to_owned())
    })
}
