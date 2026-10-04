use serde::Deserialize;
use std::{collections::BTreeMap, fs, path::Path};
use thiserror::Error;

#[derive(Debug, Clone, Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct AppConfig {
    pub suite: String,
    pub cases: String,
    pub output_dir: String,
    pub repeats: u32,
    pub concurrency: usize,
    pub max_output_tokens: u32,
    pub max_input_bytes: usize,
    pub max_retries: u32,
    pub seed: u64,
    pub model_spacing_ms: u64,
    #[serde(default)]
    pub filters: CatalogFilters,
    #[serde(default)]
    pub protocol_overrides: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogFilters {
    #[serde(default = "text")]
    pub required_input_modality: String,
    #[serde(default = "text")]
    pub required_output_modality: String,
    #[serde(default = "yes")]
    pub require_structured_output: bool,
    #[serde(default)]
    pub min_context_tokens: u64,
    #[serde(default = "max_price")]
    pub max_prompt_usd_per_million: f64,
    #[serde(default = "max_price")]
    pub max_completion_usd_per_million: f64,
    #[serde(default)]
    pub newer_than_days: u64,
    #[serde(default)]
    pub creators: Vec<String>,
    #[serde(default)]
    pub providers: Vec<String>,
    #[serde(default)]
    pub required_parameters: Vec<String>,
    #[serde(default)]
    pub include: Vec<String>,
    #[serde(default)]
    pub exclude: Vec<String>,
}

fn text() -> String {
    "text".to_owned()
}
fn yes() -> bool {
    true
}
fn max_price() -> f64 {
    f64::MAX
}

impl Default for CatalogFilters {
    fn default() -> Self {
        Self {
            required_input_modality: text(),
            required_output_modality: text(),
            require_structured_output: true,
            min_context_tokens: 0,
            max_prompt_usd_per_million: f64::MAX,
            max_completion_usd_per_million: f64::MAX,
            newer_than_days: 0,
            creators: vec![],
            providers: vec![],
            required_parameters: vec![],
            include: vec![],
            exclude: vec![],
        }
    }
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("invalid configuration: {0}")]
    Parse(String),
    #[error("invalid configuration: {0}")]
    Validation(String),
    #[error("could not read configuration file")]
    Read,
}

impl AppConfig {
    pub fn from_str(value: &str) -> Result<Self, ConfigError> {
        let config: Self = toml::from_str(value).map_err(|_| {
            // Avoid echoing source text: it can contain credentials or sensitive values.
            ConfigError::Parse("TOML syntax or fields are invalid".to_owned())
        })?;
        config.validate()?;
        Ok(config)
    }

    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let source = fs::read_to_string(path).map_err(|_| ConfigError::Read)?;
        Self::from_str(&source)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.suite.trim().is_empty()
            || self.cases.trim().is_empty()
            || self.output_dir.trim().is_empty()
        {
            return Err(ConfigError::Validation(
                "suite, cases, and output_dir must be non-empty".into(),
            ));
        }
        if self.repeats == 0
            || self.concurrency == 0
            || self.max_output_tokens == 0
            || self.max_input_bytes == 0
        {
            return Err(ConfigError::Validation(
                "repeats, concurrency, and token/input limits must be positive".into(),
            ));
        }
        if !self.filters.max_prompt_usd_per_million.is_finite()
            || self.filters.max_prompt_usd_per_million < 0.0
            || !self.filters.max_completion_usd_per_million.is_finite()
            || self.filters.max_completion_usd_per_million < 0.0
        {
            return Err(ConfigError::Validation(
                "price filters must be finite and non-negative".into(),
            ));
        }
        for protocol in self.protocol_overrides.values() {
            if protocol != "chat_completions" && protocol != "openrouter_decisions" {
                return Err(ConfigError::Validation(format!(
                    "unsupported protocol override '{protocol}'"
                )));
            }
        }
        Ok(())
    }
}
