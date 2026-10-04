use crate::{
    application::catalog::CatalogQuery,
    domain::model::{CatalogSnapshot, ModelCandidate},
    ports::{AdapterError, ModelCatalog},
};
use async_trait::async_trait;
use chrono::Utc;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};
use serde_json::Value;

pub struct OpenRouterCatalog {
    client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
}

impl OpenRouterCatalog {
    pub fn new(base_url: &str, api_key: Option<String>) -> Result<Self, AdapterError> {
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(45))
            .build()
            .map_err(|_| AdapterError::Http("could not initialize HTTP client".into()))?;
        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_owned(),
            api_key,
        })
    }

    fn normalize(raw: &Value) -> Result<ModelCandidate, AdapterError> {
        let id = raw
            .get("id")
            .and_then(Value::as_str)
            .ok_or(AdapterError::InvalidData)?
            .to_owned();
        let architecture = raw.get("architecture").unwrap_or(&Value::Null);
        let pricing = raw.get("pricing").unwrap_or(&Value::Null);
        let modalities = |key: &str| {
            architecture
                .get(key)
                .and_then(Value::as_array)
                .map(|xs| {
                    xs.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default()
        };
        Ok(ModelCandidate {
            creator: id.split('/').next().unwrap_or("unknown").to_owned(),
            name: raw
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or(&id)
                .to_owned(),
            created: raw.get("created").and_then(Value::as_i64),
            input_modalities: modalities("input_modalities"),
            output_modalities: modalities("output_modalities"),
            context_length: raw
                .get("context_length")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            supported_parameters: raw
                .get("supported_parameters")
                .and_then(Value::as_array)
                .map(|xs| {
                    xs.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default(),
            prompt_usd_per_million: parse_price(pricing.get("prompt")),
            completion_usd_per_million: parse_price(pricing.get("completion")),
            exclusion_reasons: vec![],
            id,
        })
    }
}

fn parse_price(value: Option<&Value>) -> Option<f64> {
    let token_price = match value? {
        Value::String(s) => s.parse::<f64>().ok()?,
        Value::Number(n) => n.as_f64()?,
        _ => return None,
    };
    if token_price.is_finite() && token_price >= 0.0 {
        Some(token_price * 1_000_000.0)
    } else {
        None
    }
}

#[async_trait]
impl ModelCatalog for OpenRouterCatalog {
    async fn list_models(&self, query: &CatalogQuery) -> Result<CatalogSnapshot, AdapterError> {
        let mut headers = HeaderMap::new();
        headers.insert(
            "HTTP-Referer",
            HeaderValue::from_static("https://evolving-benchmark.local"),
        );
        headers.insert("X-Title", HeaderValue::from_static("Evolving Benchmark"));
        if let Some(key) = &self.api_key {
            let value = HeaderValue::from_str(&format!("Bearer {key}"))
                .map_err(|_| AdapterError::Http("invalid API key header".into()))?;
            headers.insert(AUTHORIZATION, value);
        }
        let mut request = self
            .client
            .get(format!("{}/models", self.base_url))
            .query(&[("output_modalities", "all"), ("sort", "newest")]);
        if !query.providers.is_empty() {
            request = request.query(&[("providers", query.providers.join(","))]);
        }
        let response = request
            .headers(headers)
            .send()
            .await
            .map_err(|_| AdapterError::Http("catalog request failed".into()))?;
        if !response.status().is_success() {
            return Err(AdapterError::Http(format!(
                "catalog returned HTTP {}",
                response.status()
            )));
        }
        let raw: Value = response
            .json()
            .await
            .map_err(|_| AdapterError::InvalidData)?;
        let data = raw
            .get("data")
            .and_then(Value::as_array)
            .ok_or(AdapterError::InvalidData)?;
        let models = data
            .iter()
            .map(Self::normalize)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(CatalogSnapshot {
            source: "openrouter".into(),
            fetched_at: Utc::now(),
            models,
            raw_json: raw,
        })
    }
}
