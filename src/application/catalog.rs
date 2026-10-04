use crate::{
    config::CatalogFilters,
    domain::model::{CatalogSnapshot, ModelCandidate},
};
use chrono::{DateTime, Duration, Utc};

#[derive(Debug, Clone, Default)]
pub struct CatalogQuery {
    pub providers: Vec<String>,
}

impl CatalogQuery {
    pub fn from_filters(filters: &CatalogFilters) -> Self {
        Self {
            providers: filters.providers.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CandidateDecision {
    pub included: bool,
    pub reasons: Vec<String>,
}

pub struct CatalogPolicy;

impl CatalogPolicy {
    pub fn evaluate(
        model: &ModelCandidate,
        filters: &CatalogFilters,
        now: DateTime<Utc>,
    ) -> CandidateDecision {
        let mut reasons = Vec::new();
        if filters.exclude.iter().any(|id| id == &model.id) {
            return CandidateDecision {
                included: false,
                reasons: vec!["explicitly excluded".into()],
            };
        }
        let explicit = filters.include.iter().any(|id| id == &model.id);
        if !filters.creators.is_empty()
            && !filters.creators.iter().any(|c| c == &model.creator)
            && !explicit
        {
            reasons.push("creator does not match configured filter".into());
        }
        if !model
            .input_modalities
            .iter()
            .any(|m| m == &filters.required_input_modality)
        {
            reasons.push(format!(
                "missing {} input modality",
                filters.required_input_modality
            ));
        }
        if !model
            .output_modalities
            .iter()
            .any(|m| m == &filters.required_output_modality)
        {
            reasons.push(format!(
                "missing {} output modality",
                filters.required_output_modality
            ));
        }
        if filters.require_structured_output
            && !model
                .supported_parameters
                .iter()
                .any(|p| p == "response_format" || p == "structured_outputs")
        {
            reasons.push("missing structured output support".into());
        }
        for required in &filters.required_parameters {
            if !model.supported_parameters.contains(required) {
                reasons.push(format!("missing required parameter {required}"));
            }
        }
        if !explicit {
            if model.context_length < filters.min_context_tokens {
                reasons.push("context below minimum".into());
            }
            match model.prompt_usd_per_million {
                Some(price) if price <= filters.max_prompt_usd_per_million => (),
                Some(_) => reasons.push("prompt price above maximum".into()),
                None => reasons.push("prompt price unavailable".into()),
            }
            match model.completion_usd_per_million {
                Some(price) if price <= filters.max_completion_usd_per_million => (),
                Some(_) => reasons.push("completion price above maximum".into()),
                None => reasons.push("completion price unavailable".into()),
            }
            if filters.newer_than_days > 0 {
                let cutoff = now - Duration::days(filters.newer_than_days as i64);
                if model
                    .created
                    .and_then(|created| DateTime::<Utc>::from_timestamp(created, 0))
                    .is_none_or(|created| created < cutoff)
                {
                    reasons
                        .push("model creation date is missing or outside freshness window".into());
                }
            }
        }
        CandidateDecision {
            included: reasons.is_empty(),
            reasons,
        }
    }

    pub fn select(
        snapshot: &CatalogSnapshot,
        filters: &CatalogFilters,
        now: DateTime<Utc>,
    ) -> Vec<(ModelCandidate, CandidateDecision)> {
        snapshot
            .models
            .iter()
            .cloned()
            .map(|model| {
                let decision = Self::evaluate(&model, filters, now);
                (model, decision)
            })
            .collect()
    }
}
