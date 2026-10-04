use chrono::Utc;
use evolving_benchmark::{
    application::catalog::CatalogPolicy, config::CatalogFilters, domain::model::ModelCandidate,
};

fn model() -> ModelCandidate {
    ModelCandidate {
        id: "vendor/model".into(),
        name: "Model".into(),
        creator: "vendor".into(),
        created: Some(Utc::now().timestamp()),
        input_modalities: vec!["text".into()],
        output_modalities: vec!["text".into()],
        context_length: 8192,
        supported_parameters: vec!["response_format".into()],
        prompt_usd_per_million: Some(1.0),
        completion_usd_per_million: Some(2.0),
        exclusion_reasons: vec![],
    }
}

#[test]
fn filter_returns_explanation_for_incompatible_models() {
    let mut candidate = model();
    candidate.supported_parameters.clear();
    let decision = CatalogPolicy::evaluate(&candidate, &CatalogFilters::default(), Utc::now());
    assert!(!decision.included);
    assert!(decision
        .reasons
        .iter()
        .any(|reason| reason.contains("structured")));
}

#[test]
fn explicit_exclusion_wins_and_zero_prices_are_valid() {
    let mut candidate = model();
    candidate.prompt_usd_per_million = Some(0.0);
    let included = CatalogPolicy::evaluate(&candidate, &CatalogFilters::default(), Utc::now());
    assert!(included.included);
    let mut filters = CatalogFilters::default();
    filters.exclude.push(candidate.id.clone());
    assert!(!CatalogPolicy::evaluate(&candidate, &filters, Utc::now()).included);
}
