use chrono::Utc;
use evolving_benchmark::{
    application::{
        budget::{BudgetLedger, CostEstimator, Money},
        plan::RunPlan,
    },
    config::{AppConfig, CatalogFilters},
    domain::{model::ModelCandidate, suite::TestSuite},
};
use std::sync::Arc;

fn config() -> AppConfig {
    AppConfig {
        suite: "decision/full".into(),
        cases: "unused".into(),
        output_dir: "runs".into(),
        repeats: 1,
        concurrency: 1,
        max_output_tokens: 160,
        max_input_bytes: 16384,
        max_retries: 1,
        seed: 7,
        model_spacing_ms: 0,
        filters: CatalogFilters::default(),
        protocol_overrides: Default::default(),
    }
}
fn model() -> ModelCandidate {
    ModelCandidate {
        id: "v/m".into(),
        name: "m".into(),
        creator: "v".into(),
        created: Some(Utc::now().timestamp()),
        input_modalities: vec!["text".into()],
        output_modalities: vec!["text".into()],
        context_length: 1000,
        supported_parameters: vec!["response_format".into()],
        prompt_usd_per_million: Some(1.0),
        completion_usd_per_million: Some(2.0),
        exclusion_reasons: vec![],
    }
}

#[test]
fn plan_counts_calls_and_refuses_spend_above_budget() {
    let suite = TestSuite::load("data/decision/full/v1/suite.toml").unwrap();
    let plan = RunPlan::build(&config(), &suite, vec![model()], 5.0).unwrap();
    assert_eq!(plan.work_units.len(), 30);
    assert!(RunPlan::build(&config(), &suite, vec![model()], 0.00001).is_err());
}

#[test]
fn cost_estimator_fails_closed_without_price_and_ledger_serializes_reservations() {
    let mut candidate = model();
    candidate.prompt_usd_per_million = None;
    assert!(CostEstimator::estimate(&candidate, 100, 100).is_err());

    let ledger = Arc::new(BudgetLedger::new(Money(1_000_000)));
    let joins = (0..8)
        .map(|_| {
            let ledger = Arc::clone(&ledger);
            std::thread::spawn(move || ledger.reserve("unit", Money(200_000)).is_ok())
        })
        .collect::<Vec<_>>();
    let reservations = joins
        .into_iter()
        .map(|j| j.join().unwrap())
        .filter(|ok| *ok)
        .count();
    assert_eq!(reservations, 5);
    assert_eq!(ledger.reserved(), Money(1_000_000));
    let id = ledger.reserve("unreachable", Money(0)).unwrap_err();
    assert!(id.to_string().contains("exceed"));
}
