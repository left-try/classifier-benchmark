use async_trait::async_trait;
use chrono::Utc;
use evolving_benchmark::{
    adapters::filesystem::FilesystemRunStore,
    application::{execute::Executor, plan::RunPlan},
    config::{AppConfig, CatalogFilters},
    domain::{
        model::{CatalogSnapshot, ModelCandidate},
        run::CallResult,
        suite::TestSuite,
    },
    ports::{CallError, ModelRequest, ModelRunner, RunStore},
};
use serde_json::json;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use tempfile::tempdir;

struct FakeRunner {
    calls: Arc<AtomicUsize>,
    include_cost: bool,
}
#[async_trait]
impl ModelRunner for FakeRunner {
    async fn execute(&self, request: &ModelRequest) -> Result<CallResult, CallError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(CallResult {
            run_id: request.run_id.clone(),
            case_id: request.unit.case_id.clone(),
            scenario: request.scenario.clone(),
            model_id: request.model.id.clone(),
            requested_model_id: request.model.id.clone(),
            protocol: request.protocol.clone(),
            prediction: Some(json!({"result":"ok"})),
            valid_output: true,
            format_compliant: Some(true),
            latency_ms: 20.0,
            input_tokens: Some(10),
            output_tokens: Some(3),
            cost_usd: self.include_cost.then_some(0.0001),
            estimated_retry_cost_usd: 0.0,
            repeat: request.unit.repeat,
            timestamp: Utc::now(),
            retry_count: 0,
            error: None,
        })
    }
}

fn config(concurrency: usize) -> AppConfig {
    AppConfig {
        suite: "decision/full".into(),
        cases: "data/decision/full/v1/cases.jsonl".into(),
        output_dir: "runs".into(),
        repeats: 1,
        concurrency,
        max_output_tokens: 160,
        max_input_bytes: 16384,
        max_retries: 0,
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
        context_length: 8192,
        supported_parameters: vec!["response_format".into()],
        prompt_usd_per_million: Some(1.0),
        completion_usd_per_million: Some(2.0),
        exclusion_reasons: vec![],
    }
}
fn snapshot(candidate: &ModelCandidate) -> CatalogSnapshot {
    CatalogSnapshot {
        source: "mock".into(),
        fetched_at: Utc::now(),
        models: vec![candidate.clone()],
        raw_json: json!({"data":[]}),
    }
}

#[tokio::test]
async fn missing_cost_stops_scheduling_after_current_concurrency_batch() {
    let suite = TestSuite::load("data/decision/full/v1/suite.toml").unwrap();
    let config = config(2);
    let plan = RunPlan::build(&config, &suite, vec![model()], 1.0).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let runner = FakeRunner {
        calls: Arc::clone(&calls),
        include_cost: false,
    };
    let temp = tempdir().unwrap();
    let store = FilesystemRunStore::new(temp.path());
    let result = Executor::run(
        plan,
        &suite,
        &snapshot(&model()),
        &runner,
        &store,
        serde_json::to_value(config).unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(
        result.status,
        evolving_benchmark::domain::run::RunStatus::StoppedError
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn complete_run_can_be_resumed_without_duplicate_calls() {
    let suite = TestSuite::load("data/decision/full/v1/suite.toml").unwrap();
    let config = config(4);
    let plan = RunPlan::build(&config, &suite, vec![model()], 1.0).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let runner = FakeRunner {
        calls: Arc::clone(&calls),
        include_cost: true,
    };
    let temp = tempdir().unwrap();
    let store = FilesystemRunStore::new(temp.path());
    let first = Executor::run(
        plan,
        &suite,
        &snapshot(&model()),
        &runner,
        &store,
        serde_json::to_value(config.clone()).unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(first.completed, 30);
    let before = calls.load(Ordering::SeqCst);
    let loaded = store.load(&first.run_id).await.unwrap();
    let plan = RunPlan::build(&config, &suite, vec![model()], 1.0).unwrap();
    let second = Executor::resume(loaded, plan, &suite, &runner, &store)
        .await
        .unwrap();
    assert_eq!(second.completed, 30);
    assert_eq!(calls.load(Ordering::SeqCst), before);
}
