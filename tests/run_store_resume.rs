use chrono::Utc;
use evolving_benchmark::{
    adapters::filesystem::FilesystemRunStore,
    domain::{
        model::CatalogSnapshot,
        run::{CallResult, RunManifest, RunStatus},
    },
    ports::RunStore,
};
use serde_json::json;
use std::io::Write;
use tempfile::tempdir;

fn manifest() -> RunManifest {
    RunManifest {
        run_id: "run-test".into(),
        selected_models: vec!["v/m".into()],
        suite: "decision/full".into(),
        dataset: "v1".into(),
        scoring_policy: "decision-v1".into(),
        catalog_snapshot: "catalog.json".into(),
        config: json!({"repeats":1}),
        budget_usd: 1.0,
        seed: 7,
        concurrency: 1,
        started_at: Utc::now(),
        finished_at: None,
        status: RunStatus::Running,
    }
}
fn result() -> CallResult {
    CallResult {
        run_id: "run-test".into(),
        case_id: "TS-001".into(),
        scenario: "tool_selector".into(),
        model_id: "v/m".into(),
        requested_model_id: "v/m".into(),
        protocol: "chat_completions".into(),
        prediction: Some(json!({"tool":"none"})),
        valid_output: true,
        format_compliant: Some(true),
        latency_ms: 12.0,
        input_tokens: Some(10),
        output_tokens: Some(2),
        cost_usd: Some(0.001),
        estimated_retry_cost_usd: 0.0,
        repeat: 1,
        timestamp: Utc::now(),
        retry_count: 0,
        error: None,
    }
}

#[tokio::test]
async fn run_store_roundtrips_manifest_and_results() {
    let tmp = tempdir().unwrap();
    let store = FilesystemRunStore::new(tmp.path());
    store.create(&manifest()).await.unwrap();
    store.append("run-test", &result()).await.unwrap();
    let loaded = store.load("run-test").await.unwrap();
    assert_eq!(loaded.results.len(), 1);
    assert_eq!(loaded.manifest.status, RunStatus::Running);
}

#[tokio::test]
async fn ignores_only_a_truncated_final_jsonl_record() {
    let tmp = tempdir().unwrap();
    let store = FilesystemRunStore::new(tmp.path());
    store.create(&manifest()).await.unwrap();
    store.append("run-test", &result()).await.unwrap();
    std::fs::OpenOptions::new()
        .append(true)
        .open(tmp.path().join("run-test/results.jsonl"))
        .unwrap()
        .write_all(b"{partial")
        .unwrap();
    let loaded = store.load("run-test").await.unwrap();
    assert_eq!(loaded.results.len(), 1);
}

#[tokio::test]
async fn catalog_snapshot_is_saved_with_a_run() {
    let tmp = tempdir().unwrap();
    let store = FilesystemRunStore::new(tmp.path());
    store.create(&manifest()).await.unwrap();
    let snapshot = CatalogSnapshot {
        source: "mock".into(),
        fetched_at: Utc::now(),
        models: vec![],
        raw_json: json!({"data":[]}),
    };
    store.save_catalog("run-test", &snapshot).await.unwrap();
    let loaded = store.catalog("run-test").await.unwrap();
    assert_eq!(loaded.raw_json["data"].as_array().unwrap().len(), 0);
}
