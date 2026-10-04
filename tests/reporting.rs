use chrono::Utc;
use evolving_benchmark::{
    application::report::Report,
    domain::{
        run::{CallResult, RunManifest, RunStatus, StoredRun},
        suite::TestSuite,
    },
    reporters::{csv::CsvReporter, markdown::MarkdownReporter},
};
use serde_json::json;
use std::fs;
use tempfile::tempdir;

fn run(with_cost: bool) -> StoredRun {
    let suite = TestSuite::load("data/decision/full/v1/suite.toml").unwrap();
    let results = suite
        .cases
        .iter()
        .map(|case| CallResult {
            run_id: "run-r".into(),
            case_id: case.case_id.clone(),
            scenario: case.scenario.clone(),
            model_id: "new-provider/new-model".into(),
            requested_model_id: "new-provider/new-model".into(),
            protocol: "chat_completions".into(),
            prediction: Some(case.expected.clone()),
            valid_output: true,
            format_compliant: Some(true),
            latency_ms: 100.0,
            input_tokens: Some(10),
            output_tokens: Some(4),
            cost_usd: with_cost.then_some(0.001),
            estimated_retry_cost_usd: 0.0,
            repeat: 1,
            timestamp: Utc::now(),
            retry_count: 0,
            error: None,
        })
        .collect();
    StoredRun {
        manifest: RunManifest {
            run_id: "run-r".into(),
            selected_models: vec!["new-provider/new-model".into()],
            suite: "decision/full".into(),
            dataset: "v1".into(),
            scoring_policy: "decision-v1".into(),
            catalog_snapshot: "catalog.json".into(),
            config: json!({"repeats":1}),
            budget_usd: 1.0,
            seed: 7,
            concurrency: 1,
            started_at: Utc::now(),
            finished_at: Some(Utc::now()),
            status: RunStatus::Complete,
        },
        results,
    }
}

#[test]
fn arbitrary_model_ids_report_and_rank_only_full_priced_runs() {
    let suite = TestSuite::load("data/decision/full/v1/suite.toml").unwrap();
    let report = Report::from_run(&run(true), &suite).unwrap();
    assert!(report.rankings_available);
    assert_eq!(report.models[0].model_id, "new-provider/new-model");
    assert!((report.models[0].accuracy - 1.0).abs() < 1e-12);
    let markdown = MarkdownReporter::render(&report);
    assert!(markdown.contains("new-provider/new-model"));
    assert!(markdown.contains("decision-v1"));
}

#[test]
fn missing_cost_blocks_rankings_and_csv_can_be_written() {
    let suite = TestSuite::load("data/decision/full/v1/suite.toml").unwrap();
    let report = Report::from_run(&run(false), &suite).unwrap();
    assert!(!report.rankings_available);
    let dir = tempdir().unwrap();
    CsvReporter::write(&report, dir.path()).unwrap();
    assert!(fs::read_to_string(dir.path().join("leaderboard.csv"))
        .unwrap()
        .contains("new-provider/new-model"));
    CsvReporter::write_scenarios(&report, dir.path()).unwrap();
    assert!(dir.path().join("scenario_leaderboard.csv").exists());
}

#[test]
fn report_uses_latest_attempt_for_quality_and_all_attempts_for_spend() {
    let suite = TestSuite::load("data/decision/full/v1/suite.toml").unwrap();
    let mut run = run(true);
    let first = run.results[0].clone();
    run.results[0].cost_usd = Some(0.002);
    let mut retry = first;
    retry.prediction = Some(json!({"decision":"incorrect"}));
    retry.cost_usd = Some(0.001);
    run.results.push(retry);

    let report = Report::from_run(&run, &suite).unwrap();
    assert!(report.rankings_available);
    assert_eq!(report.models[0].calls, suite.cases.len());
    assert!((report.total_cost_usd - (suite.cases.len() as f64 + 0.002)).abs() < 1e-12);
    assert!(
        (report.models[0].accuracy - (suite.cases.len() - 1) as f64 / suite.cases.len() as f64)
            .abs()
            < 1e-12
    );
}
