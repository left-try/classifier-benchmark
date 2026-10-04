use crate::{
    domain::{
        run::{RunStatus, StoredRun},
        scoring::ScorePolicy,
        suite::TestSuite,
    },
    ports::StoreError,
};
use std::collections::{BTreeSet, HashMap, HashSet};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ReportError {
    #[error("run manifest contains invalid metrics")]
    Invalid,
    #[error("run storage error: {0}")]
    Store(#[from] StoreError),
}

#[derive(Debug, Clone)]
pub struct ModelMetrics {
    pub rank: usize,
    pub model_id: String,
    pub resolved_model_ids: Vec<String>,
    pub accuracy: f64,
    pub valid_output: f64,
    pub format_compliance: Option<f64>,
    pub mean_cost_usd: Option<f64>,
    pub mean_latency_ms: f64,
    pub score: Option<f64>,
    pub calls: usize,
    pub expected_calls: usize,
    pub complete: bool,
}

#[derive(Debug, Clone)]
pub struct ScenarioMetrics {
    pub model_id: String,
    pub scenario: String,
    pub accuracy: f64,
    pub mean_cost_usd: Option<f64>,
    pub mean_latency_ms: f64,
    pub calls: usize,
}

#[derive(Debug, Clone)]
pub struct Report {
    pub run_id: String,
    pub dataset: String,
    pub scoring_policy: String,
    pub status: RunStatus,
    pub rankings_available: bool,
    pub total_cost_usd: f64,
    pub missing_cost_calls: usize,
    pub models: Vec<ModelMetrics>,
    pub scenarios: Vec<ScenarioMetrics>,
    pub failures: Vec<String>,
}

impl Report {
    pub fn from_run(run: &StoredRun, suite: &TestSuite) -> Result<Self, ReportError> {
        let repeats = run
            .manifest
            .config
            .get("repeats")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(1) as usize;
        let expected_calls = suite.cases.len() * repeats;
        let mut ids = run.manifest.selected_models.clone();
        if ids.is_empty() {
            ids = run
                .results
                .iter()
                .map(|r| r.requested_model_id.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
        }
        let mut models = Vec::new();
        let mut all_complete = !ids.is_empty();
        let mut every_cost_known = true;
        let mut total_cost = 0.0;
        let mut missing_cost = 0;
        let mut failures = Vec::new();
        for id in ids {
            let results: Vec<_> = run
                .results
                .iter()
                .filter(|r| r.requested_model_id == id)
                .collect();
            let mut latest: HashMap<String, &crate::domain::run::CallResult> = HashMap::new();
            for result in &results {
                latest.insert(format!("{}/{}", result.case_id, result.repeat), result);
            }
            let successful: HashSet<String> = latest
                .iter()
                .filter(|(_, r)| r.successful() && r.cost_usd.is_some())
                .map(|(key, _)| key.clone())
                .collect();
            let expected_work: HashSet<String> = suite
                .cases
                .iter()
                .flat_map(|case| {
                    (1..=repeats).map(move |repeat| format!("{}/{}", case.case_id, repeat))
                })
                .collect();
            let complete = successful == expected_work;
            all_complete &= complete;
            let calls = latest.len();
            if !complete {
                failures.push(format!(
                    "{id}: incomplete coverage ({calls}/{expected_calls} calls)"
                ));
            }
            let mut correct = 0;
            let mut valid = 0;
            let mut latency = 0.0;
            let mut prices = Vec::new();
            let mut formats = Vec::new();
            for result in &results {
                if let Some(value) = result.cost_usd {
                    total_cost += value + result.estimated_retry_cost_usd;
                } else {
                    missing_cost += 1;
                    every_cost_known = false;
                }
                if let Some(error) = &result.error {
                    failures.push(format!("{} {}: {error}", id, result.case_id));
                }
            }
            for result in latest.values() {
                if result.valid_output {
                    valid += 1;
                }
                latency += result.latency_ms;
                if let Some(value) = result.cost_usd {
                    let accounted = value + result.estimated_retry_cost_usd;
                    prices.push(accounted);
                }
                if let Some(value) = result.format_compliant {
                    formats.push(if value { 1.0 } else { 0.0 });
                }
                if let (Some(case), Some(prediction)) = (
                    suite.cases.iter().find(|c| c.case_id == result.case_id),
                    result.prediction.as_ref(),
                ) {
                    if *prediction == case.expected {
                        correct += 1;
                    }
                }
            }
            let accuracy = if expected_calls == 0 {
                0.0
            } else {
                (correct as f64 / expected_calls as f64).min(1.0)
            };
            let valid_output = if calls == 0 {
                0.0
            } else {
                valid as f64 / calls as f64
            };
            let mean_cost_usd =
                (!prices.is_empty()).then(|| prices.iter().sum::<f64>() / prices.len() as f64);
            let mean_latency_ms = if calls == 0 {
                0.0
            } else {
                latency / calls as f64
            };
            let score = if let Some(cost) = mean_cost_usd {
                Some(
                    ScorePolicy::decision_v1()
                        .score(accuracy, cost, mean_latency_ms)
                        .map_err(|_| ReportError::Invalid)?
                        .score,
                )
            } else {
                None
            };
            let resolved_model_ids = results
                .iter()
                .map(|result| result.model_id.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            models.push(ModelMetrics {
                rank: 0,
                model_id: id,
                resolved_model_ids,
                accuracy,
                valid_output,
                format_compliance: (!formats.is_empty())
                    .then(|| formats.iter().sum::<f64>() / formats.len() as f64),
                mean_cost_usd,
                mean_latency_ms,
                score,
                calls,
                expected_calls,
                complete,
            });
        }
        let rankings_available =
            all_complete && every_cost_known && run.manifest.status == RunStatus::Complete;
        models.sort_by(|a, b| b.score.unwrap_or(-1.0).total_cmp(&a.score.unwrap_or(-1.0)));
        if rankings_available {
            for (index, model) in models.iter_mut().enumerate() {
                model.rank = index + 1;
            }
        }
        let scenarios = scenario_metrics(run, suite);
        Ok(Self {
            run_id: run.manifest.run_id.clone(),
            dataset: suite.dataset.clone(),
            scoring_policy: suite.scoring_policy.clone(),
            status: run.manifest.status,
            rankings_available,
            total_cost_usd: total_cost,
            missing_cost_calls: missing_cost,
            models,
            scenarios,
            failures,
        })
    }
}

fn scenario_metrics(run: &StoredRun, suite: &TestSuite) -> Vec<ScenarioMetrics> {
    let mut groups: HashMap<(String, String), Vec<_>> = HashMap::new();
    for result in &run.results {
        groups
            .entry((result.requested_model_id.clone(), result.scenario.clone()))
            .or_default()
            .push(result);
    }
    let mut out = Vec::new();
    for ((model_id, scenario), rows) in groups {
        let cases: Vec<_> = suite
            .cases
            .iter()
            .filter(|c| c.scenario == scenario)
            .collect();
        let mut latest = HashMap::new();
        for row in &rows {
            latest.insert(format!("{}/{}", row.case_id, row.repeat), *row);
        }
        let mut correct = 0;
        let mut latency = 0.0;
        let mut costs = Vec::new();
        for row in latest.values() {
            latency += row.latency_ms;
            if let Some(cost) = row.cost_usd {
                costs.push(cost + row.estimated_retry_cost_usd);
            }
            if let (Some(case), Some(prediction)) = (
                cases.iter().find(|c| c.case_id == row.case_id),
                row.prediction.as_ref(),
            ) {
                if *prediction == case.expected {
                    correct += 1;
                }
            }
        }
        out.push(ScenarioMetrics {
            model_id,
            scenario,
            accuracy: if latest.is_empty() {
                0.0
            } else {
                correct as f64 / latest.len() as f64
            },
            mean_cost_usd: (!costs.is_empty())
                .then(|| costs.iter().sum::<f64>() / costs.len() as f64),
            mean_latency_ms: if latest.is_empty() {
                0.0
            } else {
                latency / latest.len() as f64
            },
            calls: latest.len(),
        });
    }
    out.sort_by(|a, b| {
        a.scenario
            .cmp(&b.scenario)
            .then(a.model_id.cmp(&b.model_id))
    });
    out
}
