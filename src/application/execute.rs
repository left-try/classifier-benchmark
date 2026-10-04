use crate::{
    application::{
        budget::{BudgetLedger, CostEstimator, Money},
        plan::{PlanError, RunPlan},
    },
    domain::{
        model::CatalogSnapshot,
        run::{CallResult, RunManifest, RunStatus, StoredRun},
        suite::{TestSuite, WorkUnit},
    },
    ports::{CallError, ModelRequest, ModelRunner, RunStore, StoreError},
};
use chrono::Utc;
use futures::future::join_all;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ExecuteError {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error(transparent)]
    Plan(#[from] PlanError),
    #[error("run could not continue: {0}")]
    Invalid(String),
}

#[derive(Debug, Clone)]
pub struct RunSummary {
    pub run_id: String,
    pub status: RunStatus,
    pub completed: usize,
    pub total: usize,
    pub spent_usd: f64,
}

pub struct Executor;

impl Executor {
    pub async fn run(
        plan: RunPlan,
        suite: &TestSuite,
        snapshot: &CatalogSnapshot,
        runner: &dyn ModelRunner,
        store: &dyn RunStore,
        effective_config: Value,
    ) -> Result<RunSummary, ExecuteError> {
        let run_id = uuid::Uuid::new_v4().simple().to_string();
        let manifest = RunManifest {
            run_id: run_id.clone(),
            selected_models: plan.models.iter().map(|m| m.id.clone()).collect(),
            suite: "decision/full".into(),
            dataset: suite.dataset.clone(),
            scoring_policy: suite.scoring_policy.clone(),
            catalog_snapshot: "catalog.json".into(),
            config: effective_config,
            budget_usd: plan.budget.as_usd(),
            seed: plan.seed,
            concurrency: plan.concurrency,
            started_at: Utc::now(),
            finished_at: None,
            status: RunStatus::Planned,
        };
        store.create(&manifest).await?;
        store.save_catalog(&run_id, snapshot).await?;
        Self::execute_existing(plan, suite, runner, store, store.load(&run_id).await?).await
    }

    pub async fn resume(
        run: StoredRun,
        plan: RunPlan,
        suite: &TestSuite,
        runner: &dyn ModelRunner,
        store: &dyn RunStore,
    ) -> Result<RunSummary, ExecuteError> {
        Self::execute_existing(plan, suite, runner, store, run).await
    }

    async fn execute_existing(
        plan: RunPlan,
        suite: &TestSuite,
        runner: &dyn ModelRunner,
        store: &dyn RunStore,
        mut run: StoredRun,
    ) -> Result<RunSummary, ExecuteError> {
        let prior_spend = run
            .results
            .iter()
            .filter_map(|r| r.cost_usd)
            .try_fold(Money(0), |sum, dollars| {
                Money::from_usd_ceil(dollars).and_then(|money| sum.checked_add(money))
            })
            .map_err(|e| ExecuteError::Invalid(e.to_string()))?;
        if run.results.iter().any(|r| r.cost_usd.is_none()) {
            return Err(ExecuteError::Invalid(
                "previous calls contain unknown cost; refusing to spend further".into(),
            ));
        }
        let ledger = BudgetLedger::with_existing(plan.budget, prior_spend, false);
        let completed: HashSet<String> = run
            .results
            .iter()
            .filter(|r| r.successful() && r.cost_usd.is_some())
            .map(|r| r.work_unit().identity())
            .collect();
        let expected_work: HashSet<String> =
            plan.work_units.iter().map(WorkUnit::identity).collect();
        let mut pending: Vec<WorkUnit> = plan
            .work_units
            .into_iter()
            .filter(|unit| !completed.contains(&unit.identity()))
            .collect();
        deterministic_shuffle(&mut pending, plan.seed);
        store
            .set_status(&run.manifest.run_id, RunStatus::Running)
            .await?;
        run.manifest.status = RunStatus::Running;

        let models: HashMap<String, _> =
            plan.models.into_iter().map(|m| (m.id.clone(), m)).collect();
        let cases: HashMap<String, _> =
            suite.cases.iter().map(|c| (c.case_id.clone(), c)).collect();
        let schemas: HashMap<String, Value> = suite
            .scenarios()
            .into_iter()
            .map(|scenario| (scenario.clone(), suite.output_schema(&scenario)))
            .collect();
        let mut stopped_for_budget = false;
        let mut stopped_for_missing_cost = false;
        for batch in pending.chunks(plan.concurrency.max(1)) {
            let mut reserved = Vec::new();
            for unit in batch {
                let model = models
                    .get(&unit.model_id)
                    .ok_or_else(|| ExecuteError::Invalid("planned model is missing".into()))?;
                let input_bound = *plan
                    .input_token_bounds
                    .get(&unit.case_id)
                    .ok_or_else(|| ExecuteError::Invalid("case has no input token bound".into()))?;
                let estimate =
                    CostEstimator::estimate(model, input_bound, plan.max_output_tokens as u64)
                        .map_err(|e| ExecuteError::Invalid(e.to_string()))?
                        .checked_mul(plan.max_retries as u64 + 1)
                        .map_err(|e| ExecuteError::Invalid(e.to_string()))?;
                match ledger.reserve(&unit.identity(), estimate) {
                    Ok(id) => reserved.push((unit.clone(), model.clone(), id)),
                    Err(_) => {
                        stopped_for_budget = true;
                        break;
                    }
                }
            }
            if stopped_for_budget {
                for (_, _, id) in reserved {
                    ledger
                        .reconcile(id, Some(Money(0)))
                        .map_err(|e| ExecuteError::Invalid(e.to_string()))?;
                }
                break;
            }
            let futures = reserved.iter().map(|(unit, model, _)| {
                let case = cases.get(&unit.case_id).expect("suite work unit refers to loaded case");
                let input = case.input.clone();
                let protocol = plan.protocol_overrides.get(&model.id).cloned().unwrap_or_else(|| "chat_completions".into());
                let request_bytes = serde_json::to_vec(&json!({"scenario":case.scenario,"input":input,"output_schema":schemas[&case.scenario]}))
                    .map(|bytes| bytes.len().saturating_add(384)).unwrap_or(usize::MAX);
                async move {
                    if request_bytes > plan.max_input_bytes {
                        return Err(CallError::Permanent("request exceeds configured max_input_bytes".into()));
                    }
                    runner.execute(&ModelRequest {
                        run_id: run.manifest.run_id.clone(), unit: unit.clone(), model: model.clone(), protocol,
                        scenario: case.scenario.clone(), input: case.input.clone(), output_schema: schemas[&case.scenario].clone(),
                        max_output_tokens: plan.max_output_tokens,
                    }).await
                }
            });
            let results = join_all(futures).await;
            for ((unit, model, reservation), result) in reserved.into_iter().zip(results) {
                let call = match result {
                    Ok(mut call) => {
                        let per_attempt = CostEstimator::estimate(
                            &model,
                            plan.input_token_bounds[&unit.case_id],
                            plan.max_output_tokens as u64,
                        )
                        .map_err(|e| ExecuteError::Invalid(e.to_string()))?;
                        call.estimated_retry_cost_usd =
                            per_attempt.as_usd() * call.retry_count as f64;
                        call
                    }
                    Err(error) => CallResult {
                        run_id: run.manifest.run_id.clone(),
                        case_id: unit.case_id.clone(),
                        scenario: cases[&unit.case_id].scenario.clone(),
                        model_id: model.id.clone(),
                        requested_model_id: model.id.clone(),
                        protocol: plan
                            .protocol_overrides
                            .get(&model.id)
                            .cloned()
                            .unwrap_or_else(|| "chat_completions".into()),
                        prediction: None,
                        valid_output: false,
                        format_compliant: None,
                        latency_ms: 0.0,
                        input_tokens: None,
                        output_tokens: None,
                        cost_usd: None,
                        estimated_retry_cost_usd: 0.0,
                        repeat: unit.repeat,
                        timestamp: Utc::now(),
                        retry_count: 0,
                        error: Some(error.to_string()),
                    },
                };
                let actual = call
                    .cost_usd
                    .and_then(|v| Money::from_usd_ceil(v + call.estimated_retry_cost_usd).ok());
                if actual.is_none() {
                    stopped_for_missing_cost = true;
                }
                ledger
                    .reconcile(reservation, actual)
                    .map_err(|e| ExecuteError::Invalid(e.to_string()))?;
                store.append(&run.manifest.run_id, &call).await?;
                run.results.push(call);
            }
            if stopped_for_missing_cost || !ledger.may_schedule() {
                stopped_for_budget = !stopped_for_missing_cost;
                break;
            }
            if plan.model_spacing_ms > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(plan.model_spacing_ms)).await;
            }
        }
        let successful: HashSet<String> = run
            .results
            .iter()
            .filter(|r| r.successful() && r.cost_usd.is_some())
            .map(|r| r.work_unit().identity())
            .collect();
        let total = expected_work.len();
        let completed = expected_work.intersection(&successful).count();
        let complete = completed == total && !stopped_for_missing_cost;
        let status = if ledger.over_budget() {
            RunStatus::StoppedBudget
        } else if complete {
            RunStatus::Complete
        } else if stopped_for_missing_cost {
            RunStatus::StoppedError
        } else if stopped_for_budget {
            RunStatus::StoppedBudget
        } else {
            RunStatus::Incomplete
        };
        store.set_status(&run.manifest.run_id, status).await?;
        Ok(RunSummary {
            run_id: run.manifest.run_id,
            status,
            completed,
            total,
            spent_usd: ledger.spent().as_usd(),
        })
    }
}

fn deterministic_shuffle<T>(values: &mut [T], mut seed: u64) {
    for i in (1..values.len()).rev() {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        values.swap(i, (seed as usize) % (i + 1));
    }
}
