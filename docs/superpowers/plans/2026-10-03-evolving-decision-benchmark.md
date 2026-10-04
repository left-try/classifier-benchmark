# Evolving Decision Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Python-first classifier benchmark with an Evolving Benchmark Rust CLI whose first module discovers compatible Decision models through OpenRouter, runs a versioned test suite within a hard spending limit, and produces reproducible reports.

**Architecture:** Begin with one Rust Cargo package and keep domain/application logic independent of OpenRouter and filesystem details through `ModelCatalog`, `ModelRunner`, and `RunStore` ports. Add catalog discovery, plan/budget accounting, execution/resume, and reporting in that order; keep old results as historical artifacts during migration.

**Tech Stack:** Rust; Cargo; `clap` for CLI parsing, `serde`/`serde_json`/`toml` for data and config, `reqwest` with Rustls and `tokio` for asynchronous HTTP, `async-trait` for async ports, and `thiserror` for typed errors. Test dependencies: `wiremock` for a local HTTP server, `tempfile` for isolated run storage, and `assert_cmd` for binary-level CLI checks. Use standard library synchronization primitives for budget reservations unless implementation measurements justify an additional dependency.

**Spec:** `docs/superpowers/specs/2026-10-03-evolving-benchmark-design.md`

## Global Constraints

- The first benchmark module is Decision Benchmark; additional benchmark domains, provider adapters, cron, A/B testing, dashboards, and shadow traffic are out of scope.
- Preserve the current 30 labeled cases and six scenarios as the initial full Decision suite.
- Preserve the current score formula as a named and versioned scoring policy, with its inputs and constants recorded in each run manifest.
- Preserve current JSONL results, reports, charts, and other artifacts as historical outputs.
- The API key is read from `OPENROUTER_API_KEY` or an optional local `.env` file and is never serialized into run artifacts or logs.
- Every plan/run requires a finite `--budget-usd` value; unknown pricing blocks execution by default.
- Incomplete coverage or missing/invalid costs blocks ranking.
- Automated tests must run without live API calls.

## Review Focus

- Catalog entries with absent, malformed, zero, or unexpectedly high prices must be excluded or safely costed; cover them in catalog normalization and plan tests.
- A provider response may omit usage/cost, change the resolved model ID, or return malformed JSON; cover in runner adapter tests and make missing cost stop new work.
- Concurrent workers must not reserve the same remaining budget; cover with a deterministic reservation/reconciliation test.
- A process can stop during an append or leave a partial final JSONL line; cover recovery and resume behavior without repeating complete calls.
- New/unknown model IDs and empty or incomplete candidate sets must not break report generation or produce a false ranking; cover reporter and planner tests.

---

## File map

```text
Cargo.toml                         Rust package and binary metadata
src/main.rs                        CLI entry point and process exit mapping
src/cli.rs                         clap command and argument definitions
src/config.rs                      TOML configuration and validation
src/domain/model.rs                normalized model/catalog candidate types
src/domain/suite.rs                suite manifest, case types, validation
src/domain/run.rs                  run manifest, call results, run status
src/domain/scoring.rs              generic metrics and versioned scoring policy
src/application/catalog.rs         discovery and compatibility use case
src/application/plan.rs            work plan and preflight cost estimation
src/application/execute.rs         scheduling, budget reservations, run/resume
src/application/report.rs          report use case
src/ports.rs                       ModelCatalog, ModelRunner, RunStore traits
src/adapters/openrouter/catalog.rs OpenRouter model catalog HTTP adapter
src/adapters/openrouter/runner.rs  OpenRouter chat completion adapter
src/adapters/filesystem.rs         config, snapshots, JSONL run persistence
src/reporters/markdown.rs          Markdown report rendering
src/reporters/csv.rs               CSV leaderboard rendering
data/decision/full/v1/             versioned Decision suite and cases
tests/fixtures/openrouter/          catalog and response JSON fixtures
tests/                              CLI, adapter, storage, and workflow tests
README.md                           project name, setup, and CLI usage
```

Keep implementation files focused; if a listed module grows beyond one clear responsibility, split that module within its owning task. Historical Python scripts and `artifacts/` remain available during migration and are clearly labeled as previous-run outputs.

## Task 1: Rust package, CLI skeleton, and configuration

**Files:**
- Create: `Cargo.toml`
- Create: `src/main.rs`
- Create: `src/cli.rs`
- Create: `src/config.rs`
- Create: `src/error.rs`
- Create: `tests/cli_args.rs`
- Create: `examples/benchmark.toml`

**Interfaces:**
- `Cli` exposes `catalog decision`, `plan decision`, `run decision`, `resume <run-id>`, and `report <run-id>` subcommands.
- `PlanArgs` and `RunArgs` require finite positive `budget_usd: f64`; `CatalogArgs` does not require a run budget.
- `AppConfig::load(path: &Path) -> Result<AppConfig, ConfigError>` reads TOML; `AppConfig::validate() -> Result<(), ConfigError>` checks suite, limits, filters, and protocol override structure.
- Main maps typed errors to a nonzero exit code and human-readable stderr without printing credentials.

- [ ] **Step 1: Write CLI/config tests**
  - Assert each documented command parses, `plan` and `run` reject absent/nonfinite/nonpositive budgets, and malformed TOML produces a useful error.
  - Assert config serialization/redacted debug output never includes the value of `OPENROUTER_API_KEY`.
- [ ] **Step 2: Run the focused tests and verify they fail**

  Run: `cargo test --test cli_args`

  Expected: compilation/test failure because the CLI and config types do not exist.
- [ ] **Step 3: Implement the package and command/config types**
  - Set the binary/package-facing name to `evolving-benchmark`.
  - Keep credential resolution separate from serializable `AppConfig`; resolve environment/.env credentials only in the OpenRouter adapter construction task.
  - Add only dependencies required by the plan; commit `Cargo.lock` for this executable.
- [ ] **Step 4: Run focused tests and formatting**

  Run: `cargo test --test cli_args` and `cargo fmt --check`

  Expected: all CLI/config tests pass and formatting is clean.

## Task 2: Domain types and versioned Decision suite

**Files:**
- Create: `src/domain/mod.rs`
- Create: `src/domain/model.rs`
- Create: `src/domain/suite.rs`
- Create: `src/domain/run.rs`
- Create: `src/domain/scoring.rs`
- Create: `data/decision/full/v1/suite.toml`
- Create: `data/decision/full/v1/cases.jsonl`
- Create: `tests/domain_suite.rs`
- Create: `tests/domain_scoring.rs`

**Interfaces:**
- `ModelCandidate` contains ID/name/creator, creation timestamp, input/output modalities, context length, supported parameters, normalized `prompt_usd_per_million` and `completion_usd_per_million` prices, and exclusion diagnostics.
- `DecisionCase` contains `case_id`, `scenario`, `input: serde_json::Value`, and `expected: serde_json::Value`.
- `TestSuite::load(manifest_path: &Path) -> Result<TestSuite, SuiteError>` loads and validates cases; `TestSuite::work_units(model_ids: &[String], repeats: u32) -> Vec<WorkUnit>` uses stable `(model_id, case_id, repeat)` identity.
- `CallResult` records run/suite/dataset identity, requested/resolved model IDs, protocol, prediction, format status, latency, usage/cost, timestamp, retry details, and structured error.
- `ScorePolicy::decision_v1()` implements the existing formula: `cost_factor = 1 / (1 + sqrt(C / 0.001))`; `latency_factor = 1 / (1 + (max(0, L_ms - 500) / 1000)^2)`; `score = accuracy * cost_factor * latency_factor`.

- [ ] **Step 1: Add suite and scoring tests**
  - Verify the migrated suite has 30 unique case IDs, all six expected scenario names, valid JSON object expectations, and stable work-unit IDs.
  - Verify Decision v1 score fixtures match the current documented formula and invalid negative costs are rejected.
- [ ] **Step 2: Run the focused tests and verify they fail**

  Run: `cargo test --test domain_suite --test domain_scoring`

  Expected: failure because the domain and suite are not implemented.
- [ ] **Step 3: Migrate the existing cases and implement domain validation/scoring**
  - Copy the current `data/cases.jsonl` records without changing IDs, inputs, or expected values into `data/decision/full/v1/cases.jsonl`.
  - Declare suite/dataset version, scenario names, and scoring policy identifier in `suite.toml` or a run-level scoring manifest.
  - Reject duplicate case IDs, missing expected labels, malformed JSON, and unsupported suite schema versions.
- [ ] **Step 4: Run the focused domain tests**

  Run: `cargo test --test domain_suite --test domain_scoring`

  Expected: all migration, validation, work-unit, and scoring tests pass.

## Task 3: OpenRouter catalog port, adapter, and candidate policy

**Files:**
- Create: `src/ports.rs`
- Create: `src/application/catalog.rs`
- Create: `src/adapters/mod.rs`
- Create: `src/adapters/openrouter/mod.rs`
- Create: `src/adapters/openrouter/catalog.rs`
- Create: `src/adapters/filesystem.rs`
- Create: `tests/fixtures/openrouter/models.json`
- Create: `tests/catalog_adapter.rs`
- Create: `tests/catalog_policy.rs`

**Interfaces:**
- `#[async_trait] trait ModelCatalog { async fn list_models(&self, filters: &CatalogQuery) -> Result<CatalogSnapshot, AdapterError>; }`
- `CatalogSnapshot` contains `fetched_at`, source, normalized models, and retained raw response/snapshot path.
- `CatalogPolicy::evaluate(model: &ModelCandidate, filters: &CatalogFilters) -> CandidateDecision` returns include/exclude plus human-readable reasons.
- Decision v1 default compatibility requires text input/output and the configured chat-completion/structured-output capability; no model ID is hard-coded.
- OpenRouter catalog base URL is configurable for mock tests; production default is `https://openrouter.ai/api/v1`.

- [ ] **Step 1: Add catalog adapter/policy tests**
  - With fixture catalogs, verify model metadata and prices normalize correctly, missing/malformed fields are handled explicitly, configured filters apply, and every rejection has a reason.
  - Verify include/exclude selectors take precedence as documented and empty catalog results are valid but do not form a runnable plan.
  - Use a local mock HTTP server; assert no live network dependency.
- [ ] **Step 2: Run the focused tests and verify they fail**

  Run: `cargo test --test catalog_adapter --test catalog_policy`

  Expected: failure because the port and adapter do not exist.
- [ ] **Step 3: Implement catalog port, OpenRouter adapter, and policy**
  - Parse model IDs, created time, modalities, context, supported parameters, and prompt/completion prices from `/models`.
  - Treat absent/invalid/negative prices as unknown; retain valid zero price as zero and cover it explicitly.
  - Store a snapshot for each catalog operation and record the effective filters/policy version.
  - Build the HTTP client with explicit connect/request timeouts and secret-safe headers.
- [ ] **Step 4: Run catalog tests**

  Run: `cargo test --test catalog_adapter --test catalog_policy`

  Expected: fixture-driven tests pass, including malformed responses and provider errors.

## Task 4: Run planning, upper-bound estimate, and budget ledger

**Files:**
- Create: `src/application/plan.rs`
- Create: `src/application/budget.rs`
- Create: `tests/planning_budget.rs`
- Modify: `src/cli.rs`

**Interfaces:**
- `RunPlan::build(config: &AppConfig, suite: &TestSuite, candidates: &[ModelCandidate], budget_usd: f64) -> Result<RunPlan, PlanError>` contains work units, estimates, budget, selected suite/models, exclusions, and captured policy versions.
- `CostEstimator::estimate(model: &ModelCandidate, max_input_tokens: u64, max_output_tokens: u64) -> Result<Money, EstimateError>` estimates a bounded request from per-million token prices. `Money` is integer USD micro-units; do not use floating-point arithmetic for ledger comparisons.
- `BudgetLedger::reserve(work_unit: &WorkUnit, amount: Money) -> Result<ReservationId, BudgetError>` atomically reserves against remaining budget; `reconcile(id: ReservationId, actual: Option<Money>) -> Result<(), BudgetError>` records actual spend.
- `plan` prints calls, candidate count, suite/repeats, per-model and total estimated spend, remaining budget, and exclusions; it writes no API-call results.

- [ ] **Step 1: Add planner/budget tests**
  - Assert plan call count equals candidates × cases × repeats and estimates use input/output token caps.
  - Assert plans exceeding budget fail before network completion requests can be scheduled.
  - Assert missing prices/token bounds fail closed; zero-priced models are handled without division errors.
  - Assert concurrent reservations cannot exceed the configured budget; test reservation reconciliation for actual cost below/above estimate and missing actual cost.
- [ ] **Step 2: Run focused tests and verify they fail**

  Run: `cargo test --test planning_budget`

  Expected: failure because plan and ledger implementations do not exist.
- [ ] **Step 3: Implement pure planning and atomic ledger**
  - Calculate an explicit upper bound from configured request token limits and catalog prices.
  - Enforce a configured per-request input-size/token upper bound before scheduling; use the larger of the suite/request bound and the implementation's conservative UTF-8 byte-based upper bound, including fixed request framing overhead.
  - Reserve before a worker begins a paid request; retries are new work and require a new reservation.
  - On missing usage/cost, retain the result and signal the scheduler to stop queuing work.
- [ ] **Step 4: Wire and test the `plan` command**

  Run: `cargo test --test planning_budget`

  Expected: plan output contains every estimate/exclusion and rejects over-budget runs without making a generation request.

## Task 5: Chat completion runner, run persistence, and resume

**Files:**
- Create: `src/adapters/openrouter/runner.rs`
- Create: `src/application/execute.rs`
- Modify: `src/ports.rs`
- Modify: `src/adapters/filesystem.rs`
- Modify: `src/domain/run.rs`
- Create: `tests/runner_adapter.rs`
- Create: `tests/run_store_resume.rs`
- Create: `tests/execute_workflow.rs`

**Interfaces:**
- `#[async_trait] trait ModelRunner { async fn execute(&self, request: &ModelRequest) -> Result<CallResult, CallError>; }`
- `#[async_trait] trait RunStore { async fn create(&self, manifest: &RunManifest) -> Result<(), StoreError>; async fn append(&self, run_id: &RunId, result: &CallResult) -> Result<(), StoreError>; async fn load(&self, run_id: &RunId) -> Result<StoredRun, StoreError>; async fn set_status(&self, run_id: &RunId, status: RunStatus) -> Result<(), StoreError>; }`
- `Executor::run(plan: RunPlan, runner: &dyn ModelRunner, store: &dyn RunStore, ledger: BudgetLedger) -> Result<RunSummary, ExecuteError>` and `Executor::resume(run_id: &RunId, ...)` schedule only outstanding work units.
- Chat Completion runner sends JSON-mode requests where supported by the configured compatibility path, captures response model ID, end-to-end latency, token usage and provider-reported cost, and normalizes object/fenced JSON parse status.

- [ ] **Step 1: Add runner/store/execute tests**
  - Fixture tests cover valid object JSON, fenced JSON, invalid JSON, empty choices, missing cost, resolved ID changes, auth errors, rate limits, server errors, and timeouts.
  - Store tests cover append/read round-trip, truncated final JSONL record recovery, manifest redaction, and duplicate completed work units on resume.
  - Workflow tests assert actual missing cost stops new work, completed results persist immediately, and budget stops leave a resumable incomplete run.
- [ ] **Step 2: Run focused tests and verify they fail**

  Run: `cargo test --test runner_adapter --test run_store_resume --test execute_workflow`

  Expected: failure because runner, run store, and executor are not implemented.
- [ ] **Step 3: Implement request runner and durable run store**
  - Implement retries only for transient network/rate-limit/server failures with bounded attempts and backoff.
  - Append and flush each completed request result; persist secret-free manifest, catalog snapshot reference, seed, concurrency, suite/scoring versions, config, and status.
  - On resume, validate the manifest/result schema, ignore an incomplete final line safely, and compute remaining work units from successful records.
- [ ] **Step 4: Implement scheduler and budget-aware resume**
  - Enforce concurrency, per-model spacing, max output tokens, bounded retry count, and atomic reservation before every request.
  - Do not count a failed call as successful suite coverage; do account for any reported spend.
- [ ] **Step 5: Run focused execution tests**

  Run: `cargo test --test runner_adapter --test run_store_resume --test execute_workflow`

  Expected: all fixture and local-server workflow tests pass without OpenRouter credentials.

## Task 6: Generic metrics, Markdown/CSV reports, and CLI workflow

**Files:**
- Create: `src/application/report.rs`
- Create: `src/reporters/mod.rs`
- Create: `src/reporters/markdown.rs`
- Create: `src/reporters/csv.rs`
- Create: `tests/reporting.rs`
- Modify: `src/cli.rs`
- Modify: `src/main.rs`

**Interfaces:**
- `Report::from_run(run: &StoredRun, suite: &TestSuite) -> Result<Report, ReportError>` computes per-model/per-scenario metrics and run-level status.
- `MarkdownReporter::render(report: &Report) -> String` and `CsvReporter::write(report: &Report, output_dir: &Path) -> Result<(), ReportError>` contain no model/vendor-specific conditionals.
- Rank only complete model/suite coverage with valid cost on every required call; always show operational failures and partial spend in non-ranked run summaries.

- [ ] **Step 1: Add reporting tests**
  - Assert arbitrary unknown model IDs render safely, complete generic runs rank by unrounded score, and incomplete/missing-cost records block ranking.
  - Assert scenario metrics, resolved model IDs, spend totals, and run status are present in report output.
- [ ] **Step 2: Run focused report tests and verify they fail**

  Run: `cargo test --test reporting`

  Expected: failure because report models and renderers are not implemented.
- [ ] **Step 3: Implement generic report aggregation and renderers**
  - Derive all metrics from stored data plus suite/scoring metadata; do not assume any candidate exists.
  - Include suite/data/scoring versions, catalog fetch time, run date, concurrency, seed, requested/resolved IDs, and budget estimate/actual spend.
- [ ] **Step 4: Wire catalog, plan, run, resume, and report commands**
  - Inject OpenRouter/filesystem implementations at the CLI composition root; application code continues to use ports.
- [ ] **Step 5: Run reporting and CLI workflow tests**

  Run: `cargo test --test reporting --test cli_args`

  Expected: commands exercise mocked application services and report tests pass without network access.

## Task 7: Rename user-facing project surfaces and document migration

**Files:**
- Modify: `README.md`
- Modify: `.env.example`
- Modify: `.gitignore`
- Modify: `artifacts/report.md` and `artifacts/report.en.md` only to add a historical-run note if those reports remain in the main entry path
- Do not delete: `models.csv`, `results.jsonl`, `benchmark.py`, `run_benchmark.py`, `retry_failed.py`, historical charts and CSVs until the migration regression task is complete.

**Interfaces:**
- README describes the project as Evolving Benchmark and the initial module as Decision Benchmark; all runnable commands use `evolving-benchmark`.
- `.env.example` documents only the variable name and a placeholder, never a real key.
- Default output directories for new runs do not overwrite existing historical JSONL or artifacts.

- [ ] **Step 1: Add documentation smoke assertions**
  - Check README commands match CLI help, config example parses, and no hard-coded candidate list is described as the source of truth.
  - Check existing artifact reports are labeled historical if linked as current results.
- [ ] **Step 2: Update names and migration notes**
  - Document Rust installation/build/test/run steps, model filters, suite selection, budget dry-run, resume behavior, output schema, and the distinction between catalog freshness and result freshness.
  - Do not rename the current workspace folder or delete legacy artifacts as part of the source-level rename.
- [ ] **Step 3: Verify docs and example**

  Run: `cargo run -- --help`, `cargo run -- catalog decision --help`, `cargo test --test cli_args`

  Expected: help uses Evolving Benchmark names and the example config validates.

## Task 8: End-to-end regression and release gate

**Files:**
- Create: `tests/e2e_mocked_cli.rs`
- Create: `tests/fixtures/historical/score_cases.json`
- Modify: `README.md`
- Modify: `.gitignore`

**Interfaces:**
- Mocked E2E path runs catalog → plan → run → resume/report through the CLI composition root using a local HTTP fixture server and a temporary output directory.
- Historical score fixtures pin existing formula behavior without requiring model-specific report prose or live calls.

- [ ] **Step 1: Add E2E and regression tests**
  - Assert over-budget plan makes zero completion calls, in-budget run writes manifest/snapshot/results, interruption and resume skip completed work, and report ranks only full/fully-costed results.
  - Assert documented score fixtures match the prior formula within a stated floating-point tolerance.
- [ ] **Step 2: Run the full offline test suite**

  Run: `cargo test --all-targets`

  Expected: all tests pass without `OPENROUTER_API_KEY` and without internet access.
- [ ] **Step 3: Run quality and packaging checks**

  Run: `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` and `cargo build --release`

  Expected: formatting is clean, Clippy reports no warnings, and release binary builds.
- [ ] **Step 4: Perform opt-in live smoke check only when credentials are configured**
  - Run catalog-only first; run one selected model and a tiny explicit suite only after reviewing the printed plan and budget.
  - Do not make live API checks part of CI or the default test command.

## Dependency order

```text
Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6 → Task 7
                                      └──────────────────────→ Task 8
```

Tasks 1–7 are intentionally sequential because each defines interfaces consumed by the next. Task 8 is the final cross-boundary validation and depends on all implementation tasks.

## Specification coverage check

| Spec requirement | Plan task |
|---|---|
| Rust CLI and Evolving Benchmark naming | 1, 6, 7 |
| Clean Architecture ports/adapters | 3, 5, 6 |
| OpenRouter discovery, snapshots, filters, explainable eligibility | 3 |
| Versioned Decision cases/suite and generic scoring | 2, 6 |
| Cost plan, budget reservations, retries charged, stop on missing cost | 4, 5 |
| Durable results, resume, reproducibility metadata | 5 |
| Generic reports and incomplete-run rank blocking | 6 |
| Tests for all boundaries without network | 1–6, 8 |
| Historical Python artifacts preserved during migration | 2, 7, 8 |
| Cron/shadow/A-B/other domains deferred | Global constraints, 7 |
