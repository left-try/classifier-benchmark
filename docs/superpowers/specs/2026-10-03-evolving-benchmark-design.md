# Evolving Benchmark: Decision Benchmark Design

**Status:** Draft for review  
**Date:** 2026-10-03

## Purpose

Rename the project to **Evolving Benchmark** and establish it as a Rust CLI platform for repeatable model benchmarks. The first benchmark module is **Decision Benchmark**, which evaluates models on the repository's existing agentic decision cases. The CLI discovers model candidates from OpenRouter, selects them using explicit filters and compatibility rules, runs a selected test suite under a spending limit, and stores reproducible results.

This design covers the first usable Decision Benchmark implementation. It does not include other benchmark domains, hosted services, dashboards, automatic cron installation, shadow traffic, or A/B testing. The architecture must leave clear interfaces for those later additions.

## Existing behavior to preserve

- Preserve the current 30 labeled cases and six scenarios as the initial full Decision suite.
- Preserve the core measurements: exact-match accuracy, response validity, format compliance where applicable, end-to-end latency, reported cost, and per-scenario summaries.
- Preserve the current score formula as a named and versioned scoring policy, with its inputs and constants recorded in each run manifest. Avoid model-specific assumptions in evaluation and reporting.
- Preserve existing JSONL results, reports, charts, and other artifacts as historical outputs. They must not be presented as results produced by the new Rust runner.
- Preserve API credentials outside source code and command-line arguments.

## User workflow

The first release supports these operations:

```text
evolving-benchmark catalog decision --config benchmark.toml
evolving-benchmark plan decision --config benchmark.toml --budget-usd 5
evolving-benchmark run decision --config benchmark.toml --budget-usd 5
evolving-benchmark resume <run-id>
evolving-benchmark report <run-id>
```

`catalog` fetches the provider catalog and explains which models pass or fail configured filters and compatibility rules. `plan` performs a dry run, shows the selected models, case/repeat/call counts, estimated maximum spend, and exclusions, and refuses plans above the supplied budget. `run` executes a valid plan. `resume` continues incomplete work without repeating successful model/case/repeat units. `report` derives output from the saved run data and manifest.

Configuration defines catalog filters, the selected suite (default: the full Decision suite), repeats, concurrency, output location, request token limits, retries, and protocol overrides. The budget is supplied explicitly per plan/run. The API key is read from `OPENROUTER_API_KEY` or an optional local `.env` file and is never serialized into run artifacts or logs.

## Architecture

Use one Rust Cargo package for the initial release, with module boundaries that can later become workspace crates:

```text
CLI
 └─ Application services: catalog, plan, run, resume, report
     ├─ Domain: model candidates, filters, suites, cases, runs, budget, metrics
     ├─ Ports: ModelCatalog, ModelRunner, RunStore
     └─ Adapters
         ├─ OpenRouter catalog and request clients
         ├─ Filesystem configuration, snapshots, and run storage
         └─ Markdown/CSV reporter
```

Domain and application code depend on ports, not OpenRouter or a particular storage format. The OpenRouter adapter maps remote responses to internal types and owns HTTP concerns. The run store persists a manifest, model catalog snapshot, append-only per-call JSONL records, and derived reports. Each run has a unique identifier and explicit lifecycle status: planned, running, complete, incomplete, or stopped for budget/error.

## Model discovery and compatibility

The OpenRouter adapter uses the model catalog endpoint and normalizes fields relevant to this use case, including model ID, display name, creator, creation time, modalities, context length, supported parameters, and prompt/completion pricing when supplied. Catalog retrieval time and the unmodified or losslessly retained catalog snapshot are recorded for reproducibility.

Catalog presence alone does not prove that a model supports a benchmark protocol. By default, Decision candidates are text-capable models that meet configured catalog filters and can be called through the supported chat-completion path with the structured-output behavior required by the suite. Candidate inclusion is policy-based and explainable; there is no hard-coded shortlist of model IDs. Model/provider inclusion and exclusion lists may be configured by the user.

Protocol is separate from benchmark category. The default adapter path uses Chat Completions. Nonstandard paths, including OpenRouter Decisions API, are registered as explicit protocol implementations and selected through configuration/capability metadata; they are not inferred from a model name. Provider-specific protocols can be added without changing case evaluation or scoring.

Filters include creator/provider, creation-time window, context minimum, prompt/completion price constraints, required supported parameters, and explicit include/exclude selectors. The chosen filters and decision policy version are captured in the run manifest. Unsupported or incomplete catalog data is visible as an exclusion reason; the system does not silently assume missing compatibility or pricing.

## Decision test suite and evaluation

Move the current cases into a versioned, validated suite format while retaining case IDs and expected labels. Initial scenarios remain `tool_selector`, `model_router`, `guardrail`, `delegation_router`, `confidence_escalation`, and `stop_continue`. A suite may be selected by name, with `decision/full` as the default. The execution engine does not embed the cases or scenario-specific provider logic in Rust source.

Each call result records run ID, suite and dataset versions, case/scenario IDs, requested and resolved model IDs, protocol, repeat, prediction, parse/format status, latency, token usage when available, cost when available, timestamp, and structured error/retry details. Credentials and secret request headers are excluded.

Scoring is model-agnostic and deterministic. The existing score formula is retained initially as a named versioned policy; accuracy, validity, latency, cost, and component factors remain visible alongside the combined score. Reports are generated from whatever candidates are in the run and must not require known vendors or model IDs. Incomplete coverage or missing/invalid costs block ranking while still allowing an operational report of failures and spend.

## Budget and execution safety

Every plan/run requires a finite `--budget-usd` value. Before execution, estimate cost from catalog pricing, bounded request token counts, selected model/case/repeat count, and configured concurrency. Fail closed when price or token bounds are unavailable for a candidate; allow the user to exclude it or configure a safe explicit cap where supported.

At execution time, reserve estimated spend before scheduling each request so concurrent workers cannot independently consume the same remaining budget. Reconcile reservations with provider-reported cost after every response. Stop scheduling new work when the remaining budget cannot cover another reservation; preserve completed results and mark the run stopped/incomplete. Missing cost on a response is recorded and stops further work by default. Retries consume budget and are accounted for as additional calls. Per-request output token caps, concurrency limits, retry limits, and per-model spacing are configurable and recorded.

The spend cap is enforced against available pricing and usage data; the CLI must describe estimates as estimates and report any unknown/unpriced calls. It must never claim a complete ranked result when costs or required case coverage are missing.

## Resilience and reproducibility

- Apply bounded timeouts and retries only to transient transport/server/rate-limit failures; classify permanent API and request errors without retry loops.
- Persist each completed result immediately using append-only records so interruption does not discard progress.
- Resume by deriving outstanding work from the manifest and validated result records; avoid duplicate successful work units.
- Record catalog snapshot, effective config with secrets removed, suite/dataset and scoring versions, CLI version, start/end times, requested/resolved models, and run status.
- Treat provider catalog changes as new snapshots, not as edits to old runs.
- Use deterministic case/repeat scheduling when a seed is specified; record the seed and concurrency because latency is affected by harness load.

## Testing strategy

The Rust implementation is covered at each boundary:

- Unit tests for config validation, candidate filtering and exclusion explanations, compatibility policy, suite parsing/schema validation, prediction normalization, exact-match scoring, cost estimation, budget reservations/reconciliation, and run completeness.
- CLI tests for catalog/plan/run argument validation, budget refusal, secret-free output, report generation, and resume behavior.
- Adapter tests against a local mock HTTP server with catalog fixtures and success, malformed response, authentication, rate-limit, server-error, missing-price, missing-usage, and resolved-model cases.
- Storage tests for append/recovery behavior, run manifests, and avoiding duplicate work on resume.
- Regression checks that score and report calculations on compatible historical result fixtures match the documented existing behavior.
- No live API calls in the default automated test suite; live smoke checks are an explicitly separate manual operation.

## Migration and repository naming

Update repository-facing names, README, CLI/package metadata, generated report titles, and configuration examples to Evolving Benchmark / Decision Benchmark. The current workspace directory name is not assumed to be the canonical project name and should not be renamed as part of source migration. Keep current Python scripts and generated artifacts available during migration, label their outputs historical, and remove the scripts from the primary workflow only after the Rust CLI passes regression checks.

## Release acceptance criteria

1. The Rust CLI can fetch and snapshot the OpenRouter catalog, apply configured filters, and explain candidate selection without a source-code model roster.
2. A dry-run plan displays candidates, suite coverage, number of calls, price assumptions, and estimated maximum spend, and refuses a plan that exceeds its explicit budget.
3. A run executes the versioned Decision suite, stores requested/resolved IDs and per-call metrics, and respects concurrency, token, retry, and budget limits.
4. Interrupted work can be resumed without repeating completed work units.
5. Reports work for arbitrary model IDs and block rankings for incomplete coverage or missing/invalid cost data.
6. Unit, CLI, adapter, storage, and regression tests pass without network access.
7. Existing published artifacts remain identifiable as historical Python-run outputs.

## Explicitly deferred

- Additional benchmark domains beyond Decision Benchmark.
- Additional provider adapters beyond OpenRouter.
- Automated cron installation or hosted recurring execution.
- A/B testing, shadow traffic, dashboards, and online routing recommendations.
- Hidden holdout datasets or changes to the current scoring formula beyond versioning it.
- Claims that catalog freshness implies benchmark-result freshness.
