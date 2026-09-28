# Agentic Loop Classifier Benchmark

This benchmark compares models on structured decisions inside an agent loop: tool selection, model routing, delegation, guardrails, confidence escalation, and deciding whether an agent should stop or continue. It measures decision accuracy, cost, and end-to-end latency on a fixed labeled set, then produces presentation-ready tables and charts.

## What's included

- `data/cases.jsonl` — 30 labeled cases across six scenarios.
- `models.csv` — a balanced shortlist of current US and Chinese model candidates, plus Jev. Recheck model IDs, aliases, availability, and pricing before future runs; these change over time.
- `results.jsonl` — the complete scored dataset from the published run: 16 models × 30 cases × 5 starts = 2,400 calls.
- `benchmark.py` — scoring, per-scenario rankings, report generation, and PNG charts.
- `artifacts/` — Russian and English reports, leaderboard CSVs, a results infographic, charts, and rerun data for Haiku and Grok 4.3.

## Scenarios

Each scenario asks the model for a compact structured decision rather than free-form text.

1. **`tool_selector`** — choose the right tool for the goal and context, or return `none`. For example: policy lookup → `search_docs`; arithmetic → `calculator`; unauthorized deletion → `none`.
2. **`model_router`** — route a task to a suitable class of executor: a small model for field extraction, a coding model for debugging, a vision model for an image, or a stronger reasoning model for a complex plan.
3. **`guardrail`** — choose `allow`, `deny`, or `escalate` based on the request, data, and permissions. One case checks that prompt injection inside a document is treated as data.
4. **`delegation_router`** — decide whether to handle a task directly, delegate to a specialist, split it into sequential steps, or parallelize independent work. One case requires human approval for data deletion, which the available agents cannot provide.
5. **`confidence_escalation`** — given evidence and confidence, route automatically, ask for clarification, or send the case to a human.
6. **`stop_continue`** — choose the agent's next step from its current state: finish, call a tool, search again, or ask the user.

This is a small first-iteration benchmark. Before using it to make a robust model choice, expand and deduplicate the dataset, add a hidden holdout set, and version each dataset used in a publication.

## Unified score (0–1)

The same formula is used for every scenario. Accuracy is the exact-match rate. `C` is the mean per-call cost reported by OpenRouter usage. `L` is mean end-to-end latency from request submission to response receipt, including the network round trip. Each model/case pair is called five times; metrics are averaged over starts first, then over cases.

```text
cost_factor = 1 / (1 + sqrt(C / $0.001))
latency_factor = 1 / (1 + (max(0, L_ms - 500) / 1000)^2)
score = accuracy * cost_factor * latency_factor
display_score = round(score, 1)
```

The $0.001 cost reference and first 500 ms without a latency penalty are explicit starting assumptions, not universal constants. They can be calibrated before a benchmark run and versioned, but must remain fixed across models and scenarios in a comparison. Exact match is the quality measure; accuracy and both factors are also reported so the score remains interpretable. Scenario rankings use the unrounded score; displayed scores are rounded to tenths.

Cost includes prompt and completion token charges. Latency is measured client-side end to end. Requests are shuffled, and the main run uses fixed concurrency of 8. Thus, E2E latency reflects response time under the same harness load, not isolated decoding speed. Each scenario score averages its five labeled cases. Jev uses OpenRouter's Decisions API with native typed choices; other models use Chat Completions with native JSON mode.

## Run the benchmark

Requirements: Python 3.10+, `pandas`, and `matplotlib`.

```powershell
python -m pip install pandas matplotlib
python run_benchmark.py --repeats 5
python benchmark.py --results results.jsonl --out artifacts
```

`artifacts/report.md` contains an automated analysis of the overall winner, accuracy/cost/latency leaders, the Pareto set, scenario rankings, score factors, and coverage caveats. The report and charts are generated from the supplied `results.jsonl`; incomplete or failed results block ranking.

Example JSONL result:

```json
{"case_id":"TS-001","model_id":"google/gemini-2.5-flash","prediction":{"tool":"search_docs"},"latency_ms":420,"cost_usd":0.0003,"repeat":1}
```

Set `OPENROUTER_API_KEY` as an environment variable or in a root `.env` file; never pass the key as a command-line argument. By default, all IDs in `models.csv` run in a shuffled order with a fixed seed. Reduce the roster with `--model ID`. A ranking requires complete coverage and exactly five successful calls for every model/case pair. The runner saves responses as it goes and records both the requested ID and the OpenRouter-resolved model version.

Each result includes `case_id`, `model_id`, `prediction`, `latency_ms`, `cost_usd`, `repeat`, and protocol. Predictions are compared by exact structure. Any failed call or missing cost blocks ranking; run `python retry_failed.py` to retry only those requests before rebuilding the full result set.

Outputs: `leaderboard.csv`, `scenario_leaderboards.csv`, `report.md` (Russian), `report.en.md` (English), `benchmark-infographic.svg`, `summary.png`, `accuracy_cost.png`, and `latency_by_scenario.png`. The English report includes full comparison tables and embeds the infographic and charts. For presentations, use the same `results.jsonl` and include the dataset version and run date/conditions in chart captions.

## Model roster

`models.csv` includes Jev, GPT-6 Sol with reasoning effort set to low, GPT-6 Luna, GPT-5.4 Nano/Mini, Grok 4.3 as a replacement for the deprecated Grok 4.1 Fast, and rolling/pinned candidates from Anthropic, Google, xAI, DeepSeek, and Qwen. Rolling aliases and pinned snapshots are listed separately; the resolved ID is recorded for every call.

The shortlist was checked against OpenRouter catalogs for [Google](https://openrouter.ai/google), [OpenAI](https://openrouter.ai/provider/openai), [Anthropic](https://openrouter.ai/anthropic), [xAI](https://openrouter.ai/x-ai), [DeepSeek](https://openrouter.ai/deepseek), and [Qwen](https://openrouter.ai/qwen). [Jev / TypeSafe](https://typesafe.ai/blog/introducing-system-one-models-and-jev) is called through OpenRouter's Decisions API. Grok 4.1 Fast was excluded after OpenRouter marked it deprecated and recommended Grok 4.3.
