"""Retry only failed/missing-cost records in an existing full run."""
from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import run_benchmark as runner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results.jsonl"))
    parser.add_argument("--cases", type=Path, default=Path("data/cases.jsonl"))
    parser.add_argument("--models", type=Path, default=Path("models.csv"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    key = runner.load_api_key()
    if not key:
        raise SystemExit("Set OPENROUTER_API_KEY in the environment or project .env file.")
    records = [json.loads(line) for line in args.results.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = {c["case_id"]: c for c in map(json.loads, args.cases.read_text(encoding="utf-8").splitlines()) if c}
    with args.models.open(encoding="utf-8-sig", newline="") as stream:
        models = {row["model_id"]: row for row in csv.DictReader(stream)}
    schemas = {}
    for case in cases.values():
        fields = schemas.setdefault(case["scenario"], {})
        for field, label in case["expected"].items():
            fields.setdefault(field, set()).add(label)
    schemas = {scenario: {field: sorted(labels) for field, labels in fields.items()}
               for scenario, fields in schemas.items()}
    pending = [(i, r) for i, r in enumerate(records) if r.get("error") or r.get("cost_usd") is None]
    print(f"Retrying {len(pending)} failed or unpriced calls.")
    locks = {model: threading.Lock() for model in models}
    last_start = {model: 0.0 for model in models}
    for pass_number in range(1, 4):
        if not pending:
            break
        def retry(item):
            index, old = item
            model = models[old["requested_model_id"]]
            case = cases[old["case_id"]]
            with locks[model["model_id"]]:
                delay = 3.4 - (time.monotonic() - last_start[model["model_id"]])
                if delay > 0:
                    time.sleep(delay)
                last_start[model["model_id"]] = time.monotonic()
                try:
                    if model["protocol"] == "openrouter_decisions":
                        prediction, latency, cost, resolved, format_compliant = runner.call_jev(
                            model["model_id"], case, schemas[case["scenario"]], key
                        )
                    else:
                        prediction, latency, cost, resolved, format_compliant = runner.call_chat(
                            model["model_id"], model.get("reasoning_effort", ""), case,
                            schemas[case["scenario"]], key
                        )
                    error = None
                except Exception as exc:  # Preserve failed calls visibly; never drop denominator rows.
                    prediction, latency, cost, resolved, format_compliant, error = (
                        None, 0.0, None, model["model_id"], False, str(exc)
                    )
            new = {**old, "model_id": resolved, "prediction": prediction, "latency_ms": latency,
                   "cost_usd": cost, "format_compliant": format_compliant,
                   "error": error, "retry_pass": pass_number}
            return index, new

        completed = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(retry, item) for item in pending]
            for future in as_completed(futures):
                completed.append(future.result())
        for index, updated in completed:
            records[index] = updated
        pending = [(i, r) for i, r in enumerate(records) if r.get("error") or r.get("cost_usd") is None]
        print(f"Retry pass {pass_number}: remaining failures={len(pending)}.")
    temp = args.results.with_suffix(args.results.suffix + ".tmp")
    temp.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    temp.replace(args.results)
    if pending:
        raise SystemExit(f"Still have {len(pending)} failed/unpriced calls; ranking must remain blocked.")


if __name__ == "__main__":
    main()
