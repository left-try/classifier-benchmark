"""Score collected agent-classifier results and render presentation graphics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TARGET_COST_USD = 0.001
FREE_LATENCY_MS = 500


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def score_rows(results: list[dict], cases: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected = {c["case_id"]: c["expected"] for c in cases}
    scenario = {c["case_id"]: c["scenario"] for c in cases}
    normalized = []
    for result in results:
        cid = result["case_id"]
        if cid not in expected:
            raise ValueError(f"Unknown case_id: {cid}")
        pred = result.get("prediction")
        if result.get("cost_usd") is None:
            raise ValueError(f"{result.get('model_id')} / {cid} has no cost. Resolve/retry it before ranking.")
        if result.get("error"):
            raise ValueError(f"{result.get('model_id')} / {cid} failed: {result['error']}")
        normalized.append({
            "case_id": cid,
            "scenario": scenario[cid],
            "model_id": result.get("requested_model_id", result["model_id"]),
            "resolved_model_id": result["model_id"],
            "correct": int(pred == expected[cid]),
            "valid_output": int(pred is not None),
            "format_compliant": (None if result.get("protocol") == "openrouter_decisions"
                                  else int(result.get("format_compliant", pred is not None))),
            "protocol": result.get("protocol", "chat"),
            "latency_ms": float(result["latency_ms"]),
            "cost_usd": float(result["cost_usd"]),
        })
    raw = pd.DataFrame(normalized)
    if raw["cost_usd"].isna().any() or (raw["cost_usd"] < 0).any():
        raise ValueError("Some calls have no valid cost. Resolve/retry them before ranking.")
    for model_id, group in raw.groupby("model_id"):
        missing = set(expected) - set(group["case_id"])
        if missing:
            raise ValueError(f"{model_id} is missing {len(missing)} benchmark cases; do not rank partial runs.")
    # Average the five independent starts within each case, then average cases equally.
    per_case = raw.groupby(["scenario", "model_id", "case_id"], as_index=False).agg(
        accuracy=("correct", "mean"), valid_output=("valid_output", "mean"),
        format_compliant=("format_compliant", "mean"),
        latency_ms=("latency_ms", "mean"), cost_usd=("cost_usd", "mean"),
        repeat_accuracy_sd=("correct", "std"), protocol=("protocol", "first"), starts=("correct", "size"),
        resolved_model_id=("resolved_model_id", lambda values: ", ".join(sorted(set(values))))
    )
    if (per_case["starts"] != 5).any():
        raise ValueError("Every model/case must have exactly five starts before ranking.")
    per_case["repeat_accuracy_sd"] = per_case["repeat_accuracy_sd"].fillna(0)
    per_scenario = per_case.groupby(["scenario", "model_id"], as_index=False).agg(
        accuracy=("accuracy", "mean"), valid_output=("valid_output", "mean"),
        format_compliant=("format_compliant", "mean"),
        latency_ms=("latency_ms", "mean"), cost_usd=("cost_usd", "mean"),
        repeat_accuracy_sd=("repeat_accuracy_sd", "mean"), cases=("case_id", "nunique"),
        resolved_model_id=("resolved_model_id", lambda values: ", ".join(sorted(set(
            version for value in values for version in value.split(", ")
        ))))
    )
    for frame in (per_case, per_scenario):
        frame["cost_factor"] = 1 / (1 + (frame["cost_usd"] / TARGET_COST_USD).pow(0.5))
        frame["latency_factor"] = 1 / (1 + ((frame["latency_ms"] - FREE_LATENCY_MS).clip(lower=0) / 1000).pow(2))
        frame["score"] = frame["accuracy"] * frame["cost_factor"] * frame["latency_factor"]
        frame["display_score"] = frame["score"].round(1)
    per_scenario["rank"] = per_scenario.groupby("scenario")["score"].rank(method="min", ascending=False).astype(int)
    summary = per_case.groupby(["model_id"], as_index=False).agg(
        accuracy=("accuracy", "mean"), valid_output=("valid_output", "mean"),
        format_compliant=("format_compliant", "mean"),
        latency_ms=("latency_ms", "mean"), cost_usd=("cost_usd", "mean"),
        repeat_accuracy_sd=("repeat_accuracy_sd", "mean"),
        cases=("case_id", "nunique"), scenarios=("scenario", "nunique"),
        resolved_model_id=("resolved_model_id", lambda values: ", ".join(sorted(set(
            version for value in values for version in value.split(", ")
        ))))
    )
    summary["cost_factor"] = 1 / (1 + (summary["cost_usd"] / TARGET_COST_USD).pow(0.5))
    summary["latency_factor"] = 1 / (1 + ((summary["latency_ms"] - FREE_LATENCY_MS).clip(lower=0) / 1000).pow(2))
    summary["score"] = summary["accuracy"] * summary["cost_factor"] * summary["latency_factor"]
    summary["display_score"] = summary["score"].round(1)
    summary = summary.sort_values("score", ascending=False)
    return summary, per_scenario.sort_values(["scenario", "score"], ascending=[True, False])


def make_charts(summary: pd.DataFrame, per_scenario: pd.DataFrame, out: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    short = summary["model_id"].str.split("/").str[-1]
    fig, ax = plt.subplots(figsize=(12, max(5, 0.38 * len(summary))))
    colors = plt.cm.viridis(summary["score"].clip(0, 1))
    ax.barh(short.iloc[::-1], summary["score"].iloc[::-1], color=colors[::-1])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Combined score (0–1; rounded to tenths in tables)")
    ax.set_title("Agentic classifier benchmark · overall leaderboard")
    for i, value in enumerate(summary["score"].iloc[::-1]):
        ax.text(value + 0.012, i, f"{value:.1f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "summary.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(summary["cost_usd"], summary["accuracy"], c=summary["latency_ms"],
                         s=70 + 180 * summary["score"], cmap="plasma_r", edgecolors="#263238", linewidths=.6)
    zero_accuracy_offsets = [(8, 14), (8, 32), (8, 50), (8, 68)]
    low_accuracy_index = 0
    for _, row in summary.iterrows():
        offset = (5, 4)
        if row["accuracy"] < 0.001:
            offset = zero_accuracy_offsets[low_accuracy_index % len(zero_accuracy_offsets)]
            low_accuracy_index += 1
        elif row["accuracy"] < 0.1:
            offset = (8, 52)
        ax.annotate(row["model_id"].split("/")[-1], (row["cost_usd"], row["accuracy"]), xytext=offset,
                    textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Mean cost per call (USD, log scale)")
    ax.set_ylabel("Exact-match accuracy")
    ax.set_ylim(-.03, 1.08)
    ax.set_title("Accuracy vs cost · color = end-to-end latency")
    fig.colorbar(scatter, ax=ax, label="Mean end-to-end latency (ms)")
    fig.tight_layout()
    fig.savefig(out / "accuracy_cost.png", dpi=180)
    plt.close(fig)

    pivot = per_scenario.pivot(index="scenario", columns="model_id", values="latency_ms")
    fig, ax = plt.subplots(figsize=(max(10, 0.85 * len(pivot.columns)), 5.8))
    pivot.plot(kind="bar", ax=ax, width=.82)
    ax.axhline(FREE_LATENCY_MS, color="#c0392b", linestyle="--", linewidth=1, label="500 ms no-penalty threshold")
    ax.set_ylabel("Mean end-to-end latency (ms)")
    ax.set_xlabel("")
    ax.set_title("Latency by agentic scenario")
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out / "latency_by_scenario.png", dpi=180)
    plt.close(fig)


def write_report(summary: pd.DataFrame, per_scenario: pd.DataFrame, cases: list[dict], total_calls: int, total_cost_usd: float, out: Path) -> None:
    """Generate a concise, data-grounded Markdown interpretation of this run."""
    total_cases = len(cases)
    scenario_sizes = pd.Series([c["scenario"] for c in cases]).value_counts().to_dict()
    winner = summary.iloc[0]
    best_accuracy = summary.loc[summary["accuracy"].idxmax()]
    cheapest = summary.loc[summary["cost_usd"].idxmin()]
    fastest = summary.loc[summary["latency_ms"].idxmin()]
    by_model = summary.set_index("model_id")
    jev = by_model.loc["typesafe/jev-1.13"]
    gpt6_low = by_model.loc["openai/gpt-6-sol"]
    gpt6_luna = by_model.loc["openai/gpt-6-luna"]
    gemini_latest = by_model.loc["~google/gemini-flash-latest"]
    gemini_pinned = by_model.loc["google/gemini-3.8-flash"]
    haiku = by_model.loc["~anthropic/claude-haiku-latest"]
    jev_vs_luna_speed = float(gpt6_luna["latency_ms"] / jev["latency_ms"])
    jev_vs_luna_cost = float(gpt6_luna["cost_usd"] / jev["cost_usd"])
    pareto = []
    for _, candidate in summary.iterrows():
        dominated = ((summary["accuracy"] >= candidate["accuracy"]) &
                     (summary["cost_usd"] <= candidate["cost_usd"]) &
                     (summary["latency_ms"] <= candidate["latency_ms"]) &
                     ((summary["accuracy"] > candidate["accuracy"]) |
                      (summary["cost_usd"] < candidate["cost_usd"]) |
                      (summary["latency_ms"] < candidate["latency_ms"]))).any()
        if not dominated:
            pareto.append(candidate["model_id"])
    scenario_lines = []
    def percent_or_na(value) -> str:
        return "n/a" if pd.isna(value) else f"{value:.0%}"

    for scenario_name, group in per_scenario.groupby("scenario", sort=True):
        leader = group.iloc[0]
        scenario_lines.append(
            f"| {scenario_name} | {leader['model_id']} → {leader['resolved_model_id']} | {leader['display_score']:.1f} | "
            f"{leader['accuracy']:.0%} | {leader['valid_output']:.0%} | {percent_or_na(leader['format_compliant'])} | ${leader['cost_usd']:.6f} | {leader['latency_ms']:.0f} ms |"
        )
    model_lines = []
    for rank, (_, row) in enumerate(summary.iterrows(), 1):
        model_lines.append(
            f"| {rank} | {row['model_id']} | {row['resolved_model_id']} | {row['display_score']:.1f} | {row['accuracy']:.0%} | {row['valid_output']:.0%} | {percent_or_na(row['format_compliant'])} | "
            f"${row['cost_usd']:.6f} | {row['latency_ms']:.0f} ms |"
        )
    fastest_ratio = float(fastest["latency_ms"] / winner["latency_ms"]) if winner["latency_ms"] else 1
    best_score_acc_gap = float(best_accuracy["accuracy"] - winner["accuracy"])
    report = f"""# Отчёт по бенчмарку классификаторов Agentic Loop

## Краткий итог

**Лучший score:** {winner['model_id']} — {winner['display_score']:.1f}/1.0, точность exact match {winner['accuracy']:.0%}, ответ можно разобрать в структуру в {winner['valid_output']:.0%} случаев, строгий JSON без оболочки — {percent_or_na(winner['format_compliant'])}, средняя цена ${winner['cost_usd']:.6f} за вызов, средняя end-to-end задержка {winner['latency_ms']:.0f} мс.

Стоимость сохранённого рейтингового набора из {total_calls} ответов: **${total_cost_usd:.4f}**. Дополнительно оплачены 150 повторных Haiku-вызовов для исправления парсинга (ещё $0.0361); диагностические запросы в эту сумму не включены.

- Максимальная точность: **{best_accuracy['model_id']}** — {best_accuracy['accuracy']:.0%}, на {best_score_acc_gap:.1%} п.п. выше лидера по общему score.
- Минимальная средняя цена: **{cheapest['model_id']}** — ${cheapest['cost_usd']:.6f} за вызов.
- Минимальная средняя задержка: **{fastest['model_id']}** — {fastest['latency_ms']:.0f} мс ({fastest_ratio:.2f}× от задержки лидера по score).
- Парето-эффективные варианты по точности, цене и задержке: {', '.join(pareto)}.

Общий лидер выбран по заранее заданной формуле, а не только по точности. При выборе для конкретного ограничения сверяй все три метрики.

## Выводы по результатам

- **Jev — лучший практический баланс в этом наборе:** {jev['accuracy']:.0%} exact accuracy против {gpt6_luna['accuracy']:.0%} у самого точного GPT-6 Luna, при этом Jev в {jev_vs_luna_speed:.1f}× быстрее и в {jev_vs_luna_cost:.1f}× дешевле на вызов. Score — {jev['score']:.3f} против {gpt6_luna['score']:.3f}.
- **GPT-6 Low** (GPT-6 Sol с `reasoning.effort=low`) дал {gpt6_low['accuracy']:.0%} accuracy. Средняя latency {gpt6_low['latency_ms']:.0f} мс и цена ${gpt6_low['cost_usd']:.6f} дают score {gpt6_low['score']:.3f}, который отображается как {gpt6_low['display_score']:.1f}/1.0. GPT-6 Luna точнее и дешевле в этом корпусе, но по скорости уступает Jev.
- **Gemini Flash остаётся слабым кандидатом здесь:** rolling alias показал {gemini_latest['accuracy']:.0%} exact accuracy и {gemini_latest['valid_output']:.0%} разбираемых ответов; pinned 3.8 — {gemini_pinned['accuracy']:.0%} и {gemini_pinned['valid_output']:.0%}. Alias переключался между 3.7 и 3.8 во время прогона. Ошибки не сводятся только к alias: pinned версия тоже часто дала неразбираемый или неверный вывод.
- **Claude Haiku лучше по решению, чем показал первый парсер:** после снятия Markdown-оболочки JSON exact accuracy составила {haiku['accuracy']:.0%}, но strict JSON без Markdown — {haiku['format_compliant']:.0%}. Потребитель ответа должен разрешать такую оболочку или принудительно проверять native structured output.
- Показанные score округлены до десятых. Например, GPT-6 Low имеет точный score {gpt6_low['score']:.3f}, но в таблице отображается {gpt6_low['display_score']:.1f}; ранжирование использует неокруглённый score.

## Общий рейтинг

| Место | Запрошенная модель | Фактическая версия | Score | Точность | Разбираемый ответ | Строгий JSON¹ | Средняя цена/вызов | Средняя E2E latency |
|---:|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(model_lines)}

## Лидеры по сценариям

| Сценарий | Лидер (запрошенная → фактическая) | Score | Точность | Разбираемый ответ | Строгий JSON¹ | Средняя цена/вызов | Средняя E2E latency |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(scenario_lines)}

## Как читать графики

- `accuracy_cost.png` показывает компромисс точности и цены; цвет точки обозначает latency.
- `summary.png` ранжирует модели по общему score.
- `latency_by_scenario.png` показывает задержку каждой модели по сценариям и порог 500 мс.
- В этом прогоне latency-фактор составил {per_scenario['latency_factor'].min():.2f}–{per_scenario['latency_factor'].max():.2f}, cost-фактор — {per_scenario['cost_factor'].min():.2f}–{per_scenario['cost_factor'].max():.2f}.

## Покрытие и ограничения

- Корпус: {total_cases} примеров в {len(scenario_sizes)} сценариях ({'; '.join(f'{name}: {count}' for name, count in sorted(scenario_sizes.items()))}). Каждая пара модель/пример вызвана пять раз; точность, стоимость и E2E latency усреднены.
- Среднее стандартное отклонение точности между пятью стартами: {summary['repeat_accuracy_sd'].mean():.1%}; смотри `repeat_accuracy_sd` в общей таблице для каждой модели.
- Это небольшой стартовый корпус; различия стоит считать направляющими до расширения, независимой проверки разметки и оценки на скрытой holdout-выборке.
- Стандартная JSON-оболочка Markdown снимается перед exact-match проверкой: такой ответ считается разбираемым, но не соответствует строгому JSON контракту. Доля строгого JSON показывается отдельно. Chat-моделям задан provider-native JSON mode; Jev вызывается через native typed-choice API. 150 строк Haiku в финальном наборе заменены новыми замерами после исправления парсера; их latency измерена отдельным последовательным пакетом ниже RPM-лимита, поэтому она не полностью сопоставима с остальными моделями, запущенными при конкуррентности 8.
- ¹ Для Jev неприменимо: его API возвращает typed choice, а не JSON.
- Rolling alias оставлен отдельной строкой от закреплённой версии, даже если OpenRouter разрешил их в один model ID. Сверяй фактические версии перед тем, как считать такие строки независимыми моделями.
- Latency и цены провайдера меняются со временем и маршрутизацией. Для публикации сохраняй дату, requested/resolved IDs, маршрут провайдера, регион и параметры вызова.
- Jev запрашивается через OpenRouter Decisions API (`/api/alpha/decisions`), а текстовые модели — через Chat Completions. E2E latency включает round trip клиента до OpenRouter и ответ API; это сравнение продуктовых путей, не изолированной скорости декодирования модели.
"""
    (out / "report.md").write_text(report, encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True, help="Collected model responses in JSONL format")
    parser.add_argument("--cases", type=Path, default=Path("data/cases.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = load_jsonl(args.results)
    cases = load_jsonl(args.cases)
    summary, per_scenario = score_rows(results, cases)
    summary.to_csv(args.out / "leaderboard.csv", index=False)
    per_scenario.to_csv(args.out / "scenario_leaderboards.csv", index=False)
    make_charts(summary, per_scenario, args.out)
    total_cost_usd = sum(float(row["cost_usd"]) for row in results)
    write_report(summary, per_scenario, cases, len(results), total_cost_usd, args.out)
    print(f"Wrote leaderboard, report, and three charts to {args.out.resolve()}")


if __name__ == "__main__":
    main()
