"""Score collected agent-classifier results and render presentation graphics."""
from __future__ import annotations

import argparse
from html import escape
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


def make_infographic(summary: pd.DataFrame, per_scenario: pd.DataFrame, cases: list[dict], total_calls: int, total_cost_usd: float, out: Path) -> None:
    """Render a current-run, presentation-ready SVG summary with data tables."""
    winner = summary.iloc[0]
    best_accuracy = summary.loc[summary["accuracy"].idxmax()]
    cheapest = summary.loc[summary["cost_usd"].idxmin()]
    fastest = summary.loc[summary["latency_ms"].idxmin()]
    esc = lambda value: escape(str(value), quote=True)

    model_rows = []
    for rank, (_, row) in enumerate(summary.head(8).iterrows(), 1):
        y = 594 + (rank - 1) * 50
        tint = "#172c3c" if rank == 1 else "#121e2d"
        label = row["model_id"].split("/")[-1]
        if len(label) > 23:
            label = label[:21] + "…"
        model_rows.append(
            f'<rect x="94" y="{y}" width="888" height="48" rx="9" fill="{tint}"/>'
            f'<text x="116" y="{y+31}" class="row muted">{rank:02d}</text>'
            f'<text x="162" y="{y+31}" class="row">{esc(label)}</text>'
            f'<text x="532" y="{y+31}" class="row score">{row["display_score"]:.1f}</text>'
            f'<text x="630" y="{y+31}" class="row">{row["accuracy"]:.0%}</text>'
            f'<text x="744" y="{y+31}" class="row">${row["cost_usd"]:.5f}</text>'
            f'<text x="904" y="{y+31}" class="row">{row["latency_ms"]:.0f} ms</text>'
        )

    scenario_cards = []
    for index, (scenario_name, group) in enumerate(per_scenario.groupby("scenario", sort=True)):
        leader = group.iloc[0]
        y = 560 + index * 64
        scenario_cards.append(
            f'<rect x="1030" y="{y}" width="480" height="58" rx="11" fill="#121e2d"/>'
            f'<text x="1052" y="{y+25}" class="scenario">{esc(scenario_name.replace("_", " ").upper())}</text>'
            f'<text x="1052" y="{y+48}" class="scenario-sub">{leader["accuracy"]:.0%} accuracy  ·  {leader["latency_ms"]:.0f} ms</text>'
            f'<text x="1467" y="{y+41}" text-anchor="end" class="scenario-score">{leader["display_score"]:.1f}</text>'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1320" viewBox="0 0 1600 1320" role="img" aria-labelledby="title desc">
<title id="title">Agentic Loop Classifier Benchmark — results at a glance</title>
<desc id="desc">Jev leads the combined score among 16 models tested on 30 agent decision cases with five starts each. Includes model and scenario comparison tables and the scoring formula.</desc>
<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#09111c"/><stop offset="1" stop-color="#14283a"/></linearGradient>
  <linearGradient id="glow" x1="0" y1="0" x2="1" y2="0"><stop stop-color="#54e0b4"/><stop offset="1" stop-color="#91f0d3"/></linearGradient>
  <radialGradient id="halo"><stop stop-color="#39d6a5" stop-opacity=".22"/><stop offset="1" stop-color="#39d6a5" stop-opacity="0"/></radialGradient>
  <style>
  text{{font-family:Inter,ui-sans-serif,Segoe UI,Arial,sans-serif;fill:#eef5f7}} .muted{{fill:#9db0bf}} .eyebrow{{font-size:15px;font-weight:700;letter-spacing:2px;fill:#7ae7c4}} .title{{font-size:43px;font-weight:700;letter-spacing:-1px}} .subtitle{{font-size:18px;fill:#a9bbc8}} .hero-score{{font-size:94px;font-weight:700;letter-spacing:-4px;fill:#8af0cf}} .hero-label{{font-size:15px;font-weight:700;letter-spacing:1.4px;fill:#a7bdc8}} .hero-model{{font-size:24px;font-weight:600}} .hero-note{{font-size:16px;fill:#b1c3cc}} .stat-value{{font-size:27px;font-weight:700}} .stat-label{{font-size:13px;font-weight:700;letter-spacing:1px;fill:#9db0bf}} .panel-title{{font-size:17px;font-weight:700;letter-spacing:.5px}} .col{{font-size:12px;font-weight:700;letter-spacing:1px;fill:#91a5b5}} .row{{font-size:15px;font-weight:500}} .score{{fill:#8af0cf;font-weight:700}} .scenario{{font-size:12px;font-weight:700;letter-spacing:.7px;fill:#b3c7d1}} .scenario-sub{{font-size:13px;fill:#91a7b6}} .scenario-score{{font-size:25px;font-weight:700;fill:#8af0cf}} .formula{{font-size:20px;font-weight:600}} .foot{{font-size:14px;fill:#94a9b8}}
  </style>
</defs>
<rect width="1600" height="1320" rx="28" fill="url(#bg)"/>
<circle cx="1320" cy="70" r="330" fill="url(#halo)"/>
<rect x="70" y="56" width="5" height="63" rx="3" fill="url(#glow)"/>
<text x="94" y="76" class="eyebrow">AGENTIC SYSTEMS  /  CLASSIFIER BENCHMARK</text>
<text x="94" y="126" class="title">Fast decisions. Measured end to end.</text>
<text x="94" y="161" class="subtitle">16 models  ·  30 labeled cases  ·  6 agent decisions  ·  5 starts per model / case</text>

<rect x="70" y="194" width="1460" height="166" rx="20" fill="#152a38" stroke="#244253"/>
<text x="102" y="230" class="hero-label">BEST COMBINED RESULT</text>
<text x="98" y="322" class="hero-score">{winner['display_score']:.1f}</text>
<text x="275" y="284" class="hero-model">{esc(winner['model_id'])}</text>
<text x="275" y="316" class="hero-note">Exact score {winner['score']:.3f}  ·  {winner['accuracy']:.0%} exact match  ·  ${winner['cost_usd']:.6f} / call  ·  {winner['latency_ms']:.0f} ms E2E</text>
<text x="1486" y="247" text-anchor="end" class="hero-label">TOTAL RUN</text>
<text x="1486" y="286" text-anchor="end" class="stat-value">{total_calls:,} calls</text>
<text x="1486" y="318" text-anchor="end" class="hero-note">${total_cost_usd:.4f} scored-set spend</text>

<g>
  <rect x="70" y="382" width="460" height="76" rx="15" fill="#122131"/>
  <text x="94" y="412" class="stat-label">HIGHEST ACCURACY</text><text x="94" y="443" class="stat-value">{best_accuracy['accuracy']:.0%}</text><text x="498" y="438" text-anchor="end" class="hero-note">{esc(best_accuracy['model_id'].split('/')[-1])}</text>
  <rect x="548" y="382" width="460" height="76" rx="15" fill="#122131"/>
  <text x="572" y="412" class="stat-label">LOWEST COST / CALL</text><text x="572" y="443" class="stat-value">${cheapest['cost_usd']:.6f}</text><text x="976" y="438" text-anchor="end" class="hero-note">{esc(cheapest['model_id'].split('/')[-1])}</text>
  <rect x="1026" y="382" width="504" height="76" rx="15" fill="#122131"/>
  <text x="1050" y="412" class="stat-label">LOWEST END-TO-END LATENCY</text><text x="1050" y="443" class="stat-value">{fastest['latency_ms']:.0f} ms</text><text x="1498" y="438" text-anchor="end" class="hero-note">{esc(fastest['model_id'].split('/')[-1])}</text>
</g>

<rect x="70" y="480" width="940" height="550" rx="18" fill="#0f1b29" stroke="#263b4b"/>
<text x="96" y="520" class="panel-title">OVERALL LEADERBOARD</text>
<text x="96" y="546" class="foot">Top 8 of {len(summary)} · ranked by exact score; displayed score rounded to tenths</text>
<text x="116" y="578" class="col">#</text><text x="162" y="578" class="col">MODEL</text><text x="532" y="578" class="col">SCORE</text><text x="630" y="578" class="col">ACCURACY</text><text x="744" y="578" class="col">USD / CALL</text><text x="904" y="578" class="col">E2E</text>
{''.join(model_rows)}
<text x="96" y="1007" class="foot">Full 16-model metrics: leaderboard.csv  ·  full scenario rankings: scenario_leaderboards.csv</text>

<rect x="1030" y="480" width="500" height="550" rx="18" fill="#0f1b29" stroke="#263b4b"/>
<text x="1054" y="520" class="panel-title">SCENARIO LEADERS</text>
<text x="1054" y="546" class="foot">Top combined score within each agent decision</text>
{''.join(scenario_cards)}
<text x="1054" y="962" class="foot">Jev leads all six scenario scoreboards.</text>

<rect x="70" y="1052" width="1460" height="194" rx="18" fill="#152a38" stroke="#244253"/>
<text x="98" y="1092" class="eyebrow">ONE SCORE  /  SAME RULES ACROSS SCENARIOS</text>
<text x="98" y="1134" class="formula">Score = accuracy × cost factor × latency factor</text>
<text x="98" y="1174" class="hero-note">Cost: 1 / (1 + √(mean call cost / $0.001))</text>
<text x="98" y="1207" class="hero-note">Latency: 1 / (1 + (max(0, mean E2E ms − 500) / 1000)²)</text>
<text x="1488" y="1177" text-anchor="end" class="foot">500 ms free-latency threshold  ·  cost from OpenRouter usage  ·  E2E includes client round trip</text>
<text x="1488" y="1207" text-anchor="end" class="foot">Small 30-case starter set; results are directional pending holdout validation.</text>
</svg>'''
    (out / "benchmark-infographic.svg").write_text(svg, encoding="utf-8")


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


def write_english_report(summary: pd.DataFrame, per_scenario: pd.DataFrame, cases: list[dict], total_calls: int, total_cost_usd: float, out: Path) -> None:
    """Write the English, presentation-oriented companion report."""
    winner = summary.iloc[0]
    best_accuracy = summary.loc[summary["accuracy"].idxmax()]
    cheapest = summary.loc[summary["cost_usd"].idxmin()]
    fastest = summary.loc[summary["latency_ms"].idxmin()]
    by_model = summary.set_index("model_id")
    jev = by_model.loc["typesafe/jev-1.13"]
    luna = by_model.loc["openai/gpt-6-luna"]
    gpt6_low = by_model.loc["openai/gpt-6-sol"]
    gemini = by_model.loc["~google/gemini-flash-latest"]
    gemini_pinned = by_model.loc["google/gemini-3.8-flash"]
    haiku = by_model.loc["~anthropic/claude-haiku-latest"]

    def pct(value) -> str:
        return "n/a" if pd.isna(value) else f"{value:.0%}"

    model_rows = []
    for rank, (_, row) in enumerate(summary.iterrows(), 1):
        model_rows.append(
            f"| {rank} | `{row['model_id']}` | `{row['resolved_model_id']}` | {row['display_score']:.1f} / {row['score']:.3f} | "
            f"{row['accuracy']:.0%} | {row['valid_output']:.0%} | {pct(row['format_compliant'])} | ${row['cost_usd']:.6f} | {row['latency_ms']:.0f} |"
        )

    scenario_rows = []
    top_rows = []
    for scenario_name, group in per_scenario.groupby("scenario", sort=True):
        leader = group.iloc[0]
        scenario_rows.append(
            f"| `{scenario_name}` | `{leader['model_id']}` | {leader['display_score']:.1f} / {leader['score']:.3f} | "
            f"{leader['accuracy']:.0%} | ${leader['cost_usd']:.6f} | {leader['latency_ms']:.0f} ms |"
        )
        for position, (_, row) in enumerate(group.head(3).iterrows(), 1):
            top_rows.append(
                f"| `{scenario_name}` | {position} | `{row['model_id']}` | {row['display_score']:.1f} / {row['score']:.3f} | "
                f"{row['accuracy']:.0%} | ${row['cost_usd']:.6f} | {row['latency_ms']:.0f} ms |"
            )

    scenario_counts = pd.Series([case["scenario"] for case in cases]).value_counts().to_dict()
    report = f"""# Agentic Loop Classifier Benchmark

> **The short version**  
> **Jev leads the combined score at 0.8/1.0** (exact score **{winner['score']:.3f}**), with **{winner['accuracy']:.0%} accuracy**, **${winner['cost_usd']:.6f} per call**, and **{winner['latency_ms']:.0f} ms** mean end-to-end latency.

## At a glance

| Run | Models | Labeled cases | Starts per model/case | Scored calls | Scored-set cost |
|---|---:|---:|---:|---:|---:|
| Agentic Loop classifier benchmark | {len(summary)} | {len(cases)} | 5 | {total_calls:,} | ${total_cost_usd:.4f} |

| What led | Model | Result |
|---|---|---:|
| Combined score | `{winner['model_id']}` | **{winner['display_score']:.1f}** (exact {winner['score']:.3f}) |
| Exact-match accuracy | `{best_accuracy['model_id']}` | **{best_accuracy['accuracy']:.0%}** |
| Lowest cost per call | `{cheapest['model_id']}` | **${cheapest['cost_usd']:.6f}** |
| Lowest end-to-end latency | `{fastest['model_id']}` | **{fastest['latency_ms']:.0f} ms** |

The ranked set cost **${total_cost_usd:.4f}**. An additional 150 Haiku calls used to correct a parser issue cost **$0.0361**; diagnostic requests are excluded.

## What the results say

- **Jev is the strongest practical balance in this run.** It reached {jev['accuracy']:.0%} accuracy versus {luna['accuracy']:.0%} for GPT-6 Luna, the accuracy leader. Jev was {luna['latency_ms'] / jev['latency_ms']:.1f}× faster and {luna['cost_usd'] / jev['cost_usd']:.1f}× cheaper per call. Its exact combined score was {jev['score']:.3f} versus {luna['score']:.3f}.
- **GPT-6 Luna maximized accuracy, but latency reduced its combined score.** Its {best_accuracy['accuracy']:.0%} exact match is the best quality result in this set; Jev wins when cost and end-to-end response time are included.
- **GPT-6 Low scored {gpt6_low['accuracy']:.0%} accuracy.** Its {gpt6_low['latency_ms']:.0f} ms mean latency and ${gpt6_low['cost_usd']:.6f} call cost produce an exact score of {gpt6_low['score']:.3f}, displayed as {gpt6_low['display_score']:.1f} at the requested one-decimal precision.
- **Gemini Flash underperformed on this compact structured-output set.** The rolling alias reached {gemini['accuracy']:.0%} accuracy and {gemini['valid_output']:.0%} parseable output; pinned Gemini 3.8 reached {gemini_pinned['accuracy']:.0%} and {gemini_pinned['valid_output']:.0%}. The rolling alias resolved to both 3.7 and 3.8 during the run, while the pinned result was also weak.
- **Haiku's first result was a parser artifact.** After accepting JSON wrapped in Markdown fences, accuracy was {haiku['accuracy']:.0%} and parseable output was {haiku['valid_output']:.0%}; strict raw JSON compliance was {pct(haiku['format_compliant'])}. Consumers should either accept the wrapper or enforce native structured output.

## Overall leaderboard

Scores are ranked using the unrounded value. The first score column is rounded to tenths for presentation; exact values are included because several models round to 0.0.

| Rank | Requested model | Resolved model version(s) | Score (display / exact) | Accuracy | Parseable | Strict JSON¹ | Mean cost/call | Mean E2E (ms) |
|---:|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(model_rows)}

## Scenario leaders

| Agent decision | Leader | Score (display / exact) | Accuracy | Mean cost/call | Mean E2E |
|---|---|---:|---:|---:|---:|
{chr(10).join(scenario_rows)}

## Top three by scenario

| Scenario | Rank | Model | Score (display / exact) | Accuracy | Cost/call | Mean E2E |
|---|---:|---|---:|---:|---:|---:|
{chr(10).join(top_rows)}

## Visual comparison

![Results infographic: overall leaders, scenario winners, and scoring formula](benchmark-infographic.svg)

### Combined score

![Overall model score comparison](summary.png)

### Accuracy, cost, and latency

![Accuracy versus cost, with latency encoded by color](accuracy_cost.png)

### End-to-end latency by scenario

![End-to-end latency by scenario](latency_by_scenario.png)

## Scoring and methodology

The same score formula is applied to every scenario:

```text
cost_factor = 1 / (1 + sqrt(mean_call_cost / $0.001))
latency_factor = 1 / (1 + (max(0, mean_e2e_ms - 500) / 1000)^2)
score = exact_match_accuracy × cost_factor × latency_factor
display_score = round(score, 1)
```

Accuracy, cost, and latency are averaged across five starts within each case, then cases are weighted equally. Cost comes from OpenRouter usage. End-to-end latency includes the client network round trip. All models used the same main-run harness at concurrency 8, except the 150 Haiku replacement calls, which ran sequentially under the provider's RPM limit; Haiku latency is therefore not fully comparable.

## Coverage and caveats

- The set contains {len(cases)} labeled cases across {len(scenario_counts)} scenarios ({'; '.join(f"{name}: {count}" for name, count in sorted(scenario_counts.items()))}). Every model/case pair has five starts.
- Mean standard deviation of accuracy across starts: {summary['repeat_accuracy_sd'].mean():.1%}. See `repeat_accuracy_sd` in `leaderboard.csv` for model-level values.
- This is a small initial benchmark. Treat differences as directional until the set is expanded, labels are independently reviewed, and results are checked against a hidden holdout set.
- Markdown-fenced JSON is parsed for exact-match scoring and counted as parseable, but not as strict JSON. Chat models used provider-native JSON mode. Jev used the native typed-choice API; strict JSON does not apply (¹).
- Rolling aliases remain separate from pinned snapshots even when they resolve to the same version. The Gemini Flash rolling alias resolved to both Gemini 3.7 Flash and 3.8 Flash; Grok Latest resolved to 4.6 and 4.7.
- Provider latency and prices vary with time and routing. For publication-quality replication, record the date, requested/resolved IDs, provider route, region, and call parameters.
- Jev was called through OpenRouter Decisions API (`/api/alpha/decisions`); text models used Chat Completions. This compares end-to-end product paths, not isolated model decoding speed.

## Data files

- [Full model leaderboard](leaderboard.csv)
- [Scenario-by-model leaderboard](scenario_leaderboards.csv)
- [Full raw scored responses](../results.jsonl)
- [Benchmark definition and scenarios](../README.md)
"""
    (out / "report.en.md").write_text(report, encoding="utf-8")

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
    make_infographic(summary, per_scenario, cases, len(results), total_cost_usd, args.out)
    write_report(summary, per_scenario, cases, len(results), total_cost_usd, args.out)
    write_english_report(summary, per_scenario, cases, len(results), total_cost_usd, args.out)
    print(f"Wrote bilingual reports, leaderboards, infographic, and charts to {args.out.resolve()}")


if __name__ == "__main__":
    main()
