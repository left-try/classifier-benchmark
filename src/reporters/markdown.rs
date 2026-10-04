use crate::application::report::Report;

pub struct MarkdownReporter;

impl MarkdownReporter {
    pub fn render(report: &Report) -> String {
        let mut out = format!("# Decision Benchmark report\n\nRun: `{}`  \nDataset: `{}`  \nScoring: `{}`  \nStatus: `{:?}`  \nSpend: `${:.6}`\n\n",
            report.run_id, report.dataset, report.scoring_policy, report.status, report.total_cost_usd);
        if !report.rankings_available {
            out.push_str("**Rankings unavailable:** run coverage is incomplete or one or more calls have missing costs.\n\n");
        }
        out.push_str("| Rank | Requested model | Resolved model version(s) | Score | Accuracy | Valid output | Mean cost | Mean latency | Calls |\n|---:|---|---|---:|---:|---:|---:|---:|---:|\n");
        for model in &report.models {
            let rank = if report.rankings_available {
                model.rank.to_string()
            } else {
                "—".into()
            };
            out.push_str(&format!(
                "| {rank} | `{}` | `{}` | {} | {:.1}% | {:.1}% | {} | {:.0} ms | {}/{} |\n",
                model.model_id,
                model.resolved_model_ids.join(", "),
                model
                    .score
                    .map(|s| format!("{s:.4}"))
                    .unwrap_or_else(|| "—".into()),
                model.accuracy * 100.0,
                model.valid_output * 100.0,
                model
                    .mean_cost_usd
                    .map(|v| format!("${v:.6}"))
                    .unwrap_or_else(|| "—".into()),
                model.mean_latency_ms,
                model.calls,
                model.expected_calls
            ));
        }
        if !report.scenarios.is_empty() {
            out.push_str("\n## Scenario results\n\n| Scenario | Model | Accuracy | Mean cost | Mean latency | Calls |\n|---|---|---:|---:|---:|---:|\n");
            for scenario in &report.scenarios {
                out.push_str(&format!(
                    "| {} | `{}` | {:.1}% | {} | {:.0} ms | {} |\n",
                    scenario.scenario,
                    scenario.model_id,
                    scenario.accuracy * 100.0,
                    scenario
                        .mean_cost_usd
                        .map(|v| format!("${v:.6}"))
                        .unwrap_or_else(|| "—".into()),
                    scenario.mean_latency_ms,
                    scenario.calls
                ));
            }
        }
        if !report.failures.is_empty() {
            out.push_str("\n## Issues\n\n");
            for failure in &report.failures {
                out.push_str(&format!("- {failure}\n"));
            }
        }
        out
    }
}
