use crate::application::report::Report;
use std::{fs, path::Path};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CsvReportError {
    #[error("could not write report")]
    Io,
    #[error("could not serialize CSV")]
    Csv,
}

pub struct CsvReporter;

impl CsvReporter {
    pub fn write(report: &Report, output_dir: &Path) -> Result<(), CsvReportError> {
        fs::create_dir_all(output_dir).map_err(|_| CsvReportError::Io)?;
        let file =
            fs::File::create(output_dir.join("leaderboard.csv")).map_err(|_| CsvReportError::Io)?;
        let mut writer = csv::Writer::from_writer(file);
        writer
            .write_record([
                "rank",
                "model_id",
                "resolved_model_ids",
                "score",
                "accuracy",
                "valid_output",
                "mean_cost_usd",
                "mean_latency_ms",
                "calls",
                "expected_calls",
            ])
            .map_err(|_| CsvReportError::Csv)?;
        for model in &report.models {
            writer
                .write_record([
                    if report.rankings_available {
                        model.rank.to_string()
                    } else {
                        String::new()
                    },
                    model.model_id.clone(),
                    model.resolved_model_ids.join(";"),
                    model.score.map(|v| v.to_string()).unwrap_or_default(),
                    model.accuracy.to_string(),
                    model.valid_output.to_string(),
                    model
                        .mean_cost_usd
                        .map(|v| v.to_string())
                        .unwrap_or_default(),
                    model.mean_latency_ms.to_string(),
                    model.calls.to_string(),
                    model.expected_calls.to_string(),
                ])
                .map_err(|_| CsvReportError::Csv)?;
        }
        writer.flush().map_err(|_| CsvReportError::Io)
    }

    pub fn write_scenarios(report: &Report, output_dir: &Path) -> Result<(), CsvReportError> {
        fs::create_dir_all(output_dir).map_err(|_| CsvReportError::Io)?;
        let file = fs::File::create(output_dir.join("scenario_leaderboard.csv"))
            .map_err(|_| CsvReportError::Io)?;
        let mut writer = csv::Writer::from_writer(file);
        writer
            .write_record([
                "scenario",
                "model_id",
                "accuracy",
                "mean_cost_usd",
                "mean_latency_ms",
                "calls",
            ])
            .map_err(|_| CsvReportError::Csv)?;
        for row in &report.scenarios {
            writer
                .write_record([
                    row.scenario.clone(),
                    row.model_id.clone(),
                    row.accuracy.to_string(),
                    row.mean_cost_usd.map(|v| v.to_string()).unwrap_or_default(),
                    row.mean_latency_ms.to_string(),
                    row.calls.to_string(),
                ])
                .map_err(|_| CsvReportError::Csv)?;
        }
        writer.flush().map_err(|_| CsvReportError::Io)
    }
}
