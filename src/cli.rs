use clap::{builder::ValueParser, Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Debug, Clone, Parser)]
#[command(
    name = "evolving-benchmark",
    version,
    about = "Evolving Benchmark — model benchmark CLI"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Clone, Subcommand)]
pub enum Command {
    Catalog(BenchmarkArgs),
    Plan(RunArgs),
    Run(RunArgs),
    Resume(ResumeArgs),
    Report(ResumeArgs),
}

#[derive(Debug, Clone, Args)]
pub struct BenchmarkArgs {
    pub benchmark: BenchmarkKind,
    #[arg(long, default_value = "benchmark.toml")]
    pub config: PathBuf,
}

#[derive(Debug, Clone, Args)]
pub struct RunArgs {
    pub benchmark: BenchmarkKind,
    #[arg(long, default_value = "benchmark.toml")]
    pub config: PathBuf,
    #[arg(long, value_parser = ValueParser::new(parse_budget))]
    pub budget_usd: f64,
}

#[derive(Debug, Clone, Args)]
pub struct ResumeArgs {
    pub run_id: String,
    #[arg(long, default_value = "runs")]
    pub output_dir: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum BenchmarkKind {
    Decision,
}

fn parse_budget(raw: &str) -> Result<f64, String> {
    let value = raw
        .parse::<f64>()
        .map_err(|_| "budget must be a positive finite number".to_owned())?;
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err("budget must be a positive finite number".to_owned())
    }
}
