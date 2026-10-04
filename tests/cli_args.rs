use clap::Parser;
use evolving_benchmark::{cli::Cli, config::AppConfig};

#[test]
fn plan_requires_a_positive_finite_budget() {
    assert!(Cli::try_parse_from(["evolving-benchmark", "plan", "decision"]).is_err());
    assert!(Cli::try_parse_from([
        "evolving-benchmark",
        "plan",
        "decision",
        "--budget-usd",
        "0"
    ])
    .is_err());
    assert!(Cli::try_parse_from([
        "evolving-benchmark",
        "plan",
        "decision",
        "--budget-usd",
        "NaN"
    ])
    .is_err());
}

#[test]
fn all_documented_commands_parse() {
    assert!(Cli::try_parse_from([
        "evolving-benchmark",
        "catalog",
        "decision",
        "--config",
        "benchmark.toml"
    ])
    .is_ok());
    assert!(Cli::try_parse_from([
        "evolving-benchmark",
        "run",
        "decision",
        "--config",
        "benchmark.toml",
        "--budget-usd",
        "5"
    ])
    .is_ok());
    assert!(Cli::try_parse_from(["evolving-benchmark", "resume", "run-123"]).is_ok());
    assert!(Cli::try_parse_from(["evolving-benchmark", "report", "run-123"]).is_ok());
}

#[test]
fn config_error_does_not_echo_secret_values() {
    let error = AppConfig::from_str("[openrouter]\napi_key = 'do-not-leak'\nunknown = [\n")
        .unwrap_err()
        .to_string();
    assert!(!error.contains("do-not-leak"));
}
