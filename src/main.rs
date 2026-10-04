use clap::Parser;
use evolving_benchmark::{
    adapters::{
        filesystem::FilesystemRunStore,
        openrouter::{catalog::OpenRouterCatalog, runner::OpenRouterRunner},
    },
    application::{
        catalog::{CatalogPolicy, CatalogQuery},
        execute::Executor,
        plan::RunPlan,
        report::Report,
    },
    cli::{Cli, Command},
    config::AppConfig,
    domain::suite::TestSuite,
    ports::{load_api_key, ModelCatalog, RunStore},
    reporters::{csv::CsvReporter, markdown::MarkdownReporter},
};
use std::{error::Error, fs, path::Path};

const DEFAULT_API: &str = "https://openrouter.ai/api/v1";

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    if let Err(error) = dispatch(cli).await {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

async fn dispatch(cli: Cli) -> Result<(), Box<dyn Error>> {
    match cli.command {
        Command::Catalog(args) => {
            let (config, _suite) = load_config_suite(&args.config)?;
            let catalog = open_catalog()?;
            let snapshot = catalog
                .list_models(&CatalogQuery::from_filters(&config.filters))
                .await?;
            let decisions = CatalogPolicy::select(&snapshot, &config.filters, snapshot.fetched_at);
            let selected = decisions.iter().filter(|(_, d)| d.included).count();
            println!(
                "OpenRouter catalog fetched at {}: {} models, {} candidates",
                snapshot.fetched_at,
                snapshot.models.len(),
                selected
            );
            for (model, decision) in decisions {
                if decision.included {
                    println!(
                        "+ {}\t{}\tprompt=${:.4}/M\tcompletion=${:.4}/M",
                        model.id,
                        model.name,
                        model.prompt_usd_per_million.unwrap_or(0.0),
                        model.completion_usd_per_million.unwrap_or(0.0)
                    );
                } else {
                    println!(
                        "- {}\t{}\t{}",
                        model.id,
                        model.name,
                        decision.reasons.join("; ")
                    );
                }
            }
            let dir = Path::new(&config.output_dir).join("catalogs");
            fs::create_dir_all(&dir)?;
            let path = dir.join(format!(
                "{}.json",
                snapshot.fetched_at.format("%Y%m%dT%H%M%SZ")
            ));
            fs::write(&path, serde_json::to_vec_pretty(&snapshot)?)?;
            println!("Snapshot: {}", path.display());
        }
        Command::Plan(args) => {
            let (config, suite) = load_config_suite(&args.config)?;
            let snapshot = open_catalog()?
                .list_models(&CatalogQuery::from_filters(&config.filters))
                .await?;
            let models = selected_models(&snapshot, &config);
            let plan = RunPlan::build(&config, &suite, models, args.budget_usd)?;
            println!("Decision Benchmark dry run");
            println!(
                "Models: {} | cases: {} | repeats: {} | calls: {}",
                plan.models.len(),
                suite.cases.len(),
                config.repeats,
                plan.work_units.len()
            );
            println!(
                "Estimated maximum spend including retry reserves: ${:.6} | budget: ${:.6} | concurrency: {}",
                plan.estimated_total.as_usd(),
                plan.budget.as_usd(),
                plan.concurrency
            );
            for model in &plan.models {
                println!("  {}", model.id);
            }
            println!("No completion requests were sent.");
        }
        Command::Run(args) => {
            let (config, suite) = load_config_suite(&args.config)?;
            let key = load_api_key().ok_or("set OPENROUTER_API_KEY in environment or .env")?;
            let snapshot = open_catalog()?
                .list_models(&CatalogQuery::from_filters(&config.filters))
                .await?;
            let models = selected_models(&snapshot, &config);
            let plan = RunPlan::build(&config, &suite, models, args.budget_usd)?;
            println!(
                "Planned {} calls; retry-inclusive maximum estimate ${:.6} of ${:.6} budget.",
                plan.work_units.len(),
                plan.estimated_total.as_usd(),
                plan.budget.as_usd()
            );
            let store = FilesystemRunStore::new(&config.output_dir);
            let runner = OpenRouterRunner::new(&api_base(), key, config.max_retries)?;
            let effective_config = serde_json::to_value(&config)?;
            let summary =
                Executor::run(plan, &suite, &snapshot, &runner, &store, effective_config).await?;
            println!(
                "Run {}: {:?}, {}/{} calls complete, accounted cost ${:.6}",
                summary.run_id, summary.status, summary.completed, summary.total, summary.spent_usd
            );
        }
        Command::Resume(args) => {
            let store = FilesystemRunStore::new(&args.output_dir);
            let loaded = store.load(&args.run_id).await?;
            let config: AppConfig = serde_json::from_value(loaded.manifest.config.clone())?;
            let suite = load_suite(&config)?;
            let snapshot = store.catalog(&args.run_id).await?;
            let mut models = selected_models(&snapshot, &config);
            models.retain(|m| loaded.manifest.selected_models.contains(&m.id));
            let plan = RunPlan::build(&config, &suite, models, loaded.manifest.budget_usd)?;
            let key = load_api_key().ok_or("set OPENROUTER_API_KEY in environment or .env")?;
            let runner = OpenRouterRunner::new(&api_base(), key, config.max_retries)?;
            let summary = Executor::resume(loaded, plan, &suite, &runner, &store).await?;
            println!(
                "Run {}: {:?}, {}/{} calls complete, accounted cost ${:.6}",
                summary.run_id, summary.status, summary.completed, summary.total, summary.spent_usd
            );
        }
        Command::Report(args) => {
            let store = FilesystemRunStore::new(&args.output_dir);
            let loaded = store.load(&args.run_id).await?;
            let config: AppConfig = serde_json::from_value(loaded.manifest.config.clone())?;
            let suite = load_suite(&config)?;
            let report = Report::from_run(&loaded, &suite)?;
            let dir = args.output_dir.join(&args.run_id);
            fs::write(dir.join("report.md"), MarkdownReporter::render(&report))?;
            CsvReporter::write(&report, &dir)?;
            CsvReporter::write_scenarios(&report, &dir)?;
            println!("{}", MarkdownReporter::render(&report));
            println!("Reports saved under {}", dir.display());
        }
    }
    Ok(())
}

fn load_config_suite(path: &Path) -> Result<(AppConfig, TestSuite), Box<dyn Error>> {
    let config = AppConfig::load(path)?;
    let suite = load_suite(&config)?;
    Ok((config, suite))
}

fn load_suite(config: &AppConfig) -> Result<TestSuite, Box<dyn Error>> {
    if config.suite != "decision/full" {
        return Err(format!(
            "unknown Decision suite '{}'; available: decision/full",
            config.suite
        )
        .into());
    }
    Ok(TestSuite::from_file(
        "v1",
        "decision-full-v1",
        "decision-v1",
        &config.cases,
    )?)
}

fn selected_models(
    snapshot: &evolving_benchmark::domain::model::CatalogSnapshot,
    config: &AppConfig,
) -> Vec<evolving_benchmark::domain::model::ModelCandidate> {
    CatalogPolicy::select(snapshot, &config.filters, snapshot.fetched_at)
        .into_iter()
        .filter_map(|(model, decision)| decision.included.then_some(model))
        .collect()
}

fn api_base() -> String {
    std::env::var("OPENROUTER_BASE_URL").unwrap_or_else(|_| DEFAULT_API.into())
}

fn open_catalog() -> Result<OpenRouterCatalog, Box<dyn Error>> {
    Ok(OpenRouterCatalog::new(&api_base(), load_api_key())?)
}
