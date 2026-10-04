use assert_cmd::Command;
use serde_json::json;
use tempfile::tempdir;
use wiremock::{
    matchers::{method, path},
    Mock, MockServer, ResponseTemplate,
};

#[tokio::test]
async fn over_budget_plan_fetches_catalog_but_never_sends_completion_request() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data":[{
            "id":"provider/model", "name":"Model", "created":1800000000,
            "architecture":{"input_modalities":["text"],"output_modalities":["text"]},
            "context_length":8192,"supported_parameters":["response_format"],
            "pricing":{"prompt":"0.00001","completion":"0.00002"}
        }]})))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .expect(0)
        .mount(&server)
        .await;

    let temp = tempdir().unwrap();
    let cases = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data/decision/full/v1/cases.jsonl")
        .to_string_lossy()
        .replace('\\', "/");
    let output = temp
        .path()
        .join("runs")
        .to_string_lossy()
        .replace('\\', "/");
    let config = format!("suite=\"decision/full\"\ncases=\"{cases}\"\noutput_dir=\"{output}\"\nrepeats=1\nconcurrency=2\nmax_output_tokens=160\nmax_input_bytes=16384\nmax_retries=0\nseed=7\nmodel_spacing_ms=0\n\n[filters]\nmax_prompt_usd_per_million=100.0\nmax_completion_usd_per_million=100.0\n");
    let config_path = temp.path().join("benchmark.toml");
    std::fs::write(&config_path, config).unwrap();

    let mut command = Command::cargo_bin("evolving-benchmark").unwrap();
    command
        .env("OPENROUTER_BASE_URL", format!("{}/api/v1", server.uri()))
        .args(["plan", "decision", "--config"])
        .arg(config_path)
        .args(["--budget-usd", "0.0001"]);
    let assert = command.assert().failure();
    let stderr = String::from_utf8_lossy(&assert.get_output().stderr);
    assert!(
        stderr.contains("exceeds budget"),
        "unexpected stderr: {stderr}"
    );
}
