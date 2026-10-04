use chrono::Utc;
use evolving_benchmark::adapters::openrouter::runner::OpenRouterRunner;
use evolving_benchmark::{
    domain::{model::ModelCandidate, suite::WorkUnit},
    ports::{ModelRequest, ModelRunner},
};
use serde_json::json;
use wiremock::{
    matchers::{header, method, path},
    Mock, MockServer, ResponseTemplate,
};

fn request() -> ModelRequest {
    ModelRequest {
        run_id: "run-test".into(),
        unit: WorkUnit {
            model_id: "vendor/model".into(),
            case_id: "TS-001".into(),
            repeat: 1,
        },
        model: ModelCandidate {
            id: "vendor/model".into(),
            name: "Model".into(),
            creator: "vendor".into(),
            created: Some(Utc::now().timestamp()),
            input_modalities: vec!["text".into()],
            output_modalities: vec!["text".into()],
            context_length: 8192,
            supported_parameters: vec!["response_format".into()],
            prompt_usd_per_million: Some(1.0),
            completion_usd_per_million: Some(2.0),
            exclusion_reasons: vec![],
        },
        protocol: "chat_completions".into(),
        scenario: "tool_selector".into(),
        input: json!({"goal":"find docs"}),
        output_schema: json!({"tool":["search_docs","none"]}),
        max_output_tokens: 160,
    }
}

#[tokio::test]
async fn parses_response_and_records_usage_and_resolved_model() {
    let server = MockServer::start().await;
    Mock::given(method("POST")).and(path("/api/v1/chat/completions"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model":"vendor/model:snapshot", "choices":[{"message":{"content":r#"{"tool":"search_docs"}"#}}],
            "usage":{"cost":0.0002,"prompt_tokens":10,"completion_tokens":5}
        }))).mount(&server).await;
    let runner = OpenRouterRunner::new(&format!("{}/api/v1", server.uri()), "test-key", 0).unwrap();
    let result = runner.execute(&request()).await.unwrap();
    assert_eq!(result.model_id, "vendor/model:snapshot");
    assert_eq!(result.cost_usd, Some(0.0002));
    assert_eq!(result.input_tokens, Some(10));
    assert_eq!(result.prediction.unwrap()["tool"], "search_docs");
}

#[tokio::test]
async fn missing_choices_is_a_permanent_error_without_secret_leak() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"oops":true})))
        .mount(&server)
        .await;
    let runner = OpenRouterRunner::new(&format!("{}/api/v1", server.uri()), "test-key", 0).unwrap();
    let error = runner.execute(&request()).await.unwrap_err().to_string();
    assert!(!error.contains("test-key"));
}

#[tokio::test]
async fn configured_decisions_protocol_maps_typed_choice_to_expected_field() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/alpha/decisions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model":"vendor/model:snapshot",
            "answers":{"decision":{"choice":"search_docs"}},
            "usage":{"cost":0.0001}
        })))
        .mount(&server)
        .await;
    let mut request = request();
    request.protocol = "openrouter_decisions".into();
    let runner = OpenRouterRunner::new(&format!("{}/api/v1", server.uri()), "test-key", 0).unwrap();
    let result = runner.execute(&request).await.unwrap();
    assert_eq!(result.prediction.unwrap()["tool"], "search_docs");
}
