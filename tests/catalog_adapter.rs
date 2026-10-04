use evolving_benchmark::adapters::openrouter::catalog::OpenRouterCatalog;
use evolving_benchmark::ports::{CatalogQuery, ModelCatalog};
use wiremock::{
    matchers::{method, path},
    Mock, MockServer, ResponseTemplate,
};

#[tokio::test]
async fn normalizes_openrouter_catalog_and_keeps_raw_snapshot() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"data":[{
                "id":"vendor/model", "name":"Model", "created":1700000000,
                "architecture":{"input_modalities":["text"],"output_modalities":["text"]},
                "context_length":8192,"supported_parameters":["response_format"],
                "pricing":{"prompt":"0.000001","completion":"0.000002"}
            }]})),
        )
        .mount(&server)
        .await;
    let adapter = OpenRouterCatalog::new(&server.uri(), None).unwrap();
    let snapshot = adapter.list_models(&CatalogQuery::default()).await.unwrap();
    assert_eq!(snapshot.models[0].id, "vendor/model");
    assert!((snapshot.models[0].prompt_usd_per_million.unwrap() - 1.0).abs() < 1e-9);
    assert_eq!(snapshot.raw_json["data"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn rejects_malformed_catalog_without_echoing_credentials() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models"))
        .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
        .mount(&server)
        .await;
    let adapter = OpenRouterCatalog::new(&server.uri(), Some("secret-token".into())).unwrap();
    let error = adapter
        .list_models(&CatalogQuery::default())
        .await
        .unwrap_err()
        .to_string();
    assert!(!error.contains("secret-token"));
}
