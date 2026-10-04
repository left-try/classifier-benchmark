use thiserror::Error;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("configuration error: {0}")]
    Config(String),
    #[error("request failed: {0}")]
    Request(String),
    #[error("run failed: {0}")]
    Run(String),
    #[error("storage error: {0}")]
    Storage(String),
}
